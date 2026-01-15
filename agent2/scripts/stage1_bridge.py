#!/usr/bin/env python3
"""
Stage 1 bridge:
- Spawn ego vehicle + sensors from a JSON config.
- Publish sensor data to ROS2 topics.
- Subscribe /carla/hero/cmd_vel (Twist) and apply VehicleControl.
- Tick CARLA in synchronous mode and publish /clock.

Tested design target: CARLA 0.9.14+ style semantic tag table and semantic LiDAR fields.

Run (example):
  source /opt/ros/humble/setup.bash
  python3 scripts/stage1_bridge.py --config config/stage1_oracle_sensors.json

Then run autonomy:
  source /opt/ros/humble/setup.bash
  python3 scripts/stage1_oracle_autonomy.py
"""

# Allow running as either `python3 -m scripts.<module>` or `python3 scripts/<file>.py`
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import carla

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image as RosImage, PointCloud2, PointField, NavSatFix, Imu

from scripts.common.pc2_utils import (
    make_pointcloud2, fields_xyz_intensity, fields_semantic_lidar, fields_radar
)

def carla_transform(spawn_point: dict) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=float(spawn_point.get("x", 0.0)),
                       y=float(spawn_point.get("y", 0.0)),
                       z=float(spawn_point.get("z", 0.0))),
        carla.Rotation(roll=float(spawn_point.get("roll", 0.0)),
                      pitch=float(spawn_point.get("pitch", 0.0)),
                      yaw=float(spawn_point.get("yaw", 0.0))),
    )

class Stage1Bridge(Node):
    def __init__(self, cfg_path: Path, host: str, port: int):
        super().__init__("carla_stage1_bridge")

        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.host = host
        self.port = port
        # Camera follow (helps you instantly find the ego vehicle in the CARLA window)
        self.follow_camera = bool(self.cfg.get("follow_camera", True))
        self.camera_mode = str(self.cfg.get("camera_mode", "topdown")).lower()  # "topdown" or "chase"
        self.camera_height = float(self.cfg.get("camera_height", 50.0))
        self.chase_distance = float(self.cfg.get("chase_distance", 8.0))


        self.bridge = CvBridge()

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        self.world = self._load_world_and_settings()
        self.bp_lib = self.world.get_blueprint_library()

        self.actors: list[carla.Actor] = []
        self.sensors: dict[str, carla.Sensor] = {}

        self.vehicle = self._spawn_vehicle()
        self._spawn_sensors()

        # Publishers
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_gnss: dict[str, any] = {}
        self.pub_imu: dict[str, any] = {}
        self.pub_cam: dict[str, any] = {}
        self.pub_sem: dict[str, any] = {}
        self.pub_inst: dict[str, any] = {}
        self.pub_lidar: dict[str, any] = {}
        self.pub_sem_lidar: dict[str, any] = {}
        self.pub_radar: dict[str, any] = {}

        # control subscription
        self.create_subscription(Twist, "/carla/hero/cmd_vel", self._on_cmd_vel, 10)

        # Setup publishers per sensor
        for s in self.cfg.get("sensors", []):
            if not s.get("enabled", True):
                continue
            sid = s["id"]
            stype = s["type"]
            if stype.startswith("sensor.camera.rgb"):
                self.pub_cam[sid] = self.create_publisher(RosImage, f"/carla/hero/{sid}/image", 10)
            elif stype.startswith("sensor.camera.semantic_segmentation"):
                self.pub_sem[sid] = {
                    "raw": self.create_publisher(RosImage, f"/carla/hero/{sid}/image_raw", 10),
                    "color": self.create_publisher(RosImage, f"/carla/hero/{sid}/image_color", 10),
                }
            elif stype.startswith("sensor.camera.instance_segmentation"):
                self.pub_inst[sid] = self.create_publisher(RosImage, f"/carla/hero/{sid}/image_raw", 10)
            elif stype == "sensor.lidar.ray_cast":
                self.pub_lidar[sid] = self.create_publisher(PointCloud2, f"/carla/hero/{sid}/points", 10)
            elif stype == "sensor.lidar.ray_cast_semantic":
                self.pub_sem_lidar[sid] = self.create_publisher(PointCloud2, f"/carla/hero/{sid}/points", 10)
            elif stype == "sensor.other.radar":
                self.pub_radar[sid] = self.create_publisher(PointCloud2, f"/carla/hero/{sid}/detections", 10)
            elif stype == "sensor.other.gnss":
                self.pub_gnss[sid] = self.create_publisher(NavSatFix, f"/carla/hero/{sid}", 10)
            elif stype == "sensor.other.imu":
                self.pub_imu[sid] = self.create_publisher(Imu, f"/carla/hero/{sid}", 10)

        self.get_logger().info("Stage1Bridge ready. Ticking CARLA...")

    def _load_world_and_settings(self) -> carla.World:
        world_cfg = self.cfg.get("world", {})
        town = world_cfg.get("town")
        if town:
            self.get_logger().info(f"Loading map: {town}")
            self.client.load_world(town)
        world = self.client.get_world()

        # weather
        weather_name = world_cfg.get("weather", "ClearNoon")
        try:
            w = getattr(carla.WeatherParameters, weather_name)
            world.set_weather(w)
        except Exception:
            self.get_logger().warn(f"Unknown weather '{weather_name}', keeping default.")

        # settings
        settings = world.get_settings()
        settings.synchronous_mode = bool(world_cfg.get("synchronous_mode", True))
        settings.fixed_delta_seconds = float(world_cfg.get("fixed_delta_seconds", 0.05))
        world.apply_settings(settings)

        # ensure server is in sync
        world.tick()
        return world

    def _spawn_vehicle(self) -> carla.Vehicle:
        vcfg = self.cfg["vehicle"]
        vtype = vcfg["type"]
        vid = vcfg.get("id", "hero")

        bp = self.bp_lib.find(vtype)
        for k, v in (vcfg.get("attributes") or {}).items():
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points on this map.")
        transform = spawn_points[0]
        vehicle = self.world.try_spawn_actor(bp, transform)
        if vehicle is None:
            # fall back: try a few
            for tr in spawn_points[1:20]:
                vehicle = self.world.try_spawn_actor(bp, tr)
                if vehicle:
                    break
        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle.")

        vehicle.set_autopilot(False)
        self.actors.append(vehicle)
        self.get_logger().info(f"Spawned vehicle: {vtype} id={vehicle.id}")
        return vehicle

    def _spawn_sensors(self) -> None:
        for s in self.cfg.get("sensors", []):
            if not s.get("enabled", True):
                continue
            stype = s["type"]
            sid = s["id"]
            bp = self.bp_lib.find(stype)
            for k, v in (s.get("attributes") or {}).items():
                if bp.has_attribute(k):
                    bp.set_attribute(k, str(v))
            tr = carla_transform(s["spawn_point"])
            sensor = self.world.spawn_actor(bp, tr, attach_to=self.vehicle)
            self.actors.append(sensor)
            self.sensors[sid] = sensor
            sensor.listen(self._make_sensor_callback(sid, stype))
            self.get_logger().info(f"Spawned sensor: {sid} ({stype})")

    def _make_sensor_callback(self, sid: str, stype: str):
        if stype.startswith("sensor.camera.rgb"):
            def cb(image: carla.Image):
                arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
                msg = self.bridge.cv2_to_imgmsg(arr, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = sid
                self.pub_cam[sid].publish(msg)
            return cb

        if stype.startswith("sensor.camera.semantic_segmentation"):
            def cb(image: carla.Image):
                # Raw (semantic tag is encoded in R channel of BGRA; we publish as BGR without conversion)
                raw_arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
                msg_raw = self.bridge.cv2_to_imgmsg(raw_arr, encoding="bgr8")
                msg_raw.header.stamp = self.get_clock().now().to_msg()
                msg_raw.header.frame_id = sid
                self.pub_sem[sid]["raw"].publish(msg_raw)

                # Color (CityScapesPalette)
                image.convert(carla.ColorConverter.CityScapesPalette)
                color_arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
                msg_color = self.bridge.cv2_to_imgmsg(color_arr, encoding="bgr8")
                msg_color.header.stamp = msg_raw.header.stamp
                msg_color.header.frame_id = sid + "_color"
                self.pub_sem[sid]["color"].publish(msg_color)
            return cb

        if stype.startswith("sensor.camera.instance_segmentation"):
            def cb(image: carla.Image):
                arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
                msg = self.bridge.cv2_to_imgmsg(arr, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = sid
                self.pub_inst[sid].publish(msg)
            return cb

        if stype == "sensor.lidar.ray_cast":
            fields, step = fields_xyz_intensity()
            def cb(meas):
                # lidar raw_data is float32 XYZI packed (16 bytes each point)
                header = self.get_clock().now().to_msg()
                msg = PointCloud2()
                msg.header.stamp = header
                msg.header.frame_id = sid
                msg.height = 1
                msg.width = int(len(meas.raw_data) // step)
                msg.fields = fields
                msg.is_bigendian = False
                msg.point_step = step
                msg.row_step = step * msg.width
                msg.is_dense = False
                msg.data = bytes(meas.raw_data)
                self.pub_lidar[sid].publish(msg)
            return cb

        if stype == "sensor.lidar.ray_cast_semantic":
            fields, step = fields_semantic_lidar()
            def cb(meas):
                # Pack detections ourselves to avoid relying on a specific raw_data layout.
                stamp = self.get_clock().now().to_msg()
                out = np.zeros((len(meas),), dtype=np.dtype([
                    ('x','<f4'),('y','<f4'),('z','<f4'),
                    ('cos','<f4'),
                    ('object_idx','<u4'),
                    ('object_tag','<u4'),
                ]))
                for i, det in enumerate(meas):
                    p = det.point
                    out['x'][i] = float(p.x)
                    out['y'][i] = float(p.y)
                    out['z'][i] = float(p.z)
                    out['cos'][i] = float(getattr(det, "cos_inc_angle", 0.0))
                    out['object_idx'][i] = int(getattr(det, "object_idx", 0))
                    out['object_tag'][i] = int(getattr(det, "object_tag", 0))

                pc2 = PointCloud2()
                pc2.header.stamp = stamp
                pc2.header.frame_id = sid
                pc2.height = 1
                pc2.width = int(out.shape[0])
                pc2.fields = fields
                pc2.is_bigendian = False
                pc2.point_step = step
                pc2.row_step = step * pc2.width
                pc2.is_dense = False
                pc2.data = out.tobytes()
                self.pub_sem_lidar[sid].publish(pc2)
            return cb

        if stype == "sensor.other.radar":
            fields, step = fields_radar()
            def cb(radar_data):
                header = self.get_clock().now().to_msg()
                # doc: numpy [[vel, azimuth, altitude, depth]]
                pts = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')).reshape((len(radar_data), 4))
                pc2 = PointCloud2()
                pc2.header.stamp = header
                pc2.header.frame_id = sid
                pc2.height = 1
                pc2.width = int(pts.shape[0])
                pc2.fields = fields
                pc2.is_bigendian = False
                pc2.point_step = step
                pc2.row_step = step * pc2.width
                pc2.is_dense = False
                pc2.data = pts.astype('<f4').tobytes()
                self.pub_radar[sid].publish(pc2)
            return cb

        if stype == "sensor.other.gnss":
            def cb(meas):
                msg = NavSatFix()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = sid
                msg.latitude = float(meas.latitude)
                msg.longitude = float(meas.longitude)
                msg.altitude = float(meas.altitude)
                self.pub_gnss[sid].publish(msg)
            return cb

        if stype == "sensor.other.imu":
            def cb(meas):
                msg = Imu()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = sid
                # Orientation is not directly provided (only compass), so leave orientation unset.
                msg.angular_velocity.x = float(meas.gyroscope.x)
                msg.angular_velocity.y = float(meas.gyroscope.y)
                msg.angular_velocity.z = float(meas.gyroscope.z)
                msg.linear_acceleration.x = float(meas.accelerometer.x)
                msg.linear_acceleration.y = float(meas.accelerometer.y)
                msg.linear_acceleration.z = float(meas.accelerometer.z)
                self.pub_imu[sid].publish(msg)
            return cb

        # Fallback for unhandled sensor types.
        def cb(_):
            return
        return cb

    def _on_cmd_vel(self, msg: Twist):
        # Same convention as your existing sensor_setup.py:
        # linear.x: target speed (m/s), linear.y: brake (0..1), angular.z: steering degrees (-35..35)
        v_curr = self.vehicle.get_velocity()
        v_curr_mps = math.sqrt(v_curr.x**2 + v_curr.y**2 + v_curr.z**2)
        v_target = float(msg.linear.x)
        steer = float(msg.angular.z) / 35.0
        brake = float(msg.linear.y)

        err = v_target - v_curr_mps
        throttle = 0.0
        if err > 0:
            throttle = min(1.0, 0.5 * err)
        if brake > 0.0:
            throttle = 0.0

        steer = max(-1.0, min(1.0, steer))
        brake = max(0.0, min(1.0, brake))

        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
        ))


    def _update_spectator(self):
        if not getattr(self, "follow_camera", False):
            return
        if self.vehicle is None:
            return
        spec = self.world.get_spectator()
        tr = self.vehicle.get_transform()

        mode = getattr(self, "camera_mode", "topdown")
        h = float(getattr(self, "camera_height", 50.0))
        d = float(getattr(self, "chase_distance", 8.0))

        if mode == "chase":
            fwd = tr.get_forward_vector()
            loc = tr.location - fwd * d + carla.Location(z=h)
            rot = carla.Rotation(pitch=-15.0, yaw=tr.rotation.yaw, roll=0.0)
        else:
            loc = tr.location + carla.Location(z=h)
            rot = carla.Rotation(pitch=-90.0, yaw=tr.rotation.yaw, roll=0.0)

        spec.set_transform(carla.Transform(loc, rot))

    def spin(self):
        try:
            while rclpy.ok():
                self.world.tick()
                self._update_spectator()
                snap = self.world.get_snapshot()
                ros_time = Time(seconds=snap.timestamp.elapsed_seconds).to_msg()
                clk = Clock()
                clk.clock = ros_time
                self.pub_clock.publish(clk)
                rclpy.spin_once(self, timeout_sec=0.001)
        finally:
            self.cleanup()

    def cleanup(self):
        self.get_logger().info("Cleaning up actors...")
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass
        for a in reversed(self.actors):
            try:
                a.destroy()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=2000, type=int)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "stage1_oracle_sensors.json"))
    args = ap.parse_args()

    rclpy.init()
    node = Stage1Bridge(Path(args.config), args.host, args.port)
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()