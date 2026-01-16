#!/usr/bin/env python3

import argparse
import json
import logging
import time
import math
import os
import numpy as np
import carla
import cv2

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image as RosImage, PointCloud2, PointField, NavSatFix, Imu
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# ==============================================================================
# -- Publishers ----------------------------------------------------------------
# ==============================================================================

class CameraPublisher:
    """RGB 카메라 이미지를 ROS 토픽으로 발행"""
    def __init__(self, node: Node, topic_name: str, frame_id: str):
        self.node = node
        self.bridge = CvBridge()
        self.pub = node.create_publisher(RosImage, topic_name, 10)
        self.frame_id = frame_id
        self.node.get_logger().info(f"[Camera] Ready -> {topic_name}")

    def handle(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(arr, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

class SemanticDualPublisher:
    """
    시맨틱 카메라 데이터를 두 가지 형태로 발행:
    1. /image_raw   -> Lane Detection 알고리즘용 (ID값 그대로)
    2. /image_color -> 사람 눈으로 보는 확인용 (CityScapes 컬러)
    """
    def __init__(self, node: Node, base_topic: str, frame_id: str):
        self.node = node
        self.bridge = CvBridge()
        # 토픽 두 개 생성
        self.pub_raw = node.create_publisher(RosImage, base_topic + "/image_raw", 10)
        self.pub_color = node.create_publisher(RosImage, base_topic + "/image_color", 10)
        self.frame_id = frame_id
        self.node.get_logger().info(f"[Semantic] Ready -> {base_topic} (Raw & Color)")

    def handle(self, image: carla.Image):
        # 1. Raw 데이터 추출 (ID값 보존) - 중요! 변환 전에 먼저 복사해야 함
        # CARLA 이미지는 BGRA 순서이며, 시맨틱 ID는 R(2번) 채널에 들어있음
        raw_arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        raw_arr = raw_arr.reshape((image.height, image.width, 4))[:, :, :3] # BGR
        
        # Raw 메시지 발행
        msg_raw = self.bridge.cv2_to_imgmsg(raw_arr, encoding="bgr8")
        msg_raw.header.stamp = self.node.get_clock().now().to_msg()
        # 같은 센서 프레임으로 유지 (TF 설정이 단순해짐)
        msg_raw.header.frame_id = self.frame_id
        self.pub_raw.publish(msg_raw)

        # 2. Color 변환 (사람 확인용)
        image.convert(carla.ColorConverter.CityScapesPalette)
        color_arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        color_arr = color_arr.reshape((image.height, image.width, 4))[:, :, :3]

        # Color 메시지 발행
        msg_color = self.bridge.cv2_to_imgmsg(color_arr, encoding="bgr8")
        msg_color.header.stamp = self.node.get_clock().now().to_msg()
        msg_color.header.frame_id = self.frame_id
        self.pub_color.publish(msg_color)

class LidarPublisher:
    def __init__(self, node: Node, topic_name: str, frame_id: str):
        self.node = node
        self.pub = node.create_publisher(PointCloud2, topic_name, 10)
        self.frame_id = frame_id
        self.node.get_logger().info(f"[LiDAR] Ready -> {topic_name}")

    def handle(self, carla_lidar_measurement):
        header = self.node.get_clock().now().to_msg()
        lidar_bytes = carla_lidar_measurement.raw_data
        num_points = len(lidar_bytes) // 16 
        
        msg = PointCloud2()
        msg.header.stamp = header
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = num_points
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * num_points
        msg.is_dense = False
        msg.data = bytes(lidar_bytes)
        self.pub.publish(msg)


class SemanticLidarPublisher:
    """Semantic LiDAR를 PointCloud2로 발행 (x,y,z,cos_incidence,obj_idx,obj_tag)"""
    def __init__(self, node: Node, topic_name: str, frame_id: str):
        self.node = node
        self.pub = node.create_publisher(PointCloud2, topic_name, 10)
        self.frame_id = frame_id
        self.node.get_logger().info(f"[SemanticLiDAR] Ready -> {topic_name}")

    def handle(self, semantic_lidar_measurement):
        data = semantic_lidar_measurement.raw_data
        # CARLA docs: per point = x,y,z,cos_incidence (float32 x4) + obj_idx(uint32) + obj_tag(uint32)
        point_step = 24
        num_points = len(data) // point_step

        msg = PointCloud2()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = num_points
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * num_points
        msg.is_dense = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='cos_incidence', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='obj_idx', offset=16, datatype=PointField.UINT32, count=1),
            PointField(name='obj_tag', offset=20, datatype=PointField.UINT32, count=1),
        ]
        msg.data = bytes(data)
        self.pub.publish(msg)


class RadarPublisher:
    """Radar를 PointCloud2로 발행 (x,y,z,velocity)"""
    def __init__(self, node: Node, topic_name: str, frame_id: str):
        self.node = node
        self.pub = node.create_publisher(PointCloud2, topic_name, 10)
        self.frame_id = frame_id
        self.node.get_logger().info(f"[Radar] Ready -> {topic_name}")

    def handle(self, radar_data):
        # CARLA docs: raw_data -> float32 array shaped (N,4) = [vel, azimuth, altitude, depth]
        if len(radar_data) == 0:
            return

        pts = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        pts = np.reshape(pts, (len(radar_data), 4))
        vel = pts[:, 0]
        az = pts[:, 1]
        alt = pts[:, 2]
        depth = pts[:, 3]

        # Polar -> Cartesian
        x = depth * np.cos(alt) * np.cos(az)
        y = depth * np.cos(alt) * np.sin(az)
        z = depth * np.sin(alt)

        cloud = np.stack([x, y, z, vel], axis=1).astype(np.float32)
        data_bytes = cloud.tobytes()
        num_points = cloud.shape[0]

        msg = PointCloud2()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = num_points
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * num_points
        msg.is_dense = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='velocity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = data_bytes
        self.pub.publish(msg)

class GnssPublisher:
    def __init__(self, node: Node, topic_name: str, frame_id: str):
        self.node = node
        self.pub = node.create_publisher(NavSatFix, topic_name, 10)
        self.frame_id = frame_id
        self.node.get_logger().info(f"[GNSS] Ready -> {topic_name}")

    def handle(self, gnss):
        msg = NavSatFix()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.latitude = gnss.latitude
        msg.longitude = gnss.longitude
        msg.altitude = gnss.altitude
        self.pub.publish(msg)


class ImuPublisher:
    def __init__(self, node: Node, topic_name: str, frame_id: str):
        self.node = node
        self.pub = node.create_publisher(Imu, topic_name, 10)
        self.frame_id = frame_id
        self.node.get_logger().info(f"[IMU] Ready -> {topic_name}")

    def handle(self, imu_data):
        msg = Imu()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.linear_acceleration.x = float(imu_data.accelerometer.x)
        msg.linear_acceleration.y = float(imu_data.accelerometer.y)
        msg.linear_acceleration.z = float(imu_data.accelerometer.z)

        msg.angular_velocity.x = float(imu_data.gyroscope.x)
        msg.angular_velocity.y = float(imu_data.gyroscope.y)
        msg.angular_velocity.z = float(imu_data.gyroscope.z)

        # Orientation은 제공하지 않음
        msg.orientation_covariance[0] = -1.0
        self.pub.publish(msg)

# ==============================================================================
# -- Setup Helpers -------------------------------------------------------------
# ==============================================================================

def _euler_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """ROS 기준 RPY(deg) -> quaternion(x,y,z,w)."""
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def _update_spectator_follow(world, vehicle, dist=7.0, height=3.0, pitch=-15.0):
    """
    CARLA 시뮬레이터 화면(Spectator) 카메라를 차량 뒤에서 따라가게 함.
    dist: 차량 뒤로 떨어질 거리(m)
    height: 지면 기준 추가 높이(m)
    pitch: 아래로 보는 각도(음수면 아래)
    """
    spectator = world.get_spectator()

    vt = vehicle.get_transform()
    forward = vt.get_forward_vector()

    # 차량 뒤쪽(-forward)으로 dist만큼, 위로 height만큼 이동
    cam_loc = vt.location + carla.Location(
        x=-dist * forward.x,
        y=-dist * forward.y,
        z=height
    )

    cam_rot = carla.Rotation(
        pitch=pitch,
        yaw=vt.rotation.yaw,
        roll=0.0
    )

    spectator.set_transform(carla.Transform(cam_loc, cam_rot))

def _setup_vehicle(world, config, reverse=False):
    logging.info(f"Spawning vehicle: {config.get('type')}")
    bp_library = world.get_blueprint_library()
    map_ = world.get_map()

    bp = bp_library.filter(config.get("type"))[0]
    bp.set_attribute("role_name", config.get("id"))

    spawn = map_.get_spawn_points()[0]
    if reverse:
        spawn.rotation.yaw += 180.0

    return world.spawn_actor(bp, spawn)

def _setup_sensors(world, vehicle, sensors_config, node):
    actors = []
    bp_library = world.get_blueprint_library()

    # base_link -> sensor_frame 정적 TF 발행
    static_tf_broadcaster = StaticTransformBroadcaster(node)
    static_tfs = []

    # 핸들러 생성
    rgb_pubs = {}
    sem_pubs = {}
    lidar_pubs = {}
    sem_lidar_pubs = {}
    radar_pubs = {}
    gnss_pubs = {}
    imu_pubs = {}

    for sensor_conf in sensors_config:
        sType = sensor_conf.get("type")
        sID = sensor_conf.get("id")

        bps = bp_library.filter(sType)
        if not bps:
            node.get_logger().error(f"[Bridge] Blueprint not found: {sType} (skip)")
            continue
        bp = bps[0]
        for k, v in sensor_conf.get("attributes", {}).items():
            bp.set_attribute(str(k), str(v))

        sp = sensor_conf.get("spawn_point")

        # (ROS 기준) base_link -> sensor_frame TF (config 값 그대로 사용)
        try:
            t = TransformStamped()
            t.header.stamp = node.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = sID
            t.transform.translation.x = float(sp["x"])
            t.transform.translation.y = float(sp["y"])
            t.transform.translation.z = float(sp["z"])
            qx, qy, qz, qw = _euler_deg_to_quat(sp["roll"], sp["pitch"], sp["yaw"])
            t.transform.rotation.x = float(qx)
            t.transform.rotation.y = float(qy)
            t.transform.rotation.z = float(qz)
            t.transform.rotation.w = float(qw)
            static_tfs.append(t)
        except Exception as e:
            node.get_logger().warning(f"[TF] Failed to build static TF for {sID}: {e}")

        tr = carla.Transform(
            carla.Location(x=sp["x"], y=-sp["y"], z=sp["z"]),
            carla.Rotation(roll=sp["roll"], pitch=-sp["pitch"], yaw=-sp["yaw"])
        )

        sensor_actor = world.spawn_actor(bp, tr, attach_to=vehicle)
        actors.append(sensor_actor)

        if sType.startswith("sensor.camera.rgb"):
            topic = f"/carla/hero/{sID}/image_color"
            rgb_pubs[sID] = CameraPublisher(node, topic, frame_id=sID)
            sensor_actor.listen(lambda data, p=rgb_pubs[sID]: p.handle(data))

        elif sType.startswith("sensor.camera.semantic_segmentation"):
            base_topic = f"/carla/hero/{sID}"
            sem_pubs[sID] = SemanticDualPublisher(node, base_topic, frame_id=sID)
            sensor_actor.listen(lambda data, p=sem_pubs[sID]: p.handle(data))


        elif sType == "sensor.lidar.ray_cast":
            topic = f"/carla/hero/{sID}/point_cloud"
            lidar_pubs[sID] = LidarPublisher(node, topic, frame_id=sID)
            sensor_actor.listen(lambda data, p=lidar_pubs[sID]: p.handle(data))

        elif sType == "sensor.lidar.ray_cast_semantic":
            topic = f"/carla/hero/{sID}/point_cloud"
            sem_lidar_pubs[sID] = SemanticLidarPublisher(node, topic, frame_id=sID)
            sensor_actor.listen(lambda data, p=sem_lidar_pubs[sID]: p.handle(data))

        elif sType == "sensor.other.radar":
            topic = f"/carla/hero/{sID}/point_cloud"
            radar_pubs[sID] = RadarPublisher(node, topic, frame_id=sID)
            sensor_actor.listen(lambda data, p=radar_pubs[sID]: p.handle(data))

        elif sType == "sensor.other.gnss":
            topic = f"/carla/hero/{sID}"
            gnss_pubs[sID] = GnssPublisher(node, topic, frame_id=sID)
            sensor_actor.listen(lambda data, p=gnss_pubs[sID]: p.handle(data))

        elif sType == "sensor.other.imu":
            topic = f"/carla/hero/{sID}"
            imu_pubs[sID] = ImuPublisher(node, topic, frame_id=sID)
            sensor_actor.listen(lambda data, p=imu_pubs[sID]: p.handle(data))

    # /tf_static는 latched 성격이라 1회 전송로 충분
    if static_tfs:
        static_tf_broadcaster.sendTransform(static_tfs)
        node.get_logger().info(f"[TF] Published {len(static_tfs)} static transforms (base_link -> sensors)")

    return actors

# ==============================================================================
# -- Main ----------------------------------------------------------------------
# ==============================================================================

def main(args):
    rclpy.init(args=None)
    node = rclpy.create_node("carla_ros2_native_bridge")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    with open(args.file) as f:
        config = json.load(f)

    vehicle = None
    sensors = []

    try:
        vehicle = _setup_vehicle(world, config, reverse=args.reverse)
        sensors = _setup_sensors(world, vehicle, config.get("sensors", []), node)

        def on_cmd(msg: Twist):
            v_curr = math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
            v_target = msg.linear.x
            steer = msg.angular.z / 35.0 
            brake = msg.linear.y 

            err = v_target - v_curr
            throttle = 0.0
            if err > 0: throttle = min(1.0, 0.5 * err)
            if brake > 0.0: throttle = 0.0
            steer = max(-1.0, min(1.0, steer))
            
            vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake)
            ))

        node.create_subscription(Twist, '/carla/hero/cmd_vel', on_cmd, 10)
        node.get_logger().info("[Bridge] Started. Publishing Raw & Color semantics.")

        clock_pub = node.create_publisher(Clock, '/clock', 10)

        while rclpy.ok():
            world.tick()

            _update_spectator_follow(world, vehicle)  # <-- 추가

            snapshot = world.get_snapshot()
            ros_time = Time(seconds=snapshot.timestamp.elapsed_seconds).to_msg()
            clock_msg = Clock()
            clock_msg.clock = ros_time
            clock_pub.publish(clock_msg)
            rclpy.spin_once(node, timeout_sec=0.001)

    except KeyboardInterrupt:
        pass
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for s in sensors:
            if s.is_alive: s.destroy()
        if vehicle and vehicle.is_alive: vehicle.destroy()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--host', default='localhost')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('-f', '--file', required=True)
    argparser.add_argument('--reverse', action='store_true')
    
    args = argparser.parse_args()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    main(args)