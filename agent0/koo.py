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
from sensor_msgs.msg import Image as RosImage, PointCloud2, PointField, NavSatFix
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

# ==============================================================================
# -- Publishers ----------------------------------------------------------------
# ==============================================================================

class CameraPublisher:
    """RGB 카메라 이미지를 ROS 토픽으로 발행"""
    def __init__(self, node: Node, topic_name: str):
        self.node = node
        self.bridge = CvBridge()
        self.pub = node.create_publisher(RosImage, topic_name, 10)
        self.node.get_logger().info(f"[Camera] Ready -> {topic_name}")

    def handle(self, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(arr, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "camera_rgb"
        self.pub.publish(msg)

class SemanticDualPublisher:
    """
    시맨틱 카메라 데이터를 두 가지 형태로 발행:
    1. /image_raw   -> Lane Detection 알고리즘용 (ID값 그대로)
    2. /image_color -> 사람 눈으로 보는 확인용 (CityScapes 컬러)
    """
    def __init__(self, node: Node, base_topic: str):
        self.node = node
        self.bridge = CvBridge()
        # 토픽 두 개 생성
        self.pub_raw = node.create_publisher(RosImage, base_topic + "/image_raw", 10)
        self.pub_color = node.create_publisher(RosImage, base_topic + "/image_color", 10)
        self.node.get_logger().info(f"[Semantic] Ready -> {base_topic} (Raw & Color)")

    def handle(self, image: carla.Image):
        # 1. Raw 데이터 추출 (ID값 보존) - 중요! 변환 전에 먼저 복사해야 함
        # CARLA 이미지는 BGRA 순서이며, 시맨틱 ID는 R(2번) 채널에 들어있음
        raw_arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        raw_arr = raw_arr.reshape((image.height, image.width, 4))[:, :, :3] # BGR
        
        # Raw 메시지 발행
        msg_raw = self.bridge.cv2_to_imgmsg(raw_arr, encoding="bgr8")
        msg_raw.header.stamp = self.node.get_clock().now().to_msg()
        msg_raw.header.frame_id = "camera_semantic_raw"
        self.pub_raw.publish(msg_raw)

        # 2. Color 변환 (사람 확인용)
        image.convert(carla.ColorConverter.CityScapesPalette)
        color_arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        color_arr = color_arr.reshape((image.height, image.width, 4))[:, :, :3]

        # Color 메시지 발행
        msg_color = self.bridge.cv2_to_imgmsg(color_arr, encoding="bgr8")
        msg_color.header.stamp = self.node.get_clock().now().to_msg()
        msg_color.header.frame_id = "camera_semantic_color"
        self.pub_color.publish(msg_color)

class LidarPublisher:
    def __init__(self, node: Node, topic_name: str):
        self.node = node
        self.pub = node.create_publisher(PointCloud2, topic_name, 10)
        self.node.get_logger().info(f"[LiDAR] Ready -> {topic_name}")

    def handle(self, carla_lidar_measurement):
        header = self.node.get_clock().now().to_msg()
        lidar_bytes = carla_lidar_measurement.raw_data
        num_points = len(lidar_bytes) // 16 
        
        msg = PointCloud2()
        msg.header.stamp = header
        msg.header.frame_id = "lidar"
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

class GnssPublisher:
    def __init__(self, node: Node, topic_name: str):
        self.node = node
        self.pub = node.create_publisher(NavSatFix, topic_name, 10)
        self.node.get_logger().info(f"[GNSS] Ready -> {topic_name}")

    def handle(self, gnss):
        msg = NavSatFix()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "gnss"
        msg.latitude = gnss.latitude
        msg.longitude = gnss.longitude
        msg.altitude = gnss.altitude
        self.pub.publish(msg)

# ==============================================================================
# -- Setup Helpers -------------------------------------------------------------
# ==============================================================================

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

    # 핸들러 생성
    rgb_pubs = {}
    
    # [수정] Dual Publisher 사용 (Raw + Color)
    # 기본 경로: /carla/hero/camera_semantic_segmentation
    # 실제 토픽: .../image_raw, .../image_color
    semantic_handler = SemanticDualPublisher(node, "/carla/hero/camera_semantic_segmentation")
    
    lidar_pub = LidarPublisher(node, "/carla/hero/lidar/point_cloud")
    gnss_pub = GnssPublisher(node, "/carla/hero/gnss")

    for sensor_conf in sensors_config:
        sType = sensor_conf.get("type")
        sID = sensor_conf.get("id")
        
        bp = bp_library.filter(sType)[0]
        for k, v in sensor_conf.get("attributes", {}).items():
            bp.set_attribute(str(k), str(v))

        sp = sensor_conf.get("spawn_point")
        tr = carla.Transform(
            carla.Location(x=sp["x"], y=-sp["y"], z=sp["z"]),
            carla.Rotation(roll=sp["roll"], pitch=-sp["pitch"], yaw=-sp["yaw"])
        )

        sensor_actor = world.spawn_actor(bp, tr, attach_to=vehicle)
        actors.append(sensor_actor)

        if sType.startswith("sensor.camera.rgb"):
            topic = f"/carla/hero/{sID}/image_color"
            rgb_pubs[sID] = CameraPublisher(node, topic)
            sensor_actor.listen(lambda data, p=rgb_pubs[sID]: p.handle(data))
        
        elif sType == "sensor.camera.semantic_segmentation":
            sensor_actor.listen(lambda data: semantic_handler.handle(data))
            
        elif sType == "sensor.lidar.ray_cast":
            sensor_actor.listen(lambda data: lidar_pub.handle(data))
            
        elif sType == "sensor.other.gnss":
            sensor_actor.listen(lambda data: gnss_pub.handle(data))

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