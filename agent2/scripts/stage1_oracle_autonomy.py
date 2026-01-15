#!/usr/bin/env python3
"""
Stage 1 "oracle" autonomy node:
- Lane/road following from SEMANTIC SEGMENTATION (raw tag ID in R channel).
- Obstacle-aware speed control from SEMANTIC LiDAR (object_tag + distance).
- Publish Twist to /carla/hero/cmd_vel (compatible with stage1_bridge.py & your sensor_setup.py).

Algorithm (intentionally simple, for "make it drive"):
1) Semantic segmentation: take bottom ROI, use 'Roads' pixels -> compute centroid x -> steering.
2) Semantic LiDAR: in a forward corridor -> min distance to obstacle tags -> set speed/brake.

Topics expected (from stage1_bridge.py):
  /carla/hero/sem_front/image_raw
  /carla/hero/sem_lidar/points

Run:
  source /opt/ros/humble/setup.bash
  python3 scripts/stage1_oracle_autonomy.py
"""

# Allow running as either `python3 -m scripts.<module>` or `python3 scripts/<file>.py`
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import math
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image as RosImage, PointCloud2

from scripts.common.semantic_tags import Tags, OBSTACLE_TAGS
from scripts.common.pc2_utils import read_pointcloud2_semantic


class OracleAutonomy(Node):
    def __init__(self):
        super().__init__("stage1_oracle_autonomy")
        self.bridge = CvBridge()

        # Parameters (tune in code first; later convert to ROS params if needed)
        self.roi_y_start_ratio = 0.60   # bottom 40%
        self.steer_gain = 1.2           # proportional steering
        self.steer_deadband_px = 8      # ignore tiny offset
        self.max_steer_deg = 35.0

        self.v_cruise = 8.0             # m/s
        self.v_follow = 4.0             # m/s
        self.stop_distance = 7.0        # meters
        self.follow_distance = 20.0     # meters
        self.brake_hard = 0.8

        self.last_steer = 0.0
        self.latest_sem = None          # (H,W) uint8 tag image
        self.latest_sem_time = self.get_clock().now()

        self.latest_sem_lidar = None    # structured array
        self.latest_lidar_time = self.get_clock().now()

        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        self.create_subscription(RosImage, "/carla/hero/sem_front/image_raw", self.on_sem, 10)
        self.create_subscription(PointCloud2, "/carla/hero/sem_lidar/points", self.on_sem_lidar, 10)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info("OracleAutonomy ready.")

    def on_sem(self, msg: RosImage):
        # msg is bgr8; semantic tag is encoded in RED channel of original BGRA.
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        tag = img_bgr[:, :, 2].astype(np.uint8)   # R channel
        self.latest_sem = tag
        self.latest_sem_time = self.get_clock().now()

    def on_sem_lidar(self, msg: PointCloud2):
        self.latest_sem_lidar = read_pointcloud2_semantic(msg)
        self.latest_lidar_time = self.get_clock().now()

    def compute_steer_from_sem(self) -> float:
        if self.latest_sem is None:
            return self.last_steer

        tag = self.latest_sem
        h, w = tag.shape
        y0 = int(h * self.roi_y_start_ratio)
        roi = tag[y0:, :]

        road_mask = (roi == Tags.ROADS) | (roi == Tags.ROADLINE)  # include road line helps on some maps
        xs = np.where(road_mask)[1]
        if xs.size < 200:  # not enough evidence
            return self.last_steer

        x_mean = float(xs.mean())
        x_center = (w - 1) / 2.0
        offset_px = (x_mean - x_center)

        if abs(offset_px) < self.steer_deadband_px:
            steer = 0.0
        else:
            steer = -self.steer_gain * (offset_px / x_center)

        steer = float(np.clip(steer, -1.0, 1.0))
        self.last_steer = steer
        return steer

    def compute_speed_brake_from_sem_lidar(self) -> tuple[float, float, float]:
        """
        Returns: (v_target, brake, d_min)
        """
        if self.latest_sem_lidar is None or self.latest_sem_lidar.size == 0:
            return self.v_cruise, 0.0, float("inf")

        det = self.latest_sem_lidar
        # forward corridor in sensor frame (x-forward, y-right)
        x = det["x"]
        y = det["y"]
        z = det["z"]
        tag = det["object_tag"]

        in_front = x > 0.5
        corridor = np.abs(y) < 1.5
        height_ok = (z > -1.5) & (z < 2.5)
        is_obstacle = np.isin(tag, list(OBSTACLE_TAGS))
        mask = in_front & corridor & height_ok & is_obstacle

        if not np.any(mask):
            return self.v_cruise, 0.0, float("inf")

        d = np.sqrt(x[mask] ** 2 + y[mask] ** 2)
        d_min = float(np.min(d))

        if d_min < self.stop_distance:
            return 0.0, self.brake_hard, d_min
        if d_min < self.follow_distance:
            # linear slow-down
            alpha = (d_min - self.stop_distance) / max(1e-3, (self.follow_distance - self.stop_distance))
            v = self.v_follow + alpha * (self.v_cruise - self.v_follow)
            return float(v), 0.0, d_min

        return self.v_cruise, 0.0, d_min

    def control_loop(self):
        steer = self.compute_steer_from_sem()
        v_target, brake, d_min = self.compute_speed_brake_from_sem_lidar()

        # publish Twist (bridge expects deg in angular.z)
        cmd = Twist()
        cmd.linear.x = float(v_target)
        cmd.linear.y = float(brake)
        cmd.angular.z = float(steer * self.max_steer_deg)
        self.pub_cmd.publish(cmd)

        # occasional debug
        if (self.get_clock().now() - self.latest_lidar_time).nanoseconds < 200_000_000:  # 0.2s
            self.get_logger().debug(f"steer={steer:+.2f} v={v_target:.1f} brake={brake:.2f} d_min={d_min:.1f}")

def main():
    rclpy.init()
    node = OracleAutonomy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()