#!/usr/bin/env python3
"""ROS 2 node: convert live GNSS (NavSatFix) to local XY using a fixed origin from global_path.csv.

- Loads origin_lat/origin_lon from the same CSV header format used by ros2_gnss_path_maker.py.
- Converts incoming /carla/hero/gnss (lat, lon) to (x, y) meters with the SAME formula.
- Publishes:
    * /carla/localization/gnss_pose (geometry_msgs/PoseStamped, frame_id=map)
    * /carla/localization/gnss_odom (nav_msgs/Odometry, frame_id=map, child_frame_id=base_link)
    * TF map -> base_link (optional)

Note:
- CARLA's GNSS output is simulator-provided; this localizer is for consistent global-path tracking.
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry

import tf2_ros


def parse_origin_from_csv(csv_path: str) -> Tuple[float, float]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.startswith("#") and "origin_lat" in ln:
            if i + 1 < len(lines) and not lines[i + 1].startswith("#"):
                parts = [x.strip() for x in lines[i + 1].split(",")]
                if len(parts) >= 2:
                    return float(parts[0]), float(parts[1])
    raise ValueError("Failed to parse origin lat/lon from CSV header")


class GnssLocalizerFixedOrigin(Node):
    def __init__(self):
        super().__init__("gnss_localizer_fixed_origin")

        self.declare_parameter("path_file", "/mnt/data/global_path.csv")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("broadcast_tf", True)

        path_file = self.get_parameter("path_file").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.broadcast_tf = bool(self.get_parameter("broadcast_tf").value)

        self.lat0, self.lon0 = parse_origin_from_csv(path_file)
        self.cos_lat0 = math.cos(math.radians(self.lat0))
        self.get_logger().info(f"Origin fixed from CSV: lat0={self.lat0}, lon0={self.lon0}")

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.sub = self.create_subscription(NavSatFix, "/carla/hero/gnss", self.cb_gnss, qos)
        self.pub_pose = self.create_publisher(PoseStamped, "/carla/localization/gnss_pose", 10)
        self.pub_odom = self.create_publisher(Odometry, "/carla/localization/gnss_odom", 10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.prev_xy: Optional[Tuple[float, float]] = None
        self.prev_t: Optional[float] = None
        self.prev_yaw: float = 0.0

    def latlon_to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        # Same approximation used in ros2_gnss_path_maker.py
        dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
        dy = (lat - self.lat0) * 110540.0
        return float(dx), float(dy)

    @staticmethod
    def yaw_to_quat(yaw: float):
        # z-yaw quaternion
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return 0.0, 0.0, qz, qw

    def cb_gnss(self, msg: NavSatFix):
        now_msg = self.get_clock().now().to_msg()
        t = self.get_clock().now().nanoseconds * 1e-9

        x, y = self.latlon_to_xy(msg.latitude, msg.longitude)

        # Estimate yaw from motion direction
        yaw = self.prev_yaw
        vx = 0.0
        vy = 0.0

        if self.prev_xy is not None and self.prev_t is not None:
            dt = max(1e-3, t - self.prev_t)
            dx = x - self.prev_xy[0]
            dy = y - self.prev_xy[1]
            dist = math.hypot(dx, dy)
            if dist > 0.05:  # ignore tiny jitter
                yaw = math.atan2(dy, dx)
            vx = dx / dt
            vy = dy / dt

        self.prev_xy = (x, y)
        self.prev_t = t
        self.prev_yaw = yaw

        qx, qy, qz, qw = self.yaw_to_quat(yaw)

        # PoseStamped
        ps = PoseStamped()
        ps.header.stamp = now_msg
        ps.header.frame_id = self.frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pub_pose.publish(ps)

        # Odometry
        odom = Odometry()
        odom.header = ps.header
        odom.child_frame_id = self.base_frame
        odom.pose.pose = ps.pose
        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        self.pub_odom.publish(odom)

        # TF map -> base_link
        if self.broadcast_tf:
            tf = TransformStamped()
            tf.header = ps.header
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = GnssLocalizerFixedOrigin()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
