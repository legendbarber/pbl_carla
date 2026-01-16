#!/usr/bin/env python3
"""ROS 2 node: publish a saved global path (nav_msgs/Path) from CSV.

CSV format expected (same as ros2_gnss_path_maker.py):
  # origin_lat,origin_lon
  <lat0>,<lon0>
  # x[m],y[m]
  <x0>,<y0>
  <x1>,<y1>
  ...

Publishes:
  - /carla/path/global        (nav_msgs/Path, frame_id=map)
  - /carla/path/global_origin (std_msgs/Float64MultiArray: [lat0, lon0])

Uses TRANSIENT_LOCAL durability so late subscribers receive the last message.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray


@dataclass
class GlobalPathData:
    origin_lat: float
    origin_lon: float
    xy: List[Tuple[float, float]]


def load_global_path_csv(csv_path: str) -> GlobalPathData:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    xy: List[Tuple[float, float]] = []

    with p.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Find origin line: first non-comment line after a comment containing 'origin_lat'
    for i, ln in enumerate(lines):
        if ln.startswith("#") and "origin_lat" in ln:
            if i + 1 < len(lines) and not lines[i + 1].startswith("#"):
                parts = [x.strip() for x in lines[i + 1].split(",")]
                if len(parts) >= 2:
                    origin_lat = float(parts[0])
                    origin_lon = float(parts[1])
            break

    if origin_lat is None or origin_lon is None:
        raise ValueError("Failed to parse origin lat/lon from CSV header")

    # Parse x,y after the '# x[m],y[m]' marker if present; otherwise parse any non-comment pairs.
    start_idx = 0
    for i, ln in enumerate(lines):
        if ln.startswith("#") and "x" in ln and "y" in ln:
            start_idx = i + 1
            break

    for ln in lines[start_idx:]:
        if ln.startswith("#"):
            continue
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        xy.append((x, y))

    if len(xy) < 2:
        raise ValueError("Parsed path has fewer than 2 points")

    return GlobalPathData(origin_lat=origin_lat, origin_lon=origin_lon, xy=xy)


class GlobalPathLoader(Node):
    def __init__(self):
        super().__init__("global_path_loader")

        self.declare_parameter("path_file", "/mnt/data/global_path.csv")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_hz", 1.0)
        self.declare_parameter("decimate", 1)  # publish every N-th point

        path_file = self.get_parameter("path_file").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        publish_hz = float(self.get_parameter("publish_hz").value)
        self.decimate = int(self.get_parameter("decimate").value)
        self.decimate = max(1, self.decimate)

        self.data = load_global_path_csv(path_file)
        self.get_logger().info(
            f"Loaded global path: {len(self.data.xy)} points, origin=({self.data.origin_lat}, {self.data.origin_lon})"
        )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.pub_path = self.create_publisher(PathMsg, "/carla/path/global", qos)
        self.pub_origin = self.create_publisher(Float64MultiArray, "/carla/path/global_origin", qos)

        period = 1.0 / publish_hz if publish_hz > 0 else 1.0
        self.timer = self.create_timer(period, self.timer_cb)

        self._cached_path_msg: Optional[PathMsg] = None
        self._cached_origin_msg: Optional[Float64MultiArray] = None

    def build_msgs(self):
        # Origin message
        origin = Float64MultiArray()
        origin.data = [float(self.data.origin_lat), float(self.data.origin_lon)]

        # Path message
        path = PathMsg()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id

        pts = self.data.xy[:: self.decimate]
        for (x, y) in pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self._cached_path_msg = path
        self._cached_origin_msg = origin

    def timer_cb(self):
        self.build_msgs()
        assert self._cached_path_msg is not None
        assert self._cached_origin_msg is not None
        self.pub_origin.publish(self._cached_origin_msg)
        self.pub_path.publish(self._cached_path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPathLoader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
