#!/usr/bin/env python3
"""ROS 2 node: global guidance from a nav_msgs/Path and current pose.

Subscribes:
  - /carla/path/global                 (nav_msgs/Path)
  - /carla/localization/gnss_pose      (geometry_msgs/PoseStamped)

Publishes:
  - /carla/guidance/lookahead_point    (geometry_msgs/PointStamped)
  - /carla/guidance/desired_heading    (std_msgs/Float32, radians)
  - /carla/guidance/nearest_index      (std_msgs/Int32)

This does not control the vehicle; it only provides a global "hint" (lookahead + heading)
that a lane-based local planner can use for lane choice and intersection decisions.
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float32, Int32


class GlobalGuidance(Node):
    def __init__(self):
        super().__init__("global_guidance")

        self.declare_parameter("lookahead_m", 12.0)
        self.declare_parameter("search_window", 200)  # points forward/back window around last idx

        self.lookahead_m = float(self.get_parameter("lookahead_m").value)
        self.search_window = int(self.get_parameter("search_window").value)
        self.search_window = max(20, self.search_window)

        self.sub_path = self.create_subscription(Path, "/carla/path/global", self.cb_path, 10)
        self.sub_pose = self.create_subscription(PoseStamped, "/carla/localization/gnss_pose", self.cb_pose, 10)

        self.pub_point = self.create_publisher(PointStamped, "/carla/guidance/lookahead_point", 10)
        self.pub_heading = self.create_publisher(Float32, "/carla/guidance/desired_heading", 10)
        self.pub_idx = self.create_publisher(Int32, "/carla/guidance/nearest_index", 10)

        self.path: Optional[Path] = None
        self.last_nearest_idx: int = 0

        self.timer = self.create_timer(0.05, self.timer_cb)  # 20 Hz
        self.latest_pose: Optional[PoseStamped] = None

    def cb_path(self, msg: Path):
        if len(msg.poses) < 2:
            self.get_logger().warn("Received global path with <2 poses")
            return
        self.path = msg
        self.last_nearest_idx = 0

    def cb_pose(self, msg: PoseStamped):
        self.latest_pose = msg

    def find_nearest_index(self, x: float, y: float) -> int:
        assert self.path is not None
        n = len(self.path.poses)
        if n == 0:
            return 0

        # Search in a window around last index for stability/perf
        lo = max(0, self.last_nearest_idx - self.search_window)
        hi = min(n - 1, self.last_nearest_idx + self.search_window)

        best_i = lo
        best_d2 = float("inf")
        for i in range(lo, hi + 1):
            px = self.path.poses[i].pose.position.x
            py = self.path.poses[i].pose.position.y
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        # If window search got stuck far from true nearest (e.g., teleport), fall back to full search.
        if best_d2 > 25.0:  # 5m^2
            for i in range(n):
                px = self.path.poses[i].pose.position.x
                py = self.path.poses[i].pose.position.y
                d2 = (px - x) ** 2 + (py - y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i

        return best_i

    def compute_lookahead_index(self, start_idx: int, lookahead_m: float) -> int:
        assert self.path is not None
        n = len(self.path.poses)
        if start_idx >= n - 1:
            return n - 1

        acc = 0.0
        i = start_idx
        while i < n - 1 and acc < lookahead_m:
            x0 = self.path.poses[i].pose.position.x
            y0 = self.path.poses[i].pose.position.y
            x1 = self.path.poses[i + 1].pose.position.x
            y1 = self.path.poses[i + 1].pose.position.y
            acc += math.hypot(x1 - x0, y1 - y0)
            i += 1
        return i

    def timer_cb(self):
        if self.path is None or self.latest_pose is None:
            return

        x = self.latest_pose.pose.position.x
        y = self.latest_pose.pose.position.y

        nearest = self.find_nearest_index(x, y)
        self.last_nearest_idx = nearest

        look_i = self.compute_lookahead_index(nearest, self.lookahead_m)

        # Lookahead point
        p = self.path.poses[look_i].pose.position
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = self.path.header.frame_id or "map"
        pt.point = p
        self.pub_point.publish(pt)

        # Desired heading (tangent direction). Use segment (look_i -> look_i+1) if possible.
        if look_i < len(self.path.poses) - 1:
            p2 = self.path.poses[look_i + 1].pose.position
            heading = math.atan2(p2.y - p.y, p2.x - p.x)
        elif look_i > 0:
            p0 = self.path.poses[look_i - 1].pose.position
            heading = math.atan2(p.y - p0.y, p.x - p0.x)
        else:
            heading = 0.0

        h = Float32()
        h.data = float(heading)
        self.pub_heading.publish(h)

        idx = Int32()
        idx.data = int(nearest)
        self.pub_idx.publish(idx)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalGuidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
