#!/usr/bin/env python3

"""Global speed profile from a polyline global path.

Publishes a curvature-limited desired speed that your local controller can follow.

Subscribes
- /carla/path/global              (nav_msgs/Path)
- /carla/guidance/nearest_index   (std_msgs/Int32)  # from global_guidance.py

Publishes
- /carla/guidance/desired_speed   (std_msgs/Float32, m/s)
- /carla/guidance/path_curvature  (std_msgs/Float32, 1/m)

Notes
- Curvature is estimated from 3 points on the polyline around nearest index.
- Desired speed is computed from a lateral acceleration bound:
    v_curve = sqrt(a_lat_max / kappa)
  and then clamped + low-pass filtered.
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from std_msgs.msg import Int32, Float32


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class GlobalSpeedProfile(Node):
    def __init__(self):
        super().__init__("global_speed_profile")

        # --- parameters ---
        self.declare_parameter("v_max_mps", 12.0)      # straight-line max speed
        self.declare_parameter("v_min_mps", 3.0)       # never go below this due to curvature alone
        self.declare_parameter("a_lat_max", 2.0)       # lateral acceleration limit (m/s^2)
        self.declare_parameter("k_step", 5)            # index step for curvature points
        self.declare_parameter("smooth_tau", 0.6)      # seconds, 1st-order filter time constant

        self.v_max = float(self.get_parameter("v_max_mps").value)
        self.v_min = float(self.get_parameter("v_min_mps").value)
        self.a_lat_max = float(self.get_parameter("a_lat_max").value)
        self.k_step = int(self.get_parameter("k_step").value)
        self.smooth_tau = float(self.get_parameter("smooth_tau").value)

        self.k_step = max(1, self.k_step)
        self.v_max = max(0.1, self.v_max)
        self.v_min = clamp(self.v_min, 0.0, self.v_max)
        self.a_lat_max = max(0.1, self.a_lat_max)
        self.smooth_tau = max(0.0, self.smooth_tau)

        # --- IO ---
        self.sub_path = self.create_subscription(Path, "/carla/path/global", self.cb_path, 10)
        self.sub_idx = self.create_subscription(Int32, "/carla/guidance/nearest_index", self.cb_idx, 10)

        self.pub_speed = self.create_publisher(Float32, "/carla/guidance/desired_speed", 10)
        self.pub_kappa = self.create_publisher(Float32, "/carla/guidance/path_curvature", 10)

        self.path: Optional[Path] = None
        self.nearest_idx: Optional[int] = None

        self.v_filt = self.v_max
        self.last_t = self.get_clock().now().nanoseconds * 1e-9

        self.create_timer(0.05, self.timer_cb)  # 20 Hz

    def cb_path(self, msg: Path):
        if len(msg.poses) < 3:
            self.get_logger().warn("/carla/path/global has <3 poses; speed profile disabled")
            self.path = None
            return
        self.path = msg

    def cb_idx(self, msg: Int32):
        self.nearest_idx = int(msg.data)

    @staticmethod
    def curvature_from_3pts(p0, p1, p2) -> float:
        """Return curvature kappa (1/m). 0 means straight or degenerate."""
        ax = p1.x - p0.x
        ay = p1.y - p0.y
        bx = p2.x - p0.x
        by = p2.y - p0.y

        # 2*Area of triangle
        area2 = abs(ax * by - ay * bx)

        a = math.hypot(p1.x - p0.x, p1.y - p0.y)
        b = math.hypot(p2.x - p1.x, p2.y - p1.y)
        c = math.hypot(p2.x - p0.x, p2.y - p0.y)
        denom = a * b * c
        if denom < 1e-6 or area2 < 1e-9:
            return 0.0

        # curvature = 4A / (abc). Here area2 = 2A => 4A = 2*area2
        return (2.0 * area2) / denom

    def timer_cb(self):
        if self.path is None or self.nearest_idx is None:
            return

        n = len(self.path.poses)
        if n < 3:
            return

        i0 = clamp(float(self.nearest_idx), 0.0, float(n - 1))
        i0 = int(i0)
        k = self.k_step

        i1 = min(n - 1, i0 + k)
        i2 = min(n - 1, i0 + 2 * k)

        if i2 == i0:
            return

        p0 = self.path.poses[i0].pose.position
        p1 = self.path.poses[i1].pose.position
        p2 = self.path.poses[i2].pose.position

        kappa = self.curvature_from_3pts(p0, p1, p2)

        # curvature-limited speed
        if kappa <= 1e-6:
            v_des = self.v_max
        else:
            v_des = math.sqrt(self.a_lat_max / kappa)
            v_des = clamp(v_des, self.v_min, self.v_max)

        # time update
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(0.001, now - self.last_t)
        self.last_t = now

        # low-pass filter
        if self.smooth_tau <= 1e-6:
            self.v_filt = v_des
        else:
            alpha = dt / (self.smooth_tau + dt)
            self.v_filt = self.v_filt + alpha * (v_des - self.v_filt)

        # publish
        msg_v = Float32()
        msg_v.data = float(self.v_filt)
        self.pub_speed.publish(msg_v)

        msg_k = Float32()
        msg_k.data = float(kappa)
        self.pub_kappa.publish(msg_k)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalSpeedProfile()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
