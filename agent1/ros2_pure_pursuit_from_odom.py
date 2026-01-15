#!/usr/bin/env python3
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseArray

def quat_to_yaw(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class PurePursuitFromOdom(Node):
    def __init__(self):
        super().__init__("pure_pursuit_from_odom")

        # --- params ---
        self.declare_parameter("lookahead", 6.0)          # [m]
        self.declare_parameter("wheel_base", 2.7)         # [m]
        self.declare_parameter("base_speed", 6.0)         # [m/s]
        self.declare_parameter("max_steer_deg", 35.0)     # vehicle 모델에 맞춤 (ros2_native도 35deg 사용) :contentReference[oaicite:5]{index=5}
        self.declare_parameter("a_lat_max", 2.5)          # [m/s^2] 곡률 기반 감속용

        # obstacle gating (hero frame)
        self.declare_parameter("obs_lat_gate", 1.8)       # |y| < gate면 내 차선으로 간주
        self.declare_parameter("slow_dist", 12.0)         # [m]
        self.declare_parameter("ebrake_dist", 6.0)        # [m]

        self.lookahead = float(self.get_parameter("lookahead").value)
        self.wheel_base = float(self.get_parameter("wheel_base").value)
        self.base_speed = float(self.get_parameter("base_speed").value)
        self.max_steer_deg = float(self.get_parameter("max_steer_deg").value)
        self.a_lat_max = float(self.get_parameter("a_lat_max").value)

        self.obs_lat_gate = float(self.get_parameter("obs_lat_gate").value)
        self.slow_dist = float(self.get_parameter("slow_dist").value)
        self.ebrake_dist = float(self.get_parameter("ebrake_dist").value)

        # --- state ---
        self.pose_xy_yaw: Optional[Tuple[float, float, float]] = None
        self.path_xy: List[Tuple[float, float]] = []
        self.obstacles_hero: List[Tuple[float, float]] = []  # (x forward, y left)

        # --- subs/pubs ---
        self.sub_odom = self.create_subscription(Odometry, "/carla/hero/odometry", self.odom_cb, 10)
        self.sub_path = self.create_subscription(Path, "/carla/path/local", self.path_cb, 10)

        # lidar clustering이 PoseArray로 /carla/obstacles_2d를 쏨 :contentReference[oaicite:6]{index=6}
        self.sub_obs = self.create_subscription(PoseArray, "/carla/obstacles_2d", self.obs_cb, 10)

        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        self.timer = self.create_timer(0.05, self.control_loop)  # 20Hz

    def odom_cb(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.pose_xy_yaw = (x, y, yaw)

    def path_cb(self, msg: Path):
        self.path_xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def obs_cb(self, msg: PoseArray):
        self.obstacles_hero = [(p.position.x, p.position.y) for p in msg.poses]

    def clamp(self, v, lo, hi):
        return hi if v > hi else lo if v < lo else v

    def min_front_obstacle_x(self) -> Optional[float]:
        m = None
        for ox, oy in self.obstacles_hero:
            if ox > 0.0 and abs(oy) < self.obs_lat_gate:
                if m is None or ox < m:
                    m = ox
        return m

    def speed_from_curvature(self, steer_rad: float) -> float:
        # kappa ≈ tan(delta)/L
        kappa = abs(math.tan(steer_rad)) / max(self.wheel_base, 1e-3)
        if kappa < 1e-4:
            return self.base_speed
        v_curv = math.sqrt(max(self.a_lat_max / kappa, 0.0))
        return min(self.base_speed, v_curv)

    def control_loop(self):
        if self.pose_xy_yaw is None or len(self.path_xy) < 2:
            return

        x, y, yaw = self.pose_xy_yaw

        # global(map) -> vehicle frame
        c = math.cos(-yaw)
        s = math.sin(-yaw)

        target = None
        for gx, gy in self.path_xy:
            dx = gx - x
            dy = gy - y
            x_l = dx * c - dy * s
            y_l = dx * s + dy * c
            d = math.hypot(x_l, y_l)
            if x_l > 0.0 and d >= self.lookahead:
                target = (x_l, y_l)
                break

        if target is None:
            return

        xt, yt = target
        ld = math.hypot(xt, yt)
        alpha = math.atan2(yt, xt)
        delta = math.atan2(2.0 * self.wheel_base * math.sin(alpha), max(ld, 1e-3))  # [rad]
        steer_deg = math.degrees(delta)
        steer_deg = self.clamp(steer_deg, -self.max_steer_deg, self.max_steer_deg)

        # curvature-based speed
        v_ref = self.speed_from_curvature(delta)

        # obstacle-based slow / e-brake (hero frame)
        brake_manual = 0.0
        minx = self.min_front_obstacle_x()
        if minx is not None:
            if minx <= self.ebrake_dist:
                v_ref = 0.0
                brake_manual = 1.0  # ros2_native에서 수동브레이크가 우선 적용됨 :contentReference[oaicite:7]{index=7}
            elif minx <= self.slow_dist:
                # 선형 감속 (minx가 가까울수록 v_ref 감소)
                ratio = (minx - self.ebrake_dist) / max(self.slow_dist - self.ebrake_dist, 1e-3)
                v_ref = min(v_ref, self.base_speed * self.clamp(ratio, 0.0, 1.0))

        cmd = Twist()
        cmd.linear.x = float(v_ref)         # [m/s]
        cmd.linear.y = float(brake_manual)  # [0..1]
        cmd.angular.z = float(steer_deg)    # [deg]
        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitFromOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
