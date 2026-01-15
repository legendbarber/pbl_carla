#!/usr/bin/env python3
# ros2_local_planner_v2.py
#
# Global path: CSV(x,y) in map/odom-consistent frame (same as /carla/hero/odometry pose)
# Obstacles: PoseArray in HERO frame (x forward, y left) from ros2_lidar_clustering.py
# Output: /carla/path/local as nav_msgs/Path in 'map' frame
#
# Improvements vs v1:
# - N candidate offsets (e.g., 9), not just center/left/right
# - Per-point normal based on local tangent (better on curves)
# - Cost function: clearance (min distance) - offset penalty - curvature penalty

import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseArray, PoseStamped

def quat_to_yaw(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def rot2d(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

class LocalPathAvoidV2(Node):
    def __init__(self):
        super().__init__("local_path_avoid_v2")

        # ---- Params ----
        self.declare_parameter("global_path_csv", "global_path.csv")
        self.declare_parameter("output_frame", "map")
        self.declare_parameter("L", 25.0)                 # horizon length [m]
        self.declare_parameter("safe_lat", 2.0)           # only consider obstacles with |y| < safe_lat in HERO frame
        self.declare_parameter("max_offset", 3.0)         # [m]
        self.declare_parameter("n_candidates", 9)         # odd recommended (includes 0)
        self.declare_parameter("collision_radius", 1.5)   # [m]

        # cost weights
        self.declare_parameter("w_clear", 1.0)
        self.declare_parameter("w_offset", 0.25)
        self.declare_parameter("w_curv", 0.10)

        self.global_csv = str(self.get_parameter("global_path_csv").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.L = float(self.get_parameter("L").value)
        self.safe_lat = float(self.get_parameter("safe_lat").value)
        self.max_offset = float(self.get_parameter("max_offset").value)
        self.n_candidates = int(self.get_parameter("n_candidates").value)
        self.collision_r = float(self.get_parameter("collision_radius").value)

        self.w_clear = float(self.get_parameter("w_clear").value)
        self.w_offset = float(self.get_parameter("w_offset").value)
        self.w_curv = float(self.get_parameter("w_curv").value)

        # ---- State ----
        self.current_xy_yaw: Optional[Tuple[float, float, float]] = None  # (x,y,yaw) in map frame
        self.global_xy: List[Tuple[float, float]] = []
        self.obstacles_hero: List[Tuple[float, float]] = []              # (x,y) in hero frame

        # ---- ROS I/O ----
        self.sub_odom = self.create_subscription(Odometry, "/carla/hero/odometry", self.odom_cb, 10)
        self.sub_obs  = self.create_subscription(PoseArray, "/carla/obstacles_2d", self.obs_cb, 10)
        self.pub_local = self.create_publisher(Path, "/carla/path/local", 10)

        self.load_global_path(self.global_csv)
        if not self.global_xy:
            self.get_logger().warn(f"{self.global_csv} is empty or not found.")

        self.timer = self.create_timer(0.1, self.timer_cb)

    def load_global_path(self, filename: str):
        self.global_xy = []
        try:
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    x = float(parts[0])
                    y = float(parts[1])
                    self.global_xy.append((x, y))
            self.get_logger().info(f"Loaded {len(self.global_xy)} points from {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to load {filename}: {e}")

    def odom_cb(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.current_xy_yaw = (x, y, yaw)

    def obs_cb(self, msg: PoseArray):
        self.obstacles_hero = [(p.position.x, p.position.y) for p in msg.poses]

    def timer_cb(self):
        if self.current_xy_yaw is None or len(self.global_xy) < 2:
            return

        ex, ey, eyaw = self.current_xy_yaw
        idx = self.find_nearest_index(ex, ey, self.global_xy)
        if idx is None:
            return

        # obstacles -> map frame for scoring
        obs_map: List[Tuple[float, float]] = []
        for ox_h, oy_h in self.obstacles_hero:
            if 0.0 < ox_h < self.L and abs(oy_h) < self.safe_lat:
                dx_m, dy_m = rot2d(ox_h, oy_h, eyaw)
                obs_map.append((ex + dx_m, ey + dy_m))

        offsets = self.build_offsets()
        best_score = -1e18
        best_path: Optional[List[Tuple[float, float]]] = None

        for off in offsets:
            cand = self.build_candidate_path(idx, ex, ey, off)
            score = self.score_candidate(cand, obs_map, off)
            if score > best_score:
                best_score = score
                best_path = cand

        if not best_path:
            return

        out = Path()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.output_frame
        for px, py in best_path:
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = 0.0
            out.poses.append(ps)

        self.pub_local.publish(out)

    def build_offsets(self) -> List[float]:
        if self.n_candidates <= 1:
            return [0.0]
        n = max(3, self.n_candidates)
        if n % 2 == 0:
            n += 1  # ensure includes 0
        # equally spaced across [-max_offset, +max_offset]
        step = (2.0 * self.max_offset) / float(n - 1)
        return [(-self.max_offset + i * step) for i in range(n)]

    def build_candidate_path(self, start_idx: int, ex: float, ey: float, offset: float) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        s = 0.0

        i = start_idx
        prev_x, prev_y = ex, ey

        while i < len(self.global_xy) and s <= self.L:
            gx, gy = self.global_xy[i]
            seg = math.hypot(gx - prev_x, gy - prev_y)
            s += seg
            prev_x, prev_y = gx, gy

            # local tangent (use next point if possible)
            if i + 1 < len(self.global_xy):
                nxp, nyp = self.global_xy[i + 1]
                theta = math.atan2(nyp - gy, nxp - gx)
            elif i > 0:
                pxp, pyp = self.global_xy[i - 1]
                theta = math.atan2(gy - pyp, gx - pxp)
            else:
                theta = 0.0

            # normal (left)
            n_x = -math.sin(theta)
            n_y =  math.cos(theta)

            # smooth envelope: 0 -> max at mid -> 0
            env = math.sin(math.pi * min(s, self.L) / max(self.L, 1e-3))
            off_s = offset * env

            px = gx + off_s * n_x
            py = gy + off_s * n_y
            pts.append((px, py))
            i += 1

        return pts

    def score_candidate(self, path_xy: List[Tuple[float, float]], obs_map: List[Tuple[float, float]], offset: float) -> float:
        if len(path_xy) < 2:
            return -1e18

        # clearance
        min_d = float("inf")
        if obs_map:
            for px, py in path_xy:
                for ox, oy in obs_map:
                    d = math.hypot(px - ox, py - oy)
                    if d < min_d:
                        min_d = d
            if min_d < self.collision_r:
                return -1e6 + min_d
        else:
            min_d = 50.0  # no obstacles => large clearance

        # curvature penalty (heading changes)
        curv_sum = 0.0
        prev_theta = None
        for (x1, y1), (x2, y2) in zip(path_xy[:-1], path_xy[1:]):
            th = math.atan2(y2 - y1, x2 - x1)
            if prev_theta is not None:
                curv_sum += abs(wrap_pi(th - prev_theta))
            prev_theta = th

        score = (
            self.w_clear * min_d
            - self.w_offset * abs(offset)
            - self.w_curv * curv_sum
        )
        return score

    @staticmethod
    def find_nearest_index(x: float, y: float, pts: List[Tuple[float, float]]) -> Optional[int]:
        min_d = float("inf")
        idx = None
        for i, (px, py) in enumerate(pts):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < min_d:
                min_d = d
                idx = i
        return idx

def main(args=None):
    rclpy.init(args=args)
    node = LocalPathAvoidV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
