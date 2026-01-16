#!/usr/bin/env python3
import argparse
import math
import time
import struct
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path as RosPath
from geometry_msgs.msg import PoseStamped

try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    pc2 = None


def load_global_path_csv(csv_path: str):
    lat0 = lon0 = None
    xy: List[Tuple[float, float]] = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            a, b = float(parts[0]), float(parts[1])
            if lat0 is None:
                lat0, lon0 = a, b
            else:
                xy.append((a, b))
    if lat0 is None or not xy:
        raise RuntimeError(f"Invalid/empty path file: {csv_path}")
    return lat0, lon0, xy


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float):
    # Same approximation as ros2_gnss_path_maker.py
    cos_lat0 = math.cos(math.radians(lat0))
    dx = (lon - lon0) * (111320.0 * cos_lat0)  # x: East (m)
    dy = (lat - lat0) * 110540.0               # y: North (m)
    return dx, dy


class PathFollowerPurePursuit(Node):
    def __init__(self, csv_path: str):
        super().__init__("path_follower_pure_pursuit")
        self.declare_parameter("use_sim_time", True)

        # Load path
        self.lat0, self.lon0, self.path_xy = load_global_path_csv(csv_path)

        # ROS I/O
        self.sub_gnss = self.create_subscription(NavSatFix, "/carla/hero/gnss", self.on_gnss, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, "/carla/hero/lidar/point_cloud", self.on_lidar, 10)
        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)
        self.pub_path = self.create_publisher(RosPath, "/carla/path/global", 10)

        # Controller params (tune here)
        self.wheelbase = 2.7  # [m]
        self.base_speed = 8.0  # [m/s] (~28.8 km/h)
        self.min_speed = 2.0   # [m/s]
        self.max_speed = 12.0  # [m/s]
        self.lookahead_base = 5.0  # [m]
        self.lookahead_gain = 0.3  # [m per (m/s)]
        self.max_steer_deg = 35.0
        self.max_steer_rad = math.radians(self.max_steer_deg)

        # Safety (LiDAR) params
        self.lidar_slow_dist = 15.0  # [m]
        self.lidar_stop_dist = 8.0   # [m]
        self.lidar_lane_half_width = 1.6  # [m] front corridor |y| < this
        self.lidar_min_front_dist: Optional[float] = None

        # State
        self.x = None
        self.y = None
        self.yaw = 0.0  # heading in map frame, 0 along +x(East), +pi/2 along +y(North)
        self.last_xy = None
        self.last_xy_time = None
        self.closest_idx = 0

        # Timers
        self.timer_ctl = self.create_timer(0.05, self.on_control)  # 20 Hz
        self.timer_path = self.create_timer(0.5, self.on_publish_path)  # 2 Hz

        self.get_logger().info(f"Loaded {len(self.path_xy)} path points from {csv_path}")
        self.get_logger().info("Subscribing: /carla/hero/gnss, /carla/hero/lidar/point_cloud")
        self.get_logger().info("Publishing: /carla/hero/cmd_vel (Twist), /carla/path/global (Path)")
        self.get_logger().info(
            "cmd_vel mapping assumed by your bridge: linear.x=target_speed[m/s], linear.y=brake[0..1], angular.z=steer_cmd*35"
        )

    def on_publish_path(self):
        msg = RosPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for x, y in self.path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def on_gnss(self, msg: NavSatFix):
        x, y = latlon_to_xy(msg.latitude, msg.longitude, self.lat0, self.lon0)
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_xy is not None and self.last_xy_time is not None:
            dx = x - self.last_xy[0]
            dy = y - self.last_xy[1]
            dist = math.hypot(dx, dy)
            dt = max(1e-3, now - self.last_xy_time)
            speed = dist / dt

            # Update yaw only if moving a bit (to avoid noise when stopped)
            if dist > 0.2 and speed > 0.3:
                self.yaw = math.atan2(dy, dx)

        self.x, self.y = x, y
        self.last_xy = (x, y)
        self.last_xy_time = now

    def on_lidar(self, msg: PointCloud2):
        min_d = None

        if pc2 is not None:
            # points: x,y,z,intensity (float32)
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                if x <= 0.0:
                    continue
                if abs(y) > self.lidar_lane_half_width:
                    continue
                if z < -1.0 or z > 2.0:
                    continue
                d = math.hypot(x, y)
                if min_d is None or d < min_d:
                    min_d = d
        else:
            # Fallback: assume the bridge publishes packed float32 (x,y,z,intensity) per point (16 bytes)
            data = bytes(msg.data)
            step = int(msg.point_step) if msg.point_step else 16
            if step < 16:
                self.lidar_min_front_dist = None
                return
            for off in range(0, len(data) - step + 1, step):
                try:
                    x, y, z, _i = struct.unpack_from("<ffff", data, off)
                except Exception:
                    break
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                    continue
                if x <= 0.0:
                    continue
                if abs(y) > self.lidar_lane_half_width:
                    continue
                if z < -1.0 or z > 2.0:
                    continue
                d = math.hypot(x, y)
                if min_d is None or d < min_d:
                    min_d = d

        self.lidar_min_front_dist = min_d

    def _advance_closest_index(self):
        # Move closest_idx forward while it reduces distance (greedy)
        if self.x is None or self.y is None:
            return
        n = len(self.path_xy)
        i = self.closest_idx
        best_i = i
        best_d = (self.path_xy[i][0] - self.x) ** 2 + (self.path_xy[i][1] - self.y) ** 2
        # Look ahead a limited window to keep it cheap
        for j in range(i, min(i + 50, n)):
            d = (self.path_xy[j][0] - self.x) ** 2 + (self.path_xy[j][1] - self.y) ** 2
            if d < best_d:
                best_d = d
                best_i = j
        self.closest_idx = best_i

    def _find_lookahead_target(self, lookahead: float):
        # Starting from closest_idx, march forward until cumulative distance >= lookahead
        n = len(self.path_xy)
        i = self.closest_idx
        if i >= n - 1:
            return self.path_xy[-1], n - 1

        cum = 0.0
        prev = self.path_xy[i]
        for j in range(i + 1, n):
            cur = self.path_xy[j]
            seg = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            cum += seg
            if cum >= lookahead:
                return cur, j
            prev = cur
        return self.path_xy[-1], n - 1

    def on_control(self):
        if self.x is None or self.y is None:
            return

        # End-of-path stop
        last_x, last_y = self.path_xy[-1]
        if math.hypot(last_x - self.x, last_y - self.y) < 2.0 and self.closest_idx > len(self.path_xy) - 20:
            self._publish_cmd(v=0.0, steer_norm=0.0, brake=1.0)
            return

        self._advance_closest_index()

        # Speed estimate (from GNSS derivative)
        v_est = 0.0
        if self.last_xy is not None and self.last_xy_time is not None:
            # not perfect: estimate over last GNSS step
            pass

        # Lookahead grows with speed command
        lookahead = self.lookahead_base + self.lookahead_gain * self.base_speed
        target_xy, target_idx = self._find_lookahead_target(lookahead)

        # Transform target into vehicle frame (x forward, y left) using yaw in map frame
        dx = target_xy[0] - self.x
        dy = target_xy[1] - self.y
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        x_v = cy * dx + sy * dy
        y_v = -sy * dx + cy * dy

        # Pure pursuit curvature
        Ld = max(1.0, math.hypot(x_v, y_v))
        kappa = 0.0
        if Ld > 1e-3:
            kappa = 2.0 * y_v / (Ld * Ld)

        steer_angle = math.atan(self.wheelbase * kappa)
        steer_norm = max(-1.0, min(1.0, steer_angle / self.max_steer_rad))

        # Curvature-based speed reduction
        v_cmd = self.base_speed / (1.0 + 2.0 * abs(kappa))
        v_cmd = max(self.min_speed, min(self.max_speed, v_cmd))

        # LiDAR safety override
        brake = 0.0
        d_front = self.lidar_min_front_dist
        if d_front is not None:
            if d_front < self.lidar_stop_dist:
                v_cmd = 0.0
                brake = 1.0
            elif d_front < self.lidar_slow_dist:
                # linear slowdown
                ratio = (d_front - self.lidar_stop_dist) / max(1e-3, (self.lidar_slow_dist - self.lidar_stop_dist))
                v_cmd = min(v_cmd, self.min_speed + ratio * (self.base_speed - self.min_speed))

        self._publish_cmd(v=v_cmd, steer_norm=steer_norm, brake=brake)

    def _publish_cmd(self, v: float, steer_norm: float, brake: float):
        msg = Twist()
        msg.linear.x = float(max(0.0, v))
        msg.linear.y = float(max(0.0, min(1.0, brake)))
        # Your bridge divides angular.z by 35.0 to get [-1,1] steer
        msg.angular.z = float(max(-1.0, min(1.0, steer_norm)) * self.max_steer_deg)
        self.pub_cmd.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to global_path.csv")
    args, _unknown = parser.parse_known_args()

    rclpy.init()
    node = PathFollowerPurePursuit(args.csv)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
