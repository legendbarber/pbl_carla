#!/usr/bin/env python3
import math
import os
import csv
import argparse
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.time import Time

class KooLocalPlanner(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_local_planner")
        
        self.path_num = path_num

        gnss_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_gnss = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gnss_cb, gnss_qos_profile)

        self.sub_obs = self.create_subscription(
            PoseArray, "/carla/obstacles_2d", self.obs_cb, 10)

        self.pub_local_path = self.create_publisher(
            Path, "/carla/path/local", 10)

        self.pub_local_path_viz = self.create_publisher(
            Path, "/carla/path/local_viz", 10)

        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0

        self.current_xy: Optional[Tuple[float, float]] = None
        self.prev_xy: Optional[Tuple[float, float]] = None
        self.global_xy: List[Tuple[float, float]] = []   
        self.obstacles: List[Tuple[float, float]] = []   

        # ⭐ 추가: 차량 Yaw 관리
        self.vehicle_yaw = 0.0
        self.yaw_update_threshold = 0.02  # 2cm 이상 이동 시 업데이트

        # ⭐ 차량 중점 offset (GNSS가 차량 뒤쪽에 있다고 가정)
        self.vehicle_center_offset_x = 1.0  # 차량 앞쪽으로 1.5m (차량 길이의 절반)
        self.vehicle_center_offset_y = 0.0  # 좌우 중앙

        self.L = 20.0            
        # [수정 1] 탐색 범위를 회피 거리(3.0m)보다 넓게 설정하여
        # 회피 중에도 장애물을 놓치지 않도록 함 (2.0 -> 4.5)
        self.safe_lat = 4.5      
        self.max_offset = 3.0    
        self.prev_idx = 0        

        # 스마트 대기 로직 변수
        self.obs_start_time = None      
        self.last_seen_time = None      
        
        self.wait_time = 3.0            
        self.reset_timeout = 1.0        

        # [수정 2] 흔들림 방지용 스무딩 변수
        self.current_offset_ratio = 0.0  # 현재 적용 중인 회피 비율 (-1.0 ~ 1.0)
        self.target_offset_ratio = 0.0   # 목표 회피 비율
        self.alpha = 0.05                # 스무딩 계수 (작을수록 매우 부드럽게 반응)

        self.load_global_path()
        self.timer = self.create_timer(0.1, self.timer_cb)
        
        print(f"=== Koo Planner (Path: {self.path_num}) Started ===")
        print(f">> Safe Range: {self.safe_lat}m | Smoothing Alpha: {self.alpha}")

    def load_global_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"../path/global_path_{self.path_num}.csv"
        file_path = os.path.join(current_dir, filename)
        if not os.path.exists(file_path): return
        try:
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.split(',')
                    if len(parts) < 2 or line.startswith("#"): continue
                    try: self.global_xy.append((float(parts[0]), float(parts[1])))
                    except: continue
            self.get_logger().info(f"Loaded {len(self.global_xy)} points")
        except: pass

    def gnss_cb(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        if self.lat0 is None:
            self.lat0 = msg.latitude; self.lon0 = msg.longitude
            self.cos_lat0 = math.cos(math.radians(msg.latitude))
            self.current_xy = (0.0, 0.0)
            self.prev_xy = (0.0, 0.0)  # ⭐ 초기화
        else:
            dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
            dy = (lat - self.lat0) * 110540.0
            new_xy = (dx, dy)
            
            # ⭐ Yaw 계산: 이전 위치에서 현재 위치로의 방향
            if self.prev_xy is not None:
                delta_x = new_xy[0] - self.prev_xy[0]
                delta_y = new_xy[1] - self.prev_xy[1]
                distance_moved = math.hypot(delta_x, delta_y)
                
                # 충분히 이동했을 때만 yaw 업데이트 (노이즈 방지)
                if distance_moved > self.yaw_update_threshold:
                    self.vehicle_yaw = math.atan2(delta_y, delta_x)
                    self.prev_xy = new_xy
            
            self.current_xy = new_xy

    def obs_cb(self, msg): 
        self.obstacles = [(p.position.x, p.position.y) for p in msg.poses]

    def timer_cb(self):
        if self.current_xy is None or len(self.global_xy) < 2: return
        x, y = self.current_xy
        idx = self.find_nearest_index(x, y)
        if idx is None: return

        # 1. 장애물 판단 (목표 방향 결정)
        # analog_score는 사용하지 않고, 방향(direction)만 가져옴
        target_dir, debug_info = self.decide_target_direction(self.obstacles)
        
        now = self.get_clock().now()
        is_obstacle_present = (target_dir != 0.0)
        
        # Debouncing (잠깐 안보여도 유지)
        if is_obstacle_present:
            self.last_seen_time = now
            if self.obs_start_time is None:
                self.obs_start_time = now
        else:
            if self.last_seen_time is not None:
                elapsed_lost = (now - self.last_seen_time).nanoseconds / 1e9
                if elapsed_lost < self.reset_timeout:
                    # 잠깐 사라진 건 무시하고 이전 상태 유지 (목표 0으로 안 바꿈)
                    pass 
                else:
                    self.obs_start_time = None
                    self.last_seen_time = None
                    self.target_offset_ratio = 0.0 # 진짜 사라짐 -> 원복

        # 2. 대기 및 회피 로직
        if self.obs_start_time is not None:
            elapsed = (now - self.obs_start_time).nanoseconds / 1e9
            
            if elapsed < self.wait_time:
                # 3초 대기 구간: 목표는 0.0 (직진)
                self.target_offset_ratio = 0.0
                if int(elapsed * 5) % 5 == 0: 
                    print(f"[WAIT] {elapsed:.1f}s / {self.wait_time:.1f}s | {debug_info}")
            else:
                # 3초 경과: 결정된 방향으로 목표 설정 (확실하게 1.0 또는 -1.0)
                # 이 값이 계속 변하지 않도록 decide_target_direction에서 꽉 잡아줌
                if is_obstacle_present:
                    self.target_offset_ratio = target_dir
                
                print(f"[AVOID] Target:{self.target_offset_ratio:.0f} | Cur:{self.current_offset_ratio:.2f}")

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"

        path_viz = Path()
        path_viz.header.stamp = self.get_clock().now().to_msg()
        path_viz.header.frame_id = "hero"

        s = 0.0
        prev_px, prev_py = self.global_xy[idx]
        curr_idx = idx
        path_len = len(self.global_xy)

        # ⭐ 좌표 변환 파라미터 미리 계산 (맵 → 차량 좌표계)
        cos_yaw = math.cos(-self.vehicle_yaw)
        sin_yaw = math.sin(-self.vehicle_yaw)

        while curr_idx < path_len and s <= self.L:
            gx, gy = self.global_xy[curr_idx]
            s += math.hypot(gx - prev_px, gy - prev_py)
            prev_px, prev_py = gx, gy; px, py = gx, gy

            # 스무딩된 offset 적용
            if abs(self.current_offset_ratio) > 0.01:
                t = min(max(s / self.L, 0.0), 1.0); bezier = 3.0 * t * (1.0 - t)
                offset = self.max_offset * bezier * self.current_offset_ratio
                
                if curr_idx + 1 < path_len: nx, ny = self.global_xy[curr_idx+1]
                else: nx, ny = gx, gy
                path_yaw = math.atan2(ny - gy, nx - gx)
                px += offset * -math.sin(path_yaw); py += offset * math.cos(path_yaw)

            # ⭐ 맵 좌표계 Path (map frame)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.position.z = 0.0
            path.poses.append(ps)

            # ⭐ 차량 좌표계 Path (hero frame) - 차량 중점 기준
            # 1. 차량으로부터의 상대 위치 계산 (맵 좌표계)
            dx_map = px - x
            dy_map = py - y
            
            # 2. 회전 변환 (차량 yaw 기준으로 회전)
            px_local = dx_map * cos_yaw - dy_map * sin_yaw
            py_local = dx_map * sin_yaw + dy_map * cos_yaw
            
            # 3. ⭐ 차량 중점 offset 적용
            # GNSS 위치를 차량 중점으로 보정
            px_centered = px_local - self.vehicle_center_offset_x
            py_centered = py_local - self.vehicle_center_offset_y
            
            ps_viz = PoseStamped()
            ps_viz.header = path_viz.header
            ps_viz.pose.position.x = px_centered
            ps_viz.pose.position.y = -py_centered
            ps_viz.pose.position.z = 0.0
            path_viz.poses.append(ps_viz)

            curr_idx += 1

        # ⭐ 두 Path 모두 publish
        self.pub_local_path.publish(path)
        self.pub_local_path_viz.publish(path_viz)
        
        # 디버깅 로그 (필요시)
        # self.get_logger().info(f"Vehicle Yaw: {math.degrees(self.vehicle_yaw):.1f}°, Path points: {len(path.poses)}")

    def find_nearest_index(self, x, y):
        start = max(0, self.prev_idx - 50); end = min(len(self.global_xy), self.prev_idx + 50)
        min_d = float('inf'); idx = -1
        for i in range(start, end):
            d = (self.global_xy[i][0] - x)**2 + (self.global_xy[i][1] - y)**2
            if d < min_d: min_d = d; idx = i
        if idx != -1: self.prev_idx = idx; return idx
        return self.prev_idx

    def decide_target_direction(self, obs_xy) -> Tuple[float, str]:
        if not obs_xy: return 0.0, "Clean"
        
        # [수정] 감지 범위를 넓힘 (safe_lat 사용)
        relevant = [o for o in obs_xy if 0.1 < o[0] < self.L and abs(o[1]) < self.safe_lat]
        
        if not relevant: return 0.0, "Clean"

        # 단순히 개수나 점수로 '어디가 더 막혔나'만 판단
        left_score = sum(1 for o in relevant if o[1] > 0) # 왼쪽에 있는 장애물 수
        right_score = sum(1 for o in relevant if o[1] < 0) # 오른쪽에 있는 장애물 수

        # 0.5m 이내에 아주 가까운 장애물이 있다면 가중치 부여
        critical_left = any(o[1] > 0 and abs(o[1]) < 1.0 for o in relevant)
        critical_right = any(o[1] < 0 and abs(o[1]) < 1.0 for o in relevant)

        if critical_left: left_score += 10
        if critical_right: right_score += 10

        if left_score == 0 and right_score == 0: return 0.0, "Clean"

        # [핵심] 아날로그 값이 아닌, 고정된 방향(-1.0, 0.0, 1.0)만 반환
        if right_score > left_score:
            # 오른쪽에 장애물이 더 많음 -> 왼쪽으로 피함 (+1.0)
            return 1.0, f"Obs Right({right_score})->Go Left"
        else:
            # 왼쪽에 장애물이 더 많음 -> 오른쪽으로 피함 (-1.0)
            return -1.0, f"Obs Left({left_score})->Go Right"

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_num', type=int, default=1)
    ros_args, _ = parser.parse_known_args()
    node = KooLocalPlanner(path_num=ros_args.path_num)
    rclpy.spin(node); node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()