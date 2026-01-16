#!/usr/bin/env python3
"""lane_follow_controller_stageC.py

Stage C (객체별 행동) 통합 버전:
- 카메라 기반 차선 중심 주행 (기존)
- 시맨틱 세그멘테이션 ROI (기존)
- 레이더 TTC 기반 긴급 제동 (수정: CARLA radar velocity sign)
- 시맨틱 LiDAR(obj_tag/obj_idx) 기반: 보행자/차량 구분 감속/정지

출력(/carla/hero/cmd_vel):
  linear.x = 목표 속도 [m/s]
  linear.y = brake [0..1]
  angular.z = steer_cmd (기존과 동일)

주의:
- CARLA 센서 좌표계는 UE 기준(x forward, y right, z up)입니다.
  여기서는 차선/장애물 ROI 판단에 |y|만 사용하므로 좌표축 부호는 중요하지 않습니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

from cv_bridge import CvBridge


# ------------------------------
# Helpers
# ------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def moving_average(prev: Optional[float], new: float, alpha: float) -> float:
    """EMA: alpha in [0,1], bigger => smoother."""
    if prev is None:
        return new
    return alpha * prev + (1.0 - alpha) * new


@dataclass
class ObstacleState:
    ped_ahead_m: Optional[float] = None
    veh_ahead_m: Optional[float] = None
    ttc_s: Optional[float] = None
    last_reason: str = ""


# ------------------------------
# Node
# ------------------------------

class LaneFollowControllerStageC(Node):
    def __init__(self):
        super().__init__("lane_follow_controller_stageC")

        # Topics
        self.declare_parameter('rgb_topic', '/carla/hero/camera_rgb/image_color')
        self.declare_parameter('semseg_raw_topic', '/carla/hero/camera_semseg/image_raw')
        self.declare_parameter('semantic_lidar_topic', '/carla/hero/semantic_lidar/point_cloud')
        self.declare_parameter('radar_topic', '/carla/hero/radar/point_cloud')
        self.declare_parameter('path_topic', '/carla/path/global')
        self.declare_parameter('cmd_topic', '/carla/hero/cmd_vel')

        # Lane-follow gains
        self.declare_parameter('target_speed', 8.0)      # [m/s]
        self.declare_parameter('min_speed', 2.0)         # [m/s]
        self.declare_parameter('max_steer_deg', 35.0)    # [deg]

        # Vision settings
        self.declare_parameter('use_semseg_roi', True)
        self.declare_parameter('roi_y_min', 380)
        self.declare_parameter('roi_y_max', 720)

        # Semantic tag set
        # - 최신 CARLA 문서(ref_sensors)의 tag 표(0.9.14+ 변경 반영)에 맞춰 "latest" 기본값 사용.
        # - 예전(legacy) tag셋이 필요하면 tagset='legacy' 또는 road_label/태그 리스트를 직접 지정.
        self.declare_parameter('tagset', 'latest')  # 'latest' | 'legacy'
        self.declare_parameter('road_label', -1)    # -1이면 (1 vs 7) 자동 추정
        self.declare_parameter('vehicle_tags', [])  # 비어있으면 tagset으로 자동 채움
        self.declare_parameter('pedestrian_tags', [])

        # Stage C behavior parameters
        self.declare_parameter('vehicle_roi_half_width', 1.8)   # [m]
        self.declare_parameter('ped_roi_half_width', 3.0)       # [m]
        self.declare_parameter('min_obj_points', 6)             # 객체로 인정할 최소 포인트 수
        self.declare_parameter('min_forward_x', 1.0)            # [m] 센서 기준 전방 필터
        self.declare_parameter('z_min', -1.5)                   # [m] 지면 제거
        self.declare_parameter('z_max', 2.5)                    # [m]

        # Vehicle following
        self.declare_parameter('follow_min_dist', 6.0)          # [m]
        self.declare_parameter('follow_time_headway', 1.2)      # [s]
        self.declare_parameter('veh_stop_dist', 3.2)            # [m] 이하면 정지(브레이크 1.0)

        # Pedestrian yielding
        self.declare_parameter('ped_stop_dist', 8.0)            # [m]
        self.declare_parameter('ped_slow_dist', 18.0)           # [m] 이 안에서 선형 감속

        # Soft braking
        self.declare_parameter('soft_brake_gain', 0.25)         # (v_curr - v_cap) -> brake
        self.declare_parameter('soft_brake_max', 0.4)

        # Radar TTC emergency
        self.declare_parameter('use_radar_ttc', True)
        self.declare_parameter('ttc_brake_s', 1.2)              # [s] 이하면 즉시 정지
        self.declare_parameter('ttc_roi_half_width', 1.8)       # [m]
        self.declare_parameter('ttc_min_depth', 0.5)            # [m]

        # Smoothing
        self.declare_parameter('ema_alpha_dist', 0.85)          # 거리 EMA (큰 값일수록 더 부드러움)

        # Derived / internal
        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_semseg_raw: Optional[np.ndarray] = None

        self.road_label = int(self.get_parameter('road_label').value)
        self.use_semseg_roi = bool(self.get_parameter('use_semseg_roi').value)

        self.tagset = str(self.get_parameter('tagset').value).strip().lower()
        self.vehicle_tags = self._get_int_list_param('vehicle_tags')
        self.ped_tags = self._get_int_list_param('pedestrian_tags')
        self._apply_tag_defaults_if_needed()

        self.obs = ObstacleState()
        self._ema_ped: Optional[float] = None
        self._ema_veh: Optional[float] = None
        self._ema_ttc: Optional[float] = None

        # Global path
        self.global_path_xy: List[Tuple[float, float]] = []

        # Subscriptions
        self.sub_rgb = self.create_subscription(
            Image, self.get_parameter('rgb_topic').value, self.cb_rgb, 10)
        self.sub_semseg = self.create_subscription(
            Image, self.get_parameter('semseg_raw_topic').value, self.cb_semseg_raw, 10)
        self.sub_sem_lidar = self.create_subscription(
            PointCloud2, self.get_parameter('semantic_lidar_topic').value, self.cb_semantic_lidar, 10)
        self.sub_radar = self.create_subscription(
            PointCloud2, self.get_parameter('radar_topic').value, self.cb_radar, 10)
        self.sub_path = self.create_subscription(
            Path, self.get_parameter('path_topic').value, self.cb_path, 10)

        # Publisher
        self.pub_cmd = self.create_publisher(
            Twist, self.get_parameter('cmd_topic').value, 10)

        # Control loop
        self.timer = self.create_timer(0.05, self.timer_cb)  # 20 Hz

        self.get_logger().info(
            f"[StageC] Started. vehicle_tags={self.vehicle_tags}, ped_tags={self.ped_tags}, road_label={self.road_label} (auto if -1)")

    # -------- params helpers --------

    def _get_int_list_param(self, name: str) -> List[int]:
        v = self.get_parameter(name).value
        if v is None:
            return []
        # ROS2 python에서는 list가 이미 list로 들어오지만, 안전하게 변환
        out: List[int] = []
        try:
            for x in v:
                out.append(int(x))
        except Exception:
            return []
        return out

    def _apply_tag_defaults_if_needed(self) -> None:
        # tagset 기본값: CARLA 문서(ref_sensors) 기준
        if not self.vehicle_tags:
            if self.tagset == 'legacy':
                # (구) CityScapes 스타일(일부 자료에서 vehicle=10, pedestrian=4, road=7로 쓰던 셋)
                self.vehicle_tags = [10]
            else:
                # 최신: Car/Truck/Bus/Train
                self.vehicle_tags = [14, 15, 16, 17]

        if not self.ped_tags:
            if self.tagset == 'legacy':
                self.ped_tags = [4]
            else:
                # 최신: Pedestrian/Rider
                self.ped_tags = [12, 13]

    # -------- callbacks --------

    def cb_rgb(self, msg: Image) -> None:
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"RGB decode failed: {e}")

    def cb_semseg_raw(self, msg: Image) -> None:
        try:
            self.latest_semseg_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # road_label auto-detect: tag_r에서 1(Roads) vs 7(TrafficLight) 중 더 많이 보이는 값을 road로 사용
            # - 최신 tag 표에서는 Roads=1
            # - legacy tag 표에서는 Road=7
            if self.road_label < 0:
                tag_r = self.latest_semseg_raw[:, :, 2]
                c1 = int(np.count_nonzero(tag_r == 1))
                c7 = int(np.count_nonzero(tag_r == 7))
                self.road_label = 1 if c1 >= c7 else 7
                self.get_logger().info(f"[StageC] Auto road_label={self.road_label} (count1={c1}, count7={c7})")

        except Exception as e:
            self.get_logger().warn(f"Semseg decode failed: {e}")

    def cb_path(self, msg: Path) -> None:
        pts = []
        for ps in msg.poses:
            pts.append((ps.pose.position.x, ps.pose.position.y))
        if pts:
            self.global_path_xy = pts

    def cb_radar(self, msg: PointCloud2) -> None:
        if not bool(self.get_parameter('use_radar_ttc').value):
            return
        if msg.width <= 0 or not msg.data:
            return

        # RadarPublisher: fields = x,y,z,velocity (float32)
        # CARLA 문서: RadarDetection.velocity는 "Velocity towards the sensor" (양수=다가옴)
        # => closing_speed = max(0, velocity)
        try:
            dtype = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('velocity', '<f4')
            ])
            arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width)
        except Exception:
            return

        x = arr['x']
        y = arr['y']
        depth = np.sqrt(arr['x']**2 + arr['y']**2 + arr['z']**2)
        vel_towards = arr['velocity']

        half_w = float(self.get_parameter('ttc_roi_half_width').value)
        min_depth = float(self.get_parameter('ttc_min_depth').value)

        m = (x > 0.0) & (np.abs(y) < half_w) & (depth > min_depth)
        if not np.any(m):
            return

        d = depth[m]
        closing = np.maximum(0.0, vel_towards[m])
        valid = closing > 0.2
        if not np.any(valid):
            return

        ttc = d[valid] / closing[valid]
        ttc_min = float(np.min(ttc))

        alpha = float(self.get_parameter('ema_alpha_dist').value)
        self._ema_ttc = moving_average(self._ema_ttc, ttc_min, alpha)
        self.obs.ttc_s = self._ema_ttc

    def cb_semantic_lidar(self, msg: PointCloud2) -> None:
        if msg.width <= 0 or not msg.data:
            return

        # SemanticLidarPublisher: x,y,z,cos_incidence,obj_idx,obj_tag
        try:
            dtype = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'), ('obj_idx', '<u4'), ('obj_tag', '<u4')
            ])
            arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width)
        except Exception:
            return

        ped = self._nearest_object_ahead(arr, self.ped_tags, float(self.get_parameter('ped_roi_half_width').value))
        veh = self._nearest_object_ahead(arr, self.vehicle_tags, float(self.get_parameter('vehicle_roi_half_width').value))

        alpha = float(self.get_parameter('ema_alpha_dist').value)
        if ped is not None:
            self._ema_ped = moving_average(self._ema_ped, ped, alpha)
            self.obs.ped_ahead_m = self._ema_ped
        else:
            self.obs.ped_ahead_m = None
            self._ema_ped = None

        if veh is not None:
            self._ema_veh = moving_average(self._ema_veh, veh, alpha)
            self.obs.veh_ahead_m = self._ema_veh
        else:
            self.obs.veh_ahead_m = None
            self._ema_veh = None

    # -------- semantic lidar processing --------

    def _nearest_object_ahead(self, arr: np.ndarray, tags: List[int], half_width: float) -> Optional[float]:
        if not tags:
            return None

        x = arr['x']
        y = arr['y']
        z = arr['z']
        obj_idx = arr['obj_idx']
        obj_tag = arr['obj_tag']

        min_x = float(self.get_parameter('min_forward_x').value)
        z_min = float(self.get_parameter('z_min').value)
        z_max = float(self.get_parameter('z_max').value)
        min_pts = int(self.get_parameter('min_obj_points').value)

        m = (x > min_x) & (np.abs(y) < half_width) & (z > z_min) & (z < z_max) & (np.isin(obj_tag, tags))
        if not np.any(m):
            return None

        idx = obj_idx[m]
        x_m = x[m]

        # group-by obj_idx: per-object min(x) and point count
        order = np.argsort(idx, kind='mergesort')
        idx_s = idx[order]
        x_s = x_m[order]

        uniq, start = np.unique(idx_s, return_index=True)
        counts = np.diff(np.append(start, idx_s.size))

        # per-group min(x)
        minx = np.minimum.reduceat(x_s, start)

        # filter by min points
        keep = counts >= min_pts
        if not np.any(keep):
            return None

        nearest = float(np.min(minx[keep]))
        return nearest

    # -------- lane estimation (camera) --------

    def _estimate_lane_center_px(self, bgr: np.ndarray, semseg_raw: Optional[np.ndarray]) -> Tuple[Optional[int], np.ndarray]:
        h, w = bgr.shape[:2]
        y0 = int(self.get_parameter('roi_y_min').value)
        y1 = int(self.get_parameter('roi_y_max').value)
        y0 = clamp(y0, 0, h - 1)
        y1 = clamp(y1, 1, h)

        roi = bgr[y0:y1, :]

        # White-ish lane marking extraction
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 200], dtype=np.uint8)
        upper = np.array([180, 40, 255], dtype=np.uint8)
        lane_mask = cv2.inRange(hsv, lower, upper)

        # Semantic ROI: only on road pixels
        if self.use_semseg_roi and semseg_raw is not None and self.road_label >= 0:
            sem_roi = semseg_raw[y0:y1, :]
            tag_r = sem_roi[:, :, 2]
            road_mask = (tag_r == self.road_label).astype(np.uint8) * 255
            lane_mask = cv2.bitwise_and(lane_mask, road_mask)

        # Morph cleanup
        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_DILATE, kernel)

        ys, xs = np.where(lane_mask > 0)
        if len(xs) < 200:
            return None, lane_mask

        cx = int(np.mean(xs))
        return cx, lane_mask

    # -------- global path guidance (kept minimal) --------

    def _curvature_limit_speed(self) -> Optional[float]:
        """간단한 곡률 기반 속도 상한. (global_path_xy가 없으면 None)"""
        if len(self.global_path_xy) < 10:
            return None

        # 여기서는 고급 로컬라이징/최근접점 추적을 생략하고, 단순히 path의 국소 굴곡 정도로 제한
        # (기존 stageB 코드의 global_guidance 연계가 있다면 그쪽을 쓰는게 더 정확함)
        pts = np.array(self.global_path_xy[:50], dtype=np.float32)
        # 3점 곡률 근사
        curv = []
        for i in range(1, len(pts) - 1):
            p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p2 - p0)
            if a < 1e-3 or b < 1e-3 or c < 1e-3:
                continue
            s = 0.5 * (a + b + c)
            area2 = max(0.0, s * (s - a) * (s - b) * (s - c))
            area = math.sqrt(area2)
            k = 4.0 * area / (a * b * c)
            curv.append(k)
        if not curv:
            return None
        kmax = float(np.percentile(curv, 80))
        # v_max ~ sqrt(a_lat_max / k)
        a_lat_max = 2.5  # [m/s^2]
        if kmax < 1e-4:
            return None
        return float(math.sqrt(a_lat_max / kmax))

    # -------- behavior fusion (Stage C 핵심) --------

    def _apply_stage_c_behavior(self, v_ref: float, v_curr: float) -> Tuple[float, float, str]:
        """(v_target, brake, reason)"""
        brake = 0.0
        reason = ""

        # 0) Radar TTC emergency
        if bool(self.get_parameter('use_radar_ttc').value):
            ttc_brake_s = float(self.get_parameter('ttc_brake_s').value)
            if self.obs.ttc_s is not None and self.obs.ttc_s < ttc_brake_s:
                return 0.0, 1.0, f"RADAR_TTC({self.obs.ttc_s:.2f}s)"

        v_cap = v_ref

        # 1) Pedestrian: stop/slow
        dp = self.obs.ped_ahead_m
        if dp is not None:
            ped_stop = float(self.get_parameter('ped_stop_dist').value)
            ped_slow = float(self.get_parameter('ped_slow_dist').value)
            if dp <= ped_stop:
                return 0.0, 1.0, f"PED_STOP({dp:.1f}m)"
            if dp < ped_slow:
                scale = (dp - ped_stop) / max(1e-3, (ped_slow - ped_stop))
                v_cap = min(v_cap, v_ref * clamp(scale, 0.0, 1.0))
                reason = f"PED_SLOW({dp:.1f}m)"

        # 2) Vehicle: time-headway following
        dv = self.obs.veh_ahead_m
        if dv is not None:
            veh_stop = float(self.get_parameter('veh_stop_dist').value)
            if dv <= veh_stop:
                return 0.0, 1.0, f"VEH_STOP({dv:.1f}m)"

            d0 = float(self.get_parameter('follow_min_dist').value)
            thw = float(self.get_parameter('follow_time_headway').value)
            desired = d0 + thw * max(v_curr, 0.0)
            if dv < desired:
                scale = (dv - d0) / max(1e-3, (desired - d0))
                v_cap = min(v_cap, v_ref * clamp(scale, 0.0, 1.0))
                if reason:
                    reason += "+"
                reason += f"VEH_FOLLOW({dv:.1f}m<={desired:.1f}m)"

        # 3) Soft brake when v_curr >> v_cap
        if v_curr > v_cap + 0.3:
            k = float(self.get_parameter('soft_brake_gain').value)
            bmax = float(self.get_parameter('soft_brake_max').value)
            brake = clamp(k * (v_curr - v_cap), 0.0, bmax)

        return v_cap, brake, reason

    # -------- main loop --------

    def timer_cb(self) -> None:
        if self.latest_rgb is None:
            return

        target_speed = float(self.get_parameter('target_speed').value)
        min_speed = float(self.get_parameter('min_speed').value)

        cx, lane_mask = self._estimate_lane_center_px(self.latest_rgb, self.latest_semseg_raw)

        # Steering from lane center offset
        h, w = self.latest_rgb.shape[:2]
        if cx is None:
            steer_deg = 0.0
        else:
            err_px = (w / 2.0) - float(cx)
            steer_deg = clamp(0.08 * err_px, -float(self.get_parameter('max_steer_deg').value), float(self.get_parameter('max_steer_deg').value))

        # Global curvature speed limit (optional)
        v_ref = target_speed
        v_curve = self._curvature_limit_speed()
        if v_curve is not None:
            v_ref = min(v_ref, max(min_speed, v_curve))

        # We do not have v_curr from the bridge directly; use v_ref for behavior scaling.
        # (정확한 v_curr가 필요하면 /carla/hero/odometry 등을 추가해서 사용)
        v_curr = v_ref

        # Stage C behavior fusion
        v_cmd, brake, reason = self._apply_stage_c_behavior(v_ref, v_curr)

        # Publish cmd
        msg = Twist()
        msg.linear.x = float(max(0.0, v_cmd))
        msg.linear.y = float(clamp(brake, 0.0, 1.0))
        msg.angular.z = float(steer_deg)
        self.pub_cmd.publish(msg)

        # Debug overlay (optional view)
        dbg = self.latest_rgb.copy()
        if cx is not None:
            cv2.line(dbg, (cx, h), (cx, int(self.get_parameter('roi_y_min').value)), (0, 255, 0), 2)
        cv2.putText(dbg, f"v_cmd={v_cmd:.2f} brake={brake:.2f} steer={steer_deg:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(dbg, f"ped={self.obs.ped_ahead_m}m veh={self.obs.veh_ahead_m}m ttc={self.obs.ttc_s}s", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dbg, f"{reason}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        try:
            cv2.imshow("LaneFollow StageC", dbg)
            cv2.imshow("LaneMask", lane_mask)
            cv2.waitKey(1)
        except Exception:
            pass


def main():
    rclpy.init()
    node = LaneFollowControllerStageC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
