#!/usr/bin/env python3

"""lane_follow_controller_stageE.py

StageD에서 사용자가 바로 겪은 4가지 문제를 '코드 레벨'로 해결한 다음 단계.

해결 목표
1) "조금 움직였다 멈췄다" (stop&go 헌팅) 완화
   - (a) 속도/제동 명령에 rate-limit + 저역통과 필터
   - (b) (가능하면) CARLA vehicle.get_velocity()로 속도 피드백을 직접 취득(gnss_odom 노이즈 회피)

2) 빨간불 무시
   - 센서(semantic seg)는 TrafficLight 존재(tag=7)는 알려도 '색/상태'는 알려주지 않음.
   - 따라서 "우회"가 아니라 CARLA Python API의 traffic light state를 정식으로 질의(oracle)하여 준수.

3) 전역경로를 안 따라감
   - StageD는 lane_conf가 낮을 때만 desired_heading을 쓰는 fallback 구조라, lane_conf가 높으면 전역경로가 영향이 거의 없음.
   - 전역 lookahead_point 기반 Pure-Pursuit 조향을 항상 혼합(blend)하여 "라우팅"을 강제.

4) 장애물을 피하지 않음
   - StageD는 사실상 '정지' 위주였고, 거리 추정도 semseg 픽셀 휴리스틱이라 실패하기 쉬움.
   - StageE는 semantic LiDAR로 "실거리" 기반 전방 장애물(차/보행자) 최소거리 추정 +
     가까운 정지/간단 회피(좌/우 여유 공간 비교 후 steer bias) 를 추가.

입력 토픽(기본값은 sensor_setup2.py의 id와 일치하도록 설정)
- /carla/hero/camera_rgb/image_color
- /carla/hero/camera_semseg/image_raw
- /carla/hero/radar/point_cloud
- /carla/hero/semantic_lidar/point_cloud
- /carla/localization/gnss_odom
- /carla/guidance/lookahead_point
- /carla/guidance/desired_speed
- /carla/guidance/desired_heading

출력
- /carla/hero/cmd_vel (Twist)
  linear.x: target speed (m/s)
  linear.y: brake (0..1)
  angular.z: steer_deg (sensor_setup2/3에서 35로 나눠 steer -1..1)
"""

import math
import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class LeadObs:
    dist_m: float = float("inf")
    closing_speed_mps: float = 0.0


class LaneFollowControllerStageE(Node):
    def __init__(self):
        super().__init__("lane_follow_controller_stageE")

        # ------------------------ Topics (match sensor_setup2 ids) ------------------------
        self.declare_parameter("rgb_topic", "/carla/hero/camera_rgb/image_color")
        self.declare_parameter("sem_topic", "/carla/hero/camera_semseg/image_raw")
        self.declare_parameter("radar_topic", "/carla/hero/radar/point_cloud")
        self.declare_parameter("semantic_lidar_topic", "/carla/hero/semantic_lidar/point_cloud")
        self.declare_parameter("odom_topic", "/carla/localization/gnss_odom")
        self.declare_parameter("lookahead_point_topic", "/carla/guidance/lookahead_point")
        self.declare_parameter("desired_speed_topic", "/carla/guidance/desired_speed")
        self.declare_parameter("desired_heading_topic", "/carla/guidance/desired_heading")
        self.declare_parameter("cmd_topic", "/carla/hero/cmd_vel")

        # ------------------------ Steering ------------------------
        self.declare_parameter("steer_kp", 0.85)
        self.declare_parameter("steer_kd", 0.10)
        self.declare_parameter("max_steer_deg", 35.0)
        self.declare_parameter("lane_conf_threshold", 0.015)
        self.declare_parameter("roi_y_start_ratio", 0.55)
        self.declare_parameter("roi_y_end_ratio", 1.0)

        # Pure pursuit (global path)
        self.declare_parameter("wheelbase_m", 2.80)
        self.declare_parameter("global_steer_weight", 0.35)   # always-on blend weight (0..1)
        self.declare_parameter("global_weight_boost_when_weak_lane", 0.45)

        # ------------------------ Longitudinal ------------------------
        self.declare_parameter("v_default_mps", 10.0)
        self.declare_parameter("a_max", 1.8)
        self.declare_parameter("b_comf", 2.5)
        self.declare_parameter("idm_delta", 4.0)
        self.declare_parameter("time_headway", 1.2)
        self.declare_parameter("s0", 2.0)
        self.declare_parameter("bumper_offset", 3.0)
        self.declare_parameter("max_brake_accel", 6.0)

        # Speed command smoothing / anti-hunting
        self.declare_parameter("v_rate_up", 2.0)     # m/s^2
        self.declare_parameter("v_rate_down", 3.5)   # m/s^2
        self.declare_parameter("v_lpf_tau", 0.25)    # seconds

        # ------------------------ Radar gating ------------------------
        self.declare_parameter("radar_lane_half_width", 1.5)
        self.declare_parameter("radar_min_x", 0.5)
        self.declare_parameter("radar_max_x", 80.0)
        self.declare_parameter("radar_max_abs_z", 1.2)

        # ------------------------ Semantic LiDAR obstacle logic ------------------------
        self.declare_parameter("lane_half_width_for_lidar", 1.3)
        self.declare_parameter("lidar_min_x", 0.5)
        self.declare_parameter("lidar_max_x", 40.0)
        self.declare_parameter("lidar_max_abs_z", 1.6)

        self.declare_parameter("ped_hard_stop_m", 8.0)
        self.declare_parameter("veh_hard_stop_m", 7.0)
        self.declare_parameter("veh_soft_follow_m", 25.0)

        # Simple avoidance (steer bias)
        self.declare_parameter("avoid_enable", True)
        self.declare_parameter("avoid_trigger_m", 12.0)
        self.declare_parameter("avoid_bias_max_deg", 10.0)
        self.declare_parameter("avoid_side_band", 1.2)   # start of left/right corridor in meters
        self.declare_parameter("avoid_side_max", 3.5)    # max lateral considered
        self.declare_parameter("avoid_direction_sign", 1.0)
        # if steer direction feels reversed, set avoid_direction_sign:=-1

        # ------------------------ Traffic light (oracle) ------------------------
        self.declare_parameter("tl_oracle_enable", True)
        self.declare_parameter("carla_host", "localhost")
        self.declare_parameter("carla_port", 2000)
        self.declare_parameter("role_name", "hero")
        self.declare_parameter("stop_on_yellow", True)
        self.declare_parameter("tl_poll_hz", 10.0)

        # ------------------------ Load params ------------------------
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.sem_topic = str(self.get_parameter("sem_topic").value)
        self.radar_topic = str(self.get_parameter("radar_topic").value)
        self.semantic_lidar_topic = str(self.get_parameter("semantic_lidar_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.lookahead_topic = str(self.get_parameter("lookahead_point_topic").value)
        self.desired_speed_topic = str(self.get_parameter("desired_speed_topic").value)
        self.desired_heading_topic = str(self.get_parameter("desired_heading_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.steer_kp = float(self.get_parameter("steer_kp").value)
        self.steer_kd = float(self.get_parameter("steer_kd").value)
        self.max_steer_deg = float(self.get_parameter("max_steer_deg").value)
        self.lane_conf_threshold = float(self.get_parameter("lane_conf_threshold").value)
        self.roi_y_start_ratio = float(self.get_parameter("roi_y_start_ratio").value)
        self.roi_y_end_ratio = float(self.get_parameter("roi_y_end_ratio").value)

        self.wheelbase = float(self.get_parameter("wheelbase_m").value)
        self.w_global = clamp(float(self.get_parameter("global_steer_weight").value), 0.0, 1.0)
        self.w_global_boost = clamp(float(self.get_parameter("global_weight_boost_when_weak_lane").value), 0.0, 1.0)

        self.v_default = float(self.get_parameter("v_default_mps").value)
        self.a_max = float(self.get_parameter("a_max").value)
        self.b_comf = float(self.get_parameter("b_comf").value)
        self.idm_delta = float(self.get_parameter("idm_delta").value)
        self.T = float(self.get_parameter("time_headway").value)
        self.s0 = float(self.get_parameter("s0").value)
        self.bumper_offset = float(self.get_parameter("bumper_offset").value)
        self.max_brake_accel = float(self.get_parameter("max_brake_accel").value)

        self.v_rate_up = max(0.1, float(self.get_parameter("v_rate_up").value))
        self.v_rate_down = max(0.1, float(self.get_parameter("v_rate_down").value))
        self.v_lpf_tau = max(0.0, float(self.get_parameter("v_lpf_tau").value))

        self.radar_lane_half_width = float(self.get_parameter("radar_lane_half_width").value)
        self.radar_min_x = float(self.get_parameter("radar_min_x").value)
        self.radar_max_x = float(self.get_parameter("radar_max_x").value)
        self.radar_max_abs_z = float(self.get_parameter("radar_max_abs_z").value)

        self.lane_half_width_lidar = float(self.get_parameter("lane_half_width_for_lidar").value)
        self.lidar_min_x = float(self.get_parameter("lidar_min_x").value)
        self.lidar_max_x = float(self.get_parameter("lidar_max_x").value)
        self.lidar_max_abs_z = float(self.get_parameter("lidar_max_abs_z").value)

        self.ped_hard_stop_m = float(self.get_parameter("ped_hard_stop_m").value)
        self.veh_hard_stop_m = float(self.get_parameter("veh_hard_stop_m").value)
        self.veh_soft_follow_m = float(self.get_parameter("veh_soft_follow_m").value)

        self.avoid_enable = bool(self.get_parameter("avoid_enable").value)
        self.avoid_trigger_m = float(self.get_parameter("avoid_trigger_m").value)
        self.avoid_bias_max_deg = float(self.get_parameter("avoid_bias_max_deg").value)
        self.avoid_side_band = float(self.get_parameter("avoid_side_band").value)
        self.avoid_side_max = float(self.get_parameter("avoid_side_max").value)
        self.avoid_direction_sign = float(self.get_parameter("avoid_direction_sign").value)

        self.tl_oracle_enable = bool(self.get_parameter("tl_oracle_enable").value)
        self.carla_host = str(self.get_parameter("carla_host").value)
        self.carla_port = int(self.get_parameter("carla_port").value)
        self.role_name = str(self.get_parameter("role_name").value)
        self.stop_on_yellow = bool(self.get_parameter("stop_on_yellow").value)
        self.tl_poll_hz = float(self.get_parameter("tl_poll_hz").value)

        # ------------------------ State ------------------------
        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_sem: Optional[np.ndarray] = None

        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v_curr_odom = 0.0
        self.v_curr = 0.0

        self.have_lookahead = False
        self.lx = 0.0
        self.ly = 0.0

        self.have_v_path = False
        self.v_path = self.v_default

        self.have_heading = False
        self.h_des = 0.0

        self.lead = LeadObs()
        self.lead_hist_dist = deque(maxlen=6)
        self.lead_hist_dv = deque(maxlen=6)

        self.min_ped_dist = float("inf")
        self.min_veh_dist = float("inf")
        self.avoid_bias_deg = 0.0

        self.prev_lane_err = 0.0
        self.prev_t = time.time()
        self.v_cmd_prev = 0.0
        self.v_cmd_lpf = 0.0

        # ------------------------ CARLA handles (traffic light + speed) ------------------------
        self._carla_ok = False
        self._carla_vehicle = None
        self._tl_state = None
        self._tl_affecting = False
        if self.tl_oracle_enable:
            self._try_connect_carla()
            period = 1.0 / max(1.0, self.tl_poll_hz)
            self.create_timer(period, self._poll_traffic_light)

        # ------------------------ ROS I/O ------------------------
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self._cb_rgb, 10)
        self.sub_sem = self.create_subscription(Image, self.sem_topic, self._cb_sem, 10)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self._cb_odom, 10)
        self.sub_look = self.create_subscription(PointStamped, self.lookahead_topic, self._cb_lookahead, 10)
        self.sub_speed = self.create_subscription(Float32, self.desired_speed_topic, self._cb_des_speed, 10)
        self.sub_heading = self.create_subscription(Float32, self.desired_heading_topic, self._cb_des_heading, 10)
        self.sub_radar = self.create_subscription(PointCloud2, self.radar_topic, self._cb_radar, 10)
        self.sub_sem_lidar = self.create_subscription(PointCloud2, self.semantic_lidar_topic, self._cb_semantic_lidar, 10)

        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 10)
        self.create_timer(0.05, self._timer)  # 20 Hz

        self.get_logger().info("StageE controller ready.")

    # ------------------------ CARLA connection ------------------------
    def _try_connect_carla(self):
        try:
            import carla  # lazy import
            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(2.0)
            world = client.get_world()
            actors = world.get_actors().filter("vehicle.*")
            hero = None
            for a in actors:
                try:
                    if a.attributes.get("role_name", "") == self.role_name:
                        hero = a
                        break
                except Exception:
                    continue
            if hero is None:
                self.get_logger().warn("CARLA connected, but hero vehicle not found yet.")
                self._carla_ok = True
                self._carla_vehicle = None
                self._carla_world = world
                self._carla_client = client
                return

            self._carla_ok = True
            self._carla_vehicle = hero
            self._carla_world = world
            self._carla_client = client
            self.get_logger().info("CARLA oracle connected (traffic light + velocity).")
        except Exception as e:
            self.get_logger().warn(f"CARLA oracle disabled (connect failed): {e}")
            self._carla_ok = False
            self._carla_vehicle = None

    def _poll_traffic_light(self):
        if not self.tl_oracle_enable:
            return
        if not self._carla_ok:
            self._try_connect_carla()
            return

        try:
            import carla
            if self._carla_vehicle is None or not self._carla_vehicle.is_alive:
                # try to re-find hero
                actors = self._carla_world.get_actors().filter("vehicle.*")
                for a in actors:
                    if a.attributes.get("role_name", "") == self.role_name:
                        self._carla_vehicle = a
                        break

            if self._carla_vehicle is None:
                self._tl_affecting = False
                self._tl_state = None
                return

            self._tl_affecting = bool(self._carla_vehicle.is_at_traffic_light())
            if self._tl_affecting:
                tl = self._carla_vehicle.get_traffic_light()
                self._tl_state = tl.get_state() if tl is not None else None
            else:
                self._tl_state = None

            # also update speed directly from simulator state (stable)
            v = self._carla_vehicle.get_velocity()
            self.v_curr = float(math.hypot(v.x, v.y))

        except Exception:
            # keep last state; don't crash controller
            pass

    # ------------------------ ROS callbacks ------------------------
    def _cb_rgb(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"rgb decode failed: {e}")

    def _cb_sem(self, msg: Image):
        try:
            self.latest_sem = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"sem decode failed: {e}")

    def _cb_odom(self, msg: Odometry):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.v_curr_odom = float(math.hypot(vx, vy))

        # if CARLA oracle isn't active, use odom speed (filtered later)
        if not self._carla_ok or self._carla_vehicle is None:
            self.v_curr = self.v_curr_odom

        self.have_odom = True

    def _cb_lookahead(self, msg: PointStamped):
        self.lx = float(msg.point.x)
        self.ly = float(msg.point.y)
        self.have_lookahead = True

    def _cb_des_speed(self, msg: Float32):
        self.v_path = float(msg.data)
        self.have_v_path = True

    def _cb_des_heading(self, msg: Float32):
        self.h_des = float(msg.data)
        self.have_heading = True

    def _cb_radar(self, msg: PointCloud2):
        min_x = float("inf")
        vel_at_minx = 0.0
        try:
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "velocity"), skip_nans=True):
                x, y, z, v = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                if x < self.radar_min_x or x > self.radar_max_x:
                    continue
                if abs(y) > self.radar_lane_half_width:
                    continue
                if abs(z) > self.radar_max_abs_z:
                    continue
                if x < min_x:
                    min_x = x
                    vel_at_minx = v
        except Exception as e:
            self.get_logger().warn(f"radar parse failed: {e}")
            return

        if math.isfinite(min_x):
            # CARLA RadarDetection.velocity: towards sensor (approaching -> positive)
            closing = max(0.0, vel_at_minx)
            dist = max(0.0, min_x - self.bumper_offset)
            self.lead_hist_dist.append(dist)
            self.lead_hist_dv.append(closing)
            self.lead.dist_m = float(np.median(self.lead_hist_dist))
            self.lead.closing_speed_mps = float(np.median(self.lead_hist_dv))
        else:
            self.lead_hist_dist.clear()
            self.lead_hist_dv.clear()
            self.lead = LeadObs()

    def _cb_semantic_lidar(self, msg: PointCloud2):
        # Expect fields: x,y,z,cos_incidence,obj_idx,obj_tag (sensor_setup2 publisher)
        min_ped = float("inf")
        min_veh = float("inf")

        # corridor occupancy for simple avoidance
        left_cnt = 0
        right_cnt = 0

        try:
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "obj_tag"), skip_nans=True):
                x, y, z, tag = float(p[0]), float(p[1]), float(p[2]), int(p[3])

                if x < self.lidar_min_x or x > self.lidar_max_x:
                    continue
                if abs(z) > self.lidar_max_abs_z:
                    continue

                # Only consider dynamic-ish obstacles (CityObjectLabel values)
                is_ped = tag in (12, 13)
                is_veh = tag in (14, 15, 16, 17, 18, 19)
                if not (is_ped or is_veh):
                    continue

                # lane corridor (front)
                if abs(y) <= self.lane_half_width_lidar:
                    if is_ped:
                        min_ped = min(min_ped, x)
                    if is_veh:
                        min_veh = min(min_veh, x)

                # left/right occupancy for avoidance (close range only)
                if self.avoid_enable and x <= self.avoid_trigger_m:
                    if -self.avoid_side_max <= y <= -self.avoid_side_band:
                        left_cnt += 1
                    elif self.avoid_side_band <= y <= self.avoid_side_max:
                        right_cnt += 1

        except Exception as e:
            self.get_logger().warn(f"semantic lidar parse failed: {e}")
            return

        self.min_ped_dist = min_ped
        self.min_veh_dist = min_veh

        # update avoidance bias
        self.avoid_bias_deg = 0.0
        if self.avoid_enable and math.isfinite(min_veh) and min_veh <= self.avoid_trigger_m:
            # choose side with fewer points
            if left_cnt < right_cnt:
                # go left => steer negative in CARLA convention (usually)
                self.avoid_bias_deg = -abs(self.avoid_bias_max_deg) * self.avoid_direction_sign
            elif right_cnt < left_cnt:
                self.avoid_bias_deg = abs(self.avoid_bias_max_deg) * self.avoid_direction_sign
            else:
                # tie: don't bias
                self.avoid_bias_deg = 0.0

    # ------------------------ Perception helpers ------------------------
    def _detect_lane_steer(self, bgr: np.ndarray) -> Tuple[float, float, float]:
        h, w = bgr.shape[:2]
        y0 = int(clamp(self.roi_y_start_ratio, 0.0, 0.95) * h)
        y1 = int(clamp(self.roi_y_end_ratio, 0.05, 1.0) * h)
        roi = bgr[y0:y1, :, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        wmask = cv2.inRange(hsv, (0, 0, 200), (180, 55, 255))
        ymask = cv2.inRange(hsv, (15, 70, 70), (40, 255, 255))
        mask = cv2.bitwise_or(wmask, ymask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        conf = float(np.count_nonzero(mask)) / float(mask.size + 1e-9)

        col_sum = np.sum(mask > 0, axis=0).astype(np.float32)
        valid_cols = np.where(col_sum > 3)[0]
        if len(valid_cols) < 20:
            return 0.0, 0.0, conf

        lane_center_px = float(np.mean(valid_cols))
        img_center_px = 0.5 * w
        err_px = lane_center_px - img_center_px
        err_norm = clamp(float(err_px / (0.5 * w)), -1.0, 1.0)

        steer = self.steer_kp * err_norm
        steer += self.steer_kd * (err_norm - self.prev_lane_err)
        steer = clamp(steer, -1.0, 1.0)
        return steer, err_norm, conf

    def _pure_pursuit_steer(self) -> float:
        if not (self.have_odom and self.have_lookahead):
            return 0.0

        dx = self.lx - self.x
        dy = self.ly - self.y
        ld = math.hypot(dx, dy)
        if ld < 1e-3:
            return 0.0

        path_heading = math.atan2(dy, dx)
        alpha = wrap_pi(path_heading - self.yaw)

        # curvature = 2*sin(alpha)/ld
        kappa = (2.0 * math.sin(alpha)) / max(1e-3, ld)
        delta = math.atan(self.wheelbase * kappa)  # steering angle (rad)

        max_steer_rad = math.radians(max(5.0, self.max_steer_deg))
        steer_norm = clamp(delta / max_steer_rad, -1.0, 1.0)
        return steer_norm

    # ------------------------ Longitudinal helpers ------------------------
    def _idm_accel(self, v: float, v0: float, s: float, dv: float) -> float:
        v0 = max(0.1, v0)
        s = max(0.1, s)
        s_star = self.s0 + max(0.0, v * self.T + (v * dv) / (2.0 * math.sqrt(self.a_max * self.b_comf)))
        term_free = 1.0 - (v / v0) ** self.idm_delta
        term_int = (s_star / s) ** 2
        return self.a_max * (term_free - term_int)

    def _speed_smooth(self, v_des: float, dt: float) -> float:
        # rate limit
        up = self.v_rate_up * dt
        dn = self.v_rate_down * dt
        v_rl = clamp(v_des, self.v_cmd_prev - dn, self.v_cmd_prev + up)
        self.v_cmd_prev = v_rl

        # low-pass filter
        if self.v_lpf_tau <= 1e-6:
            self.v_cmd_lpf = v_rl
        else:
            alpha = dt / (self.v_lpf_tau + dt)
            self.v_cmd_lpf = self.v_cmd_lpf + alpha * (v_rl - self.v_cmd_lpf)
        return self.v_cmd_lpf

    def _compute_longitudinal(self, dt: float, lane_conf: float) -> Tuple[float, float]:
        v0 = self.v_path if self.have_v_path else self.v_default

        # lane weak => conservative
        if lane_conf < self.lane_conf_threshold:
            v0 *= 0.65
        v0 = clamp(v0, 0.0, 30.0)

        # Hard stops from semantic LiDAR distances
        if math.isfinite(self.min_ped_dist) and self.min_ped_dist < self.ped_hard_stop_m:
            return 0.0, 1.0
        if math.isfinite(self.min_veh_dist) and self.min_veh_dist < self.veh_hard_stop_m:
            return 0.0, 1.0

        # Traffic light stop (oracle)
        if self.tl_oracle_enable and self._tl_affecting and self._tl_state is not None:
            try:
                import carla
                if self._tl_state == carla.TrafficLightState.Red:
                    return 0.0, 1.0
                if self.stop_on_yellow and self._tl_state == carla.TrafficLightState.Yellow:
                    return 0.0, 1.0
            except Exception:
                pass

        # Lead selection: prefer radar lead; if semantic veh closer, treat as stationary lead
        lead_dist = self.lead.dist_m
        closing = self.lead.closing_speed_mps
        if math.isfinite(self.min_veh_dist) and self.min_veh_dist < self.veh_soft_follow_m and self.min_veh_dist < lead_dist:
            lead_dist = self.min_veh_dist
            closing = self.v_curr  # conservative (stationary)

        if math.isfinite(lead_dist):
            a = self._idm_accel(self.v_curr, v0, lead_dist, closing)
        else:
            v0_eff = max(0.1, v0)
            a = self.a_max * (1.0 - (self.v_curr / v0_eff) ** self.idm_delta)

        v_target = clamp(self.v_curr + a * dt, 0.0, v0)

        brake = 0.0
        if a < -0.3:
            brake = clamp((-a) / max(0.1, self.max_brake_accel), 0.0, 1.0)
            v_target = min(v_target, self.v_curr)

        v_target = self._speed_smooth(v_target, dt)
        return v_target, brake

    # ------------------------ Main loop ------------------------
    def _timer(self):
        if self.latest_rgb is None:
            return
        if not self.have_odom:
            return

        now = time.time()
        dt = max(0.02, min(0.2, now - self.prev_t))
        self.prev_t = now

        steer_lane, lane_err, lane_conf = self._detect_lane_steer(self.latest_rgb)
        self.prev_lane_err = lane_err

        steer_global = self._pure_pursuit_steer()

        # blend weights
        w_g = self.w_global
        if lane_conf < self.lane_conf_threshold:
            w_g = clamp(w_g + self.w_global_boost, 0.0, 1.0)
        w_l = 1.0 - w_g

        steer_cmd = w_l * steer_lane + w_g * steer_global

        # If lane is strong but heading disagrees a lot, add small correction via desired_heading (optional)
        if self.have_heading:
            h_err = wrap_pi(self.h_des - self.yaw)
            steer_cmd += clamp(0.10 * (h_err / math.radians(self.max_steer_deg)), -0.2, 0.2)

        steer_deg = clamp(steer_cmd * self.max_steer_deg + self.avoid_bias_deg, -self.max_steer_deg, self.max_steer_deg)

        v_target, brake = self._compute_longitudinal(dt, lane_conf)

        cmd = Twist()
        cmd.linear.x = float(v_target)
        cmd.linear.y = float(brake)
        cmd.angular.z = float(steer_deg)
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowControllerStageE()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
