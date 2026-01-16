#!/usr/bin/env python3

"""lane_follow_controller_stageD.py

(D) 단계 목표
- (C) 단계의 로컬 차선중심 주행 + 시맨틱 기반 안전정지를 유지하면서,
  1) GNSS Odom(속도) 기반으로 종방향 제어를 실제 속도 피드백으로 닫고
  2) Radar(거리+접근속도) 기반 ACC (간단 IDM)로 선행차량 추종
  3) Global path 곡률 기반 desired_speed를 따라 커브에서 감속
  4) 차선 인식이 흔들릴 때 global desired_heading로 steering fallback

필요 입력 토픽(기본값)
- /carla/hero/camera_rgb_front/image_color                 (sensor_msgs/Image)
- /carla/hero/camera_semseg_front/image_raw                (sensor_msgs/Image)
- /carla/hero/radar_front/point_cloud                      (sensor_msgs/PointCloud2)
- /carla/localization/gnss_odom                             (nav_msgs/Odometry)
- /carla/guidance/desired_speed                             (std_msgs/Float32)
- /carla/guidance/desired_heading                           (std_msgs/Float32, rad)

출력
- /carla/hero/cmd_vel (geometry_msgs/Twist)
  - linear.x : target speed (m/s)
  - linear.y : brake (0..1)
  - angular.z: steer_deg (CARLA bridge에서 35로 나눠 steer -1..1로 사용)

주의
- sensor_setup2.py에서 radar를 PointCloud2(x,y,z,velocity)로 발행하도록 구현되어 있어야 합니다.
- CARLA RadarDetection.velocity 는 "센서 방향으로의 속도(approaching positive)" 입니다.
"""

import math
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

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
    # yaw (Z) from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class LeadObs:
    dist_m: float = float("inf")
    closing_speed_mps: float = 0.0  # ego - lead (approx), + means we are closing


class LaneFollowControllerStageD(Node):
    def __init__(self):
        super().__init__("lane_follow_controller_stageD")

        # ------------------------ Parameters ------------------------
        self.declare_parameter("rgb_topic", "/carla/hero/camera_rgb_front/image_color")
        self.declare_parameter("sem_topic", "/carla/hero/camera_semseg_front/image_raw")
        self.declare_parameter("radar_topic", "/carla/hero/radar_front/point_cloud")
        self.declare_parameter("odom_topic", "/carla/localization/gnss_odom")
        self.declare_parameter("desired_speed_topic", "/carla/guidance/desired_speed")
        self.declare_parameter("desired_heading_topic", "/carla/guidance/desired_heading")
        self.declare_parameter("cmd_topic", "/carla/hero/cmd_vel")

        # Steering
        self.declare_parameter("steer_kp", 0.85)
        self.declare_parameter("steer_kd", 0.10)
        self.declare_parameter("max_steer_deg", 35.0)
        self.declare_parameter("lane_conf_threshold", 0.015)  # ratio of lane pixels in ROI
        self.declare_parameter("heading_kp", 1.8)              # rad -> steer (normalized) gain

        # Lane ROI
        self.declare_parameter("roi_y_start_ratio", 0.55)
        self.declare_parameter("roi_y_end_ratio", 1.0)

        # Longitudinal (IDM-ish)
        self.declare_parameter("v_default_mps", 10.0)
        self.declare_parameter("a_max", 1.8)          # accel
        self.declare_parameter("b_comf", 2.5)         # comfortable braking
        self.declare_parameter("idm_delta", 4.0)
        self.declare_parameter("time_headway", 1.2)   # T
        self.declare_parameter("s0", 2.0)             # minimum spacing
        self.declare_parameter("bumper_offset", 3.0)  # approx front bumper -> radar origin
        self.declare_parameter("max_brake_accel", 6.0) # for mapping accel->brake

        # Radar gating
        self.declare_parameter("radar_lane_half_width", 1.5)
        self.declare_parameter("radar_min_x", 0.5)
        self.declare_parameter("radar_max_x", 80.0)
        self.declare_parameter("radar_max_abs_z", 1.2)

        # Semantic safety
        self.declare_parameter("ped_hard_stop_m", 8.0)
        self.declare_parameter("veh_soft_follow_m", 25.0)  # if semantic says a vehicle ahead within this, treat as lead (stationary)

        # Uncertainty slow-down
        self.declare_parameter("uncertain_speed_scale", 0.6)

        # ------------------------ Load params ------------------------
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.sem_topic = str(self.get_parameter("sem_topic").value)
        self.radar_topic = str(self.get_parameter("radar_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.desired_speed_topic = str(self.get_parameter("desired_speed_topic").value)
        self.desired_heading_topic = str(self.get_parameter("desired_heading_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.steer_kp = float(self.get_parameter("steer_kp").value)
        self.steer_kd = float(self.get_parameter("steer_kd").value)
        self.max_steer_deg = float(self.get_parameter("max_steer_deg").value)
        self.lane_conf_threshold = float(self.get_parameter("lane_conf_threshold").value)
        self.heading_kp = float(self.get_parameter("heading_kp").value)

        self.roi_y_start_ratio = float(self.get_parameter("roi_y_start_ratio").value)
        self.roi_y_end_ratio = float(self.get_parameter("roi_y_end_ratio").value)

        self.v_default = float(self.get_parameter("v_default_mps").value)
        self.a_max = float(self.get_parameter("a_max").value)
        self.b_comf = float(self.get_parameter("b_comf").value)
        self.idm_delta = float(self.get_parameter("idm_delta").value)
        self.T = float(self.get_parameter("time_headway").value)
        self.s0 = float(self.get_parameter("s0").value)
        self.bumper_offset = float(self.get_parameter("bumper_offset").value)
        self.max_brake_accel = float(self.get_parameter("max_brake_accel").value)

        self.radar_lane_half_width = float(self.get_parameter("radar_lane_half_width").value)
        self.radar_min_x = float(self.get_parameter("radar_min_x").value)
        self.radar_max_x = float(self.get_parameter("radar_max_x").value)
        self.radar_max_abs_z = float(self.get_parameter("radar_max_abs_z").value)

        self.ped_hard_stop_m = float(self.get_parameter("ped_hard_stop_m").value)
        self.veh_soft_follow_m = float(self.get_parameter("veh_soft_follow_m").value)
        self.uncertain_speed_scale = float(self.get_parameter("uncertain_speed_scale").value)

        # ------------------------ State ------------------------
        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_sem: Optional[np.ndarray] = None

        self.v_curr: float = 0.0
        self.yaw_curr: float = 0.0
        self.have_odom: bool = False

        self.v_path: float = self.v_default
        self.have_v_path: bool = False

        self.h_des: float = 0.0
        self.have_heading: bool = False

        self.lead = LeadObs()
        self.lead_hist_dist = deque(maxlen=6)
        self.lead_hist_dv = deque(maxlen=6)

        self.prev_lane_err = 0.0
        self.prev_t = self.get_clock().now().nanoseconds * 1e-9

        # ------------------------ ROS I/O ------------------------
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.cb_rgb, 10)
        self.sub_sem = self.create_subscription(Image, self.sem_topic, self.cb_sem, 10)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 10)
        self.sub_speed = self.create_subscription(Float32, self.desired_speed_topic, self.cb_des_speed, 10)
        self.sub_heading = self.create_subscription(Float32, self.desired_heading_topic, self.cb_des_heading, 10)
        self.sub_radar = self.create_subscription(PointCloud2, self.radar_topic, self.cb_radar, 10)

        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 10)

        self.timer = self.create_timer(0.05, self.timer_cb)  # 20 Hz

        self.get_logger().info("StageD controller ready.")

    # ------------------------ Callbacks ------------------------
    def cb_rgb(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"rgb decode failed: {e}")

    def cb_sem(self, msg: Image):
        try:
            self.latest_sem = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"sem decode failed: {e}")

    def cb_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.v_curr = math.hypot(vx, vy)

        q = msg.pose.pose.orientation
        self.yaw_curr = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        self.have_odom = True

    def cb_des_speed(self, msg: Float32):
        self.v_path = float(msg.data)
        self.have_v_path = True

    def cb_des_heading(self, msg: Float32):
        self.h_des = float(msg.data)
        self.have_heading = True

    def cb_radar(self, msg: PointCloud2):
        # Read points: x,y,z,velocity (float32)
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
            # CARLA RadarDetection.velocity: velocity towards the sensor. Approaching -> positive.
            closing = max(0.0, vel_at_minx)
            dist = max(0.0, min_x - self.bumper_offset)

            self.lead_hist_dist.append(dist)
            self.lead_hist_dv.append(closing)

            self.lead.dist_m = float(np.median(self.lead_hist_dist))
            self.lead.closing_speed_mps = float(np.median(self.lead_hist_dv))
        else:
            # No detection
            self.lead_hist_dist.clear()
            self.lead_hist_dv.clear()
            self.lead = LeadObs()

    # ------------------------ Perception helpers ------------------------
    def detect_lane_steer(self, bgr: np.ndarray) -> Tuple[float, float, float]:
        """Return (steer_norm [-1,1], lane_err_norm [-1,1], confidence [0..1])."""
        h, w = bgr.shape[:2]
        y0 = int(clamp(self.roi_y_start_ratio, 0.0, 0.95) * h)
        y1 = int(clamp(self.roi_y_end_ratio, 0.05, 1.0) * h)
        roi = bgr[y0:y1, :, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # white-ish
        wmask = cv2.inRange(hsv, (0, 0, 200), (180, 55, 255))
        # yellow-ish
        ymask = cv2.inRange(hsv, (15, 70, 70), (40, 255, 255))
        mask = cv2.bitwise_or(wmask, ymask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # confidence: lane pixels ratio in ROI
        conf = float(np.count_nonzero(mask)) / float(mask.size + 1e-9)

        # compute center of mass of lane pixels per column
        col_sum = np.sum(mask > 0, axis=0).astype(np.float32)
        valid_cols = np.where(col_sum > 3)[0]

        if len(valid_cols) < 20:
            return 0.0, 0.0, conf

        # weighted average of valid columns
        lane_center_px = float(np.mean(valid_cols))
        img_center_px = 0.5 * w

        err_px = lane_center_px - img_center_px
        err_norm = float(err_px / (0.5 * w))
        err_norm = clamp(err_norm, -1.0, 1.0)

        # PD control
        steer = self.steer_kp * err_norm
        # derivative on error
        steer += self.steer_kd * (err_norm - self.prev_lane_err)
        steer = clamp(steer, -1.0, 1.0)

        return steer, err_norm, conf

    def detect_semantic_obstacles(self, sem_bgr: np.ndarray) -> Tuple[float, float]:
        """Return (ped_dist_m, veh_dist_m). inf if not found.

        sem_bgr is BGR with class ID stored in R channel (index 2) in your publisher.
        Tag IDs follow CARLA CityScapes tags (0.9.14+):
          Pedestrian=12, Rider=13, Car=14, Truck=15, Bus=16, Train=17, Motorcycle=18, Bicycle=19
        """
        h, w = sem_bgr.shape[:2]
        roi_y0 = int(0.45 * h)
        roi_y1 = int(0.95 * h)
        roi_x0 = int(0.35 * w)
        roi_x1 = int(0.65 * w)

        roi = sem_bgr[roi_y0:roi_y1, roi_x0:roi_x1]
        tags = roi[:, :, 2].astype(np.uint8)  # R

        ped_mask = (tags == 12) | (tags == 13)
        veh_mask = (tags == 14) | (tags == 15) | (tags == 16) | (tags == 17) | (tags == 18) | (tags == 19)

        ped_dist = float("inf")
        veh_dist = float("inf")

        # crude pixel-row to distance mapping (same spirit as StageC)
        # top of ROI: far, bottom: near
        def row_to_dist(row_idx: int, total_rows: int) -> float:
            t = 1.0 - (row_idx / max(1, total_rows - 1))
            # map [0..1] -> [1.5..30]
            return 1.5 + (30.0 - 1.5) * t

        if np.any(ped_mask):
            rows = np.where(np.any(ped_mask, axis=1))[0]
            ped_dist = min(row_to_dist(int(rows.max()), ped_mask.shape[0]), 30.0)

        if np.any(veh_mask):
            rows = np.where(np.any(veh_mask, axis=1))[0]
            veh_dist = min(row_to_dist(int(rows.max()), veh_mask.shape[0]), 30.0)

        return ped_dist, veh_dist

    # ------------------------ Control helpers ------------------------
    def idm_accel(self, v: float, v0: float, s: float, dv: float) -> float:
        """IDM acceleration.

        v  : ego speed (m/s)
        v0 : desired speed (m/s)
        s  : gap to lead (m)
        dv : closing speed (ego - lead) (m/s). + means closing.
        """
        v0 = max(0.1, v0)
        s = max(0.1, s)

        # desired dynamic gap
        s_star = self.s0 + max(0.0, v * self.T + (v * dv) / (2.0 * math.sqrt(self.a_max * self.b_comf)))

        term_free = 1.0 - (v / v0) ** self.idm_delta
        term_int = (s_star / s) ** 2
        return self.a_max * (term_free - term_int)

    def compute_longitudinal(self, dt: float, lane_conf: float, ped_dist: float, sem_veh_dist: float) -> Tuple[float, float]:
        """Return (v_target, brake)."""
        # path speed upper bound
        v0 = self.v_path if self.have_v_path else self.v_default

        # if lane perception is weak, slow down proactively
        if lane_conf < self.lane_conf_threshold:
            v0 *= self.uncertain_speed_scale

        v0 = clamp(v0, 0.0, 30.0)

        # Hard pedestrian stop
        if ped_dist < self.ped_hard_stop_m:
            return 0.0, 1.0

        # Lead from radar (preferred for vehicles)
        lead_dist = self.lead.dist_m
        closing = self.lead.closing_speed_mps

        # If semantic suggests something (vehicle-ish) closer than radar (or radar missing), treat as stationary lead
        if sem_veh_dist < self.veh_soft_follow_m and sem_veh_dist < lead_dist:
            lead_dist = sem_veh_dist
            closing = self.v_curr  # assume stationary obstacle, conservative

        # If we have a lead, run IDM. Else accelerate towards v0.
        if math.isfinite(lead_dist):
            a = self.idm_accel(self.v_curr, v0, lead_dist, closing)
        else:
            # free-road acceleration shape
            v0_eff = max(0.1, v0)
            a = self.a_max * (1.0 - (self.v_curr / v0_eff) ** self.idm_delta)

        v_target = self.v_curr + a * dt
        v_target = clamp(v_target, 0.0, v0)

        # map negative accel to brake (bridge uses brake to cut throttle)
        brake = 0.0
        if a < -0.3:
            brake = clamp((-a) / max(0.1, self.max_brake_accel), 0.0, 1.0)
            v_target = min(v_target, self.v_curr)

        return v_target, brake

    # ------------------------ Main timer ------------------------
    def timer_cb(self):
        if self.latest_rgb is None or self.latest_sem is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(0.01, now - self.prev_t)
        self.prev_t = now

        # --- steering ---
        steer_lane, lane_err, lane_conf = self.detect_lane_steer(self.latest_rgb)
        self.prev_lane_err = lane_err

        steer_cmd = steer_lane

        # fallback / blend with desired heading when lane is weak
        if (lane_conf < self.lane_conf_threshold) and self.have_heading and self.have_odom:
            h_err = wrap_pi(self.h_des - self.yaw_curr)
            max_steer_rad = math.radians(max(5.0, self.max_steer_deg))
            steer_head = clamp(self.heading_kp * (h_err / max_steer_rad), -1.0, 1.0)
            # blend: when lane_conf is very low, trust heading more
            w_lane = clamp(lane_conf / max(self.lane_conf_threshold, 1e-6), 0.0, 1.0)
            steer_cmd = w_lane * steer_lane + (1.0 - w_lane) * steer_head

        steer_deg = steer_cmd * self.max_steer_deg

        # --- obstacle perception from semantics ---
        ped_dist, sem_veh_dist = self.detect_semantic_obstacles(self.latest_sem)

        # --- longitudinal ---
        v_target, brake = self.compute_longitudinal(dt, lane_conf, ped_dist, sem_veh_dist)

        # publish command
        cmd = Twist()
        cmd.linear.x = float(v_target)
        cmd.linear.y = float(brake)
        cmd.angular.z = float(steer_deg)
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowControllerStageD()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
