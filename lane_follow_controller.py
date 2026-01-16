#!/usr/bin/env python3
"""Lane-based local planner + controller for CARLA ROS2 native bridge.

This node outputs geometry_msgs/Twist on /carla/hero/cmd_vel using the convention
implemented by sensor_setup2.py:
  - linear.x  : target speed (m/s)
  - angular.z : steering command in DEGREES (bridge divides by 35.0 -> [-1, 1])
  - linear.y  : brake [0..1]

Subscriptions:
  - /carla/hero/camera_rgb/image_color           (sensor_msgs/Image, bgr8)
  - /carla/hero/camera_semseg/image_raw         (sensor_msgs/Image, bgr8; semantic id stored in R channel)
  - /carla/guidance/desired_heading             (std_msgs/Float32, radians)
  - /carla/localization/gnss_pose               (geometry_msgs/PoseStamped)
  - /carla/hero/semantic_lidar/point_cloud      (sensor_msgs/PointCloud2) [optional]
  - /carla/hero/radar/point_cloud               (sensor_msgs/PointCloud2) [optional]

Publishes:
  - /carla/hero/cmd_vel                         (geometry_msgs/Twist)
  - /carla/debug/lane_image                     (sensor_msgs/Image) debug overlay

Notes:
  - Semantic segmentation does NOT directly label lane markings; here we use it primarily
    as a road/sidewalk ROI mask + a robust fallback "road center" estimate.
  - Obstacle guard uses semantic_lidar + radar if available; if not, it degrades gracefully.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32

try:
    from cv_bridge import CvBridge
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "cv_bridge is required. Install with: sudo apt install ros-humble-cv-bridge"
    ) from e

# PointCloud2 helpers (ROS2)
try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    pc2 = None


def wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw from quaternion (Z-up)."""
    # Standard yaw extraction
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class LaneEstimate:
    ok: bool
    target_x: float
    target_y: float
    lane_width_px: float
    method: str


class LaneFollowController(Node):
    def __init__(self):
        super().__init__("lane_follow_controller")

        # --- Parameters ---
        self.declare_parameter("rgb_topic", "/carla/hero/camera_rgb/image_color")
        self.declare_parameter("wide_topic", "/carla/hero/camera_wide/image_color")
        self.declare_parameter("use_wide_fallback", True)
        self.declare_parameter("use_semseg_roi", True)
        self.declare_parameter("semseg_topic", "/carla/hero/camera_semseg/image_raw")
        self.declare_parameter("cmd_topic", "/carla/hero/cmd_vel")
        self.declare_parameter("debug_topic", "/carla/debug/lane_image")

        self.declare_parameter("target_speed", 8.0)  # m/s
        self.declare_parameter("turn_speed", 5.0)    # m/s when turning

        self.declare_parameter("steer_deg_limit", 35.0)  # bridge expects +/-35 deg
        self.declare_parameter("k_lane_deg", 20.0)       # deg per normalized pixel error
        self.declare_parameter("k_heading_deg", 12.0)    # deg per rad heading error

        self.declare_parameter("roi_y_start", 0.55)      # fraction of height
        self.declare_parameter("roi_y_end", 0.95)
        self.declare_parameter("hough_min_line_len", 40)
        self.declare_parameter("hough_max_line_gap", 60)

        # Semantic segmentation label id used as "road" in CARLA (default = 7 in CARLA semantic tag set).
        # If yours differs, set parameter road_label.
        self.declare_parameter("road_label", 7)

        # Turn bias based on global desired_heading - current yaw
        self.declare_parameter("turn_bias_px", 120)      # pixels to bias target on sharp turns
        self.declare_parameter("turn_yaw_thresh", 0.25)  # radians

        # Obstacle guard
        self.declare_parameter("use_obstacle_guard", True)
        self.declare_parameter("semantic_lidar_topic", "/carla/hero/semantic_lidar/point_cloud")
        self.declare_parameter("radar_topic", "/carla/hero/radar/point_cloud")
        self.declare_parameter("roi_half_width_m", 1.8)  # forward corridor half width
        self.declare_parameter("stop_distance_m", 6.0)
        self.declare_parameter("slow_distance_m", 14.0)
        self.declare_parameter("ttc_brake_s", 1.6)

        # Semantic LiDAR object tags to consider obstacles (defaults aligned with CARLA semantic tags)
        # vehicles=10, pedestrians=4, bicycle/motorcycle might appear under "Vehicles" in some maps.
        self.declare_parameter("obstacle_tags", [10, 4])

        # --- State ---
        self.bridge = CvBridge()
        self.last_rgb: Optional[np.ndarray] = None
        self.last_wide: Optional[np.ndarray] = None
        self.last_sem: Optional[np.ndarray] = None
        self.last_img_t: float = 0.0

        self.desired_heading: float = 0.0
        self.have_heading: bool = False

        self.current_yaw: float = 0.0
        self.have_pose: bool = False

        self.last_lane_width_px: float = 650.0

        self.last_obstacle_dist: Optional[float] = None
        self.last_radar_ttc: Optional[float] = None
        self.last_pc_t: float = 0.0

        # --- ROS I/O ---
        qos_sensor = QoSProfile(depth=5)
        qos_sensor.reliability = ReliabilityPolicy.BEST_EFFORT

        rgb_topic = self.get_parameter("rgb_topic").value
        sem_topic = self.get_parameter("semseg_topic").value
        self.create_subscription(Image, rgb_topic, self.cb_rgb, qos_sensor)

        self.use_wide = bool(self.get_parameter("use_wide_fallback").value)
        if self.use_wide:
            wide_topic = self.get_parameter("wide_topic").value
            self.create_subscription(Image, wide_topic, self.cb_wide, qos_sensor)
        self.create_subscription(Image, sem_topic, self.cb_semseg, qos_sensor)
        self.create_subscription(Float32, "/carla/guidance/desired_heading", self.cb_heading, 10)
        self.create_subscription(PoseStamped, "/carla/localization/gnss_pose", self.cb_pose, 10)

        self.pub_cmd = self.create_publisher(Twist, self.get_parameter("cmd_topic").value, 10)
        self.pub_dbg = self.create_publisher(Image, self.get_parameter("debug_topic").value, 1)

        self.use_guard = bool(self.get_parameter("use_obstacle_guard").value)

        if self.use_guard and pc2 is not None:
            self.create_subscription(
                PointCloud2,
                self.get_parameter("semantic_lidar_topic").value,
                self.cb_semantic_lidar,
                qos_sensor,
            )
            self.create_subscription(
                PointCloud2,
                self.get_parameter("radar_topic").value,
                self.cb_radar,
                qos_sensor,
            )
        elif self.use_guard and pc2 is None:
            self.get_logger().warn("sensor_msgs_py not available; obstacle guard disabled")
            self.use_guard = False

        self.timer = self.create_timer(0.05, self.loop)  # 20Hz

        self.get_logger().info("LaneFollowController ready.")

    # --- Callbacks ---
    def cb_rgb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB decode failed: {e}")
            return
        self.last_rgb = img
        self.last_img_t = time.time()

    def cb_wide(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Wide RGB decode failed: {e}")
            return
        self.last_wide = img
        self.last_img_t = time.time()

    def cb_semseg(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"SemSeg decode failed: {e}")
            return
        self.last_sem = img
        self.last_img_t = time.time()

    def cb_heading(self, msg: Float32):
        self.desired_heading = float(msg.data)
        self.have_heading = True

    def cb_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        self.current_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.have_pose = True

    def cb_semantic_lidar(self, msg):
        # msg is sensor_msgs/PointCloud2
        self.last_pc_t = time.time()
        self.last_obstacle_dist = self.compute_obstacle_distance_semantic_lidar(msg)

    def cb_radar(self, msg):
        self.last_pc_t = time.time()
        self.last_radar_ttc = self.compute_ttc_from_radar(msg)

    # --- Core loop ---
    def loop(self):
        if self.last_rgb is None:
            return

        # Guard against stale images
        if time.time() - self.last_img_t > 0.5:
            return

        rgb = self.last_rgb
        sem = self.last_sem  # may be None

        lane = self.estimate_lane_target(rgb, sem)

        # Wide-angle fallback (useful near intersections / sharp turns)
        if (not lane.ok) and getattr(self, "use_wide", False) and self.last_wide is not None:
            wide = self.last_wide
            lane2 = self.estimate_lane_target(wide, None)
            if lane2.ok:
                rgb = wide
                lane = lane2
                lane.method = "wide_" + lane.method



        # Heading error
        yaw_err = 0.0
        if self.have_heading and self.have_pose:
            yaw_err = wrap_pi(self.desired_heading - self.current_yaw)

        # Steering
        w = rgb.shape[1]
        cx = w * 0.5
        err_norm = (lane.target_x - cx) / (w * 0.5)

        k_lane = float(self.get_parameter("k_lane_deg").value)
        k_head = float(self.get_parameter("k_heading_deg").value)

        steer_deg = k_lane * float(err_norm) + k_head * float(yaw_err)

        steer_lim = float(self.get_parameter("steer_deg_limit").value)
        steer_deg = float(max(-steer_lim, min(steer_lim, steer_deg)))

        # Speed plan
        v_ref = float(self.get_parameter("target_speed").value)
        v_turn = float(self.get_parameter("turn_speed").value)
        if abs(yaw_err) > float(self.get_parameter("turn_yaw_thresh").value):
            v_ref = min(v_ref, v_turn)

        brake = 0.0

        # Obstacle guard (optional)
        if self.use_guard:
            v_ref, brake = self.apply_obstacle_guard(v_ref)

        # Publish cmd_vel
        cmd = Twist()
        cmd.linear.x = float(max(0.0, v_ref))
        cmd.linear.y = float(max(0.0, min(1.0, brake)))
        cmd.angular.z = float(steer_deg)
        self.pub_cmd.publish(cmd)

        # Debug image
        self.publish_debug(rgb, sem, lane, steer_deg, v_ref, brake, yaw_err)

    # --- Lane estimation ---
    def estimate_lane_target(self, rgb: np.ndarray, sem: Optional[np.ndarray]) -> LaneEstimate:
        h, w = rgb.shape[:2]

        y0 = int(float(self.get_parameter("roi_y_start").value) * h)
        y1 = int(float(self.get_parameter("roi_y_end").value) * h)
        y0 = max(0, min(h - 2, y0))
        y1 = max(y0 + 1, min(h - 1, y1))

        roi = rgb[y0:y1, :]

        road_mask = None
        if sem is not None and sem.shape[:2] == rgb.shape[:2]:
            sem_roi = sem[y0:y1, :]
            # Semantic id is in the R channel (BGR -> index 2)
            sem_id = sem_roi[:, :, 2]
            road_label = int(self.get_parameter("road_label").value)
            road_mask = (sem_id == road_label).astype(np.uint8) * 255

        lane_center_x, lane_width_px, method = self.detect_lane_center_px(roi, road_mask)

        # Convert roi-local y to full image y
        target_y = int((y0 + y1) * 0.5)
        target_x = float(lane_center_x)

        # Apply turn bias based on global heading
        if self.have_heading and self.have_pose:
            yaw_err = wrap_pi(self.desired_heading - self.current_yaw)
            thr = float(self.get_parameter("turn_yaw_thresh").value)
            if abs(yaw_err) > thr:
                bias = float(self.get_parameter("turn_bias_px").value)
                target_x += math.copysign(bias, yaw_err)  # left turn -> negative yaw_err? depends map; bias to sign

        # Clamp
        target_x = float(max(0.0, min(w - 1.0, target_x)))

        return LaneEstimate(
            ok=True,
            target_x=target_x,
            target_y=float(target_y),
            lane_width_px=float(lane_width_px),
            method=method,
        )

    def detect_lane_center_px(self, roi_bgr: np.ndarray, road_mask: Optional[np.ndarray]) -> Tuple[float, float, str]:
        """Return (lane_center_x in full-image coords), lane_width_px, method."""
        h, w = roi_bgr.shape[:2]

        # Color threshold for lane markings (white + yellow)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # white-ish
        w_lo = np.array([0, 0, 180], dtype=np.uint8)
        w_hi = np.array([180, 60, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, w_lo, w_hi)

        # yellow-ish
        y_lo = np.array([15, 60, 120], dtype=np.uint8)
        y_hi = np.array([40, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, y_lo, y_hi)

        mask = cv2.bitwise_or(mask_white, mask_yellow)

        if road_mask is not None and bool(self.get_parameter("use_semseg_roi").value):
            mask = cv2.bitwise_and(mask, road_mask)

        # Morphology to connect broken markings
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(mask, 60, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=50,
            minLineLength=int(self.get_parameter("hough_min_line_len").value),
            maxLineGap=int(self.get_parameter("hough_max_line_gap").value),
        )

        left = []
        right = []

        if lines is not None:
            for (x1, y1, x2, y2) in lines.reshape(-1, 4):
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                if abs(dx) < 1.0:
                    continue
                slope = dy / dx
                if abs(slope) < 0.35 or abs(slope) > 10.0:
                    continue
                length = math.hypot(dx, dy)
                if slope < 0:
                    left.append((x1, y1, x2, y2, length))
                else:
                    right.append((x1, y1, x2, y2, length))

        y_eval = int(h * 0.65)

        x_left = None
        x_right = None

        if left:
            x_left = self.fit_x_at_y(left, y_eval)
        if right:
            x_right = self.fit_x_at_y(right, y_eval)

        if x_left is not None and x_right is not None and (x_right - x_left) > 120:
            lane_width = float(x_right - x_left)
            self.last_lane_width_px = 0.7 * self.last_lane_width_px + 0.3 * lane_width
            lane_center = float(0.5 * (x_left + x_right))
            return lane_center, self.last_lane_width_px, "hough_lines"

        # If only one side detected, estimate the other side from history
        if x_left is not None and x_right is None:
            lane_center = float(x_left + 0.5 * self.last_lane_width_px)
            return lane_center, self.last_lane_width_px, "one_side_left"
        if x_right is not None and x_left is None:
            lane_center = float(x_right - 0.5 * self.last_lane_width_px)
            return lane_center, self.last_lane_width_px, "one_side_right"

        # Fallback: semantic road centroid in a band
        if road_mask is not None:
            band = road_mask[int(h * 0.25) : int(h * 0.85), :]
            xs = np.where(band > 0)[1]
            if xs.size > 2000:
                lane_center = float(xs.mean())
                return lane_center, self.last_lane_width_px, "road_centroid"

        # Ultimate fallback: image center
        return float(w * 0.5), self.last_lane_width_px, "center_fallback"

    @staticmethod
    def fit_x_at_y(lines_with_len, y_eval: int) -> Optional[float]:
        # Weighted least squares fit x = a*y + b using endpoints
        # Convert each segment to points, weight by length
        pts = []
        ws = []
        for x1, y1, x2, y2, length in lines_with_len:
            pts.append((float(x1), float(y1)))
            pts.append((float(x2), float(y2)))
            ws.append(float(length))
            ws.append(float(length))

        if len(pts) < 4:
            return None

        Y = np.array([p[1] for p in pts], dtype=np.float32)
        X = np.array([p[0] for p in pts], dtype=np.float32)
        W = np.array(ws, dtype=np.float32)

        # Solve weighted linear regression: X ~ a*Y + b
        # [a,b] = (A^T W A)^-1 A^T W X
        A = np.stack([Y, np.ones_like(Y)], axis=1)
        Aw = A * W[:, None]
        try:
            params = np.linalg.lstsq(Aw, X * W, rcond=None)[0]
        except Exception:
            return None
        a, b = float(params[0]), float(params[1])
        x = a * float(y_eval) + b
        return float(x)

    # --- Obstacle guard ---
    def apply_obstacle_guard(self, v_ref: float) -> Tuple[float, float]:
        """Return (v_ref, brake)."""
        stop_d = float(self.get_parameter("stop_distance_m").value)
        slow_d = float(self.get_parameter("slow_distance_m").value)
        ttc_brake = float(self.get_parameter("ttc_brake_s").value)

        # If we haven't received recent pointclouds, don't brake.
        if time.time() - self.last_pc_t > 0.8:
            return v_ref, 0.0

        # TTC-based hard brake
        if self.last_radar_ttc is not None and self.last_radar_ttc < ttc_brake:
            return 0.0, 1.0

        # Distance-based slow/stop
        if self.last_obstacle_dist is None:
            return v_ref, 0.0

        d = self.last_obstacle_dist
        if d <= stop_d:
            return 0.0, 1.0
        if d <= slow_d:
            # linear slowdown
            s = (d - stop_d) / max(1e-3, (slow_d - stop_d))
            return max(1.0, v_ref * float(s)), 0.0

        return v_ref, 0.0

    def compute_obstacle_distance_semantic_lidar(self, msg) -> Optional[float]:
        if pc2 is None:
            return None

        tags = set(int(x) for x in self.get_parameter("obstacle_tags").value)
        half_w = float(self.get_parameter("roi_half_width_m").value)

        min_d = None
        # Read points; field names must match our publisher
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "obj_tag"), skip_nans=True):
            x, y, z, tag = float(p[0]), float(p[1]), float(p[2]), int(p[3])
            if x < 0.5:
                continue
            if abs(y) > half_w:
                continue
            if z < -2.0 or z > 3.0:
                continue
            if tag not in tags:
                continue
            d = math.hypot(x, y)
            if (min_d is None) or (d < min_d):
                min_d = d
        return min_d

    def compute_ttc_from_radar(self, msg) -> Optional[float]:
        if pc2 is None:
            return None

        half_w = float(self.get_parameter("roi_half_width_m").value)

        best_ttc = None
        for p in pc2.read_points(msg, field_names=("x", "y", "velocity"), skip_nans=True):
            x, y, vel = float(p[0]), float(p[1]), float(p[2])
            if x < 0.5 or abs(y) > half_w:
                continue

            # CARLA radar velocity is radial; approaching targets usually have negative velocity.
            closing = -vel
            if closing <= 0.2:
                continue

            d = math.hypot(x, y)
            ttc = d / closing
            if (best_ttc is None) or (ttc < best_ttc):
                best_ttc = ttc

        return best_ttc

    # --- Debug ---
    def publish_debug(
        self,
        rgb: np.ndarray,
        sem: Optional[np.ndarray],
        lane: LaneEstimate,
        steer_deg: float,
        v_ref: float,
        brake: float,
        yaw_err: float,
    ):
        try:
            dbg = rgb.copy()
            h, w = dbg.shape[:2]

            # target point
            cv2.circle(dbg, (int(lane.target_x), int(lane.target_y)), 8, (0, 255, 0), -1)
            cv2.line(dbg, (int(w * 0.5), h), (int(lane.target_x), int(lane.target_y)), (0, 255, 0), 2)

            # text overlay
            txt1 = f"method={lane.method} steer_deg={steer_deg:+.1f} v_ref={v_ref:.1f} brake={brake:.2f}"
            txt2 = f"yaw_err={yaw_err:+.2f} rad obs_d={self.last_obstacle_dist} ttc={self.last_radar_ttc}"
            cv2.putText(dbg, txt1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(dbg, txt2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_rgb"
            self.pub_dbg.publish(msg)
        except Exception:
            # Never allow debug failure to stop control
            return


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
