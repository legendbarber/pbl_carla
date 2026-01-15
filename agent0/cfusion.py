#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import cv2
import numpy as np
import os
import sys
import time
import json
import math

from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, Imu
from cv_bridge import CvBridge
from ultralytics import YOLO


def pointcloud2_to_array(cloud_msg: PointCloud2):
    """
    NOTE: 이 구현은 PointCloud2가 x,y,z가 float32로 연속 저장되어 있다는 가정.
    CARLA 기본 토픽에선 대개 동작하지만, 포맷이 다르면 수정 필요.
    """
    cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.float32)
    num_points = cloud_msg.width * cloud_msg.height
    arr = cloud_arr.reshape(num_points, -1)
    return arr[:, :3]  # x,y,z


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        print("=== [Fusion with Display] ===")
        print(">> Camera: FOV 110 (Detection) + FOV 90 (View)")

        # === 시각화 on/off (원격/도커에서 창 문제 있으면 False로) ===
        self.enable_viz = True

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "model")

        # 네가 받은 신호등 모델
        path_tl = os.path.join(model_dir, "best.pt")
        if not os.path.exists(path_tl):
            # 혹시 model 폴더가 아니라면 현재 폴더도 한 번 봄
            alt = os.path.join(current_dir, "best.pt")
            path_tl = alt if os.path.exists(alt) else path_tl

        try:
            # 차량: COCO 사전학습 YOLOv8 (처음 실행 시 자동 다운로드)
            # 더 빠름: yolov8n.pt / 더 정확: yolov8s.pt
            self.model_vehicle = YOLO("yolov8n.pt", task='detect')

            # 신호등: best.pt
            self.model_traffic = YOLO(path_tl, task='detect')
            print("traffic names:", self.model_traffic.names)
            print(">> 모델 로딩 완료")
            print(f"   - vehicle model: yolov8n.pt")
            print(f"   - traffic model: {path_tl}")

        except Exception as e:
            print(f"[오류] 모델 로딩 실패: {e}")
            sys.exit(1)

        # === 디바이스 선택: CUDA 있으면 0, 없으면 cpu ===
        self.device = "cpu"
        try:
            import torch  # noqa
            self.device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

        # COCO 클래스 ID: car=2, motorcycle=3, bus=5, truck=7
        self.classes_vehicle = [2, 3, 5, 7]

        # traffic best.pt는 클래스 구성이 제각각이라 "전체 허용"으로 두고,
        # 내부에서 이름 기반으로 red/yellow/green 매핑
        self.classes_traffic = None

        self.bridge = CvBridge()
        self.decision_pub = self.create_publisher(String, "/fusion/decision", 10)
        self.result_pub = self.create_publisher(Image, "/fusion/result", 10)

        qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # [구독 1] 전방 카메라 (인식용)
        self.sub_img = self.create_subscription(
            Image, "/carla/hero/camera_front/image_color", self.image_callback, qos_profile)

        # [구독 2] 3인칭 카메라 (화면 표시용)
        self.sub_view = self.create_subscription(
            Image, "/carla/hero/camera_view/image_color", self.view_callback, qos_profile)

        # [구독 3] 라이다
        self.sub_lidar = self.create_subscription(
            PointCloud2, "/carla/hero/lidar/point_cloud", self.lidar_callback, qos_profile)

        # IMU
        self.sub_imu = self.create_subscription(
            Imu, "/carla/hero/imu", self.imu_callback, 10)

        self.latest_img = None
        self.latest_view = None
        self.latest_lidar = None

        self.memory_buffer = {}
        self.memory_ttl = 1.0  # 초 (오래된 트랙 제거)

        # IMU 데이터 저장
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.yaw_rate = 0.0
        self.lateral_accel = 0.0
        self.forward_accel = 0.0

        # 좌표 변환 행렬
        self.R_lidar2cam = np.array([[0, -1, 0],
                                     [0,  0, -1],
                                     [1,  0, 0]], dtype=np.float32)
        self.T_lidar2cam = np.array([0, -0.7, -1.6], dtype=np.float32)

        self.K = None
        self.timer = self.create_timer(0.05, self.fusion_loop)

    def image_callback(self, msg):
        self.latest_img = msg
        if self.K is None:
            w, h = msg.width, msg.height
            fov = 110.0
            f = w / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
            self.K = np.array([[f, 0, w/2.0],
                               [0, f, h/2.0],
                               [0, 0, 1]], dtype=np.float32)

    def view_callback(self, msg):
        self.latest_view = msg

    def lidar_callback(self, msg):
        self.latest_lidar = msg

    def imu_callback(self, msg: Imu):
        """IMU 콜백: 차량 자세 및 관성 정보"""
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w

        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        self.current_roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch
        sinp = 2 * (w * y - z * x)
        self.current_pitch = math.asin(max(-1.0, min(1.0, sinp)))

        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # 각속도
        self.yaw_rate = msg.angular_velocity.z

        # 가속도
        self.forward_accel = msg.linear_acceleration.x
        self.lateral_accel = msg.linear_acceleration.y

    # ---------- Traffic Light helpers ----------
    def map_traffic_name_to_cls(self, name: str):
        """
        best.pt의 클래스 이름을 red/yellow/green으로 해석해서
        기존 로직 호환 cls로 변환:
          red -> 4, yellow -> 5, green -> 3
        """
        if not name:
            return None
        n = name.strip().lower()

        # 영어
        if "red" in n or "stop" in n:
            return 4
        if "yellow" in n or "amber" in n:
            return 5
        if "green" in n or "go" in n:
            return 3

        # 한국어(혹시 데이터셋이 한글일 경우)
        if "빨" in n or "적" in n:
            return 4
        if "노" in n or "황" in n:
            return 5
        if "초" in n or "녹" in n:
            return 3

        return None

    def infer_tl_color_cls_hsv(self, crop_bgr):
        """이름 매핑이 실패할 때: HSV로 빨/노/초 추정해서 cls로 리턴"""
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)
        yellow = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
        green = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))

        total = crop_bgr.shape[0] * crop_bgr.shape[1]
        counts = {
            "red": int(cv2.countNonZero(red)),
            "yellow": int(cv2.countNonZero(yellow)),
            "green": int(cv2.countNonZero(green)),
        }
        best = max(counts, key=counts.get)

        # 너무 애매하면 버림
        if counts[best] < 0.01 * total:
            return None

        return 4 if best == "red" else 5 if best == "yellow" else 3

    # ---------- Core detection ----------
    def process_detection(self, cv_img, model, classes, lidar_points, prefix, detect_traffic=False):
        if lidar_points is None or self.K is None:
            return [], cv_img

        # 1) LiDAR Projection
        p_cam = np.dot(lidar_points, self.R_lidar2cam.T) + self.T_lidar2cam
        valid = (p_cam[:, 2] > 0.5) & (p_cam[:, 2] < 100.0)
        p_cam = p_cam[valid]

        p_2d = np.dot(p_cam, self.K.T)
        p_2d[:, 0] /= p_2d[:, 2]
        p_2d[:, 1] /= p_2d[:, 2]

        u = p_2d[:, 0].astype(np.int32)
        v = p_2d[:, 1].astype(np.int32)
        d = p_cam[:, 2]

        h, w = cv_img.shape[:2]
        in_view = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, d = u[in_view], v[in_view], d[in_view]

        # 2) YOLO Inference (track 시도 -> 실패하면 predict로 fallback)
        results = None
        try:
            kwargs = dict(
                persist=True,
                verbose=False,
                conf=0.45,
                device=self.device,
                imgsz=960
            )
            if classes is not None:
                kwargs["classes"] = classes

            # 차량은 track이 유리. 신호등은 track 없어도 되지만 그냥 통일.
            results = model.track(cv_img, tracker="bytetrack.yaml", **kwargs)
        except Exception:
            try:
                kwargs = dict(
                    verbose=False,
                    conf=0.45,
                    device=self.device,
                    imgsz=960
                )
                if classes is not None:
                    kwargs["classes"] = classes
                results = model.predict(cv_img, **kwargs)
            except Exception:
                return [], cv_img

        detected_objects = []
        fx = self.K[0, 0]
        cx = self.K[0, 2]

        if not results or len(results) == 0:
            return [], cv_img

        r0 = results[0]
        if getattr(r0, "boxes", None) is None or len(r0.boxes) == 0:
            return [], cv_img

        for box in r0.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if getattr(box, "id", None) is not None else -1

            # 중심 각도 계산
            box_cx = (x1 + x2) / 2.0
            angle_deg = np.degrees(np.arctan((box_cx - cx) / fx))

            # 거리 측정 (LiDAR ROI 최소거리)
            roi_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            roi_depths = d[roi_mask]

            dist = 999.0
            if len(roi_depths) > 0:
                valid_d = roi_depths[roi_depths > 1.5]
                if len(valid_d) > 0:
                    dist = float(np.min(valid_d))

            cls_id = raw_cls_id
            label_txt = ""

            # --- Traffic light: best.pt 클래스 이름을 red/yellow/green으로 매핑 ---
            if detect_traffic:
                name = ""
                try:
                    # Ultralytics names는 dict 또는 list 형태
                    names = getattr(model, "names", {})
                    if isinstance(names, dict):
                        name = names.get(raw_cls_id, "")
                    elif isinstance(names, list) and raw_cls_id < len(names):
                        name = names[raw_cls_id]
                except Exception:
                    name = ""

                mapped = self.map_traffic_name_to_cls(str(name))
                if mapped is None:
                    # 이름으로 못 맞추면 HSV로 fallback
                    crop = cv_img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    mapped = self.infer_tl_color_cls_hsv(crop)

                if mapped is None:
                    continue  # 신호등 색을 못 정하면 버림

                cls_id = mapped
                label_txt = "Red" if cls_id == 4 else "Yellow" if cls_id == 5 else "Green"

            else:
                label_txt = "Vehicle"

            obj_info = {
                "type": "traffic" if detect_traffic else "vehicle",
                "cls": int(cls_id),
                "dist": float(dist),
                "angle": float(angle_deg),
                "id": f"{prefix}_{track_id if track_id >= 0 else raw_cls_id}_{x1}_{y1}"
            }
            detected_objects.append(obj_info)

            # 시각화
            if self.enable_viz:
                if detect_traffic:
                    if cls_id == 4:
                        color = (0, 0, 255)
                    elif cls_id == 5:
                        color = (0, 255, 255)
                    else:
                        color = (0, 255, 0)
                else:
                    color = (255, 100, 0)

                cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                if dist < 100:
                    info = f"{label_txt} {dist:.1f}m {angle_deg:.0f}dg"
                    cv2.putText(cv_img, info, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return detected_objects, cv_img

    def draw_imu_overlay(self, frame):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 150), (420, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        y_offset = h - 130
        line_height = 25

        texts = [
            f"Yaw: {math.degrees(self.current_yaw):>6.1f}deg",
            f"YawRate: {self.yaw_rate:>5.2f}r/s",
            f"Pitch: {math.degrees(self.current_pitch):>6.1f}deg",
            f"Roll: {math.degrees(self.current_roll):>6.1f}deg",
            f"Accel(F/L): {self.forward_accel:>4.1f}/{self.lateral_accel:>4.1f}m/s2"
        ]

        for i, text in enumerate(texts):
            color = (0, 255, 255)
            if "YawRate" in text and abs(self.yaw_rate) > 0.8:
                color = (0, 0, 255)
            elif "Accel" in text and (abs(self.forward_accel) > 5 or abs(self.lateral_accel) > 5):
                color = (0, 165, 255)

            cv2.putText(frame, text, (20, y_offset + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        center_x, center_y = w - 100, h - 75
        arrow_len = 50
        end_x = int(center_x + arrow_len * math.cos(self.current_yaw - math.pi / 2))
        end_y = int(center_y + arrow_len * math.sin(self.current_yaw - math.pi / 2))
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                        (0, 255, 0), 3, tipLength=0.3)
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)

        return frame

    def cleanup_memory(self, now_t):
        # 오래된 key 제거 (가볍게)
        dead = [k for k, v in self.memory_buffer.items() if now_t - v["time"] > self.memory_ttl]
        for k in dead:
            del self.memory_buffer[k]

    def fusion_loop(self):
        # 3인칭 화면
        if self.enable_viz and self.latest_view is not None:
            try:
                cv_view = self.bridge.imgmsg_to_cv2(self.latest_view, "bgr8")
                cv2.imshow("Spectator View", cv_view)
            except Exception:
                pass

        if self.latest_img is None or self.latest_lidar is None or self.K is None:
            if self.enable_viz:
                cv2.waitKey(1)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
            lidar_pts = pointcloud2_to_array(self.latest_lidar)
            current_time = time.time()
            self.cleanup_memory(current_time)

            all_objects = []

            # 1) 신호등 탐지 (best.pt)
            objs_light, frame = self.process_detection(
                frame, self.model_traffic, self.classes_traffic,
                lidar_pts, "light", detect_traffic=True
            )
            all_objects.extend(objs_light)

            # 2) 차량 탐지 (COCO yolov8)
            objs_car, frame = self.process_detection(
                frame, self.model_vehicle, self.classes_vehicle,
                lidar_pts, "veh", detect_traffic=False
            )
            all_objects.extend(objs_car)

            frame = self.draw_imu_overlay(frame)

            # 3) 판단 로직
            min_light_dist = 999.0
            closest_light = "none"
            min_vehicle_dist = 999.0
            closest_vehicle_angle = 0.0

            for obj in all_objects:
                d = obj["dist"]
                if d >= 999.0:
                    continue

                # 스무딩
                key = obj["id"]
                if key in self.memory_buffer:
                    d = self.memory_buffer[key]["dist"] * 0.4 + d * 0.6
                self.memory_buffer[key] = {"dist": float(d), "time": current_time}

                if obj["type"] == "traffic":
                    status = "traffic_green"
                    if obj["cls"] == 4:
                        status = "traffic_red"
                    elif obj["cls"] == 5:
                        status = "traffic_yellow"

                    if d < min_light_dist:
                        min_light_dist = d
                        closest_light = status

                elif obj["type"] == "vehicle":
                    if d < min_vehicle_dist:
                        min_vehicle_dist = d
                        closest_vehicle_angle = obj["angle"]

            # 4) 발행
            msg_data = {
                "light": closest_light,
                "light_dist": float(min_light_dist) if min_light_dist < 999 else -1.0,
                "vehicle_dist": float(min_vehicle_dist) if min_vehicle_dist < 999 else -1.0,
                "vehicle_angle": float(closest_vehicle_angle),
            }
            self.decision_pub.publish(String(data=json.dumps(msg_data)))

            info_txt = f"Li:{closest_light} | Car:{msg_data['vehicle_dist']:.1f}m({closest_vehicle_angle:.0f}dg)"
            cv2.putText(frame, info_txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if self.enable_viz:
                cv2.imshow("Detection Result", frame)
                cv2.waitKey(1)

            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.result_pub.publish(out_msg)

        except Exception:
            # 디버깅 필요하면 여기서 print(e)로 확인
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
