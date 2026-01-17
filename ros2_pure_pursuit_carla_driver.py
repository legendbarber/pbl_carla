#!/usr/bin/env python3
"""ros2_pure_pursuit_carla_driver.py

지역경로(/carla/path/local)를 받아 CARLA 차량을 실제로 주행시키는 ROS2 노드.

- sensor_setup3.py는 setup-only(차량/센서 스폰 + 센서 토픽 발행 + world.tick())만 담당
- 이 노드는 별도 CARLA client로 접속해서 role_name(vehicle_role) 차량을 찾고 apply_control()을 수행

입력:
- /carla/path/local (nav_msgs/Path)        : 지역경로 (기본 frame_id='map' 가정)
- /carla/hero/gnss (sensor_msgs/NavSatFix) : 자차 위치(원점은 path 제작과 동일해야 함)

출력:
- (옵션) /carla/hero/cmd_vel (geometry_msgs/Twist) : 디버깅용 (기본 off)

주의:
- yaw는 GNSS의 연속 차분으로 근사한다(MVP). 정지/저속에서 튈 수 있으므로 min_move_m을 둔다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import carla

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Twist


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class EgoPose2D:
    x: float
    y: float
    yaw: float


class PurePursuitCarlaDriver(Node):
    def __init__(self):
        super().__init__("pure_pursuit_carla_driver")

        # ---------- params ----------
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("vehicle_role", "hero")

        self.declare_parameter("local_path_topic", "/carla/path/local")
        self.declare_parameter("gnss_topic", "/carla/hero/gnss")

        # GNSS origin 수동 고정 옵션 (전역/지역경로 origin과 다르면 맞춰야 함)
        self.declare_parameter("origin_lat", float('nan'))
        self.declare_parameter("origin_lon", float('nan'))

        # control config
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("wheel_base_m", 2.7)
        self.declare_parameter("target_speed_mps", 5.0)

        # lookahead
        self.declare_parameter("lookahead_base_m", 5.5)
        self.declare_parameter("lookahead_gain", 0.35)  # ld = base + gain*speed

        # steering
        self.declare_parameter("max_steer_deg", 35.0)

        # speed PID
        self.declare_parameter("kp", 0.45)
        self.declare_parameter("ki", 0.10)
        self.declare_parameter("kd", 0.00)
        self.declare_parameter("i_limit", 5.0)
        self.declare_parameter("throttle_min_moving", 0.22)
        self.declare_parameter("throttle_smooth_alpha", 0.25)

        # yaw update
        self.declare_parameter("min_move_m", 0.2)

        # debug
        self.declare_parameter("publish_cmd_vel", False)

        # ---------- state ----------
        self._path_xy: List[Tuple[float, float]] = []

        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._cos_lat0: float = 1.0

        self._ego_curr_xy: Optional[Tuple[float, float]] = None
        self._ego_prev_xy: Optional[Tuple[float, float]] = None
        self._ego_yaw: Optional[float] = None

        # speed PID state
        self._pid_i = 0.0
        self._prev_err = 0.0
        self._prev_t = time.time()
        self._throttle_prev = 0.0

        # CARLA
        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._vehicle: Optional[carla.Vehicle] = None

        # ---------- ROS I/O ----------
        self.create_subscription(Path, self.get_parameter("local_path_topic").value, self._on_path, 10)
        self.create_subscription(NavSatFix, self.get_parameter("gnss_topic").value, self._on_gnss, 10)

        self._pub_cmd = None
        if bool(self.get_parameter("publish_cmd_vel").value):
            self._pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        # ---------- timers ----------
        rate = float(self.get_parameter("control_rate_hz").value)
        self._timer = self.create_timer(1.0 / max(rate, 1.0), self._control_once)
        self._timer_find = self.create_timer(1.0, self._try_find_vehicle)

        self._connect_carla()
        self.get_logger().info("PurePursuit CARLA driver ready.")

    # ---------------- CARLA ----------------

    def _connect_carla(self) -> None:
        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        self._client = carla.Client(host, port)
        self._client.set_timeout(5.0)
        self._world = self._client.get_world()

    def _try_find_vehicle(self) -> None:
        if self._world is None:
            return
        if self._vehicle is not None and self._vehicle.is_alive:
            return

        role = str(self.get_parameter("vehicle_role").value)
        for v in self._world.get_actors().filter("vehicle.*"):
            try:
                if v.attributes.get("role_name") == role:
                    self._vehicle = v
                    self.get_logger().info(f"Found vehicle role_name='{role}' id={v.id}")
                    return
            except Exception:
                continue

    # ---------------- GNSS / coordinate ----------------

    def _maybe_init_origin(self, lat: float, lon: float) -> None:
        origin_lat = float(self.get_parameter("origin_lat").value)
        origin_lon = float(self.get_parameter("origin_lon").value)
        if not math.isnan(origin_lat) and not math.isnan(origin_lon):
            if self._lat0 is None:
                self._lat0 = origin_lat
                self._lon0 = origin_lon
                self._cos_lat0 = math.cos(math.radians(self._lat0))
                self.get_logger().info(
                    f"Using fixed GNSS origin from params: lat0={self._lat0:.8f}, lon0={self._lon0:.8f}"
                )
            return

        if self._lat0 is None:
            self._lat0 = lat
            self._lon0 = lon
            self._cos_lat0 = math.cos(math.radians(lat))
            self.get_logger().info(f"GNSS origin set from first fix: lat0={lat:.8f}, lon0={lon:.8f}")

    def _latlon_to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        assert self._lat0 is not None and self._lon0 is not None
        dx = (lon - self._lon0) * (111320.0 * self._cos_lat0)
        dy = (lat - self._lat0) * 110540.0
        return float(dx), float(dy)

    def _on_gnss(self, msg: NavSatFix) -> None:
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        self._maybe_init_origin(lat, lon)

        if self._lat0 is None:
            return

        fixed = (not math.isnan(float(self.get_parameter("origin_lat").value))) and (not math.isnan(float(self.get_parameter("origin_lon").value)))
        if fixed:
            xy = self._latlon_to_xy(lat, lon)
        else:
            if self._ego_curr_xy is None:
                xy = (0.0, 0.0)
            else:
                xy = self._latlon_to_xy(lat, lon)

        self._ego_prev_xy = self._ego_curr_xy
        self._ego_curr_xy = xy

        if self._ego_prev_xy is not None:
            px, py = self._ego_prev_xy
            x, y = self._ego_curr_xy
            dist = math.hypot(x - px, y - py)
            if dist > float(self.get_parameter("min_move_m").value):
                self._ego_yaw = math.atan2(y - py, x - px)

    # ---------------- path ----------------

    def _on_path(self, msg: Path) -> None:
        self._path_xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    # ---------------- control ----------------

    def _control_once(self) -> None:
        if self._vehicle is None or (not self._vehicle.is_alive):
            return
        if len(self._path_xy) < 2:
            return

        if self._ego_curr_xy is None:
            return

        # 정지 상태에서는 GNSS 차분으로 yaw가 만들어지지 않아 제어가 시작되지 않는다.
        # 로컬 경로의 진행방향(가까운 점의 탄젠트)로 초기 yaw를 잡아 출발을 가능하게 한다.
        if self._ego_yaw is None:
            yaw0 = self._init_yaw_from_path(self._path_xy, self._ego_curr_xy[0], self._ego_curr_xy[1])
            if yaw0 is None:
                return
            self._ego_yaw = yaw0

        ego = EgoPose2D(x=self._ego_curr_xy[0], y=self._ego_curr_xy[1], yaw=self._ego_yaw)

        # path -> local
        pts_local = [self._map_to_local(ego, gx, gy) for gx, gy in self._path_xy]

        # choose target point by lookahead
        v = self._vehicle.get_velocity()
        speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

        ld = float(self.get_parameter("lookahead_base_m").value) + float(self.get_parameter("lookahead_gain").value) * speed
        ld = _clamp(ld, 3.0, 25.0)

        target = self._find_lookahead_point(pts_local, ld)
        if target is None:
            # no forward point -> full brake
            self._apply_control(throttle=0.0, steer=0.0, brake=1.0)
            return

        xt, yt = target

        wheel_base = float(self.get_parameter("wheel_base_m").value)
        alpha = math.atan2(yt, xt)
        delta = math.atan2(2.0 * wheel_base * math.sin(alpha), ld)
        steer_deg = math.degrees(delta)

        max_steer_deg = float(self.get_parameter("max_steer_deg").value)
        steer_deg = _clamp(steer_deg, -max_steer_deg, max_steer_deg)
        steer_norm = float(steer_deg / max_steer_deg) if max_steer_deg > 1e-6 else 0.0

        # speed PID -> throttle/brake
        target_speed = float(self.get_parameter("target_speed_mps").value)

        now = time.time()
        dt = _clamp(now - self._prev_t, 0.02, 0.20)
        self._prev_t = now

        # stop logic
        if target_speed < 0.05:
            self._reset_pid()
            self._apply_control(throttle=0.0, steer=steer_norm, brake=1.0)
            return

        err = target_speed - speed
        kp = float(self.get_parameter("kp").value)
        ki = float(self.get_parameter("ki").value)
        kd = float(self.get_parameter("kd").value)
        i_lim = float(self.get_parameter("i_limit").value)

        self._pid_i = _clamp(self._pid_i + err * dt, -i_lim, i_lim)
        derr = (err - self._prev_err) / dt
        self._prev_err = err

        u = kp * err + ki * self._pid_i + kd * derr
        # tiny FF to overcome stiction
        u += _clamp(target_speed / 25.0, 0.0, 0.35)

        throttle = _clamp(u, 0.0, 1.0)
        throttle_min = float(self.get_parameter("throttle_min_moving").value)
        if target_speed > 1.0 and speed < 0.5:
            throttle = max(throttle, throttle_min)

        alpha_smooth = float(self.get_parameter("throttle_smooth_alpha").value)
        throttle = (1.0 - alpha_smooth) * self._throttle_prev + alpha_smooth * throttle
        self._throttle_prev = throttle

        self._apply_control(throttle=throttle, steer=steer_norm, brake=0.0)

        # optional cmd_vel publish for debugging (compatible with your old bridge convention)
        if self._pub_cmd is not None:
            cmd = Twist()
            cmd.linear.x = target_speed
            cmd.linear.y = 0.0
            cmd.angular.z = steer_deg
            self._pub_cmd.publish(cmd)

    def _reset_pid(self) -> None:
        self._pid_i = 0.0
        self._prev_err = 0.0
        self._throttle_prev = 0.0

    def _apply_control(self, throttle: float, steer: float, brake: float) -> None:
        assert self._vehicle is not None
        ctrl = carla.VehicleControl(
            throttle=float(_clamp(throttle, 0.0, 1.0)),
            steer=float(_clamp(steer, -1.0, 1.0)),
            brake=float(_clamp(brake, 0.0, 1.0)),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self._vehicle.apply_control(ctrl)

    @staticmethod
    def _init_yaw_from_path(path_xy: List[Tuple[float, float]], x: float, y: float) -> Optional[float]:
        """정지 상태에서 사용할 초기 yaw 추정.

        경로에서 (x,y)에 가장 가까운 점을 찾고, 그 점의 진행방향(다음 점 방향)을 yaw로 반환.
        """
        if len(path_xy) < 2:
            return None
        best_i = None
        best_d2 = float("inf")
        for i, (px, py) in enumerate(path_xy):
            dx = px - x
            dy = py - y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        if best_i is None:
            return None

        if best_i < len(path_xy) - 1:
            x1, y1 = path_xy[best_i]
            x2, y2 = path_xy[best_i + 1]
        elif best_i > 0:
            x1, y1 = path_xy[best_i - 1]
            x2, y2 = path_xy[best_i]
        else:
            return None

        if (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) < 1e-6:
            return None
        return math.atan2(y2 - y1, x2 - x1)

    @staticmethod
    def _map_to_local(ego: EgoPose2D, gx: float, gy: float) -> Tuple[float, float]:
        dx = gx - ego.x
        dy = gy - ego.y
        c = math.cos(ego.yaw)
        s = math.sin(ego.yaw)
        xl = c * dx + s * dy
        yl = -s * dx + c * dy
        return xl, yl

    @staticmethod
    def _find_lookahead_point(pts_local: List[Tuple[float, float]], ld: float) -> Optional[Tuple[float, float]]:
        for x, y in pts_local:
            if x <= 0.0:
                continue
            d = math.hypot(x, y)
            if d >= ld:
                return (x, y)
        return None


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitCarlaDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
