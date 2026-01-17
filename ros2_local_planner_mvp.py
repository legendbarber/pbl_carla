#!/usr/bin/env python3
"""ros2_local_planner_mvp.py

전역경로 + 센서 기반으로 지역경로(/carla/path/local)를 생성하는 ROS2 노드.

입력:
- /carla/path/global (nav_msgs/Path)                 : 전역경로(기준선)
- /carla/hero/gnss (sensor_msgs/NavSatFix)           : 자차 위치(lat/lon)
- /carla/hero/lidar/point_cloud (sensor_msgs/PointCloud2)
- /carla/hero/semantic_lidar/point_cloud (sensor_msgs/PointCloud2, optional)
- /carla/hero/camera_semseg/image_raw (sensor_msgs/Image, bgr8, optional)

출력:
- /carla/path/local (nav_msgs/Path): 선택된 지역경로 (frame_id='map')

핵심 아이디어(MVP):
- 전역경로에서 현재 위치 근처부터 horizon 만큼 잘라(seg) local frame으로 변환
- 좌우 오프셋 후보(lattice)들을 생성
- LiDAR 점군으로 만든 2D occupancy에서 충돌/여유거리(cost) 평가
- (옵션) semantic 기반 drivable evidence grid를 만들어 off-road penalty를 추가

주의/가정:
- 이 노드는 TF를 쓰지 않는다(MVP). 대신 "센서 포인트를 ego local 좌표로 간주"하는 근사 사용.
- CARLA LiDAR/SemanticLiDAR PointCloud2는 기본적으로 y가 "우측(+)"인 경우가 많다.
  Pure pursuit 및 본 플래너의 local frame을 "좌측(+)"로 쓰려면 y 부호를 뒤집어야 한다.
  -> flip_sensor_y 파라미터로 제어.
- GNSS yaw는 연속 차분으로 근사한다. 정지 상태에서 yaw가 없으면 전역경로 탄젠트로 초기화한다.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix, PointCloud2, Image


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class EgoPose2D:
    x: float
    y: float
    yaw: float  # rad


class LocalPlannerMVP(Node):
    def __init__(self):
        super().__init__("local_path_planner_mvp")

        # ---------- topics ----------
        self.declare_parameter("global_path_topic", "/carla/path/global")
        self.declare_parameter("local_path_topic", "/carla/path/local")
        self.declare_parameter("gnss_topic", "/carla/hero/gnss")
        self.declare_parameter("lidar_topic", "/carla/hero/lidar/point_cloud")
        self.declare_parameter("semantic_lidar_topic", "/carla/hero/semantic_lidar/point_cloud")
        self.declare_parameter("semantic_camera_raw_topic", "/carla/hero/camera_semseg/image_raw")

        # GNSS origin
        self.declare_parameter("origin_lat", float("nan"))
        self.declare_parameter("origin_lon", float("nan"))

        # ---------- planning config ----------
        self.declare_parameter("plan_rate_hz", 10.0)
        self.declare_parameter("horizon_m", 60.0)
        self.declare_parameter("lookahead_start_m", 3.0)
        self.declare_parameter("offset_candidates", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

        # local frame convention
        # Default: False
        # - Set True if your LiDAR/SemLiDAR points look mirrored left/right in RViz or the vehicle avoids obstacles to the wrong side.
        self.declare_parameter("flip_sensor_y", False)  # True: sensor y(+) right -> local y(+) left

        # obstacle / lidar
        self.declare_parameter("safety_radius_m", 1.6)
        self.declare_parameter("clearance_search_m", 3.0)
        self.declare_parameter("lidar_roi_x", 35.0)
        self.declare_parameter("lidar_roi_y", 12.0)
        self.declare_parameter("lidar_stride", 6)
        self.declare_parameter("max_lidar_points", 30000)
        self.declare_parameter("lidar_z_min", -3.5)
        self.declare_parameter("lidar_z_max", 1.5)

        # drivable evidence
        self.declare_parameter("use_semantic_lidar", True)
        self.declare_parameter("use_semantic_camera", False)  # camera IPM은 근사라 기본 off
        self.declare_parameter("drivable_roi_x", 40.0)
        self.declare_parameter("drivable_roi_y", 14.0)
        self.declare_parameter("drivable_cell", 0.5)
        self.declare_parameter("drivable_neighbor_cells", 2)
        self.declare_parameter("min_drivable_points", 200)

        # CARLA semantic tags (Roads=1, RoadLine=24 default)
        self.declare_parameter("drivable_tags", [1, 24])

        # semantic lidar parsing
        self.declare_parameter("sem_lidar_stride", 6)
        self.declare_parameter("sem_lidar_max_points", 60000)
        self.declare_parameter("sem_lidar_z_min", -6.0)
        self.declare_parameter("sem_lidar_z_max", -0.2)

        # semantic camera rough IPM
        self.declare_parameter("sem_cam_width", 1280)
        self.declare_parameter("sem_cam_height", 720)
        self.declare_parameter("sem_cam_fov_deg", 110.0)
        self.declare_parameter("sem_cam_x", 1.6)
        self.declare_parameter("sem_cam_y", 0.0)
        self.declare_parameter("sem_cam_z", 1.7)
        self.declare_parameter("sem_cam_sample_step", 12)
        self.declare_parameter("sem_cam_v_start_ratio", 0.55)
        self.declare_parameter("sem_cam_v_end_ratio", 0.95)
        self.declare_parameter("sem_cam_max_points", 9000)

        # costs
        self.declare_parameter("w_offset", 1.0)
        self.declare_parameter("w_curvature", 1.0)
        self.declare_parameter("w_clearance", 6.0)
        self.declare_parameter("w_offroad", 80.0)

        # ---------- state ----------
        self._global_path_xy: List[Tuple[float, float]] = []

        self._ego_curr_xy: Optional[Tuple[float, float]] = None
        self._ego_prev_xy: Optional[Tuple[float, float]] = None
        self._ego_yaw: Optional[float] = None

        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._cos_lat0: float = 1.0

        # obstacle points (ego local frame: x forward, y left)
        self._lidar_xy: List[Tuple[float, float]] = []

        # drivable evidence points (ego local frame)
        self._sem_lidar_drivable_xy: List[Tuple[float, float]] = []
        self._sem_cam_drivable_xy: List[Tuple[float, float]] = []

        self._tick = 0

        # ---------- pubs/subs ----------
        self._pub_local_path = self.create_publisher(Path, self.get_parameter("local_path_topic").value, 10)

        self.create_subscription(Path, self.get_parameter("global_path_topic").value, self._on_global_path, 10)
        self.create_subscription(NavSatFix, self.get_parameter("gnss_topic").value, self._on_gnss, 10)
        self.create_subscription(PointCloud2, self.get_parameter("lidar_topic").value, self._on_lidar, 10)

        # optional semantic inputs
        self.create_subscription(PointCloud2, self.get_parameter("semantic_lidar_topic").value, self._on_sem_lidar, 2)
        self.create_subscription(Image, self.get_parameter("semantic_camera_raw_topic").value, self._on_sem_raw, 2)

        rate = float(self.get_parameter("plan_rate_hz").value)
        self._timer = self.create_timer(1.0 / max(rate, 1.0), self._plan_once)

        self.get_logger().info(
            "Local planner ready. global_path+GNSS+LiDAR(+semantic) -> /carla/path/local"
        )

    # ---------------- GNSS / coordinate helpers ----------------

    def _maybe_init_origin(self, lat: float, lon: float) -> None:
        origin_lat = float(self.get_parameter("origin_lat").value)
        origin_lon = float(self.get_parameter("origin_lon").value)
        if not math.isnan(origin_lat) and not math.isnan(origin_lon):
            if self._lat0 is None:
                self._lat0 = origin_lat
                self._lon0 = origin_lon
                self._cos_lat0 = math.cos(math.radians(self._lat0))
                self.get_logger().info(
                    f"Using fixed GNSS origin: lat0={self._lat0:.8f}, lon0={self._lon0:.8f}"
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

        fixed = (not math.isnan(float(self.get_parameter("origin_lat").value))) and (
            not math.isnan(float(self.get_parameter("origin_lon").value))
        )

        if fixed:
            xy = self._latlon_to_xy(lat, lon)
        else:
            if self._ego_curr_xy is None:
                xy = (0.0, 0.0)
            else:
                xy = self._latlon_to_xy(lat, lon)

        self._ego_prev_xy = self._ego_curr_xy
        self._ego_curr_xy = xy

        # yaw from GNSS displacement (MVP)
        if self._ego_prev_xy is not None:
            px, py = self._ego_prev_xy
            x, y = self._ego_curr_xy
            dist = math.hypot(x - px, y - py)
            if dist > 0.2:
                self._ego_yaw = math.atan2(y - py, x - px)

    # ---------------- path / sensors ----------------

    def _on_global_path(self, msg: Path) -> None:
        self._global_path_xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _on_lidar(self, msg: PointCloud2) -> None:
        stride = int(self.get_parameter("lidar_stride").value)
        max_pts = int(self.get_parameter("max_lidar_points").value)
        roi_x = float(self.get_parameter("lidar_roi_x").value)
        roi_y = float(self.get_parameter("lidar_roi_y").value)
        z_min = float(self.get_parameter("lidar_z_min").value)
        z_max = float(self.get_parameter("lidar_z_max").value)
        flip_y = bool(self.get_parameter("flip_sensor_y").value)

        if msg.point_step < 12 or msg.width == 0:
            return

        data = msg.data
        step = msg.point_step
        n = msg.width

        # offsets
        off_x, off_y, off_z = 0, 4, 8
        for f in msg.fields:
            if f.name == "x":
                off_x = f.offset
            elif f.name == "y":
                off_y = f.offset
            elif f.name == "z":
                off_z = f.offset

        pts: List[Tuple[float, float]] = []
        fmt_f = "<f"
        for i in range(0, n, max(1, stride)):
            base = i * step
            if base + max(off_x, off_y, off_z) + 4 > len(data):
                break

            x = struct.unpack_from(fmt_f, data, base + off_x)[0]
            y = struct.unpack_from(fmt_f, data, base + off_y)[0]
            z = struct.unpack_from(fmt_f, data, base + off_z)[0]

            if x < 0.0 or x > roi_x or abs(y) > roi_y or z < z_min or z > z_max:
                continue

            # Make y positive = left (ROS base_link convention)
            if flip_y:
                y = -y

            pts.append((float(x), float(y)))
            if len(pts) >= max_pts:
                break

        self._lidar_xy = pts

    def _on_sem_lidar(self, msg: PointCloud2) -> None:
        if not bool(self.get_parameter("use_semantic_lidar").value):
            return

        stride = int(self.get_parameter("sem_lidar_stride").value)
        max_pts = int(self.get_parameter("sem_lidar_max_points").value)
        z_min = float(self.get_parameter("sem_lidar_z_min").value)
        z_max = float(self.get_parameter("sem_lidar_z_max").value)
        roi_x = float(self.get_parameter("drivable_roi_x").value)
        roi_y = float(self.get_parameter("drivable_roi_y").value)
        tags = set(int(t) for t in self.get_parameter("drivable_tags").value)
        flip_y = bool(self.get_parameter("flip_sensor_y").value)

        if msg.point_step < 24 or msg.width == 0:
            return

        data = msg.data
        step = msg.point_step
        n = msg.width

        # offsets
        off_x, off_y, off_z, off_tag = 0, 4, 8, 20
        for f in msg.fields:
            if f.name == "x":
                off_x = f.offset
            elif f.name == "y":
                off_y = f.offset
            elif f.name == "z":
                off_z = f.offset
            elif f.name == "obj_tag":
                off_tag = f.offset

        pts: List[Tuple[float, float]] = []
        fmt_f = "<f"
        fmt_u = "<I"
        for i in range(0, n, max(1, stride)):
            base = i * step
            if base + max(off_x, off_y, off_z, off_tag) + 4 > len(data):
                break

            tag = struct.unpack_from(fmt_u, data, base + off_tag)[0]
            if int(tag) not in tags:
                continue

            x = struct.unpack_from(fmt_f, data, base + off_x)[0]
            y = struct.unpack_from(fmt_f, data, base + off_y)[0]
            z = struct.unpack_from(fmt_f, data, base + off_z)[0]

            if x < 0.0 or x > roi_x or abs(y) > roi_y or z < z_min or z > z_max:
                continue

            if flip_y:
                y = -y

            pts.append((float(x), float(y)))
            if len(pts) >= max_pts:
                break

        self._sem_lidar_drivable_xy = pts

    def _on_sem_raw(self, msg: Image) -> None:
        if not bool(self.get_parameter("use_semantic_camera").value):
            return
        if msg.width == 0 or msg.height == 0:
            return
        if msg.encoding.lower() != "bgr8":
            # sensor_setup3.py publishes bgr8
            return

        w = int(self.get_parameter("sem_cam_width").value)
        h = int(self.get_parameter("sem_cam_height").value)
        if msg.width != w or msg.height != h:
            w = int(msg.width)
            h = int(msg.height)

        fov = float(self.get_parameter("sem_cam_fov_deg").value)
        cam_x = float(self.get_parameter("sem_cam_x").value)
        cam_y = float(self.get_parameter("sem_cam_y").value)
        cam_z = float(self.get_parameter("sem_cam_z").value)
        step = int(self.get_parameter("sem_cam_sample_step").value)
        v0r = float(self.get_parameter("sem_cam_v_start_ratio").value)
        v1r = float(self.get_parameter("sem_cam_v_end_ratio").value)
        max_pts = int(self.get_parameter("sem_cam_max_points").value)
        roi_x = float(self.get_parameter("drivable_roi_x").value)
        roi_y = float(self.get_parameter("drivable_roi_y").value)
        tags = set(int(t) for t in self.get_parameter("drivable_tags").value)
        flip_y = bool(self.get_parameter("flip_sensor_y").value)

        # intrinsics
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        f = (w * 0.5) / math.tan(math.radians(fov) * 0.5)

        # decode image -> np.uint8 (h,w,3)
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            arr = arr.reshape((h, w, 3))
        except Exception:
            return

        v0 = int(_clamp(v0r, 0.0, 1.0) * h)
        v1 = int(_clamp(v1r, 0.0, 1.0) * h)
        v0 = max(0, min(h - 1, v0))
        v1 = max(0, min(h, v1))

        pts: List[Tuple[float, float]] = []

        # CARLA semantic raw: ID is in R channel (BGR -> index 2)
        # Rough IPM: assume camera axes aligned with ego: x forward, y right, z up.
        for v in range(v0, v1, max(1, step)):
            for u in range(0, w, max(1, step)):
                tag = int(arr[v, u, 2])
                if tag not in tags:
                    continue

                # ray direction in camera/ego frame
                du = (u - cx) / f
                dv = (v - cy) / f
                dir_x = 1.0
                dir_y = du
                dir_z = -dv  # v down -> ray points downward (negative z)

                if dir_z >= -1e-3:
                    continue

                t = -cam_z / dir_z
                if t <= 0.0 or t > 120.0:
                    continue

                x = cam_x + t * dir_x
                y = cam_y + t * dir_y

                # convert y positive = left if needed
                if flip_y:
                    y = -y

                if x < 0.0 or x > roi_x or abs(y) > roi_y:
                    continue

                pts.append((float(x), float(y)))
                if len(pts) >= max_pts:
                    self._sem_cam_drivable_xy = pts
                    return

        self._sem_cam_drivable_xy = pts

    # ---------------- planning core ----------------

    def _plan_once(self) -> None:
        if not self._global_path_xy or self._ego_curr_xy is None:
            return

        # yaw init: if no GNSS yaw yet, derive from global path tangent
        if self._ego_yaw is None:
            ni = self._find_nearest_index(self._global_path_xy, self._ego_curr_xy[0], self._ego_curr_xy[1])
            if ni is None:
                return
            self._ego_yaw = self._yaw_from_path_tangent(self._global_path_xy, ni)

        assert self._ego_yaw is not None
        ego = EgoPose2D(x=self._ego_curr_xy[0], y=self._ego_curr_xy[1], yaw=self._ego_yaw)

        # 1) global path: nearest index
        nearest_i = self._find_nearest_index(self._global_path_xy, ego.x, ego.y)
        if nearest_i is None:
            return

        # 2) slice horizon ahead
        horizon_m = float(self.get_parameter("horizon_m").value)
        start_m = float(self.get_parameter("lookahead_start_m").value)
        seg = self._slice_path(self._global_path_xy, nearest_i, horizon_m, start_m)
        if len(seg) < 5:
            return

        # 3) map -> ego(local)
        seg_local = [self._map_to_local(ego, gx, gy) for gx, gy in seg]

        # 4) build occupancy from lidar
        occ_cell = float(self.get_parameter("drivable_cell").value)
        occ = self._build_occupancy(self._lidar_xy, cell=occ_cell)

        safety_r = float(self.get_parameter("safety_radius_m").value)
        search_r = float(self.get_parameter("clearance_search_m").value)

        # 5) build drivable evidence grid (optional)
        drivable = self._build_drivable_grid()

        # 6) evaluate lattice candidates
        offsets = [float(d) for d in self.get_parameter("offset_candidates").value]
        w_offset = float(self.get_parameter("w_offset").value)
        w_curv = float(self.get_parameter("w_curvature").value)
        w_clear = float(self.get_parameter("w_clearance").value)
        w_offroad = float(self.get_parameter("w_offroad").value)

        best_cost = float("inf")
        best_local: Optional[List[Tuple[float, float]]] = None
        best_dbg = ""

        for d in offsets:
            cand = self._offset_path(seg_local, d)

            collides, min_clear = self._path_clearance(cand, occ, safety_r, search_r, cell=occ_cell)
            if collides:
                cost = 1e9 + 10.0 * abs(d)
                dbg = f"d={d:+.2f} COLL"
            else:
                # clearance cost: smaller when far from obstacles
                clear_cost = 1.0 / max(min_clear, 0.1)
                offroad_cost = self._offroad_cost(cand, drivable, cell=occ_cell) if drivable is not None else 0.0
                curv_cost = self._curvature_cost(cand)

                cost = (w_offset * abs(d)) + (w_clear * clear_cost) + (w_curv * curv_cost) + (w_offroad * offroad_cost)
                dbg = f"d={d:+.2f} clr={min_clear:.2f} cc={clear_cost:.2f} off={offroad_cost:.2f} curv={curv_cost:.2f} -> {cost:.2f}"

            if cost < best_cost:
                best_cost = cost
                best_local = cand
                best_dbg = dbg

        if best_local is None:
            return

        # 7) local -> map, publish Path(frame_id='map')
        best_map = [self._local_to_map(ego, xl, yl) for xl, yl in best_local]
        out = Path()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"

        poses: List[PoseStamped] = []
        for x, y in best_map:
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)
        out.poses = poses
        self._pub_local_path.publish(out)

        self._tick += 1
        if self._tick % 10 == 0:
            n_occ = len(self._lidar_xy)
            n_drv = 0 if drivable is None else len(drivable)
            self.get_logger().info(f"best: {best_dbg} | occ_pts={n_occ} drivable_cells={n_drv}")

    # ---------------- drivable grid ----------------

    def _build_drivable_grid(self) -> Optional[Set[Tuple[int, int]]]:
        min_pts = int(self.get_parameter("min_drivable_points").value)
        use_sem_l = bool(self.get_parameter("use_semantic_lidar").value)
        use_sem_c = bool(self.get_parameter("use_semantic_camera").value)
        cell = float(self.get_parameter("drivable_cell").value)
        neigh = int(self.get_parameter("drivable_neighbor_cells").value)

        pts: List[Tuple[float, float]] = []
        if use_sem_l:
            pts.extend(self._sem_lidar_drivable_xy)
        if use_sem_c:
            pts.extend(self._sem_cam_drivable_xy)

        if len(pts) < min_pts:
            return None

        inv = 1.0 / max(cell, 1e-6)
        grid: Set[Tuple[int, int]] = set()

        for x, y in pts:
            ix = int(math.floor(x * inv))
            iy = int(math.floor(y * inv))
            grid.add((ix, iy))

        if neigh > 0:
            dil: Set[Tuple[int, int]] = set(grid)
            for ix, iy in grid:
                for dx in range(-neigh, neigh + 1):
                    for dy in range(-neigh, neigh + 1):
                        dil.add((ix + dx, iy + dy))
            grid = dil

        return grid

    @staticmethod
    def _offroad_cost(path_local: List[Tuple[float, float]], drivable: Set[Tuple[int, int]], cell: float) -> float:
        if not drivable:
            return 0.0
        inv = 1.0 / max(cell, 1e-6)
        bad = 0
        total = 0
        for i in range(0, len(path_local), 2):
            x, y = path_local[i]
            if x < 0.0:
                continue
            total += 1
            ix = int(math.floor(x * inv))
            iy = int(math.floor(y * inv))
            if (ix, iy) not in drivable:
                bad += 1
        if total == 0:
            return 0.0
        return float(bad) / float(total)

    # ---------------- geometry helpers ----------------

    @staticmethod
    def _find_nearest_index(path_xy: List[Tuple[float, float]], x: float, y: float) -> Optional[int]:
        best_i = None
        best_d2 = float("inf")
        for i, (px, py) in enumerate(path_xy):
            dx = px - x
            dy = py - y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    @staticmethod
    def _yaw_from_path_tangent(path_xy: List[Tuple[float, float]], i: int) -> float:
        if len(path_xy) < 2:
            return 0.0
        i0 = max(0, min(len(path_xy) - 2, i))
        x1, y1 = path_xy[i0]
        x2, y2 = path_xy[i0 + 1]
        return math.atan2(y2 - y1, x2 - x1)

    @staticmethod
    def _slice_path(path_xy: List[Tuple[float, float]], start_i: int, horizon_m: float, start_m: float) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        dist = 0.0
        prev = path_xy[start_i]

        # advance to start_m
        i = start_i
        while i + 1 < len(path_xy) and dist < start_m:
            cur = path_xy[i + 1]
            dist += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            prev = cur
            i += 1

        out.append(prev)
        dist = 0.0
        while i + 1 < len(path_xy) and dist < horizon_m:
            cur = path_xy[i + 1]
            dist += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            out.append(cur)
            prev = cur
            i += 1

        return out

    @staticmethod
    def _map_to_local(ego: EgoPose2D, gx: float, gy: float) -> Tuple[float, float]:
        dx = gx - ego.x
        dy = gy - ego.y
        c = math.cos(ego.yaw)
        s = math.sin(ego.yaw)
        # local: x forward, y left
        xl = c * dx + s * dy
        yl = -s * dx + c * dy
        return xl, yl

    @staticmethod
    def _local_to_map(ego: EgoPose2D, xl: float, yl: float) -> Tuple[float, float]:
        c = math.cos(ego.yaw)
        s = math.sin(ego.yaw)
        gx = ego.x + (c * xl - s * yl)
        gy = ego.y + (s * xl + c * yl)
        return gx, gy

    @staticmethod
    def _offset_path(path_local: List[Tuple[float, float]], d: float) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        if len(path_local) < 2:
            return list(path_local)
        for i in range(len(path_local)):
            if i == len(path_local) - 1:
                x1, y1 = path_local[i - 1]
                x2, y2 = path_local[i]
            else:
                x1, y1 = path_local[i]
                x2, y2 = path_local[i + 1]
            theta = math.atan2(y2 - y1, x2 - x1)
            nx = -math.sin(theta)
            ny = math.cos(theta)
            x, y = path_local[i]
            out.append((x + d * nx, y + d * ny))
        return out

    @staticmethod
    def _build_occupancy(pts_xy: List[Tuple[float, float]], cell: float) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
        occ: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        inv = 1.0 / max(cell, 1e-6)
        for x, y in pts_xy:
            ix = int(math.floor(x * inv))
            iy = int(math.floor(y * inv))
            occ.setdefault((ix, iy), []).append((x, y))
        return occ

    @staticmethod
    def _path_clearance(
        path_local: List[Tuple[float, float]],
        occ: Dict[Tuple[int, int], List[Tuple[float, float]]],
        safety_r: float,
        search_r: float,
        cell: float,
    ) -> Tuple[bool, float]:
        """Return (collides, min_clearance_m)."""
        if not occ:
            return (False, float("inf"))

        inv = 1.0 / max(cell, 1e-6)
        sr2 = safety_r * safety_r
        min_d2 = float("inf")

        max_cells = int(math.ceil(search_r * inv))

        for i in range(0, len(path_local), 2):
            x, y = path_local[i]
            if x < 0.0:
                continue

            ix = int(math.floor(x * inv))
            iy = int(math.floor(y * inv))

            for dx in range(-max_cells, max_cells + 1):
                for dy in range(-max_cells, max_cells + 1):
                    cell_pts = occ.get((ix + dx, iy + dy))
                    if not cell_pts:
                        continue
                    for ox, oy in cell_pts:
                        ddx = ox - x
                        ddy = oy - y
                        d2 = ddx * ddx + ddy * ddy
                        if d2 < sr2:
                            return (True, 0.0)
                        if d2 < min_d2:
                            min_d2 = d2

        if min_d2 == float("inf"):
            return (False, float("inf"))

        return (False, math.sqrt(min_d2))

    @staticmethod
    def _curvature_cost(path_local: List[Tuple[float, float]]) -> float:
        if len(path_local) < 3:
            return 0.0
        cost = 0.0
        prev_theta = None
        for i in range(1, len(path_local)):
            x1, y1 = path_local[i - 1]
            x2, y2 = path_local[i]
            theta = math.atan2(y2 - y1, x2 - x1)
            if prev_theta is not None:
                d = theta - prev_theta
                while d > math.pi:
                    d -= 2.0 * math.pi
                while d < -math.pi:
                    d += 2.0 * math.pi
                cost += abs(d)
            prev_theta = theta
        return cost


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerMVP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
