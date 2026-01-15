from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

# Note: We keep these helpers ROS2-only, but pure numpy packing.

def make_pointcloud2(header, points: np.ndarray, fields: list[PointField], point_step: int) -> PointCloud2:
    """
    points: (N, point_step) bytes view OR structured array with .tobytes()
    """
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = int(point_step)
    msg.row_step = int(point_step * msg.width)
    msg.is_dense = False
    msg.data = points.tobytes()
    return msg

def fields_xyz_intensity() -> tuple[list[PointField], int]:
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    return fields, 16

def fields_semantic_lidar() -> tuple[list[PointField], int]:
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='cos_inc_angle', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='object_idx', offset=16, datatype=PointField.UINT32, count=1),
        PointField(name='object_tag', offset=20, datatype=PointField.UINT32, count=1),
    ]
    return fields, 24

def fields_radar() -> tuple[list[PointField], int]:
    fields = [
        PointField(name='velocity', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='azimuth', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='altitude', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='depth', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    return fields, 16

def read_pointcloud2_xyz(points_msg: PointCloud2) -> np.ndarray:
    """Return float32 Nx3 view of x,y,z for PointCloud2 whose first 12 bytes are XYZ float32."""
    data = np.frombuffer(points_msg.data, dtype=np.uint8)
    step = points_msg.point_step
    n = points_msg.width * points_msg.height
    data = data.reshape((n, step))
    xyz = data[:, 0:12].view(np.float32).reshape((n, 3))
    return xyz

def read_pointcloud2_semantic(points_msg: PointCloud2) -> np.ndarray:
    """Return structured array with x,y,z,cos_inc_angle,object_idx,object_tag."""
    n = points_msg.width * points_msg.height
    step = points_msg.point_step
    raw = np.frombuffer(points_msg.data, dtype=np.uint8).reshape((n, step))
    dt = np.dtype([
        ('x','<f4'),('y','<f4'),('z','<f4'),
        ('cos','<f4'),
        ('object_idx','<u4'),
        ('object_tag','<u4'),
    ])
    return raw[:, :24].view(dt).reshape((n,))
