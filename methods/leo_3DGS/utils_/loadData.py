from __future__ import annotations
import struct
from dataclasses import dataclass
from pathlib import Path
import numpy as np

'''
加载 COLMAP 生成的点云数据，文件格式为二进制 .bin 文件
这里加载到的rgb颜色会除以255将其归一化到[0, 1]范围内，方便后续处理
'''


@dataclass(frozen=True)
class PointCloudData:
    xyz: np.ndarray
    colors: np.ndarray

def load_colmap_points3d_bin(path: str | Path) -> PointCloudData:
    path = Path(path)

    point_xyz = []
    point_colors = []

    with path.open("rb") as f:
        # num_points 是点云数量
        num_points = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_points):
            _point_id = struct.unpack("<Q", f.read(8))[0]

            xyz = struct.unpack("<3d", f.read(24))
            colors = struct.unpack("<3B", f.read(3))

            _error = struct.unpack("<d", f.read(8))[0]

            track_length = struct.unpack("<Q", f.read(8))[0]
            
            f.read(8 * track_length)

            point_xyz.append(xyz)
            point_colors.append(colors)

    return PointCloudData(
        xyz=np.asarray(point_xyz, dtype=np.float32),
        # 颜色归一化
        colors=np.asarray(point_colors, dtype=np.float32) / 255.0,
    )
    

if __name__ == "__main__":
    data = load_colmap_points3d_bin("src/datasets/SeathruNeRF_dataset/Curasao/sparse/0/points3D.bin")
    print(data.xyz.shape, data.colors.shape)
