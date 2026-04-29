"""PLY 点云读写，内置 3DGS checkpoint 属性命名规范支持。

支持 ASCII 和 binary_little_endian 两种格式。
无 PyTorch/JAX 依赖，仅用 NumPy 和标准库。
"""

import os
import struct
from typing import Dict, Tuple

import numpy as np


# PLY 属性类型 → (struct 格式符, numpy dtype, 字节数)
_PLY_DTYPE_MAP = {
    "char":   ("<b",  np.int8,    1),
    "uchar":  ("<B",  np.uint8,   1),
    "short":  ("<h",  np.int16,   2),
    "ushort": ("<H",  np.uint16,  2),
    "int":    ("<i",  np.int32,   4),
    "uint":   ("<I",  np.uint32,  4),
    "float":  ("<f",  np.float32, 4),
    "double": ("<d",  np.float64, 8),
}


def read_ply(path: str) -> Dict[str, np.ndarray]:
    """读取 PLY 文件，返回 {属性名: ndarray} 字典。

    标准点云属性：'x','y','z','red','green','blue','nx','ny','nz'
    3DGS checkpoint 额外属性：'f_dc_0..2','f_rest_0..N','opacity',
                              'scale_0..2','rot_0..3'
    """
    with open(path, "rb") as f:
        # 解析 header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # 提取格式与属性
        fmt = "ascii"
        num_vertices = 0
        properties = []   # [(name, dtype_str)]
        in_vertex = False
        for line in header_lines:
            if line.startswith("format"):
                parts = line.split()
                fmt = parts[1]
            elif line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                dtype_str = parts[1]
                name = parts[2]
                properties.append((name, dtype_str))

        # 读取数据
        if fmt == "ascii":
            data = _read_ply_ascii(f, num_vertices, properties)
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            data = _read_ply_binary(f, num_vertices, properties, fmt)
        else:
            raise ValueError(f"不支持的 PLY 格式：{fmt}")

    return data


def _read_ply_ascii(f, num_vertices: int, properties) -> Dict[str, np.ndarray]:
    cols = {name: [] for name, _ in properties}
    for _ in range(num_vertices):
        vals = f.readline().decode("ascii").split()
        for (name, dtype_str), val in zip(properties, vals):
            _, np_dtype, _ = _PLY_DTYPE_MAP[dtype_str]
            cols[name].append(np_dtype(val))
    return {name: np.array(arr) for name, arr in cols.items()}


def _read_ply_binary(f, num_vertices: int, properties, fmt: str) -> Dict[str, np.ndarray]:
    endian = "<" if fmt == "binary_little_endian" else ">"
    row_fmt = endian + "".join(
        _PLY_DTYPE_MAP[dtype_str][0].lstrip("<>") for _, dtype_str in properties
    )
    row_size = struct.calcsize(row_fmt)
    raw = f.read(row_size * num_vertices)
    if len(raw) < row_size * num_vertices:
        raise IOError("PLY 文件数据不完整")

    # 批量解析所有行
    arr = np.frombuffer(raw, dtype=np.uint8)
    result = {}
    offset = 0
    for name, dtype_str in properties:
        _, np_dtype, nbytes = _PLY_DTYPE_MAP[dtype_str]
        fmt_char = _PLY_DTYPE_MAP[dtype_str][0].lstrip("<>")
        # 从交错字节流中提取该列
        col_raw = np.lib.stride_tricks.as_strided(
            arr[offset:],
            shape=(num_vertices,),
            strides=(row_size,),
        )
        # 逐元素解包（支持跨步读取）
        col_data = np.frombuffer(
            b"".join(
                raw[i * row_size + offset: i * row_size + offset + nbytes]
                for i in range(num_vertices)
            ),
            dtype=np.dtype(endian + fmt_char),
        )
        result[name] = col_data.astype(np_dtype)
        offset += nbytes
    return result


def write_ply(
    path: str,
    data: Dict[str, np.ndarray],
    binary: bool = True,
) -> None:
    """写出 PLY 文件。

    Args:
        path: 输出路径。
        data: {属性名: ndarray}，所有数组长度必须相同。
        binary: True 则写二进制小端格式，False 则写 ASCII。
    """
    names = list(data.keys())
    arrays = [data[n] for n in names]
    num_vertices = len(arrays[0])

    # 推断每列的 PLY 类型
    _NP_TO_PLY = {
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
        np.dtype("uint8"):   "uchar",
        np.dtype("int8"):    "char",
        np.dtype("int16"):   "short",
        np.dtype("uint16"):  "ushort",
        np.dtype("int32"):   "int",
        np.dtype("uint32"):  "uint",
    }

    def _infer_ply_type(arr: np.ndarray) -> str:
        dt = arr.dtype
        if dt in _NP_TO_PLY:
            return _NP_TO_PLY[dt]
        if np.issubdtype(dt, np.floating):
            return "float"
        if np.issubdtype(dt, np.unsignedinteger):
            return "uint"
        return "int"

    ply_types = [_infer_ply_type(a) for a in arrays]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "wb") as f:
        # 写 header
        header = ["ply"]
        if binary:
            header.append("format binary_little_endian 1.0")
        else:
            header.append("format ascii 1.0")
        header.append(f"element vertex {num_vertices}")
        for name, ply_type in zip(names, ply_types):
            header.append(f"property {ply_type} {name}")
        header.append("end_header")
        f.write(("\n".join(header) + "\n").encode("ascii"))

        if binary:
            row_fmt = "<" + "".join(
                _PLY_DTYPE_MAP[t][0].lstrip("<>") for t in ply_types
            )
            casted = [
                a.astype(_PLY_DTYPE_MAP[t][1]) for a, t in zip(arrays, ply_types)
            ]
            for i in range(num_vertices):
                row = struct.pack(row_fmt, *[c[i] for c in casted])
                f.write(row)
        else:
            for i in range(num_vertices):
                row = " ".join(str(a[i]) for a in arrays)
                f.write((row + "\n").encode("ascii"))


# ---------------------------------------------------------------------------
# 3DGS checkpoint 转换工具
# ---------------------------------------------------------------------------

def gaussians_to_ply_dict(
    means: np.ndarray,        # (N, 3)
    scales: np.ndarray,       # (N, 3) log-scale
    rotations: np.ndarray,    # (N, 4) wxyz
    opacities: np.ndarray,    # (N, 1) pre-sigmoid
    sh_dc: np.ndarray,        # (N, 3) DC SH 系数（对应 f_dc_0/1/2）
    sh_rest: np.ndarray,      # (N, K, 3) 高阶 SH 系数
) -> Dict[str, np.ndarray]:
    """将高斯参数数组转换为 PLY 属性字典，匹配官方 3DGS 命名规范。"""
    N = len(means)
    data: Dict[str, np.ndarray] = {}

    data["x"] = means[:, 0].astype(np.float32)
    data["y"] = means[:, 1].astype(np.float32)
    data["z"] = means[:, 2].astype(np.float32)

    # DC SH 系数：shape (N, 1, 3) 或 (N, 3)
    dc = sh_dc.reshape(N, -1)   # → (N, 3)
    for i in range(dc.shape[1]):
        data[f"f_dc_{i}"] = dc[:, i].astype(np.float32)

    # 高阶 SH 系数：shape (N, K, 3) → f_rest_0 .. f_rest_{3K-1}
    rest = sh_rest.reshape(N, -1)   # → (N, K*3)
    for i in range(rest.shape[1]):
        data[f"f_rest_{i}"] = rest[:, i].astype(np.float32)

    data["opacity"] = opacities.reshape(N).astype(np.float32)

    for i in range(3):
        data[f"scale_{i}"] = scales[:, i].astype(np.float32)

    for i in range(4):
        data[f"rot_{i}"] = rotations[:, i].astype(np.float32)

    return data


def ply_dict_to_gaussians(
    data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """gaussians_to_ply_dict 的逆操作。

    Returns:
        (means(N,3), scales(N,3), rotations(N,4),
         opacities(N,1), sh_dc(N,1,3), sh_rest(N,K,3))
    """
    means = np.stack([data["x"], data["y"], data["z"]], axis=1)  # (N,3)
    N = len(means)

    # DC SH
    dc_keys = sorted([k for k in data if k.startswith("f_dc_")], key=lambda k: int(k.split("_")[-1]))
    sh_dc = np.stack([data[k] for k in dc_keys], axis=1).reshape(N, 1, 3)

    # 高阶 SH
    rest_keys = sorted([k for k in data if k.startswith("f_rest_")], key=lambda k: int(k.split("_")[-1]))
    if rest_keys:
        rest_flat = np.stack([data[k] for k in rest_keys], axis=1)  # (N, K*3)
        K = rest_flat.shape[1] // 3
        sh_rest = rest_flat.reshape(N, K, 3)
    else:
        sh_rest = np.zeros((N, 0, 3), dtype=np.float32)

    opacities = data["opacity"].reshape(N, 1)

    scale_keys = sorted([k for k in data if k.startswith("scale_")], key=lambda k: int(k.split("_")[-1]))
    scales = np.stack([data[k] for k in scale_keys], axis=1)  # (N,3)

    rot_keys = sorted([k for k in data if k.startswith("rot_")], key=lambda k: int(k.split("_")[-1]))
    rotations = np.stack([data[k] for k in rot_keys], axis=1)  # (N,4)

    return means, scales, rotations, opacities, sh_dc, sh_rest
