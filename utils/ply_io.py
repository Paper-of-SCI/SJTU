"""PLY helpers for 3DGS-style Gaussian checkpoints.

The writer emits standard 3DGS PLY files that common viewers such as
SuperSplat can import: binary little-endian, named Gaussian properties, and
the Graphdeco high-order SH coefficient layout.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np


PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def gaussians_to_ply_dict(
    means: np.ndarray,
    log_scales: np.ndarray,
    quats: np.ndarray,
    logit_opacities: np.ndarray,
    features_dc: np.ndarray,
    features_rest: np.ndarray,
) -> Dict[str, np.ndarray]:
    n = means.shape[0]
    data: Dict[str, np.ndarray] = {
        "x": means[:, 0].astype(np.float32),
        "y": means[:, 1].astype(np.float32),
        "z": means[:, 2].astype(np.float32),
        "nx": np.zeros(n, dtype=np.float32),
        "ny": np.zeros(n, dtype=np.float32),
        "nz": np.zeros(n, dtype=np.float32),
    }
    dc = features_dc.reshape(n, -1)
    for i in range(dc.shape[1]):
        data[f"f_dc_{i}"] = dc[:, i].astype(np.float32)
    rest = features_rest.transpose(0, 2, 1).reshape(n, -1)
    for i in range(rest.shape[1]):
        data[f"f_rest_{i}"] = rest[:, i].astype(np.float32)
    data["opacity"] = logit_opacities.reshape(n).astype(np.float32)
    for i in range(3):
        data[f"scale_{i}"] = log_scales[:, i].astype(np.float32)
    for i in range(4):
        data[f"rot_{i}"] = quats[:, i].astype(np.float32)
    return data


def ply_dict_to_gaussians(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    n = means.shape[0]
    dc_keys = _numbered_keys(data, "f_dc_")
    rest_keys = _numbered_keys(data, "f_rest_")
    scale_keys = _numbered_keys(data, "scale_")
    rot_keys = _numbered_keys(data, "rot_")
    features_dc = np.stack([data[key] for key in dc_keys], axis=1).reshape(n, 1, 3).astype(np.float32)
    if rest_keys:
        rest = np.stack([data[key] for key in rest_keys], axis=1)
        features_rest = rest.reshape(n, 3, rest.shape[1] // 3).transpose(0, 2, 1).astype(np.float32)
    else:
        features_rest = np.zeros((n, 0, 3), dtype=np.float32)
    log_scales = np.stack([data[key] for key in scale_keys], axis=1).astype(np.float32)
    quats = np.stack([data[key] for key in rot_keys], axis=1).astype(np.float32)
    opacities = data["opacity"].reshape(n, 1).astype(np.float32)
    return means, log_scales, quats, opacities, features_dc, features_rest


def read_ply(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as handle:
        header = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = "ascii"
        count = 0
        properties = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
                in_vertex = True
            elif parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[0] == "property":
                properties.append((parts[2], parts[1]))
        if fmt == "ascii":
            rows = [handle.readline().decode("ascii").split() for _ in range(count)]
            return {name: np.array([float(row[i]) for row in rows], dtype=np.float32) for i, (name, _) in enumerate(properties)}
        if fmt != "binary_little_endian":
            raise ValueError(f"不支持的 PLY 格式: {fmt}")
        dtype = _structured_dtype(properties)
        table = np.fromfile(handle, dtype=dtype, count=count)
    result = {name: table[name].astype(np.float32) for name, _ in properties}
    return result


def write_ply(path: str, data: Dict[str, np.ndarray], binary: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    names = list(data.keys())
    count = len(data[names[0]]) if names else 0
    if not binary:
        with open(path, "w", encoding="ascii") as handle:
            _write_header(handle, names, count, fmt="ascii")
            for i in range(count):
                handle.write(" ".join(str(float(data[name][i])) for name in names) + "\n")
        return

    dtype = [(name, "<f4") for name in names]
    table = np.empty(count, dtype=dtype)
    for name in names:
        table[name] = np.asarray(data[name], dtype=np.float32)
    with open(path, "wb") as handle:
        header = _format_header(names, count, fmt="binary_little_endian").encode("ascii")
        handle.write(header)
        table.tofile(handle)


def _numbered_keys(data: Dict[str, np.ndarray], prefix: str) -> list[str]:
    return sorted([key for key in data if key.startswith(prefix)], key=lambda key: int(key[len(prefix) :]))


def _structured_dtype(properties: list[tuple[str, str]]) -> np.dtype:
    fields = []
    for name, ply_type in properties:
        dtype = PLY_DTYPE_MAP.get(ply_type)
        if dtype is None:
            raise ValueError(f"不支持的 PLY 属性类型: {ply_type} ({name})")
        fields.append((name, dtype))
    return np.dtype(fields)


def _write_header(handle, names: list[str], count: int, fmt: str) -> None:
    handle.write(_format_header(names, count, fmt))


def _format_header(names: list[str], count: int, fmt: str) -> str:
    lines = ["ply", f"format {fmt} 1.0", f"element vertex {count}"]
    lines.extend(f"property float {name}" for name in names)
    lines.append("end_header")
    return "\n".join(lines) + "\n"
