#!/usr/bin/env python3
"""Collect Jetson Nano / CUDA runtime evidence for the embedded report.

The default mode reports the real machine state.  It intentionally does not
spoof ARM64 or CUDA results on non-Jetson machines.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def run_command(command: list[str], timeout: float = 5.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "command": " ".join(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }

    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - environment dependent.
        return {
            "available": True,
            "command": " ".join(command),
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }

    return {
        "available": True,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_text_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore").strip("\x00\n ")
    except OSError:
        return ""


def collect_torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "error": repr(exc),
        }

    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", ""),
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": 0,
        "devices": [],
    }

    if torch.cuda.is_available():
        try:
            info["device_count"] = int(torch.cuda.device_count())
            devices = []
            for index in range(info["device_count"]):
                capability = torch.cuda.get_device_capability(index)
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "compute_capability": f"{capability[0]}.{capability[1]}",
                    }
                )
            info["devices"] = devices
        except Exception as exc:  # pragma: no cover - environment dependent.
            info["device_error"] = repr(exc)

    return info


def collect_environment() -> dict[str, Any]:
    machine = platform.machine()
    device_tree_model = read_text_file("/proc/device-tree/model")
    nv_tegra_release = read_text_file("/etc/nv_tegra_release")
    compatible = read_text_file("/proc/device-tree/compatible")
    os_release = read_text_file("/etc/os-release")
    uname = run_command(["uname", "-a"])
    nvcc = run_command(["nvcc", "--version"])
    tegrastats_path = shutil.which("tegrastats")
    nvidia_smi = run_command(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], timeout=5.0)
    torch_info = collect_torch_info()

    is_arm64 = machine.lower() in {"aarch64", "arm64"}
    jetson_markers = [
        "jetson" in device_tree_model.lower(),
        "nvidia" in device_tree_model.lower() and "nano" in device_tree_model.lower(),
        bool(nv_tegra_release),
        "tegra" in compatible.lower(),
        tegrastats_path is not None,
    ]
    is_probable_jetson = any(jetson_markers)

    cuda_device_names = [
        device.get("name", "")
        for device in torch_info.get("devices", [])
        if isinstance(device, dict)
    ]
    has_cuda_runtime = bool(torch_info.get("cuda_available"))
    has_nano_compute_capability = any(
        device.get("compute_capability") == "5.3"
        for device in torch_info.get("devices", [])
        if isinstance(device, dict)
    )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": machine,
            "processor": platform.processor(),
            "uname": uname,
            "os_release": os_release,
        },
        "jetson": {
            "device_tree_model": device_tree_model,
            "compatible": compatible,
            "nv_tegra_release": nv_tegra_release,
            "tegrastats_path": tegrastats_path or "",
            "is_arm64": is_arm64,
            "is_probable_jetson": is_probable_jetson,
        },
        "cuda": {
            "nvcc": nvcc,
            "nvidia_smi": nvidia_smi,
            "torch": torch_info,
            "device_names": cuda_device_names,
            "has_cuda_runtime": has_cuda_runtime,
            "has_nano_compute_capability": has_nano_compute_capability,
        },
        "verdict": {
            "arm64": is_arm64,
            "jetson_marker": is_probable_jetson,
            "cuda_available": has_cuda_runtime,
            "jetson_nano_like": is_arm64 and is_probable_jetson and has_cuda_runtime,
            "strict_jetson_nano_cuda": is_arm64
            and is_probable_jetson
            and has_cuda_runtime
            and has_nano_compute_capability,
        },
    }


def print_command_block(title: str, result: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    print(f"$ {result.get('command', '')}")
    if not result.get("available", False):
        print(f"not available: {result.get('stderr', '')}")
        return
    stdout = str(result.get("stdout", "")).strip()
    stderr = str(result.get("stderr", "")).strip()
    returncode = result.get("returncode")
    print(f"returncode={returncode}")
    if stdout:
        print(stdout)
    if stderr:
        print("stderr:")
        print(stderr)


def print_human_report(info: dict[str, Any]) -> None:
    platform_info = info["platform"]
    jetson = info["jetson"]
    cuda = info["cuda"]
    torch_info = cuda["torch"]
    verdict = info["verdict"]

    print("=" * 78)
    print("Jetson Nano / CUDA 环境检查报告（真实采集）")
    print("=" * 78)
    print(f"采集时间: {info['timestamp']}")
    print(f"Python: {info['python']}")
    print(f"系统: {platform_info['system']} {platform_info['release']}")
    print(f"CPU 架构: {platform_info['machine']}")
    print(f"处理器: {platform_info['processor'] or 'unknown'}")
    print()

    print("[板端识别]")
    print(f"ARM64/aarch64: {'是' if jetson['is_arm64'] else '否'}")
    print(f"疑似 Jetson: {'是' if jetson['is_probable_jetson'] else '否'}")
    print(f"设备树 model: {jetson['device_tree_model'] or '未检测到'}")
    print(f"NVIDIA L4T 信息: {jetson['nv_tegra_release'] or '未检测到'}")
    print(f"tegrastats: {jetson['tegrastats_path'] or '未检测到'}")
    print()

    print("[PyTorch / CUDA]")
    if not torch_info.get("available"):
        print(f"PyTorch: 未安装或导入失败: {torch_info.get('error', '')}")
    else:
        print(f"PyTorch 版本: {torch_info.get('version')}")
        print(f"PyTorch CUDA build: {torch_info.get('cuda_build')}")
        print(f"torch.cuda.is_available(): {torch_info.get('cuda_available')}")
        print(f"CUDA device count: {torch_info.get('device_count')}")
        for device in torch_info.get("devices", []):
            print(
                f"  GPU {device['index']}: {device['name']} "
                f"(compute capability {device['compute_capability']})"
            )
    print()

    print("[结论]")
    print(f"ARM64: {'PASS' if verdict['arm64'] else 'FAIL'}")
    print(f"Jetson 标记: {'PASS' if verdict['jetson_marker'] else 'FAIL'}")
    print(f"CUDA 可用: {'PASS' if verdict['cuda_available'] else 'FAIL'}")
    print(
        "Jetson Nano + CUDA 严格判定: "
        f"{'PASS' if verdict['strict_jetson_nano_cuda'] else 'FAIL'}"
    )
    if not verdict["strict_jetson_nano_cuda"]:
        print(
            "说明: 当前运行环境不能作为 Jetson Nano ARM64/CUDA 实测截图；"
            "请在真实 Jetson Nano B01 上运行本脚本。"
        )

    print_command_block("uname -a", platform_info["uname"])
    print_command_block("nvcc --version", cuda["nvcc"])
    print_command_block("nvidia-smi", cuda["nvidia_smi"])


def print_example_jetson() -> None:
    print("=" * 78)
    print("Jetson Nano 环境检查模板（待板端采集）")
    print("=" * 78)
    print("采集时间: <待板端采集>")
    print("Python: <待板端采集>")
    print("系统: <待板端采集>")
    print("CPU 架构: <待板端采集>")
    print("处理器: <待板端采集>")
    print()
    print("[板端识别]")
    print("ARM64/aarch64: <待板端采集>")
    print("疑似 Jetson: <待板端采集>")
    print("设备树 model: <待板端采集>")
    print("NVIDIA L4T 信息: <待板端采集>")
    print("tegrastats: <待板端采集>")
    print()
    print("[PyTorch / CUDA]")
    print("PyTorch 版本: <待板端采集>")
    print("PyTorch CUDA build: <待板端采集>")
    print("torch.cuda.is_available(): <待板端采集>")
    print("CUDA device count: <待板端采集>")
    print("  GPU 0: <待板端采集>")
    print()
    print("[结论]")
    print("ARM64: <待板端采集>")
    print("Jetson 标记: <待板端采集>")
    print("CUDA 可用: <待板端采集>")
    print("Jetson Nano + CUDA 严格判定: <待板端采集>")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect real Jetson Nano / CUDA environment evidence."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the real collected result as JSON.",
    )
    parser.add_argument(
        "--example-jetson",
        action="store_true",
        help="Print a blank Jetson report template with fields to fill on the board.",
    )
    args = parser.parse_args()

    if args.example_jetson:
        print_example_jetson()
        return

    info = collect_environment()
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print_human_report(info)


if __name__ == "__main__":
    main()
