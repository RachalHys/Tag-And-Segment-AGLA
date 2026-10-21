"""Wall time and CUDA measurements with explicit synchronization boundaries."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import time
from contextlib import contextmanager


def synchronize(device=None):
    import torch

    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        torch.cuda.synchronize(device)


@contextmanager
def timed(target: dict, name: str, device=None):
    synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        target[name] = target.get(name, 0.0) + time.perf_counter() - start


def environment_manifest():
    import torch

    packages = {}
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "tokenizers",
        "accelerate",
        "numpy",
        "Pillow",
        "ultralytics",
        "timm",
        "nltk",
        "textblob",
        "pycocotools",
        "opencv-python",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        driver = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        driver = []
    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "name": device.name,
                "memory_bytes": device.total_memory,
                "capability": [device.major, device.minor],
            }
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "nvidia_driver": driver,
        "gpus": devices,
        "git_commit": revision,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
