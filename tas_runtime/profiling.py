"""Wall time and CUDA measurements with explicit synchronization boundaries."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess


def synchronize(device=None):
    import torch

    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        torch.cuda.synchronize(device)


def profile_call(call, directory, *, device):
    """One diagnostic call; profiler timings never select the execution policy."""
    from pathlib import Path
    import torch
    from .artifacts import atomic_json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table_path = directory / "operators.txt"
    operators_path = directory / "operators.json"
    result_path = directory / "profile.json"
    table_path.unlink(missing_ok=True)
    operators_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda = str(device).startswith("cuda")
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    call_failed = False
    try:
        with torch.profiler.profile(activities=activities) as profiler:
            try:
                call()
            except Exception as exc:
                call_failed = True
                atomic_json(result_path, {"status": "failed", "error": str(exc)})
                raise
        averages = profiler.key_averages()
        # Retain tiny synchronization/transfer operators even when large GEMMs
        # push them out of the printed top twenty. No per-token instrumentation.
        operators = [
            {
                "name": event.key,
                "count": event.count,
                "self_cpu_time_us": event.self_cpu_time_total,
                "self_device_time_us": getattr(event, "self_device_time_total", 0.0),
            }
            for event in averages
            if any(name in event.key.lower() for name in (
                "item", "local_scalar", "synchronize", "memcpy", "nonzero",
                "aten::cat", "aten::mm", "aten::addmm", "aten::bmm", "scaled_dot_product",
                "multinomial",
            ))
        ]
        atomic_json(operators_path, {"time_unit": "microseconds", "operators": operators})
        cpu_table = averages.table(sort_by="self_cpu_time_total", row_limit=20)
        cuda_table = (
            averages.table(sort_by="self_cuda_time_total", row_limit=20)
            if cuda else "CUDA profiling was not requested."
        )
        table_path.write_text(
            "Profiler adds overhead; use unprofiled pilot for throughput.\n\n"
            + cpu_table + "\n\n" + cuda_table
        )
        result = {
            "status": "complete", "table": "operators.txt", "operator_totals": "operators.json",
            "scope": "one warmed pilot-length call; no shape, stack or memory tracing",
            "throughput": "use unprofiled execution-policy pilot, not profiler timings",
        }
        print(cpu_table, flush=True)
        if cuda:
            print(cuda_table, flush=True)
    except RuntimeError as exc:
        if call_failed:
            raise
        # CUPTI can be unavailable in a hosted runtime. This diagnostic never
        # bypasses or changes the independently completed P0 correctness gate.
        result = {"status": "unavailable", "error": str(exc)}
        print(f"Operator profiler unavailable: {exc}", flush=True)
    atomic_json(result_path, result)
    return result


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
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
