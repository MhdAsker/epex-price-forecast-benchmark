"""Reproducible environment and artifact diagnostics."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path


def resource_report() -> dict:
    import torch

    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    disk = shutil.disk_usage(Path.cwd())
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "nvidia_smi": gpu,
        "disk_free_bytes": disk.free,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
