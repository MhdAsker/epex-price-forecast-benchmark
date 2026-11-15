"""Install the exact official Kumo code and model artifacts registered by this study."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "kumo_finetuning.json").read_text(encoding="utf-8"))
DEP = ROOT / ".research_deps" / "Kumo-TS"


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    model = CONFIG["model"]
    if not DEP.exists():
        DEP.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", model["official_repository"], str(DEP))
    run("git", "fetch", "origin", model["official_commit"], cwd=DEP)
    run("git", "checkout", "--detach", model["official_commit"], cwd=DEP)
    forecast = DEP / "Kumo-Forecast"
    run("uv", "sync", "--frozen", cwd=forecast)
    python = forecast / ".venv" / "Scripts" / "python.exe"
    run("uv", "pip", "install", "-r", str(ROOT / "requirements-kumo.txt"), cwd=forecast)
    code = f"""
from huggingface_hub import snapshot_download
snapshot_download({model['checkpoint_repo']!r}, revision={model['checkpoint_revision']!r},
 allow_patterns=[{model['checkpoint_file']!r}, {model['standardizer_file']!r}], cache_dir='../hf_cache')
snapshot_download({model['backbone_repo']!r}, revision={model['backbone_revision']!r},
 allow_patterns=['config.json', 'model.safetensors'], cache_dir='../hf_cache')
"""
    run(str(python), "-c", code, cwd=forecast)
    print("Pinned Kumo environment is ready.")


if __name__ == "__main__":
    main()
