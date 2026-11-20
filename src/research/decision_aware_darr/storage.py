"""Atomic, checksum-verified batch persistence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def output_record(path: Path, root: Path, rows: int | None = None) -> dict:
    record = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def verify_output(record: dict, root: Path) -> tuple[bool, str]:
    path = root / record["path"]
    if not path.exists():
        return False, "missing"
    if path.stat().st_size != int(record["bytes"]):
        return False, "size mismatch"
    if sha256_file(path) != record["sha256"]:
        return False, "checksum mismatch"
    return True, "verified"


def verify_batch_manifest(path: Path, root: Path, config_hash: str, source_hash: str) -> tuple[bool, str]:
    if not path.exists():
        return False, "manifest missing"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return False, f"manifest unreadable: {error}"
    if manifest.get("status") != "complete":
        return False, "status is not complete"
    if manifest.get("config_sha256") != config_hash:
        return False, "configuration hash conflict"
    if manifest.get("source_sha256") != source_hash:
        return False, "source hash conflict"
    for record in manifest.get("outputs", []):
        valid, reason = verify_output(record, root)
        if not valid:
            return False, f"{record.get('path')}: {reason}"
    return True, "verified"
