"""Append-only pseudo-live forecast packets with a hash-chained index."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_packet(root: Path, packet: dict) -> Path:
    required = {"forecast_origin", "delivery_date", "model", "predictions", "source_checksums"}
    missing = required - packet.keys()
    if missing:
        raise ValueError(f"Missing packet fields: {sorted(missing)}")
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(packet)
    payload.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    payload["sha256"] = digest
    path = root / f"{payload['delivery_date']}_{payload['model'].replace(' ', '_')}_{digest[:12]}.json"
    if path.exists():
        raise FileExistsError(f"Immutable packet already exists: {path}")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_index(root: Path, output: Path) -> dict:
    previous = "0" * 64
    entries = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stated = payload.pop("sha256")
        actual = hashlib.sha256(_canonical(payload)).hexdigest()
        if stated != actual:
            raise ValueError(f"Packet checksum mismatch: {path}")
        chain = hashlib.sha256((previous + stated).encode()).hexdigest()
        entries.append({"file": path.name, "sha256": stated, "previous": previous, "chain": chain})
        previous = chain
    index = {"entries": entries, "head": previous, "immutable": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index
