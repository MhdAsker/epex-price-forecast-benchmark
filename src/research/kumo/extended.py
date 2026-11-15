"""Pure planning, selection and comparability helpers for extended Kumo study.

This module does not train any model and deliberately has no Kumo/PyTorch imports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def config_digest(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def history_candidates(train_origins: list, requested: list) -> list[tuple[str, list]]:
    """Return fixed chronological suffixes without fabricating unavailable years."""
    if not train_origins:
        raise ValueError("No issuance-safe training origins")
    candidates: list[tuple[str, list]] = []
    for item in requested:
        if item == "all_available":
            suffix = train_origins
            label = "all_available"
        else:
            count = int(item)
            if count > len(train_origins):
                raise ValueError(f"Requested {count} days but only {len(train_origins)} are eligible")
            suffix = train_origins[-count:]
            label = str(count)
        if not suffix:
            raise ValueError(f"Empty training history {label}")
        candidates.append((label, list(suffix)))
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Duplicate history candidate labels")
    return candidates


def choose_validation_candidate(records: list[dict], expected_labels: list[str]) -> dict:
    """Selection must use 2015 validation MAE, never retrospective test outcomes."""
    by_label = {}
    for record in records:
        label = record["history_label"]
        if label in by_label:
            raise ValueError(f"Duplicate candidate: {label}")
        if "test_mae" in record or "retrospective_mae" in record:
            raise ValueError("Do not use test outcomes during model selection")
        score = float(record["validation_operational_mae_eur_mwh"])
        if not np.isfinite(score):
            raise ValueError(f"Nonfinite validation MAE: {label}")
        by_label[label] = record
    if set(by_label) != set(expected_labels):
        raise ValueError(f"Missing/unexpected candidates: expected {expected_labels}, got {list(by_label)}")
    return dict(min(by_label.values(), key=lambda x: (float(x["validation_operational_mae_eur_mwh"]), x["history_label"])))


def assert_identical_model_panel(panel: pd.DataFrame, models: list[str]) -> None:
    """Reject biased comparisons on different eligible hours or targets."""
    required = {"model", "delivery_date", "hour", "actual", "prediction"}
    if not required.issubset(panel.columns):
        raise ValueError(f"Missing panel columns: {sorted(required - set(panel.columns))}")
    canonical = None
    for model in models:
        group = panel.loc[panel.model == model].sort_values(["delivery_date", "hour"])
        if group.empty:
            raise ValueError(f"Missing model predictions: {model}")
        if group.duplicated(["delivery_date", "hour"]).any():
            raise ValueError(f"Duplicate delivery product for {model}")
        keys = list(zip(group.delivery_date.astype(str), group.hour.astype(int)))
        targets = group.actual.to_numpy(dtype=float)
        predictions = group.prediction.to_numpy(dtype=float)
        if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
            raise ValueError(f"Nonfinite values for {model}")
        if canonical is None:
            canonical = (keys, targets)
        elif keys != canonical[0] or not np.allclose(targets, canonical[1], rtol=0, atol=1e-9):
            raise ValueError(f"Noncomparable product coverage/actual prices: {model}")


def validation_seasons(panel: pd.DataFrame) -> pd.DataFrame:
    """Descriptive 2015 seasonal diagnostics, not independent model selection."""
    work = panel.copy()
    months = pd.to_datetime(work.delivery_date).dt.month
    work["season"] = np.select(
        [months.isin([12, 1, 2]), months.isin([3, 4, 5]), months.isin([6, 7, 8])],
        ["winter", "spring", "summer"],
        default="autumn",
    )
    work["absolute_error"] = (work.actual - work.prediction).abs()
    return work.groupby(["model", "season"], as_index=False).agg(
        mae_eur_mwh=("absolute_error", "mean"),
        n_hours=("hour", "size"),
        n_days=("delivery_date", "nunique"),
    )


def ensure_new_immutable_selection(path: Path, payload: dict) -> None:
    """Never silently revise a checkpoint choice after seeing test results."""
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("Frozen selection already exists and differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
