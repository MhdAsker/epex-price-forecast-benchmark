"""Normalization-controlled Kumo adaptation diagnostics."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.kumo.data import PriceStandardizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_standardizer(
    frame: pd.DataFrame,
    standardizer: PriceStandardizer,
    *,
    name: str,
    fit_start: str | None,
    fit_end_exclusive: str | None,
) -> dict:
    values = frame["price_eur_mwh"].to_numpy(dtype=np.float32)
    probe = np.array([values.min(), values.mean(), values.max()], dtype=np.float32)
    restored = standardizer.inverse(standardizer.transform(probe))
    record = {
        "name": name, "mean": standardizer.mean, "std": standardizer.std,
        "roundtrip_max_abs_error_eur_mwh": float(np.max(np.abs(probe - restored))),
        "fit_start": fit_start, "fit_end_exclusive": fit_end_exclusive,
    }
    if fit_start is not None and fit_end_exclusive is not None:
        eligible = frame.loc[
            (frame.index >= pd.Timestamp(fit_start)) & (frame.index < pd.Timestamp(fit_end_exclusive)),
            "price_eur_mwh",
        ]
        record.update({
            "fit_observations": int(len(eligible)),
            "fit_last_timestamp": eligible.index[-1].isoformat(),
            "recomputed_mean": float(eligible.mean()),
            "recomputed_std_population": float(eligible.to_numpy().std()),
        })
    return record


def extreme_metrics(points: pd.DataFrame, training_prices: np.ndarray) -> pd.DataFrame:
    """Score tails against thresholds fixed from training prices, not validation."""
    low, high = np.quantile(np.asarray(training_prices, dtype=float), [0.05, 0.95])
    rows = []
    for model, group in points.groupby("model"):
        error = np.abs(group.prediction.to_numpy() - group.actual.to_numpy())
        actual = group.actual.to_numpy()
        rows.append({
            "model": model,
            "training_price_q05": float(low), "training_price_q95": float(high),
            "lower_tail_MAE": float(error[actual <= low].mean()) if np.any(actual <= low) else np.nan,
            "upper_tail_MAE": float(error[actual >= high].mean()) if np.any(actual >= high) else np.nan,
            "two_tail_MAE": float(error[(actual <= low) | (actual >= high)].mean()),
            "lower_tail_hours": int(np.sum(actual <= low)),
            "upper_tail_hours": int(np.sum(actual >= high)),
        })
    return pd.DataFrame(rows).sort_values("two_tail_MAE")


def factorial_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return signed MAE contrasts for the registered 2x2 design."""
    value = metrics.set_index("cell")["MAE"].to_dict()
    required = set("ABCD")
    if set(value) != required:
        raise ValueError(f"factorial table requires exactly A/B/C/D, got {sorted(value)}")
    rows = [
        ("normalization_untrained_B_minus_A", value["B"] - value["A"]),
        ("weight_update_german_C_minus_B", value["C"] - value["B"]),
        ("weight_update_nvidia_D_minus_A", value["D"] - value["A"]),
        ("normalization_finetuned_C_minus_D", value["C"] - value["D"]),
        ("interaction", (value["C"] - value["B"]) - (value["D"] - value["A"])),
    ]
    return pd.DataFrame(rows, columns=["contrast", "mae_difference_eur_mwh"])
