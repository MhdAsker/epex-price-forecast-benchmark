"""Registered Phase 2 ablation specifications."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.research.backtest import BacktestSpec, run_point_backtest
from src.research.features import DailyDataset


def feature_variant(data: DailyDataset, variant: str) -> DailyDataset:
    X = data.X.copy()
    if variant == "full":
        pass
    elif variant == "no_calendar":
        X[:, 72:] = 0.0
    elif variant == "no_d2_curve":
        X[:, 24:48] = 0.0
    elif variant == "no_weekly_curve":
        X[:, 48:72] = 0.0
    else:
        raise KeyError(variant)
    return DailyDataset(data.dates, X, data.y, data.feature_names)


def run_ablation_suite(
    data: DailyDataset,
    model: str = "LightGBM",
    base_spec: BacktestSpec = BacktestSpec(),
    max_test_days: int | None = None,
) -> pd.DataFrame:
    """Feature, window, and refit-frequency ablations on identical origins."""
    frames: list[pd.DataFrame] = []
    scenarios: list[tuple[str, DailyDataset, BacktestSpec]] = []
    for variant in ("full", "no_calendar", "no_d2_curve", "no_weekly_curve"):
        scenarios.append((f"feature:{variant}", feature_variant(data, variant), base_spec))
    for days in (730, 1095, 1461):
        scenarios.append((f"window:{days}", data, replace(base_spec, calibration_days=days)))
    for frequency in (1, 7, 28):
        scenarios.append((f"retrain:{frequency}", data, replace(base_spec, retrain_every_days=frequency)))
    for label, variant_data, spec in scenarios:
        prediction, _ = run_point_backtest(
            variant_data, (model,), spec, max_test_days=max_test_days
        )
        prediction = prediction[prediction["model"] == model].copy()
        prediction["ablation"] = label
        frames.append(prediction)
    return pd.concat(frames, ignore_index=True)
