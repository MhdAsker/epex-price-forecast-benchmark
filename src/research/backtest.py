"""Rolling-origin, daily-recalibrated point benchmark with temporal tuning."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.features import DailyDataset
from src.research.models import MODEL_CLASSES, MODEL_GRIDS, DailyModel

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "results" / "predictions"


@dataclass(frozen=True)
class BacktestSpec:
    test_years: tuple[int, ...] = (2016, 2017)
    calibration_days: int = 1461
    validation_days: int = 182
    retrain_every_days: int = 1
    hyperparameter_refresh_days: int = 28
    seed: int = 42


def _temporal_select(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    validation_days: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    if len(X) <= validation_days + 90:
        raise ValueError("Not enough observations for nested temporal validation")
    cut = len(X) - validation_days
    candidates: list[dict] = []
    for params in MODEL_GRIDS[model_name]:
        model = MODEL_CLASSES[model_name](seed=seed, **params)
        started = time.perf_counter()
        model.fit(X[:cut], y[:cut])
        pred = model.predict(X[cut:])
        candidates.append(
            {
                "params": params,
                "validation_mae": float(np.mean(np.abs(y[cut:] - pred))),
                "fit_seconds": time.perf_counter() - started,
            }
        )
    best = min(candidates, key=lambda row: row["validation_mae"])
    return dict(best["params"]), candidates


def _prediction_rows(
    date: pd.Timestamp,
    actual: np.ndarray,
    prediction: np.ndarray,
    model: str,
    spec: BacktestSpec,
    fit_seconds: float,
) -> list[dict]:
    return [
        {
            "delivery_date": date.date().isoformat(),
            "product": hour + 1,
            "hour": hour,
            "actual": float(actual[hour]),
            "prediction": float(prediction[hour]),
            "model": model,
            "seed": spec.seed,
            "calibration_days": spec.calibration_days,
            "retrain_every_days": spec.retrain_every_days,
            "fit_seconds": fit_seconds,
        }
        for hour in range(24)
    ]


def run_point_backtest(
    data: DailyDataset,
    models: tuple[str, ...],
    spec: BacktestSpec,
    *,
    max_test_days: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run outer rolling-origin evaluation and inner blocked validation.

    At each test origin, training contains only earlier delivery days. Model
    parameters are re-estimated at the configured daily frequency. Model
    hyperparameters are selected on the final blocked validation window and
    refreshed periodically without touching outer-test outcomes.
    """
    unknown = set(models) - set(MODEL_CLASSES)
    if unknown:
        raise KeyError(f"Unknown models: {sorted(unknown)}")
    test_mask = np.isin(data.dates.year, spec.test_years)
    test_indices = np.flatnonzero(test_mask)
    if max_test_days is not None:
        test_indices = test_indices[:max_test_days]
    if not len(test_indices):
        raise ValueError("No requested test days in dataset")

    rows: list[dict] = []
    tuning_log: dict[str, list[dict]] = {name: [] for name in models}
    fitted: dict[str, DailyModel] = {}
    params_by_model: dict[str, dict] = {}

    # Baselines are exact columns of the declared lag design.
    for idx in test_indices:
        rows.extend(_prediction_rows(data.dates[idx], data.y[idx], data.X[idx, :24], "Naive D-1", spec, 0.0))
        rows.extend(_prediction_rows(data.dates[idx], data.y[idx], data.X[idx, 48:72], "Naive D-7", spec, 0.0))

    for model_name in models:
        print(f"Backtesting {model_name} on {len(test_indices)} daily origins", flush=True)
        last_fit_position = -10**9
        for position, idx in enumerate(test_indices):
            if position % 28 == 0:
                print(
                    f"  {model_name}: {position + 1}/{len(test_indices)} "
                    f"({data.dates[idx].date()})",
                    flush=True,
                )
            start = max(0, idx - spec.calibration_days)
            X_train, y_train = data.X[start:idx], data.y[start:idx]
            refresh = position % spec.hyperparameter_refresh_days == 0
            if refresh or model_name not in params_by_model:
                params, candidates = _temporal_select(
                    model_name, X_train, y_train, spec.validation_days, spec.seed
                )
                params_by_model[model_name] = params
                tuning_log[model_name].append(
                    {
                        "forecast_date": data.dates[idx].date().isoformat(),
                        "selected": params,
                        "candidates": candidates,
                    }
                )
            if position - last_fit_position >= spec.retrain_every_days or model_name not in fitted:
                model = MODEL_CLASSES[model_name](seed=spec.seed, **params_by_model[model_name])
                started = time.perf_counter()
                model.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - started
                fitted[model_name] = model
                last_fit_position = position
            else:
                fit_seconds = 0.0
            pred = fitted[model_name].predict(data.X[idx : idx + 1])[0]
            rows.extend(
                _prediction_rows(
                    data.dates[idx], data.y[idx], pred, model_name, spec, fit_seconds
                )
            )
    predictions = pd.DataFrame(rows)
    return predictions, tuning_log


def save_backtest(
    predictions: pd.DataFrame,
    tuning_log: dict,
    name: str,
) -> tuple[Path, Path]:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PREDICTIONS_DIR / f"{name}.parquet"
    tune_path = PREDICTIONS_DIR / f"{name}_tuning.json"
    predictions.to_parquet(pred_path, index=False)
    tune_path.write_text(json.dumps(tuning_log, indent=2), encoding="utf-8")
    return pred_path, tune_path
