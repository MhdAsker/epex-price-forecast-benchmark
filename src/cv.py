"""
Walk-forward (expanding-window) cross-validation for energy price forecasting.

Each fold extends the training set by one period and evaluates on the next
fixed-length window — the standard evaluation protocol for energy forecasting
papers (see e.g. Nowotarski & Weron, 2018; Lago et al., 2021).

Usage
-----
    from src.cv import walk_forward_cv
    from src.models.lgbm_model import LGBMModel
    from src.models.xgb_model import XGBModel

    cv_results = walk_forward_cv(
        df            = df[df.index < "2024-01-01"],
        model_classes = {"LightGBM": LGBMModel, "XGBoost": XGBModel},
        target_col    = "price_eur_mwh",
        exclude_cols  = {"price_eur_mwh", "negative_price_flag"},
        n_splits      = 5,
        min_train_months = 6,
        test_months   = 3,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def walk_forward_day_ahead_cv(
    raw_df: pd.DataFrame,
    n_splits: int = 5,
    min_train_months: int = 6,
    test_months: int = 3,
    val_fraction: float = 0.10,
) -> dict[str, dict]:
    """Expanding-window CV on the primary noon-D-1 information set.

    This is the research-facing CV entry point.  Unlike ``walk_forward_cv``,
    it constructs the auction-safe matrix internally and cannot accidentally
    inherit ``price_lag_1h`` or unverifiable post-gate ENTSO-E forecasts.
    """
    from src.day_ahead import build_day_ahead_features, feature_columns
    from src.models.day_ahead_models import fit_lgbm
    from src.protocol import assert_primary_features

    frame = build_day_ahead_features(raw_df).sort_index()
    cols = feature_columns(frame)
    assert_primary_features(cols)
    start = pd.Timestamp(frame.index.min())
    end = pd.Timestamp(frame.index.max())
    fold_specs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(n_splits):
        test_start = start + pd.DateOffset(months=min_train_months + i * test_months)
        test_end = min(
            test_start + pd.DateOffset(months=test_months),
            end + pd.Timedelta(hours=1),
        )
        if test_start >= end:
            break
        fold_specs.append((test_start, test_end))
    if not fold_specs:
        raise ValueError("No valid day-ahead folds could be constructed")

    fold_maes: list[float] = []
    fold_rmses: list[float] = []
    fold_times: list[float] = []
    fold_meta: list[dict] = []
    for number, (test_start, test_end) in enumerate(fold_specs, start=1):
        train = frame[frame.index < test_start]
        test = frame[(frame.index >= test_start) & (frame.index < test_end)]
        cut = int(len(train) * (1.0 - val_fraction))
        fit, val = train.iloc[:cut], train.iloc[cut:]
        if len(fit) < 500 or len(val) < 24 or len(test) < 24:
            continue
        t0 = time.time()
        model = fit_lgbm(
            fit[cols].to_numpy(dtype=float),
            fit["price_eur_mwh"].to_numpy(dtype=float),
            val[cols].to_numpy(dtype=float),
            val["price_eur_mwh"].to_numpy(dtype=float),
        )
        elapsed = time.time() - t0
        actual = test["price_eur_mwh"].to_numpy(dtype=float)
        pred = model.predict(test[cols].to_numpy(dtype=float))
        fold_maes.append(float(np.mean(np.abs(actual - pred))))
        fold_rmses.append(float(np.sqrt(np.mean((actual - pred) ** 2))))
        fold_times.append(elapsed)
        fold_meta.append(
            {
                "fold": number,
                "train_start": train.index.min(),
                "train_end": train.index.max(),
                "test_start": test.index.min(),
                "test_end": test.index.max(),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
    if not fold_maes:
        raise ValueError("All day-ahead folds were too small")
    return {
        "DA Price History": {
            "fold_maes": fold_maes,
            "fold_rmses": fold_rmses,
            "fold_times": fold_times,
            "fold_meta": fold_meta,
            "mean_mae": float(np.mean(fold_maes)),
            "std_mae": float(np.std(fold_maes)),
            "mean_rmse": float(np.mean(fold_rmses)),
            "std_rmse": float(np.std(fold_rmses)),
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    model_classes: dict[str, Type],
    target_col: str = "price_eur_mwh",
    exclude_cols: set[str] | None = None,
    n_splits: int = 5,
    min_train_months: int = 6,
    test_months: int = 3,
    val_fraction: float = 0.10,
) -> dict[str, dict]:
    """
    Expanding-window walk-forward CV.

    Parameters
    ----------
    df               : Feature DataFrame with DatetimeIndex (training data only).
    model_classes    : Mapping of model name → class that supports .fit() / .predict().
    target_col       : Name of the regression target column.
    exclude_cols     : Columns excluded from feature matrix (target + flags).
    n_splits         : Number of folds.
    min_train_months : Minimum months of training data before the first test fold.
    test_months      : Length of each test fold in months.
    val_fraction     : Fraction of each training set held back as validation.

    Returns
    -------
    dict keyed by model name, each containing::

        {
            "fold_maes":  [float, ...],
            "fold_rmses": [float, ...],
            "fold_times": [float, ...],
            "fold_meta":  [{"fold": int, "train_start": ..., "test_end": ...}, ...],
            "mean_mae":   float,
            "std_mae":    float,
            "mean_rmse":  float,
            "std_rmse":   float,
        }
    """
    if exclude_cols is None:
        exclude_cols = {target_col, "negative_price_flag"}

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    start_date = df.index.min()

    # ── Build fold boundaries ────────────────────────────────────────────────
    folds: list[dict] = []
    for i in range(n_splits):
        test_start = start_date + pd.DateOffset(months=min_train_months + i * test_months)
        test_end   = start_date + pd.DateOffset(months=min_train_months + (i + 1) * test_months)

        if test_start >= df.index.max():
            logger.warning("Fold %d: test_start beyond data — stopping at %d folds.", i + 1, i)
            break

        # Clamp to available data
        test_end = min(test_end, df.index.max() + pd.Timedelta(hours=1))

        train_df = df[df.index < test_start]
        test_df  = df[(df.index >= test_start) & (df.index < test_end)]

        if len(train_df) < 500 or len(test_df) < 24:
            logger.warning("Fold %d: not enough data (train=%d, test=%d) — skipping.",
                           i + 1, len(train_df), len(test_df))
            continue

        folds.append({
            "fold":        i + 1,
            "train_start": train_df.index.min(),
            "train_end":   train_df.index.max(),
            "test_start":  test_df.index.min(),
            "test_end":    test_df.index.max(),
            "train_df":    train_df,
            "test_df":     test_df,
        })

    if not folds:
        raise ValueError("No valid folds could be constructed. Check min_train_months / test_months.")

    print(f"\n  Walk-forward CV: {len(folds)} folds, "
          f"test window = {test_months} months each")

    results: dict[str, dict] = {}

    # ── Run each model ───────────────────────────────────────────────────────
    for model_name, ModelClass in model_classes.items():
        fold_maes:  list[float] = []
        fold_rmses: list[float] = []
        fold_times: list[float] = []
        fold_meta:  list[dict]  = []

        print(f"\n  ▸ {model_name}")
        print(f"    {'Fold':<6} {'Train period':<28} {'Test period':<22} "
              f"{'MAE':>7} {'RMSE':>7} {'Time':>6}")
        print("    " + "─" * 76)

        for fold in folds:
            train_df = fold["train_df"]
            test_df  = fold["test_df"]

            # Val = last val_fraction of training set (time-ordered)
            val_start_idx = int(len(train_df) * (1.0 - val_fraction))
            train_part = train_df.iloc[:val_start_idx]
            val_part   = train_df.iloc[val_start_idx:]

            X_train = train_part[feature_cols].values
            y_train = train_part[target_col].values
            X_val   = val_part[feature_cols].values
            y_val   = val_part[target_col].values
            X_test  = test_df[feature_cols].values
            y_test  = test_df[target_col].values

            # Fit
            model = ModelClass()
            t0 = time.time()
            model.fit(X_train, y_train, X_val, y_val)
            elapsed = time.time() - t0

            # Evaluate
            preds = model.predict(X_test)
            mae   = float(np.mean(np.abs(y_test - preds)))
            rmse  = float(np.sqrt(np.mean((y_test - preds) ** 2)))

            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_times.append(elapsed)
            fold_meta.append({
                "fold":        fold["fold"],
                "train_start": fold["train_start"],
                "train_end":   fold["train_end"],
                "test_start":  fold["test_start"],
                "test_end":    fold["test_end"],
                "n_train":     len(train_df),
                "n_test":      len(test_df),
            })

            train_period = (f"{fold['train_start'].strftime('%Y-%m-%d')} – "
                            f"{fold['train_end'].strftime('%Y-%m-%d')}")
            test_period  = (f"{fold['test_start'].strftime('%Y-%m-%d')} – "
                            f"{fold['test_end'].strftime('%Y-%m-%d')}")
            print(f"    {fold['fold']:<6} {train_period:<28} {test_period:<22} "
                  f"{mae:>7.2f} {rmse:>7.2f} {elapsed:>5.1f}s")

        mean_mae  = float(np.mean(fold_maes))
        std_mae   = float(np.std(fold_maes))
        mean_rmse = float(np.mean(fold_rmses))
        std_rmse  = float(np.std(fold_rmses))

        print(f"    {'MEAN':<6} {'':<28} {'':<22} "
              f"{mean_mae:>7.2f} {mean_rmse:>7.2f}")
        print(f"    {'STD':<6} {'':<28} {'':<22} "
              f"{std_mae:>7.2f} {std_rmse:>7.2f}")

        results[model_name] = {
            "fold_maes":  fold_maes,
            "fold_rmses": fold_rmses,
            "fold_times": fold_times,
            "fold_meta":  fold_meta,
            "mean_mae":   mean_mae,
            "std_mae":    std_mae,
            "mean_rmse":  mean_rmse,
            "std_rmse":   std_rmse,
        }

    return results

