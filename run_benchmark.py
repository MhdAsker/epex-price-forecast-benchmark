"""
EPEX Price Forecast Benchmark — extended diagnostics entry point.

The point/quantile models below retain the original hourly information set and
are not valid noon-D-1 forecasts. The research-facing result is produced by
``run_day_ahead.py``; this script keeps the legacy comparison for auditability.

Runs the full pipeline:
  1.  Load features
  2.  Train / val / test split
  3.  Walk-forward cross-validation  (LightGBM + XGBoost)           [Upgrade #3]
  4.  Train base point-forecast models  (SARIMA · LGBM · XGB · LSTM)
  5.  Train quantile models             (LGBM + XGB q10/q50/q90)    [Upgrade #2]
  6.  Print leaderboard + save CSV
  7.  Generate all plots:
        • Actual vs predicted (30 days)
        • Error distribution
        • Feature importance (split-count + gain)
        • Negative price analysis
        • SHAP beeswarm  (LGBM + XGB)                               [Upgrade #1]
        • Quantile fan chart  (LGBM + XGB)                          [Upgrade #2]
        • Walk-forward CV  (MAE + RMSE per fold)                     [Upgrade #3]
        • Day-ahead auction protocol (hourly experts + tails)       [Upgrade #4]

Usage
-----
    python run_benchmark.py           # full run
    python run_benchmark.py --quick   # skip CV, quantile, day-ahead (fast demo)
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows terminals (cp1252 can't encode box-drawing chars)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cv import walk_forward_day_ahead_cv
from src.day_ahead import run_comparison
from src.evaluate import compute_metrics, pinball_loss, print_leaderboard, save_metrics
from src.models.lgbm_model import LGBMModel
from src.models.lgbm_quantile_model import LGBMQuantileModel
from src.models.lstm_model import LSTMModel
from src.models.sarima_model import SARIMAModel
from src.models.xgb_model import XGBModel
from src.visualize import (
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_feature_importance,
    plot_negative_price_analysis,
    plot_quantile_forecast,
    plot_shap_summary,
    plot_walk_forward_cv,
)

# ── constants ────────────────────────────────────────────────────────────────
SPLIT_DATE   = "2024-01-01"
TARGET_COL   = "price_eur_mwh"
EXCLUDE_COLS = {"price_eur_mwh", "negative_price_flag"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 3} {title} {'─' * max(0, 44 - len(title))}")


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EPEX Price Forecast Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Skip walk-forward CV, quantile models, and the day-ahead protocol "
            "for a fast demo run (point-forecast models + basic plots only)."
        ),
    )
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    load_dotenv()
    print(
        "WARNING: legacy hourly point/quantile scores are diagnostic only; "
        "the primary noon-D-1 leaderboard is produced by run_day_ahead.py."
    )

    # ── 1. Load features ──────────────────────────────────────────────────────
    _section("Load features")
    features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        print("ERROR: data/processed/features.parquet not found.")
        print("Run:  python src/features.py  (after fetching raw data)")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    print(f"Loaded features: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 2. Train / val / test split ───────────────────────────────────────────
    _section("Split")
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

    train_full = df[df.index <  SPLIT_DATE]
    test       = df[df.index >= SPLIT_DATE]

    # Validation = last 10 % of training set (time-ordered, no shuffle)
    val_start = int(len(train_full) * 0.90)
    train = train_full.iloc[:val_start]
    val   = train_full.iloc[val_start:]

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_val   = val[FEATURE_COLS].values
    y_val   = val[TARGET_COL].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET_COL].values

    n_features = len(FEATURE_COLS)
    n_train    = len(train_full)
    n_test     = len(test)
    test_dates = test.index

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {n_test:,}")
    if args.quick:
        print("  [--quick mode: CV, quantile forecasts, and day-ahead protocol will be skipped]")

    # ── 3. Walk-forward cross-validation ─────────────────────────────────────
    cv_results: dict | None = None

    if not args.quick:
        _section("Auction-safe walk-forward cross-validation")
        cv_results = walk_forward_day_ahead_cv(
            raw_df=train_full,
            n_splits=5,
            min_train_months=6,
            test_months=3,
            val_fraction=0.10,
        )
        print("\n  CV Summary:")
        for name, res in cv_results.items():
            print(f"    {name}: MAE = {res['mean_mae']:.2f} ± {res['std_mae']:.2f} €/MWh  "
                  f"| RMSE = {res['mean_rmse']:.2f} ± {res['std_rmse']:.2f} €/MWh")

    # ── 4. Point-forecast models ──────────────────────────────────────────────
    metrics_list: list[dict]       = []
    predictions:  dict[str, np.ndarray] = {}

    # 4a. SARIMA (price series only — no feature matrix)
    _section("SARIMA")
    sarima = SARIMAModel()
    t0 = time.time()
    sarima_preds = sarima.fit_predict(train_full[TARGET_COL], test[TARGET_COL])
    sarima_time  = time.time() - t0
    predictions["SARIMA"] = sarima_preds
    metrics_list.append(compute_metrics(y_test, sarima_preds, "SARIMA", sarima_time))

    # 4b. LightGBM
    _section("LightGBM")
    lgbm = LGBMModel()
    t0 = time.time()
    lgbm.fit(X_train, y_train, X_val, y_val)
    lgbm_time = time.time() - t0
    lgbm_preds = lgbm.predict(X_test)
    predictions["LightGBM"] = lgbm_preds
    metrics_list.append(compute_metrics(y_test, lgbm_preds, "LightGBM", lgbm_time))

    # 4c. XGBoost
    _section("XGBoost")
    xgb_model = XGBModel()
    t0 = time.time()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_time = time.time() - t0
    xgb_preds = xgb_model.predict(X_test)
    predictions["XGBoost"] = xgb_preds
    metrics_list.append(compute_metrics(y_test, xgb_preds, "XGBoost", xgb_time))

    # 4d. LSTM
    _section("LSTM")
    lstm = LSTMModel(input_dim=n_features, hidden_dim=64)
    t0 = time.time()
    lstm.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.001)
    lstm_time  = time.time() - t0
    lstm_preds = lstm.predict(X_test)

    # Metrics: align y_test to the (shorter) LSTM output
    lstm_offset  = len(y_test) - len(lstm_preds)
    y_test_lstm  = y_test[lstm_offset:]
    metrics_list.append(compute_metrics(y_test_lstm, lstm_preds, "LSTM", lstm_time))

    # BUG FIX: pad LSTM predictions with NaN so all models share the same date
    # axis.  The first `lstm_offset` hours are left as NaN (warm-up window).
    lstm_preds_padded = np.concatenate([np.full(lstm_offset, np.nan), lstm_preds])
    predictions["LSTM"] = lstm_preds_padded

    # ── 5. Quantile / probabilistic models ────────────────────────────────────
    quantile_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    quantile_summary: list[str] = []

    if not args.quick:
        _section("Quantile models  [Upgrade #2]")

        # LightGBM quantile
        print("  LightGBM q10 / q50 / q90 …")
        lgbm_q = LGBMQuantileModel()
        lgbm_q.fit(X_train, y_train, X_val, y_val)
        lgbm_q10, lgbm_q50, lgbm_q90 = lgbm_q.predict_interval(X_test)
        quantile_preds["LightGBM"] = (lgbm_q10, lgbm_q50, lgbm_q90)

        lgbm_pb = (pinball_loss(y_test, lgbm_q10, 0.10) + pinball_loss(y_test, lgbm_q90, 0.90)) / 2
        lgbm_cov = float(np.mean((y_test >= lgbm_q10) & (y_test <= lgbm_q90)) * 100)
        quantile_summary.append(
            f"  LightGBM  — Pinball (avg q10+q90): {lgbm_pb:.3f} €/MWh  |  "
            f"Coverage: {lgbm_cov:.1f}%  (target 80%)"
        )

        # XGBoost quantile
        print("  XGBoost q10 / q50 / q90 …")
        try:
            from src.models.xgb_quantile_model import XGBQuantileModel
            xgb_q = XGBQuantileModel()
            xgb_q.fit(X_train, y_train, X_val, y_val)
            xgb_q10, xgb_q50, xgb_q90 = xgb_q.predict_interval(X_test)
            quantile_preds["XGBoost"] = (xgb_q10, xgb_q50, xgb_q90)

            xgb_pb = (pinball_loss(y_test, xgb_q10, 0.10) + pinball_loss(y_test, xgb_q90, 0.90)) / 2
            xgb_cov = float(np.mean((y_test >= xgb_q10) & (y_test <= xgb_q90)) * 100)
            quantile_summary.append(
                f"  XGBoost   — Pinball (avg q10+q90): {xgb_pb:.3f} €/MWh  |  "
                f"Coverage: {xgb_cov:.1f}%  (target 80%)"
            )
        except (ImportError, Exception) as exc:
            print(f"  [XGBoost quantile skipped: {exc}]")

    # ── 5b. Day-ahead auction protocol ────────────────────────────────────────
    if not args.quick:
        _section("Day-ahead auction protocol  [Upgrade #4]")
        run_comparison(df)

    # ── 6. Leaderboard ────────────────────────────────────────────────────────
    _section("Legacy hourly diagnostic leaderboard")
    print_leaderboard(metrics_list)
    save_metrics(metrics_list)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    _section("Plots")

    print("  Point-forecast diagnostics …")
    plot_actual_vs_predicted(y_test, predictions, test_dates)
    plot_error_distribution(y_test, predictions)
    plot_feature_importance(lgbm, xgb_model, FEATURE_COLS)
    plot_negative_price_analysis(test, predictions)

    print("  SHAP beeswarm …  [Upgrade #1]")
    plot_shap_summary(lgbm, xgb_model, X_test, FEATURE_COLS)

    if quantile_preds:
        print("  Quantile fan chart …  [Upgrade #2]")
        plot_quantile_forecast(y_test, quantile_preds, test_dates)

    if cv_results:
        print("  Walk-forward CV chart …  [Upgrade #3]")
        plot_walk_forward_cv(cv_results)

    # ── 8. Final summary ──────────────────────────────────────────────────────
    sorted_metrics = sorted(metrics_list, key=lambda x: x["MAE"])
    best    = sorted_metrics[0]
    sarima_m = next(m for m in metrics_list if m["model"] == "SARIMA")
    improvement = (sarima_m["MAE"] - best["MAE"]) / sarima_m["MAE"] * 100

    lgbm_imp  = lgbm.get_feature_importance(FEATURE_COLS)
    top_feat  = lgbm_imp.iloc[0]["feature"] if not lgbm_imp.empty else "N/A"

    neg_hours = int((test[TARGET_COL] < 0).sum())
    neg_pct   = 100 * neg_hours / n_test

    bar = "═" * 54
    print(f"\n{bar}")
    print("LEGACY HOURLY DIAGNOSTIC — NOT A NOON D-1 LEADERBOARD")
    print(bar)
    print(f"Dataset     : DE-LU bidding zone, 2022–2023 train")
    print(f"Test period : {test.index.min().date()} to {test.index.max().date()}")
    print(f"Features    : {n_features} engineered features")
    print(f"Train rows  : {n_train:,}  |  Test rows: {n_test:,}")
    print()
    print(f"{'Model':<14} {'MAE':>7} {'RMSE':>7} {'MAPE':>8}  {'Time':>7}")
    print("─" * 48)
    for m in sorted_metrics:
        tag = "  ← WINNER" if m["model"] == best["model"] else ""
        print(
            f"{m['model']:<14} {m['MAE']:>7.2f} {m['RMSE']:>7.2f}"
            f" {m['MAPE']:>7.1f}%  {m['training_time_s']:>6.1f}s{tag}"
        )
    print()
    print(f"  {best['model']} beat SARIMA by {improvement:.1f}% on MAE")
    print(f"  Top LGBM feature: {top_feat}")
    print(f"  Negative price hours: {neg_hours} ({neg_pct:.1f}%)")

    if quantile_summary:
        print()
        print("Probabilistic Forecast (80% interval):")
        for line in quantile_summary:
            print(line)

    if cv_results:
        print()
        print("Walk-Forward CV (5 folds, 3-month test windows):")
        for name, res in cv_results.items():
            print(f"  {name}: MAE = {res['mean_mae']:.2f} ± {res['std_mae']:.2f} €/MWh")

    n_plots  = 4 + (1 if not args.quick else 0)  # base + SHAP (2 files) + extras
    print()
    print(bar)
    print("Plots  → results/plots/")
    print("Metrics → results/metrics/leaderboard.csv")


if __name__ == "__main__":
    main()
