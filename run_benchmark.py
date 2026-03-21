"""
EPEX Price Forecast Benchmark -- main entry point.
Trains SARIMA, LightGBM, XGBoost, LSTM on DE-LU day-ahead prices
and produces a leaderboard + diagnostic plots.
"""

import io
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows terminals (cp1252 can't encode box-drawing chars)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- project imports --------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluate import compute_metrics, print_leaderboard, save_metrics
from src.models.lgbm_model import LGBMModel
from src.models.lstm_model import LSTMModel
from src.models.sarima_model import SARIMAModel
from src.models.xgb_model import XGBModel
from src.visualize import (
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_feature_importance,
    plot_negative_price_analysis,
)

# ---------------------------------------------------------------------------

SPLIT_DATE = "2024-01-01"
TARGET_COL = "price_eur_mwh"
EXCLUDE_COLS = {"price_eur_mwh", "negative_price_flag"}


def main() -> None:
    load_dotenv()

    # -----------------------------------------------------------------------
    # 1. Load features
    # -----------------------------------------------------------------------
    features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        print("ERROR: data/processed/features.parquet not found.")
        print("Run:  python src/features.py  (after fetching raw data)")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    print(f"Loaded features: {df.shape[0]} rows × {df.shape[1]} columns")

    # -----------------------------------------------------------------------
    # 2. Train / val / test split
    # -----------------------------------------------------------------------
    FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

    train_full = df[df.index < SPLIT_DATE]
    test = df[df.index >= SPLIT_DATE]

    # val = last 10% of train (time-ordered, no shuffle)
    val_start = int(len(train_full) * 0.90)
    train = train_full.iloc[:val_start]
    val = train_full.iloc[val_start:]

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_val = val[FEATURE_COLS].values
    y_val = val[TARGET_COL].values
    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    n_features = len(FEATURE_COLS)
    n_train = len(train_full)
    n_test = len(test)

    test_dates = test.index

    print(
        f"Train: {len(train)} rows | Val: {len(val)} rows | Test: {n_test} rows"
    )

    metrics_list: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    # -----------------------------------------------------------------------
    # 3a. SARIMA  (price series only)
    # -----------------------------------------------------------------------
    print("\n─── SARIMA ───────────────────────────────")
    sarima = SARIMAModel()
    t0 = time.time()
    sarima_preds = sarima.fit_predict(
        train_full[TARGET_COL],   # fit on full train (no val split needed)
        test[TARGET_COL],
    )
    sarima_time = time.time() - t0

    predictions["SARIMA"] = sarima_preds
    metrics_list.append(
        compute_metrics(y_test, sarima_preds, "SARIMA", sarima_time)
    )

    # -----------------------------------------------------------------------
    # 3b. LightGBM
    # -----------------------------------------------------------------------
    print("\n─── LightGBM ─────────────────────────────")
    lgbm = LGBMModel()
    t0 = time.time()
    lgbm.fit(X_train, y_train, X_val, y_val)
    lgbm_time = time.time() - t0

    lgbm_preds = lgbm.predict(X_test)
    predictions["LightGBM"] = lgbm_preds
    metrics_list.append(
        compute_metrics(y_test, lgbm_preds, "LightGBM", lgbm_time)
    )

    # -----------------------------------------------------------------------
    # 3c. XGBoost
    # -----------------------------------------------------------------------
    print("\n─── XGBoost ──────────────────────────────")
    xgb_model = XGBModel()
    t0 = time.time()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_time = time.time() - t0

    xgb_preds = xgb_model.predict(X_test)
    predictions["XGBoost"] = xgb_preds
    metrics_list.append(
        compute_metrics(y_test, xgb_preds, "XGBoost", xgb_time)
    )

    # -----------------------------------------------------------------------
    # 3d. LSTM
    # -----------------------------------------------------------------------
    print("\n─── LSTM ─────────────────────────────────")
    lstm = LSTMModel(input_dim=n_features, hidden_dim=64)
    t0 = time.time()
    lstm.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.001)
    lstm_time = time.time() - t0

    lstm_preds = lstm.predict(X_test)
    # LSTM predictions are shorter by SEQUENCE_LEN — align y_test accordingly
    lstm_offset = len(y_test) - len(lstm_preds)
    y_test_lstm = y_test[lstm_offset:]
    predictions["LSTM"] = lstm_preds
    metrics_list.append(
        compute_metrics(y_test_lstm, lstm_preds, "LSTM", lstm_time)
    )

    # -----------------------------------------------------------------------
    # 4. Leaderboard
    # -----------------------------------------------------------------------
    print("\n")
    print_leaderboard(metrics_list)

    # -----------------------------------------------------------------------
    # 5. Save CSV
    # -----------------------------------------------------------------------
    save_metrics(metrics_list)

    # -----------------------------------------------------------------------
    # 6. Plots
    # -----------------------------------------------------------------------
    print("\nGenerating plots...")

    # For plots that align all models we use y_test; LSTM is internally shorter
    # but plot functions handle length mismatches via min(len(y), len(pred))
    plot_actual_vs_predicted(y_test, predictions, test_dates)
    plot_error_distribution(y_test, predictions)
    plot_feature_importance(lgbm, xgb_model, FEATURE_COLS)
    plot_negative_price_analysis(test, predictions)

    # -----------------------------------------------------------------------
    # 7. Final summary
    # -----------------------------------------------------------------------
    sorted_metrics = sorted(metrics_list, key=lambda x: x["MAE"])
    best = sorted_metrics[0]
    sarima_m = next(m for m in metrics_list if m["model"] == "SARIMA")
    improvement = (sarima_m["MAE"] - best["MAE"]) / sarima_m["MAE"] * 100

    # Top LGBM feature by importance
    lgbm_imp = lgbm.get_feature_importance(FEATURE_COLS)
    top_feature = lgbm_imp.iloc[0]["feature"] if not lgbm_imp.empty else "N/A"

    neg_hours = int((test[TARGET_COL] < 0).sum())
    neg_pct = 100 * neg_hours / n_test

    bar = "═" * 48
    print(f"\n{bar}")
    print("EPEX PRICE FORECAST BENCHMARK — RESULTS")
    print(bar)
    print(f"Dataset    : DE-LU bidding zone, 2022–2023 train")
    print(f"Test period: {test.index.min().date()} to {test.index.max().date()}")
    print(f"Features   : {n_features} engineered features")
    print(f"Train rows : {n_train}  |  Test rows: {n_test}")
    print()
    print(f"{'Model':<14} {'MAE':>7} {'RMSE':>7} {'MAPE':>8} {'Time':>8}")
    print("─" * 46)
    for m in sorted_metrics:
        winner_tag = "  ← WINNER" if m["model"] == best["model"] else ""
        print(
            f"{m['model']:<14} {m['MAE']:>7.2f} {m['RMSE']:>7.2f}"
            f" {m['MAPE']:>7.1f}% {m['training_time_s']:>7.1f}s{winner_tag}"
        )
    print()
    print(f"{best['model']} beat SARIMA baseline by {improvement:.1f}% on MAE")
    print(f"Top feature: {top_feature} (importance rank #1)")
    print(f"Negative price hours in test: {neg_hours} ({neg_pct:.1f}%)")
    print(bar)
    print("Plots saved to results/plots/")
    print("Metrics saved to results/metrics/leaderboard.csv")


if __name__ == "__main__":
    main()
