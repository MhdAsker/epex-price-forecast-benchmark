import csv
from pathlib import Path

import numpy as np
import pandas as pd

METRICS_COLS = ["model", "MAE", "RMSE", "MAPE", "training_time_s"]

# ── Pinball (Quantile) Loss ──────────────────────────────────────────────────

def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """
    Mean pinball (quantile) loss at a given quantile level.

    L_q(y, ŷ) = q * (y − ŷ)   if y ≥ ŷ
              = (1−q) * (ŷ − y) if y < ŷ

    Lower is better.  A well-calibrated 80% interval [q10, q90] has coverage ≈ 80%.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)))


def regime_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    high_threshold: float = 100.0,
) -> dict:
    """
    Point-forecast scores that MAE alone hides.

    Negative hours and hours at or above ``high_threshold`` €/MWh are scored
    separately. ``neg_recall`` is the share of truly negative hours whose
    prediction is also negative.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = np.abs(y_true - y_pred)
    negative = y_true < 0
    high = y_true >= high_threshold
    predicted_negative = y_pred < 0

    if negative.any():
        mae_negative = float(error[negative].mean())
        neg_recall = float(np.mean(predicted_negative[negative]))
    else:
        mae_negative = float("nan")
        neg_recall = float("nan")

    if predicted_negative.any():
        neg_precision = float(np.mean(negative[predicted_negative]))
    else:
        neg_precision = float("nan")

    mae_high = float(error[high].mean()) if high.any() else float("nan")
    return {
        "MAE": round(float(error.mean()), 4),
        "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
        "MAE_negative": round(mae_negative, 4),
        "MAE_high_100": round(mae_high, 4),
        "neg_recall": round(neg_recall, 4),
        "neg_precision": round(neg_precision, 4),
        "n_negative": int(negative.sum()),
        "n_high": int(high.sum()),
    }


def mae_by_hour(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray,
) -> pd.DataFrame:
    """MAE for each local hour 0–23. Hours with no rows are omitted."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    hours = np.asarray(hours)
    rows = []
    for hour in range(24):
        mask = hours == hour
        if not np.any(mask):
            continue
        rows.append({
            "local_hour": hour,
            "MAE": float(np.mean(np.abs(y_true[mask] - y_pred[mask]))),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)

METRICS_DIR = Path(__file__).resolve().parent.parent / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    training_time: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE: skip hours where |y_true| < 1 €/MWh.
    # Using != 0 is insufficient — near-zero negative prices (e.g. −0.01 €/MWh)
    # cause |y_true| → 0 and MAPE → ∞, which still distorts the metric.
    # Threshold of 1 €/MWh is standard in German/European energy forecasting literature.
    valid = np.abs(y_true) >= 1.0
    if valid.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs(y_true[valid] - y_pred[valid]) / np.abs(y_true[valid])) * 100
        )

    return {
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "training_time_s": round(training_time, 2),
    }


def save_metrics(
    metrics_list: list[dict],
    path: str = "results/metrics/leaderboard.csv",
) -> None:
    sorted_metrics = sorted(metrics_list, key=lambda x: x["MAE"])
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLS)
        writer.writeheader()
        writer.writerows(sorted_metrics)


def print_leaderboard(metrics_list: list[dict]) -> None:
    sorted_metrics = sorted(metrics_list, key=lambda x: x["MAE"])

    # Column widths
    W_MODEL = 14
    W_MAE   = 10
    W_RMSE  = 7
    W_MAPE  = 7
    W_TIME  = 12

    def row(model, mae, rmse, mape, time, sep="║"):
        return (
            f"{sep} {model:<{W_MODEL}}"
            f"{sep} {mae:<{W_MAE}}"
            f"{sep} {rmse:<{W_RMSE}}"
            f"{sep} {mape:<{W_MAPE}}"
            f"{sep} {time:<{W_TIME}}{sep}"
        )

    def hline(left, mid, right, fill="═"):
        segments = [
            fill * (W_MODEL + 2),
            fill * (W_MAE + 2),
            fill * (W_RMSE + 2),
            fill * (W_MAPE + 2),
            fill * (W_TIME + 2),
        ]
        return left + mid.join(segments) + right

    print(hline("╔", "╦", "╗"))
    print(row("Model", "MAE", "RMSE", "MAPE", "Train Time"))
    print(hline("╠", "╬", "╣"))

    for m in sorted_metrics:
        mae_str  = f"{m['MAE']:.2f}"
        rmse_str = f"{m['RMSE']:.2f}"
        mape_str = f"{m['MAPE']:.1f}%"
        time_str = f"{m['training_time_s']:.1f}s"
        print(row(m["model"], mae_str, rmse_str, mape_str, time_str))

    print(hline("╚", "╩", "╝"))

    # Winner line — compare against SARIMA baseline
    best = sorted_metrics[0]
    sarima = next((m for m in sorted_metrics if m["model"].upper() == "SARIMA"), None)

    if sarima and best["model"].upper() != "SARIMA" and sarima["MAE"] > 0:
        improvement = (sarima["MAE"] - best["MAE"]) / sarima["MAE"] * 100
        print(
            f"\nWinner: {best['model']} — {improvement:.1f}% lower MAE than SARIMA baseline"
        )
    else:
        print(f"\nWinner: {best['model']} — MAE {best['MAE']:.2f} €/MWh")
