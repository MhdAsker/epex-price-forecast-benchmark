import csv
from pathlib import Path

import numpy as np
import pandas as pd

METRICS_COLS = ["model", "MAE", "RMSE", "MAPE", "training_time_s"]

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

    # MAPE: skip hours where y_true == 0 to avoid division by zero
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(y_true[nonzero] - y_pred[nonzero]) / np.abs(y_true[nonzero])) * 100)

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
