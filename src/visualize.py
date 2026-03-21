from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "SARIMA": "#4C72B0",
    "LightGBM": "#DD8452",
    "XGBoost": "#55A868",
    "LSTM": "#C44E52",
}


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> None:
    """4-panel subplot, one per model. First 30 days of test set."""
    hours_30d = 30 * 24
    models = list(predictions_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("Actual vs Predicted — Day-Ahead Price (First 30 Days of Test Set)", fontsize=14, fontweight="bold")

    for ax, model_name in zip(axes.flat, models):
        preds = predictions_dict[model_name]
        # align lengths (LSTM predictions are shorter by SEQUENCE_LEN)
        n = min(len(y_test), len(preds), hours_30d)
        actual = y_test[:n]
        predicted = preds[:n]
        plot_dates = dates[:n]

        mae = mean_absolute_error(actual, predicted)

        ax.plot(plot_dates, actual, color="#2166ac", linewidth=0.8, label="Actual", alpha=0.9)
        ax.plot(plot_dates, predicted, color=MODEL_COLORS.get(model_name, "#d6604d"),
                linewidth=0.8, label="Predicted", alpha=0.85, linestyle="--")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_title(f"{model_name}   MAE = {mae:.2f} €/MWh", fontsize=11, fontweight="bold")
        ax.set_ylabel("Price (€/MWh)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    for ax in axes.flat[len(models):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "actual_vs_predicted.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_error_distribution(
    y_test: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
) -> None:
    """4 histograms of prediction errors (y_pred - y_true)."""
    models = list(predictions_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Prediction Error Distribution (y_pred − y_true)", fontsize=14, fontweight="bold")

    for ax, model_name in zip(axes.flat, models):
        preds = predictions_dict[model_name]
        n = min(len(y_test), len(preds))
        errors = preds[:n] - y_test[:n]
        mean_err = errors.mean()

        color = MODEL_COLORS.get(model_name, "#888888")
        ax.hist(errors, bins=60, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Zero error")
        ax.axvline(mean_err, color="red", linewidth=1.2, linestyle="-",
                   label=f"Mean = {mean_err:.2f}")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Error (€/MWh)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[len(models):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "error_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_feature_importance(
    lgbm_model,
    xgb_model,
    feature_names: list[str],
) -> None:
    """Side-by-side horizontal bar charts: LightGBM vs XGBoost top-15 features."""
    lgbm_imp = lgbm_model.get_feature_importance(feature_names).head(15)
    xgb_imp = xgb_model.get_feature_importance(feature_names).head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Feature Importance — Top 15 Features", fontsize=14, fontweight="bold")

    ax1.barh(lgbm_imp["feature"][::-1], lgbm_imp["importance"][::-1],
             color=MODEL_COLORS["LightGBM"], alpha=0.85)
    ax1.set_title("LightGBM", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Importance (split count)")
    ax1.grid(True, axis="x", alpha=0.3)

    ax2.barh(xgb_imp["feature"][::-1], xgb_imp["importance"][::-1],
             color=MODEL_COLORS["XGBoost"], alpha=0.85)
    ax2.set_title("XGBoost", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Importance (gain)")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_negative_price_analysis(
    df_test: pd.DataFrame,
    predictions_dict: dict[str, np.ndarray],
) -> None:
    """
    Scatter: actual price vs predicted price, coloured by hour of day.
    Negative actual price hours are highlighted with red markers on top.
    """
    actual = df_test["price_eur_mwh"].values
    hours = df_test.index.hour
    neg_mask = actual < 0

    models = list(predictions_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Actual vs Predicted — Coloured by Hour (Red = Negative Price Hours)",
                 fontsize=13, fontweight="bold")

    cmap = cm.get_cmap("twilight", 24)

    for ax, model_name in zip(axes.flat, models):
        preds = predictions_dict[model_name]
        n = min(len(actual), len(preds))
        act_n = actual[:n]
        pred_n = preds[:n]
        hrs_n = hours[:n]
        neg_n = neg_mask[:n]

        # All points coloured by hour
        sc = ax.scatter(act_n[~neg_n], pred_n[~neg_n],
                        c=hrs_n[~neg_n], cmap=cmap, vmin=0, vmax=23,
                        alpha=0.4, s=6, linewidths=0)

        # Negative price hours on top
        ax.scatter(act_n[neg_n], pred_n[neg_n],
                   color="red", alpha=0.85, s=20, linewidths=0,
                   zorder=5, label=f"Negative ({neg_n.sum()}h)")

        # Diagonal reference line
        lo = min(act_n.min(), pred_n.min())
        hi = max(act_n.max(), pred_n.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6, label="Perfect")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")

        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual (€/MWh)")
        ax.set_ylabel("Predicted (€/MWh)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

        plt.colorbar(sc, ax=ax, label="Hour of day")

    for ax in axes.flat[len(models):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "negative_price_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
