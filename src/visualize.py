"""
Visualization module for the EPEX Price Forecast Benchmark.

Plot catalogue
--------------
Existing (point-forecast diagnostics):
  plot_actual_vs_predicted()      — 4-panel time series, first 30 days
  plot_error_distribution()       — 4-panel error histogram
  plot_feature_importance()       — LightGBM (split) vs XGBoost (gain)
  plot_negative_price_analysis()  — scatter coloured by hour, neg hours highlighted

New (complexity upgrades):
  plot_shap_summary()             — SHAP beeswarm for LGBM + XGBoost  [Upgrade #1]
  plot_quantile_forecast()        — fan chart: 80% interval + median    [Upgrade #2]
  plot_walk_forward_cv()          — per-fold MAE bar chart               [Upgrade #3]
  plot_day_ahead_by_hour()        — MAE by auction hour                  [Upgrade #4]
  plot_day_ahead_regimes()        — overall / tail / negative-price hit  [Upgrade #4]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS: dict[str, str] = {
    "SARIMA":    "#4C72B0",
    "LightGBM":  "#DD8452",
    "XGBoost":   "#55A868",
    "LSTM":      "#C44E52",
    "Leaky LGBM":  "#8172B3",
    "DA Global":   "#DD8452",
    "DA Hourly":   "#55A868",
    "DA Spike":    "#C44E52",
    "DA Ensemble": "#4C72B0",
}

# ──────────────────────────────────────────────────────────────────────────────
# Existing diagnostic plots
# ──────────────────────────────────────────────────────────────────────────────


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> None:
    """4-panel subplot, one per model. First 30 days of test set.

    LSTM predictions are padded with NaN for the first SEQUENCE_LEN hours so
    all models share the same date axis (no silent window shift).
    """
    hours_30d = 30 * 24
    models = list(predictions_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(
        "Actual vs Predicted — Day-Ahead Price (First 30 Days of Test Set)",
        fontsize=14, fontweight="bold",
    )

    for ax, model_name in zip(axes.flat, models):
        preds = predictions_dict[model_name]
        n = min(len(y_test), len(preds), hours_30d)
        actual    = y_test[:n]
        predicted = preds[:n]
        plot_dates = dates[:n]

        # Compute MAE ignoring NaN padding (LSTM first 24 h)
        valid = ~np.isnan(predicted)
        mae = mean_absolute_error(actual[valid], predicted[valid]) if valid.any() else float("nan")

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
    print(f"  Saved {out.name}")


def plot_error_distribution(
    y_test: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
) -> None:
    """4 histograms of prediction errors (y_pred - y_true)."""
    models = list(predictions_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Prediction Error Distribution (y_pred − y_true)",
        fontsize=14, fontweight="bold",
    )

    for ax, model_name in zip(axes.flat, models):
        preds = predictions_dict[model_name]
        n = min(len(y_test), len(preds))
        # Skip NaN padding (LSTM)
        valid   = ~np.isnan(preds[:n])
        errors  = preds[:n][valid] - y_test[:n][valid]
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
    print(f"  Saved {out.name}")


def plot_feature_importance(
    lgbm_model,
    xgb_model,
    feature_names: list[str],
) -> None:
    """Side-by-side horizontal bar charts: LightGBM vs XGBoost top-15 features."""
    lgbm_imp = lgbm_model.get_feature_importance(feature_names).head(15)
    xgb_imp  = xgb_model.get_feature_importance(feature_names).head(15)

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
    print(f"  Saved {out.name}")


def plot_negative_price_analysis(
    df_test: pd.DataFrame,
    predictions_dict: dict[str, np.ndarray],
) -> None:
    """
    Scatter: actual price vs predicted price, coloured by hour of day.
    Negative actual price hours are highlighted with red markers on top.
    """
    actual   = df_test["price_eur_mwh"].values
    hours    = df_test.index.hour
    neg_mask = actual < 0

    models = list(predictions_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Actual vs Predicted — Coloured by Hour (Red = Negative Price Hours)",
        fontsize=13, fontweight="bold",
    )

    # Use the modern colormap accessor (matplotlib ≥ 3.7 deprecates cm.get_cmap)
    cmap = matplotlib.colormaps["twilight"].resampled(24)

    for ax, model_name in zip(axes.flat, models):
        preds = predictions_dict[model_name]
        n     = min(len(actual), len(preds))
        act_n  = actual[:n]
        pred_n = np.where(np.isnan(preds[:n]), act_n, preds[:n])  # replace NaN for scatter
        hrs_n  = hours[:n]
        neg_n  = neg_mask[:n]

        sc = ax.scatter(act_n[~neg_n], pred_n[~neg_n],
                        c=hrs_n[~neg_n], cmap=cmap, vmin=0, vmax=23,
                        alpha=0.4, s=6, linewidths=0)
        ax.scatter(act_n[neg_n], pred_n[neg_n],
                   color="red", alpha=0.85, s=20, linewidths=0,
                   zorder=5, label=f"Negative ({neg_n.sum()}h)")

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
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Upgrade #1 — SHAP beeswarm summary
# ──────────────────────────────────────────────────────────────────────────────


def plot_shap_summary(
    lgbm_model,
    xgb_model,
    X_test: np.ndarray,
    feature_names: list[str],
    max_display: int = 15,
    sample_size: int = 2000,
) -> None:
    """
    SHAP beeswarm summary plots for LightGBM and XGBoost.

    Unlike split-count or gain importance, SHAP values measure each feature's
    actual marginal contribution to individual predictions, respecting feature
    interactions and direction of effect.

    Saves results/plots/shap_lgbm.png  and  results/plots/shap_xgb.png.

    Parameters
    ----------
    sample_size : Number of test rows used for SHAP (subsample for speed).
                  Full test set (~8 k rows) is fast for tree models; 2000 is
                  used as a conservative default.
    """
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap not installed — skipping. Run: pip install shap")
        return

    rng = np.random.default_rng(42)
    n   = min(sample_size, len(X_test))
    idx = np.sort(rng.choice(len(X_test), size=n, replace=False))
    X_sample = X_test[idx]

    for model_obj, model_name, fname in [
        (lgbm_model.model, "LightGBM", "shap_lgbm.png"),
        (xgb_model.model,  "XGBoost",  "shap_xgb.png"),
    ]:
        explainer  = shap.TreeExplainer(model_obj)
        shap_vals  = explainer.shap_values(X_sample)

        # summary_plot creates its own figure; we capture and save it.
        shap.summary_plot(
            shap_vals,
            X_sample,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        fig = plt.gcf()
        fig.suptitle(
            f"SHAP Feature Importance — {model_name}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        out = PLOTS_DIR / fname
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Upgrade #2 — Probabilistic forecast fan chart
# ──────────────────────────────────────────────────────────────────────────────


def plot_quantile_forecast(
    y_test: np.ndarray,
    quantile_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dates: pd.DatetimeIndex,
    window_days: int = 14,
) -> None:
    """
    Fan chart: 80% prediction interval [q10, q90] + median (q50) + actual.

    Parameters
    ----------
    quantile_preds : {"ModelName": (q10_array, q50_array, q90_array), ...}
    window_days    : How many days of the test set to display (default 14).
    """
    n_hours = window_days * 24
    models  = list(quantile_preds.keys())

    fig, axes = plt.subplots(len(models), 1, figsize=(16, 5 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    fig.suptitle(
        f"Probabilistic Forecast — 80% Prediction Interval (First {window_days} Days)",
        fontsize=14, fontweight="bold",
    )

    for ax, model_name in zip(axes, models):
        q10, q50, q90 = quantile_preds[model_name]
        n_plot    = min(n_hours, len(y_test), len(q10))
        plot_dates = dates[:n_plot]
        actual_n   = y_test[:n_plot]
        q10_n      = q10[:n_plot]
        q50_n      = q50[:n_plot]
        q90_n      = q90[:n_plot]

        # Calibration metrics for this window
        coverage = float(np.mean((actual_n >= q10_n) & (actual_n <= q90_n)) * 100)
        interval_width = float(np.mean(q90_n - q10_n))

        color = MODEL_COLORS.get(model_name, "#888888")

        ax.fill_between(
            plot_dates, q10_n, q90_n,
            alpha=0.22, color=color,
            label=f"80% interval  [q10–q90]  (width {interval_width:.1f} €/MWh avg)",
        )
        ax.plot(plot_dates, q10_n, color=color, linewidth=0.5, linestyle=":", alpha=0.6)
        ax.plot(plot_dates, q90_n, color=color, linewidth=0.5, linestyle=":", alpha=0.6)
        ax.plot(plot_dates, q50_n, color=color, linewidth=1.1,
                linestyle="--", label="Median forecast (q50)", alpha=0.9)
        ax.plot(plot_dates, actual_n, color="#2166ac", linewidth=0.85,
                label="Actual", alpha=0.95)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

        coverage_colour = "green" if 75 <= coverage <= 85 else "darkorange"
        ax.set_title(
            f"{model_name}  —  Coverage: "
            f"$\\bf{{{coverage:.1f}\\%}}$  (target 80%)",
            fontsize=11, fontweight="bold", color=coverage_colour,
        )
        ax.set_ylabel("Price (€/MWh)")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = PLOTS_DIR / "quantile_forecast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Upgrade #3 — Walk-forward CV results chart
# ──────────────────────────────────────────────────────────────────────────────


def plot_walk_forward_cv(cv_results: dict[str, dict]) -> None:
    """
    Two-panel figure:
      Top  — per-fold MAE bar chart with dashed mean line.
      Bottom — per-fold RMSE bar chart with dashed mean line.

    Parameters
    ----------
    cv_results : Output of ``src.cv.walk_forward_cv()``.
    """
    model_names = list(cv_results.keys())
    n_folds = max(len(cv_results[m]["fold_maes"]) for m in model_names)
    x = np.arange(1, n_folds + 1)
    bar_width = 0.35

    fig, (ax_mae, ax_rmse) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "Walk-Forward Expanding-Window Cross-Validation",
        fontsize=14, fontweight="bold",
    )

    for metric, ax, ylabel in [
        ("fold_maes",  ax_mae,  "MAE (€/MWh)"),
        ("fold_rmses", ax_rmse, "RMSE (€/MWh)"),
    ]:
        mean_key = "mean_mae" if metric == "fold_maes" else "mean_rmse"
        std_key  = "std_mae"  if metric == "fold_maes" else "std_rmse"

        for i, model_name in enumerate(model_names):
            vals     = cv_results[model_name][metric]
            mean_val = cv_results[model_name][mean_key]
            std_val  = cv_results[model_name][std_key]
            color    = MODEL_COLORS.get(model_name, f"C{i}")
            offset   = (i - len(model_names) / 2 + 0.5) * bar_width

            ax.bar(
                x[:len(vals)] + offset, vals, bar_width,
                label=f"{model_name}  (μ = {mean_val:.2f} ± {std_val:.2f})",
                color=color, alpha=0.80, edgecolor="white", linewidth=0.5,
            )
            ax.axhline(mean_val, color=color, linewidth=1.6, linestyle="--", alpha=0.75)

        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    ax_rmse.set_xlabel("Fold (expanding training window)")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels([f"Fold {i}" for i in x])

    plt.tight_layout()
    out = PLOTS_DIR / "walk_forward_cv.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Upgrade #4 — Day-ahead auction protocol
# ──────────────────────────────────────────────────────────────────────────────


def plot_day_ahead_by_hour(by_hour: pd.DataFrame) -> None:
    """Line chart of test MAE against the Europe/Berlin delivery hour."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for model_name, group in by_hour.groupby("model", sort=False):
        group = group.sort_values("local_hour")
        ax.plot(
            group["local_hour"],
            group["MAE"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=model_name,
            color=MODEL_COLORS.get(str(model_name)),
        )
    ax.set_xticks(range(24))
    ax.set_xlabel("Delivery hour (Europe/Berlin)")
    ax.set_ylabel("MAE (€/MWh)")
    ax.set_title(
        "Day-ahead protocol — test MAE by delivery hour",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout()
    out = PLOTS_DIR / "day_ahead_by_hour.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_day_ahead_regimes(leaderboard: pd.DataFrame) -> None:
    """Overall MAE, tail MAE, and negative-price recall for each protocol."""
    if "rankable" in leaderboard.columns:
        leaderboard = leaderboard[leaderboard["rankable"]].copy()
    models = leaderboard["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, (ax_err, ax_hit) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle(
        "Day-ahead protocol — error by price regime",
        fontsize=13,
        fontweight="bold",
    )

    series = [
        ("MAE", "All hours", "#4C72B0"),
        ("MAE_negative", "Negative hours", "#C44E52"),
        ("MAE_high_100", "Hours ≥ 100 €/MWh", "#DD8452"),
    ]
    for offset, (column, label, color) in enumerate(series):
        ax_err.bar(
            x + (offset - 1) * width,
            leaderboard[column].to_numpy(),
            width,
            label=label,
            color=color,
            alpha=0.9,
        )
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(models, rotation=15)
    ax_err.set_ylabel("MAE (€/MWh)")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, axis="y", alpha=0.3)

    recall = leaderboard["neg_recall"].to_numpy() * 100
    colors = [MODEL_COLORS.get(name, "#888888") for name in models]
    ax_hit.bar(x, recall, color=colors, alpha=0.9)
    ax_hit.set_xticks(x)
    ax_hit.set_xticklabels(models, rotation=15)
    ax_hit.set_ylabel("Negative-hour recall (%)")
    ax_hit.set_ylim(0, 100)
    ax_hit.grid(True, axis="y", alpha=0.3)
    for pos, value in zip(x, recall):
        ax_hit.text(pos, value + 1.5, f"{value:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = PLOTS_DIR / "day_ahead_regimes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
