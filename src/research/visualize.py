"""Paper-oriented Phase 2 diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "results" / "plots"


def plot_point_results(metrics: pd.DataFrame) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(metrics.sort_values("MAE")["model"]))
    years = sorted(metrics["test_year"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(years)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for offset, year in enumerate(years):
        values = metrics[metrics["test_year"] == year].set_index("model")["MAE"].reindex(models)
        ax.bar(x + (offset - (len(years) - 1) / 2) * width, values, width, label=str(year))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Rolling-origin point accuracy by complete test year", fontweight="bold")
    ax.legend(title="Test year")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = PLOTS / "phase2_point_mae_by_year.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_probabilistic_calibration(metrics: pd.DataFrame) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    nominal = np.array([0.5, 0.8, 0.9])
    for (method, year), group in metrics.groupby(["method", "test_year"]):
        row = group.iloc[0]
        empirical = np.array([row["coverage_50"], row["coverage_80"], row["coverage_90"]])
        ax.plot(nominal, empirical, marker="o", label=f"{method} — {year}")
    ax.plot([0.45, 0.95], [0.45, 0.95], linestyle="--", color="black", label="Ideal")
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Nominal interval coverage")
    ax.set_ylabel("Empirical interval coverage")
    ax.set_title("Online probabilistic calibration", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = PLOTS / "phase2_probabilistic_calibration.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dm_matrix(tests: pd.DataFrame, alpha: float = 0.05) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    models = sorted(set(tests["model_1"]) | set(tests["model_2"]))
    matrix = np.full((len(models), len(models)), np.nan)
    lookup = {model: idx for idx, model in enumerate(models)}
    np.fill_diagonal(matrix, 0.0)
    for row in tests.itertuples():
        i, j = lookup[row.model_1], lookup[row.model_2]
        if row.DM_pvalue < alpha:
            matrix[i, j] = row.mean_loss_difference
            matrix[j, i] = -row.mean_loss_difference
    bound = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(models)), models, rotation=45, ha="right")
    ax.set_yticks(range(len(models)), models)
    ax.set_title("Significant daily MAE differences (DM p < 0.05)", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Row model loss minus column model loss")
    fig.tight_layout()
    path = PLOTS / "phase2_dm_matrix.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
