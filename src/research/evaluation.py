"""Point/probabilistic scores and dependence-aware forecast comparisons."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def point_metrics(points: pd.DataFrame, spike_threshold: float = 100.0) -> pd.DataFrame:
    rows: list[dict] = []
    baseline = points[points["model"] == "Naive D-1"].copy()
    baseline["abs_error"] = np.abs(baseline["actual"] - baseline["prediction"])
    baseline_mae = baseline.groupby(["seed", baseline["delivery_date"].str[:4]])["abs_error"].mean()
    for (model, seed, year), group in points.groupby(["model", "seed", points["delivery_date"].str[:4]]):
        actual = group["actual"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        error = pred - actual
        negative = actual < 0
        spike = actual >= spike_threshold
        curve = group.sort_values(["delivery_date", "hour"])
        actual_curves = curve["actual"].to_numpy().reshape(-1, 24)
        pred_curves = curve["prediction"].to_numpy().reshape(-1, 24)
        ramp_mae = np.mean(np.abs(np.diff(actual_curves, axis=1) - np.diff(pred_curves, axis=1)))
        training_time = float(group.groupby("delivery_date")["fit_seconds"].first().sum()) if "fit_seconds" in group else np.nan
        mae = float(np.mean(np.abs(error)))
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "test_year": int(year),
                "MAE": mae,
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "rMAE_D1": mae / float(baseline_mae.loc[(seed, year)]),
                "MAE_negative": float(np.mean(np.abs(error[negative]))) if negative.any() else np.nan,
                "negative_recall": float(np.mean(pred[negative] < 0)) if negative.any() else np.nan,
                "MAE_spike": float(np.mean(np.abs(error[spike]))) if spike.any() else np.nan,
                "ramp_MAE": float(ramp_mae),
                "training_time_s": training_time,
                "n_hours": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_year", "MAE"]).reset_index(drop=True)


def _pinball(actual: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    error = actual - pred
    return np.maximum(q * error, (q - 1) * error)


def probabilistic_metrics(quantiles: pd.DataFrame) -> pd.DataFrame:
    quantiles = quantiles.copy()
    quantiles["test_year"] = pd.to_datetime(quantiles["delivery_date"]).dt.year
    rows: list[dict] = []
    group_keys = ["method", "test_year"]
    if "seed" in quantiles.columns:
        group_keys.append("seed")
    for key, group in quantiles.groupby(group_keys):
        if "seed" in quantiles.columns:
            method, test_year, seed = key
        else:
            method, test_year = key
            seed = None
        pivot = group.pivot_table(
            index=["delivery_date", "hour", "actual"], columns="quantile", values="prediction"
        ).reset_index()
        actual = pivot["actual"].to_numpy(dtype=float)
        qcols = sorted(c for c in pivot.columns if isinstance(c, float))
        losses = [_pinball(actual, pivot[q].to_numpy(dtype=float), q) for q in qcols]
        row = {
            "method": method,
            "test_year": int(test_year),
            "mean_pinball": float(np.mean(np.column_stack(losses))),
            "CRPS_approx": float(2 * np.mean(np.column_stack(losses))),
            "n_hours": len(pivot),
        }
        if seed is not None:
            row["seed"] = int(seed)
        for lower, upper in ((0.05, 0.95), (0.10, 0.90), (0.25, 0.75)):
            if lower not in pivot or upper not in pivot:
                continue
            lo, hi = pivot[lower].to_numpy(), pivot[upper].to_numpy()
            nominal = upper - lower
            covered = (actual >= lo) & (actual <= hi)
            width = hi - lo
            alpha = 1 - nominal
            interval_score = width + (2 / alpha) * (lo - actual) * (actual < lo) + (2 / alpha) * (actual - hi) * (actual > hi)
            label = int(round(nominal * 100))
            row[f"coverage_{label}"] = float(np.mean(covered))
            row[f"width_{label}"] = float(np.mean(width))
            row[f"interval_score_{label}"] = float(np.mean(interval_score))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("CRPS_approx").reset_index(drop=True)


def daily_absolute_losses(points: pd.DataFrame) -> pd.DataFrame:
    copy = points.copy()
    copy["loss"] = np.abs(copy["actual"] - copy["prediction"])
    return copy.groupby(["delivery_date", "model"], as_index=False)["loss"].mean()


def moving_block_ci(
    differential: np.ndarray,
    block_days: int = 7,
    replications: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    values = np.asarray(differential, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = np.arange(max(1, n - block_days + 1))
    means = np.empty(replications)
    blocks_needed = int(np.ceil(n / block_days))
    for i in range(replications):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[s : s + block_days] for s in chosen])[:n]
        means[i] = sample.mean()
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def diebold_mariano(differential: np.ndarray, hac_lags: int = 7) -> tuple[float, float]:
    """Two-sided DM test with a Bartlett/Newey-West long-run variance."""
    d = np.asarray(differential, dtype=float)
    centered = d - d.mean()
    n = len(d)
    gamma0 = float(centered @ centered / n)
    long_run = gamma0
    for lag in range(1, min(hac_lags, n - 1) + 1):
        gamma = float(centered[lag:] @ centered[:-lag] / n)
        long_run += 2 * (1 - lag / (hac_lags + 1)) * gamma
    statistic = d.mean() / np.sqrt(max(long_run, 1e-12) / n)
    pvalue = 2 * stats.norm.sf(abs(statistic))
    return float(statistic), float(pvalue)


def pairwise_tests(points: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    daily = daily_absolute_losses(points).pivot(index="delivery_date", columns="model", values="loss")
    rows: list[dict] = []
    for first, second in itertools.combinations(daily.columns, 2):
        aligned = daily[[first, second]].dropna()
        differential = aligned[first].to_numpy() - aligned[second].to_numpy()
        low, high = moving_block_ci(differential, seed=seed)
        dm, pvalue = diebold_mariano(differential)
        rows.append(
            {
                "model_1": first,
                "model_2": second,
                "mean_loss_difference": float(differential.mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "DM_statistic": dm,
                "DM_pvalue": pvalue,
                "n_days": len(aligned),
            }
        )
    return pd.DataFrame(rows)
