"""Economic diagnostics beyond aggregate battery revenue."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.research.decision import BatterySpec, optimize_day, realized_value
from src.research.evaluation import moving_block_ci


def dispatch_diagnostics(points: pd.DataFrame, spec: BatterySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for date, day in points.groupby("delivery_date", sort=True):
        actual_day = day.drop_duplicates("hour").sort_values("hour")
        actual = actual_day.actual.to_numpy(dtype=float)
        oracle_charge, oracle_discharge, _ = optimize_day(actual, spec)
        oracle_value = realized_value(actual, oracle_charge, oracle_discharge, spec)
        actual_spread = float(actual.max() - actual.min())
        for model, model_day in day.groupby("model"):
            prediction = model_day.sort_values("hour").prediction.to_numpy(dtype=float)
            charge, discharge, _ = optimize_day(prediction, spec)
            revenue = realized_value(actual, charge, discharge, spec)
            throughput = float(np.sum(charge + discharge))
            charge_den = max(float(np.sum(oracle_charge)), 1e-12)
            discharge_den = max(float(np.sum(oracle_discharge)), 1e-12)
            rows.append({
                "delivery_date": date, "model": model, "revenue_eur": revenue,
                "oracle_eur": oracle_value, "regret_eur": oracle_value - revenue,
                "throughput_mwh": throughput,
                "degradation_cost_eur": float(spec.degradation_cost_eur_mwh * throughput),
                "revenue_per_throughput_eur_mwh": revenue / throughput if throughput else np.nan,
                "charge_timing_overlap": float(np.minimum(charge, oracle_charge).sum() / charge_den),
                "discharge_timing_overlap": float(np.minimum(discharge, oracle_discharge).sum() / discharge_den),
                "actual_spread_eur_mwh": actual_spread,
                "predicted_spread_eur_mwh": float(prediction.max() - prediction.min()),
                "spread_error_eur_mwh": float((prediction.max() - prediction.min()) - actual_spread),
                "actual_low_hour": int(np.argmin(actual)), "predicted_low_hour": int(np.argmin(prediction)),
                "actual_high_hour": int(np.argmax(actual)), "predicted_high_hour": int(np.argmax(prediction)),
            })
    daily = pd.DataFrame(rows).sort_values(["model", "delivery_date"])
    return daily, summarize_dispatch_daily(daily)


def summarize_dispatch_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize already optimized daily schedules without solving them again."""
    summaries = []
    for model, group in daily.groupby("model"):
        cumulative = group.revenue_eur.cumsum().to_numpy()
        drawdown = np.maximum.accumulate(np.r_[0.0, cumulative]) - np.r_[0.0, cumulative]
        total_throughput = float(group.throughput_mwh.sum())
        summaries.append({
            "model": model, "revenue_eur": float(group.revenue_eur.sum()),
            "revenue_per_day_eur": float(group.revenue_eur.mean()),
            "throughput_mwh": total_throughput,
            "revenue_per_throughput_eur_mwh": float(group.revenue_eur.sum() / total_throughput),
            "degradation_cost_eur": float(group.degradation_cost_eur.sum()),
            "regret_eur": float(group.regret_eur.sum()),
            "max_drawdown_eur": float(drawdown.max()),
            "charge_timing_overlap": float(group.charge_timing_overlap.mean()),
            "discharge_timing_overlap": float(group.discharge_timing_overlap.mean()),
            "low_hour_accuracy": float((group.actual_low_hour == group.predicted_low_hour).mean()),
            "high_hour_accuracy": float((group.actual_high_hour == group.predicted_high_hour).mean()),
            "mean_spread_error_eur_mwh": float(group.spread_error_eur_mwh.mean()),
            "n_days": int(len(group)),
        })
    return pd.DataFrame(summaries).sort_values("revenue_eur", ascending=False)


def pairwise_revenue_intervals(daily: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
    pivot = daily.pivot(index="delivery_date", columns="model", values="revenue_eur")
    rows = []
    for first, second in itertools.combinations(pivot.columns, 2):
        aligned = pivot[[first, second]].dropna()
        difference = aligned[first].to_numpy() - aligned[second].to_numpy()
        low, high = moving_block_ci(difference, block_days=7, replications=2000, seed=seed)
        rows.append({
            "model_1": first, "model_2": second,
            "mean_daily_revenue_difference_eur": float(difference.mean()),
            "bootstrap_ci_low": low, "bootstrap_ci_high": high, "n_days": len(aligned),
        })
    return pd.DataFrame(rows)
