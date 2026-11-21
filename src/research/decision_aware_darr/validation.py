"""Development evaluation and dependence-aware uncertainty for Phase 2."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research.decision import BatterySpec, optimize_day, realized_value
from src.research.evaluation import diebold_mariano, moving_block_ci


def evaluate_forecasts(
    forecasts: dict[str, dict[str, np.ndarray]],
    query_features: pd.DataFrame,
    query_labels: pd.DataFrame,
    spec: BatterySpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queries = query_features.merge(query_labels, on="query__id", validate="one_to_one").set_index("query__id")
    daily_rows, hourly_rows = [], []
    for method, method_forecasts in forecasts.items():
        for query_id, prediction in method_forecasts.items():
            query = queries.loc[query_id]
            actual = np.asarray(query["label__actual_24"], dtype=float)
            prediction = np.asarray(prediction, dtype=float)
            error = prediction - actual
            charge, discharge, _ = optimize_day(prediction, spec)
            oracle_charge, oracle_discharge, _ = optimize_day(actual, spec)
            revenue = realized_value(actual, charge, discharge, spec)
            oracle = realized_value(actual, oracle_charge, oracle_discharge, spec)
            rho = spearmanr(actual, prediction).statistic
            daily_rows.append({
                "query__id": query_id, "delivery_date": query["query__delivery_date"], "method": method,
                "MAE": float(np.mean(np.abs(error))), "MSE": float(np.mean(np.square(error))),
                "bias": float(np.mean(error)), "rank_correlation": float(rho) if np.isfinite(rho) else 0.0,
                "lowest_hour_correct": int(np.argmin(prediction) == np.argmin(actual)),
                "highest_hour_correct": int(np.argmax(prediction) == np.argmax(actual)),
                "spread_error": float(np.ptp(prediction) - np.ptp(actual)),
                "revenue_eur": revenue, "oracle_eur": oracle, "regret_eur": oracle - revenue,
                "throughput_mwh": float(np.sum(charge + discharge)),
                "degradation_eur": float(np.sum(charge + discharge) * spec.degradation_cost_eur_mwh),
                "transaction_eur": float(np.sum(charge + discharge) * spec.transaction_cost_eur_mwh),
                "charge_overlap": float(np.minimum(charge, oracle_charge).sum() / max(oracle_charge.sum(), 1e-12)),
                "discharge_overlap": float(np.minimum(discharge, oracle_discharge).sum() / max(oracle_discharge.sum(), 1e-12)),
                "actual_negative_hours": int(np.sum(actual < 0)),
                "predicted_negative_true_positive": int(np.sum((actual < 0) & (prediction < 0))),
            })
            for hour, (a, p) in enumerate(zip(actual, prediction)):
                hourly_rows.append({
                    "query__id": query_id, "delivery_date": query["query__delivery_date"],
                    "hour": hour, "method": method, "actual": a, "prediction": p,
                })
    daily = pd.DataFrame(daily_rows)
    hourly = pd.DataFrame(hourly_rows)
    summary_rows = []
    for method, group in daily.groupby("method", sort=False):
        ordered = group.sort_values("delivery_date")
        cumulative = ordered.revenue_eur.cumsum()
        drawdown = cumulative - cumulative.cummax()
        positive = ordered.revenue_eur.clip(lower=0).sort_values(ascending=False)
        top_days = max(1, int(np.ceil(len(group) * 0.10)))
        summary_rows.append({
            "method": method, "MAE": group.MAE.mean(), "RMSE": float(np.sqrt(group.MSE.mean())),
            "bias": group.bias.mean(), "daily_rank_correlation": group.rank_correlation.mean(),
            "lowest_hour_accuracy": group.lowest_hour_correct.mean(),
            "highest_hour_accuracy": group.highest_hour_correct.mean(),
            "mean_spread_error": group.spread_error.mean(),
            "negative_price_recall": group.predicted_negative_true_positive.sum() / max(group.actual_negative_hours.sum(), 1),
            "battery_revenue_eur": group.revenue_eur.sum(), "revenue_per_day_eur": group.revenue_eur.mean(),
            "throughput_mwh": group.throughput_mwh.sum(),
            "revenue_per_throughput_eur_mwh": group.revenue_eur.sum() / max(group.throughput_mwh.sum(), 1e-12),
            "degradation_eur": group.degradation_eur.sum(), "transaction_eur": group.transaction_eur.sum(),
            "regret_eur": group.regret_eur.sum(), "charge_timing_overlap": group.charge_overlap.mean(),
            "discharge_timing_overlap": group.discharge_overlap.mean(),
            "losing_days": int((group.revenue_eur < 0).sum()), "worst_day_revenue_eur": group.revenue_eur.min(),
            "maximum_drawdown_eur": float(drawdown.min()),
            "top_10pct_positive_revenue_share": float(positive.head(top_days).sum() / max(positive.sum(), 1e-12)),
            "n_queries": group.query__id.nunique(),
        })
    return daily, pd.DataFrame(summary_rows).sort_values("MAE"), hourly


def paired_uncertainty(daily: pd.DataFrame, reference: str, seed: int, replications: int = 2000) -> pd.DataFrame:
    rows = []
    for method in sorted(set(daily.method) - {reference}):
        pair = daily[daily.method.isin([method, reference])].pivot(index="delivery_date", columns="method")
        for metric in ("MAE", "revenue_eur", "regret_eur"):
            differential = pair[metric][method].to_numpy() - pair[metric][reference].to_numpy()
            low, high = moving_block_ci(differential, block_days=7, replications=replications, seed=seed)
            dm, pvalue = (diebold_mariano(differential) if metric == "MAE" else (np.nan, np.nan))
            rows.append({
                "method": method, "reference": reference, "metric": metric,
                "mean_daily_difference": float(differential.mean()),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "DM_statistic": dm, "DM_pvalue": pvalue, "n_days": len(differential),
            })
    return pd.DataFrame(rows)


def seasonal_summary(daily: pd.DataFrame) -> pd.DataFrame:
    copy = daily.copy()
    dates = pd.to_datetime(copy.delivery_date)
    copy["season"] = dates.dt.month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"})
    copy["regime"] = np.where(copy.actual_negative_hours > 0, "negative_price_day", "ordinary")
    season = copy.groupby(["season", "method"], as_index=False).agg(
        MAE=("MAE", "mean"), revenue_eur=("revenue_eur", "sum"), n_queries=("query__id", "nunique")
    ).rename(columns={"season": "slice"})
    regime = copy.groupby(["regime", "method"], as_index=False).agg(
        MAE=("MAE", "mean"), revenue_eur=("revenue_eur", "sum"), n_queries=("query__id", "nunique")
    ).rename(columns={"regime": "slice"})
    return pd.concat([season, regime], ignore_index=True)
