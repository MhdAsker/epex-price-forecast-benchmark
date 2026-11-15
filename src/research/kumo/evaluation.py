"""Electricity-market diagnostics for Kumo prediction panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def market_metrics(points: pd.DataFrame, spike_threshold: float = 100.0) -> pd.DataFrame:
    records = []
    group_columns = ["model"] + (["test_year"] if "test_year" in points else [])
    for keys, group in points.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        model = keys[0]
        year = keys[1] if len(keys) > 1 else None
        baseline = points[points.model == "Naive D-1"]
        if year is not None:
            baseline = baseline[baseline.test_year == year]
        baseline_mae = float(np.mean(np.abs(baseline.actual - baseline.prediction))) if len(baseline) else np.nan
        group = group.sort_values(["delivery_date", "hour"])
        actual = group.actual.to_numpy(); prediction = group.prediction.to_numpy()
        error = prediction - actual
        day_ranks = []
        low_hits = []; high_hits = []; spread_errors = []
        for _, day in group.groupby("delivery_date"):
            a = day.actual.to_numpy(); p = day.prediction.to_numpy()
            day_ranks.append(float(spearmanr(a, p).statistic))
            low_hits.append(int(np.argmin(a) == np.argmin(p)))
            high_hits.append(int(np.argmax(a) == np.argmax(p)))
            spread_errors.append(abs((p.max() - p.min()) - (a.max() - a.min())))
        actual_negative = actual < 0; predicted_negative = prediction < 0
        tp = int(np.sum(actual_negative & predicted_negative))
        fp = int(np.sum(~actual_negative & predicted_negative))
        fn = int(np.sum(actual_negative & ~predicted_negative))
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
        actual_spike = actual >= spike_threshold; predicted_spike = prediction >= spike_threshold
        row = {
            "model": model, "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error)),
            "rMAE_D1": float(np.mean(np.abs(error)) / baseline_mae) if np.isfinite(baseline_mae) else np.nan,
            "daily_rank_correlation": float(np.nanmean(day_ranks)),
            "lowest_hour_accuracy": float(np.mean(low_hits)),
            "highest_hour_accuracy": float(np.mean(high_hits)),
            "spread_MAE": float(np.mean(spread_errors)),
            "negative_precision": precision, "negative_recall": recall, "negative_f1": f1,
            "negative_events": int(actual_negative.sum()),
            "spike_recall": float(np.sum(actual_spike & predicted_spike) / actual_spike.sum()) if actual_spike.any() else np.nan,
            "spike_events": int(actual_spike.sum()),
            "mean_inference_seconds": float(group.inference_seconds_per_origin.mean()) if "inference_seconds_per_origin" in group else np.nan,
            "n_hours": len(group), "n_days": group.delivery_date.nunique(),
        }
        if year is not None:
            row["test_year"] = int(year)
        records.append(row)
    return pd.DataFrame(records).sort_values("MAE").reset_index(drop=True)


def add_persistence(frame: pd.DataFrame, kumo_points: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_index(); lookup = ordered["price_eur_mwh"]
    rows = []
    for date in sorted(kumo_points.delivery_date.unique()):
        delivery = pd.Timestamp(date); actual = lookup.loc[delivery:delivery + pd.Timedelta(hours=23)].to_numpy()
        d1 = lookup.loc[delivery - pd.Timedelta(days=1):delivery - pd.Timedelta(days=1) + pd.Timedelta(hours=23)].to_numpy()
        for hour, (a, p) in enumerate(zip(actual, d1)):
            rows.append({"delivery_date": date, "hour": hour, "actual": a, "prediction": p, "model": "Naive D-1", "seed": 13})
    return pd.concat([kumo_points, pd.DataFrame(rows)], ignore_index=True)


def regime_metrics(points: pd.DataFrame) -> pd.DataFrame:
    work = points.copy()
    daily_volatility = work.drop_duplicates(["delivery_date", "hour"]).groupby("delivery_date").actual.std()
    volatility_cutoff = float(daily_volatility.quantile(0.75))
    work["date"] = pd.to_datetime(work.delivery_date)
    labels = {
        "all": pd.Series(True, index=work.index),
        "negative_price_hours": work.actual < 0,
        "high_volatility_days_top_quartile": work.delivery_date.map(daily_volatility) >= volatility_cutoff,
        "weekend": work.date.dt.dayofweek >= 5,
        "weekday": work.date.dt.dayofweek < 5,
    }
    rows = []
    for regime, mask in labels.items():
        for model, group in work[mask].groupby("model"):
            rows.append({
                "regime": regime, "model": model,
                "MAE": float(np.mean(np.abs(group.actual - group.prediction))),
                "bias": float(np.mean(group.prediction - group.actual)),
                "n_hours": len(group), "n_days": group.delivery_date.nunique(),
                "volatility_cutoff_eur_mwh": volatility_cutoff,
            })
    return pd.DataFrame(rows)
