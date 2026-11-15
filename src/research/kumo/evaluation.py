"""Electricity-market diagnostics for Kumo prediction panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter
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
    holiday_dates: set[pd.Timestamp] = set()
    for year in work.date.dt.year.unique():
        easter_sunday = pd.Timestamp(easter(int(year)))
        holiday_dates.update({
            pd.Timestamp(year=int(year), month=1, day=1),
            easter_sunday - pd.Timedelta(days=2), easter_sunday + pd.Timedelta(days=1),
            pd.Timestamp(year=int(year), month=5, day=1),
            easter_sunday + pd.Timedelta(days=39), easter_sunday + pd.Timedelta(days=50),
            pd.Timestamp(year=int(year), month=10, day=3),
            pd.Timestamp(year=int(year), month=12, day=25),
            pd.Timestamp(year=int(year), month=12, day=26),
        })
    month = work.date.dt.month
    labels = {
        "all": pd.Series(True, index=work.index),
        "negative_price_hours": work.actual < 0,
        "high_volatility_days_top_quartile": work.delivery_date.map(daily_volatility) >= volatility_cutoff,
        "price_spike_hours_ge_100": work.actual >= 100,
        "german_public_holiday": work.date.dt.normalize().isin(holiday_dates),
        "weekend": work.date.dt.dayofweek >= 5,
        "weekday": work.date.dt.dayofweek < 5,
        "winter_DJF": month.isin([12, 1, 2]),
        "spring_MAM": month.isin([3, 4, 5]),
        "summer_JJA": month.isin([6, 7, 8]),
        "autumn_SON": month.isin([9, 10, 11]),
    }
    rows = []
    group_columns = ["model"] + (["test_year"] if "test_year" in work else [])
    for regime, mask in labels.items():
        for keys, group in work[mask].groupby(group_columns):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = {
                "regime": regime, "model": keys[0],
                "MAE": float(np.mean(np.abs(group.actual - group.prediction))),
                "bias": float(np.mean(group.prediction - group.actual)),
                "n_hours": len(group), "n_days": group.delivery_date.nunique(),
                "volatility_cutoff_eur_mwh": volatility_cutoff,
            }
            if len(keys) > 1:
                row["test_year"] = int(keys[1])
            rows.append(row)
    return pd.DataFrame(rows)
