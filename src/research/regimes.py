"""Ex-post regime labels for honest conditional error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _holiday_dates(year: int) -> set[pd.Timestamp]:
    # Fixed-date holidays common to the benchmark zones. Movable holidays are
    # intentionally omitted rather than guessed across national calendars.
    return {pd.Timestamp(year, month, day) for month, day in ((1, 1), (5, 1), (12, 25), (12, 26))}


def regime_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    dates = pd.to_datetime(work["delivery_date"])
    absolute = work["actual"].abs()
    spike_cutoff = float(absolute.quantile(0.95))
    holidays = set().union(*(_holiday_dates(int(y)) for y in dates.dt.year.unique()))
    labels = {
        "all": np.ones(len(work), dtype=bool),
        "negative_price": work["actual"].to_numpy() < 0,
        "price_spike_top_5pct": absolute.to_numpy() >= spike_cutoff,
        "peak_hours_08_20": work["hour"].between(8, 20).to_numpy(),
        "weekend": (dates.dt.dayofweek >= 5).to_numpy(),
        "fixed_date_holiday": dates.dt.normalize().isin(holidays).to_numpy(),
    }
    group_cols = (["zone"] if "zone" in work else []) + ["test_year", "model"]
    records = []
    for regime, mask in labels.items():
        subset = work.loc[mask].copy()
        subset["absolute_error"] = (subset["actual"] - subset["prediction"]).abs()
        for keys, group in subset.groupby(group_cols, sort=False):
            values = keys if isinstance(keys, tuple) else (keys,)
            records.append({
                **dict(zip(group_cols, values)), "regime": regime,
                "MAE": float(group["absolute_error"].mean()), "n_hours": len(group),
            })
    return pd.DataFrame(records)
