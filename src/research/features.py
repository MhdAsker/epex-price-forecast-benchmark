"""Daily 24-product design matrix for the published EPEX-DE benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyDataset:
    dates: pd.DatetimeIndex
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]

    def subset(self, mask: np.ndarray) -> "DailyDataset":
        return DailyDataset(self.dates[mask], self.X[mask], self.y[mask], self.feature_names)


def to_daily_curves(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Convert a continuous, canonical 24-hour benchmark into daily curves."""
    if "price_eur_mwh" not in frame:
        raise KeyError("price_eur_mwh is required")
    ordered = frame.sort_index()
    groups = list(ordered["price_eur_mwh"].groupby(ordered.index.normalize()))
    bad = [(date, len(values)) for date, values in groups if len(values) != 24]
    if bad:
        raise ValueError(f"Daily-curve benchmark requires 24 products per day; first bad={bad[0]}")
    dates = pd.DatetimeIndex([date for date, _ in groups])
    curves = np.vstack([values.to_numpy(dtype=float) for _, values in groups])
    return dates, curves


def build_daily_dataset(
    frame: pd.DataFrame,
    price_lags: tuple[int, ...] = (1, 2, 7),
) -> DailyDataset:
    """Build one row per forecast day using only prior complete price curves."""
    dates, curves = to_daily_curves(frame)
    max_lag = max(price_lags)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    kept_dates: list[pd.Timestamp] = []
    names: list[str] = []
    for lag in price_lags:
        names.extend(f"price_d{lag}_h{hour:02d}" for hour in range(24))
    names += [
        "d1_mean", "d1_std", "d1_min", "d1_max", "d1_range",
        "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend",
    ]
    for i in range(max_lag, len(dates)):
        lag_curves = [curves[i - lag] for lag in price_lags]
        d1 = curves[i - 1]
        date = dates[i]
        dow = date.dayofweek
        doy = date.dayofyear
        calendar = np.array(
            [
                d1.mean(), d1.std(), d1.min(), d1.max(), np.ptp(d1),
                np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                float(dow >= 5),
            ],
            dtype=float,
        )
        rows.append(np.concatenate([*lag_curves, calendar]))
        targets.append(curves[i])
        kept_dates.append(date)
    return DailyDataset(
        dates=pd.DatetimeIndex(kept_dates),
        X=np.vstack(rows),
        y=np.vstack(targets),
        feature_names=tuple(names),
    )


def stack_hourly(X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Repeat daily predictors for 24 hours and append cyclical hour features."""
    n_days = len(X)
    hours = np.tile(np.arange(24), n_days)
    repeated = np.repeat(X, 24, axis=0)
    hour_features = np.column_stack(
        [hours, np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)]
    )
    target = None if y is None else np.asarray(y, dtype=float).reshape(-1)
    return np.column_stack([repeated, hour_features]), target
