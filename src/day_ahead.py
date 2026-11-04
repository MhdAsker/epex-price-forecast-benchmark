"""
Day-ahead auction information set for DE-LU prices.

EPEX SPOT clears the German day-ahead auction at 12:00 Europe/Berlin on day
D-1. The bid covers every hour of delivery day D. At gate closure the prices
of day D are unknown, including the earlier hours of that same day.

The original feature matrix shifts rolling windows by one hour. That blocks
the target hour itself, and it still feeds hours 1–23 of day D their own
earlier auction prices through ``price_lag_1h``. Those prices are published
with the auction result, so a model that sees them is not a day-ahead forecast.

This module rebuilds the matrix from what is actually known at noon on D-1:

* Prices from local days D-1, D-2, D-7 and D-14 only. The full 24-hour price
  curve of day D-1 is known, so it is used as a profile, not as a 1-hour lag.
ENTSO-E renewable and scheduled-generation forecasts are *not* primary inputs.
EU 543/2013 permits publication at 18:00 D-1, after the noon forecast origin,
and the API extract does not preserve an auditable publication vintage. They
can be included only in an explicitly labelled post-gate oracle sensitivity.

The clock is Europe/Berlin. The stored index is timezone-naive UTC.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Price-derived columns. The leakage check scrambles same-day prices and
# requires every one of these to stay put on the scrambled day.
PRICE_FEATURE_COLS: list[str] = [
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_lag_336h",
    "price_shape_24h",
    "yest_mean",
    "yest_std",
    "yest_min",
    "yest_max",
    "yest_neg_hours",
    "yest_ramp",
    "week_ago_mean",
    "yest_h02_fold1",
] + [f"yest_h{hour:02d}" for hour in range(24)]

EXCLUDE_FROM_MODEL = {
    "price_eur_mwh",
    "negative_price_flag",
    "local_date",
    "local_hour",
}

# Exogenous forecast columns copied from the hourly frame when present.
# These are day-ahead forecasts, so the delivery hour itself is allowed.
FORECAST_COLS = (
    "wind_onshore_mw",
    "wind_offshore_mw",
    "wind_total_mw",
    "solar_mw",
    "gen_forecast_mw",
)


def _berlin_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Treat a naive index as UTC and convert it to Europe/Berlin."""
    if index.tz is None:
        utc = index.tz_localize("UTC")
    else:
        utc = index.tz_convert("UTC")
    return utc.tz_convert("Europe/Berlin")


def _market_clock(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Return local time, repeated-hour fold, and 1-based product position."""
    local = _berlin_clock(index)
    wall = pd.Series(local.tz_localize(None))
    fold = wall.groupby(wall, sort=False).cumcount().to_numpy(dtype=int)
    dates = pd.Series([ts.date() for ts in local])
    position = dates.groupby(dates, sort=False).cumcount().to_numpy(dtype=int) + 1
    return local, fold, position


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric model inputs. Drops the target, flags, and the local clock."""
    cols: list[str] = []
    for col in frame.columns:
        if col in EXCLUDE_FROM_MODEL:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        cols.append(col)
    return cols


def build_day_ahead_features(
    df: pd.DataFrame,
    *,
    include_oracle_forecasts: bool = False,
) -> pd.DataFrame:
    """
    Build one row per delivery hour with an auction-safe feature set.

    Parameters
    ----------
    df : Hourly frame indexed by timezone-naive UTC timestamps. Must contain
         ``price_eur_mwh``. Renewable forecast columns are included when present.

    Returns
    -------
    DataFrame indexed like the surviving rows of ``df``. Rows that do not have
    a full 14-day price history are dropped.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("build_day_ahead_features expects a DatetimeIndex")
    if "price_eur_mwh" not in df.columns:
        raise KeyError("price_eur_mwh is required")

    local, folds, product_positions = _market_clock(df.index)
    price = df["price_eur_mwh"].to_numpy(dtype=float)
    dates = np.array([ts.date() for ts in local])
    hours = np.asarray(local.hour, dtype=int)
    n = len(df)

    # Keep the two autumn 02:00 market products separate.
    price_at: dict[tuple[dt.date, int, int], float] = {}
    daily_values: dict[dt.date, list[float]] = {}
    for day, hour, fold, value in zip(dates, hours, folds, price):
        price_at[(day, int(hour), int(fold))] = float(value)
        daily_values.setdefault(day, []).append(float(value))

    def previous_price(day: dt.date, hour: int, fold: int) -> float:
        # A fold-1 product normally has no counterpart on the previous day.
        return price_at.get((day, hour, fold), price_at.get((day, hour, 0), np.nan))

    daily_stats: dict[dt.date, dict[str, float]] = {}
    for day, vals in daily_values.items():
        arr = np.asarray(vals, dtype=float)
        daily_stats[day] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "neg": float(np.sum(arr < 0)),
            "ramp": float(arr.max() - arr.min()),
        }

    lag_24 = np.empty(n)
    lag_48 = np.empty(n)
    lag_168 = np.empty(n)
    lag_336 = np.empty(n)
    yesterday = np.empty((n, 24))
    yest_mean = np.empty(n)
    yest_std = np.empty(n)
    yest_min = np.empty(n)
    yest_max = np.empty(n)
    yest_neg = np.empty(n)
    yest_ramp = np.empty(n)
    week_mean = np.empty(n)
    yest_h02_fold1 = np.empty(n)

    for i, (day, hour, fold) in enumerate(zip(dates, hours, folds)):
        day_1 = day - dt.timedelta(days=1)
        day_2 = day - dt.timedelta(days=2)
        day_7 = day - dt.timedelta(days=7)
        day_14 = day - dt.timedelta(days=14)
        lag_24[i] = previous_price(day_1, int(hour), int(fold))
        lag_48[i] = previous_price(day_2, int(hour), int(fold))
        lag_168[i] = previous_price(day_7, int(hour), int(fold))
        lag_336[i] = previous_price(day_14, int(hour), int(fold))
        for clock_hour in range(24):
            yesterday[i, clock_hour] = price_at.get((day_1, clock_hour, 0), np.nan)
        yest_h02_fold1[i] = price_at.get((day_1, 2, 1), np.nan)
        stats_1 = daily_stats.get(day_1)
        if stats_1 is None:
            yest_mean[i] = np.nan
            yest_std[i] = np.nan
            yest_min[i] = np.nan
            yest_max[i] = np.nan
            yest_neg[i] = np.nan
            yest_ramp[i] = np.nan
        else:
            yest_mean[i] = stats_1["mean"]
            yest_std[i] = stats_1["std"]
            yest_min[i] = stats_1["min"]
            yest_max[i] = stats_1["max"]
            yest_neg[i] = stats_1["neg"]
            yest_ramp[i] = stats_1["ramp"]
        stats_7 = daily_stats.get(day_7)
        week_mean[i] = stats_7["mean"] if stats_7 is not None else np.nan

    out = pd.DataFrame(index=df.index)
    out["price_eur_mwh"] = price
    out["negative_price_flag"] = (price < 0).astype(int)
    out["local_hour"] = hours
    out["local_date"] = dates
    out["local_fold"] = folds
    out["product_position"] = product_positions
    out["hour"] = hours
    out["day_of_week"] = np.asarray(local.dayofweek, dtype=int)
    out["month"] = np.asarray(local.month, dtype=int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out["price_lag_24h"] = lag_24
    out["price_lag_48h"] = lag_48
    out["price_lag_168h"] = lag_168
    out["price_lag_336h"] = lag_336
    out["price_shape_24h"] = lag_24 - yest_mean
    out["yest_mean"] = yest_mean
    out["yest_std"] = yest_std
    out["yest_min"] = yest_min
    out["yest_max"] = yest_max
    out["yest_neg_hours"] = yest_neg
    out["yest_ramp"] = yest_ramp
    out["week_ago_mean"] = week_mean
    for clock_hour in range(24):
        out[f"yest_h{clock_hour:02d}"] = yesterday[:, clock_hour]
    out["yest_h02_fold1"] = yest_h02_fold1

    if include_oracle_forecasts:
        _add_forecast_features(df, out, dates)

    needed = ["price_lag_24h", "price_lag_48h", "price_lag_168h", "price_lag_336h", "yest_mean"]
    out = out.dropna(subset=needed)
    return out


def _add_forecast_features(
    source: pd.DataFrame,
    out: pd.DataFrame,
    dates: np.ndarray,
) -> None:
    """Copy day-ahead forecasts and add delivery-day aggregates.

    The whole delivery-day forecast curve is published before gate closure,
    so same-day sums and maxima of wind and solar are allowed. Prices are not.
    """
    n = len(out)
    for col in FORECAST_COLS:
        if col in source.columns:
            out[col] = source[col].to_numpy(dtype=float)

    if "solar_mw" in source.columns:
        solar = source["solar_mw"].to_numpy(dtype=float)
        sums: dict[dt.date, float] = {}
        maxima: dict[dt.date, float] = {}
        for day, value in zip(dates, solar):
            if np.isnan(value):
                continue
            sums[day] = sums.get(day, 0.0) + float(value)
            maxima[day] = max(maxima.get(day, -np.inf), float(value))
        out["solar_day_sum"] = np.array([sums.get(day, np.nan) for day in dates])
        out["solar_day_max"] = np.array([maxima.get(day, np.nan) for day in dates])

    if "wind_total_mw" in source.columns:
        wind = source["wind_total_mw"].to_numpy(dtype=float)
        totals: dict[dt.date, float] = {}
        counts: dict[dt.date, int] = {}
        for day, value in zip(dates, wind):
            if np.isnan(value):
                continue
            totals[day] = totals.get(day, 0.0) + float(value)
            counts[day] = counts.get(day, 0) + 1
        out["wind_day_mean"] = np.array([
            totals[day] / counts[day] if counts.get(day, 0) else np.nan
            for day in dates
        ])

    if {"gen_forecast_mw", "wind_total_mw", "solar_mw"} <= set(out.columns):
        out["thermal_proxy_mw"] = (
            out["gen_forecast_mw"] - out["wind_total_mw"] - out["solar_mw"]
        )

    if len(out) != n:
        raise RuntimeError("forecast feature block changed the frame length")


def assert_auction_information_set(df: pd.DataFrame) -> None:
    """
    Fail if a same-day price permutation changes that day's price features.

    The next delivery day must change, otherwise the features are ignoring
    history and the check would pass for an empty feature set.
    """
    da = build_day_ahead_features(df)
    counts = da.groupby("local_date").size()
    candidates: list[dt.date] = []
    for day in counts.index:
        nxt = day + dt.timedelta(days=1)
        if int(counts.get(day, 0)) >= 20 and int(counts.get(nxt, 0)) >= 20:
            candidates.append(day)
    if len(candidates) < 3:
        raise AssertionError(
            "Not enough complete local days to test the day-ahead information set"
        )

    target = candidates[len(candidates) // 2]
    nxt = target + dt.timedelta(days=1)

    shuffled = df.copy()
    local_dates = np.array([ts.date() for ts in _berlin_clock(shuffled.index)])
    mask = local_dates == target
    original = shuffled.loc[mask, "price_eur_mwh"].to_numpy(dtype=float).copy()
    perm = original.copy()
    rng = np.random.default_rng(7)
    for _ in range(10):
        rng.shuffle(perm)
        if not np.allclose(perm, original):
            break
    else:
        raise AssertionError("Could not build a different permutation of same-day prices")
    shuffled.loc[mask, "price_eur_mwh"] = perm

    da_shuffled = build_day_ahead_features(shuffled)
    profile_cols = ["price_lag_24h"] + [f"yest_h{hour:02d}" for hour in range(24)]

    same = da.loc[da["local_date"] == target, PRICE_FEATURE_COLS].to_numpy(dtype=float)
    same_after = da_shuffled.loc[
        da_shuffled["local_date"] == target, PRICE_FEATURE_COLS
    ].to_numpy(dtype=float)
    if same.shape != same_after.shape or not np.allclose(same, same_after, equal_nan=True):
        raise AssertionError(
            "Permuting prices inside a delivery day changed that day's price features"
        )

    nxt_before = da.loc[da["local_date"] == nxt, profile_cols].to_numpy(dtype=float)
    nxt_after = da_shuffled.loc[
        da_shuffled["local_date"] == nxt, profile_cols
    ].to_numpy(dtype=float)
    if np.allclose(nxt_before, nxt_after, equal_nan=True):
        raise AssertionError(
            "The next delivery day ignored a permutation of yesterday's prices"
        )


def run_comparison(df: pd.DataFrame, split_date: dt.date | None = None) -> pd.DataFrame:
    """
    Compare the original lag-1 feature set with the auction-safe protocol.

    Models, all LightGBM with the same tree hyperparameters:
      Leaky LGBM    — original features, including price_lag_1h
      DA Price History — primary model using only price history and calendar data
      DA Hourly     — 24 experts, one per Europe/Berlin hour
      DA Spike      — auction-safe set, higher weight on negative and high prices
      DA Ensemble   — mean of DA Price History and DA Hourly
      Post-gate Oracle — sensitivity using unverifiable ENTSO-E forecasts

    The split is the local delivery date, default 2024-01-01. Rows are aligned
    so every model is scored on the same hours.
    """
    from src.evaluate import mae_by_hour, regime_scores
    from src.models.day_ahead_models import (
        EnsembleAverage,
        HourlyExpertLGBM,
        fit_lgbm,
    )
    from src.protocol import assert_primary_features, load_protocol
    from src.visualize import plot_day_ahead_by_hour, plot_day_ahead_regimes

    if split_date is None:
        split_date = dt.date(2024, 1, 1)

    print("  Checking that price features ignore same-day prices …")
    assert_auction_information_set(df)

    da = build_day_ahead_features(df).sort_index()
    oracle = build_day_ahead_features(
        df, include_oracle_forecasts=True
    ).loc[da.index]
    aligned = df.loc[da.index]
    leaky_cols = [
        col for col in aligned.columns
        if col not in {"price_eur_mwh", "negative_price_flag"}
        and pd.api.types.is_numeric_dtype(aligned[col])
    ]
    da_cols = feature_columns(da)
    oracle_cols = feature_columns(oracle)
    assert_primary_features(da_cols)

    train_mask = np.array([day < split_date for day in da["local_date"]])
    test_mask = ~train_mask
    if train_mask.sum() < 1000 or test_mask.sum() < 24:
        raise RuntimeError(
            f"Day-ahead split {split_date} left train={train_mask.sum()} test={test_mask.sum()}"
        )

    # Last 10% of the training window, in time order, matches the main benchmark.
    train_idx = np.flatnonzero(train_mask)
    cut = int(len(train_idx) * 0.90)
    fit_idx = train_idx[:cut]
    val_idx = train_idx[cut:]
    test_idx = np.flatnonzero(test_mask)

    y = da["price_eur_mwh"].to_numpy(dtype=float)
    hours = da["local_hour"].to_numpy(dtype=int)
    X_leaky = aligned[leaky_cols].to_numpy(dtype=float)
    X_da = da[da_cols].to_numpy(dtype=float)
    X_oracle = oracle[oracle_cols].to_numpy(dtype=float)

    print(
        f"  Auction-safe matrix: {len(da_cols)} features, "
        f"train {len(fit_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}"
    )
    print(f"  Leaky reference matrix: {len(leaky_cols)} features on the same hours")

    import time

    specs: list[tuple[str, np.ndarray, np.ndarray]] = []
    reports: list[dict] = []

    def _score(name: str, pred: np.ndarray, elapsed: float, n_features: int) -> None:
        report = regime_scores(y[test_idx], pred)
        report["model"] = name
        report["training_time_s"] = round(elapsed, 2)
        report["n_features"] = n_features
        reports.append(report)
        specs.append((name, pred, hours[test_idx]))
        print(
            f"    {name:<14} MAE {report['MAE']:6.2f}  "
            f"neg {report['MAE_negative']:6.2f}  "
            f"high {report['MAE_high_100']:6.2f}  "
            f"neg-recall {report['neg_recall'] * 100:5.1f}%  "
            f"({elapsed:.1f}s)"
        )

    _score("Naive D-1", da["price_lag_24h"].to_numpy(dtype=float)[test_idx], 0.0, 1)
    _score("Naive D-7", da["price_lag_168h"].to_numpy(dtype=float)[test_idx], 0.0, 1)

    print("  Leaky LGBM (diagnostic only) …")
    t0 = time.time()
    leaky = fit_lgbm(
        X_leaky[fit_idx], y[fit_idx], X_leaky[val_idx], y[val_idx],
        save_path=PROCESSED_DIR.parent.parent / "models" / "day_ahead_leaky_lgbm.pkl",
    )
    _score("Leaky LGBM", leaky.predict(X_leaky[test_idx]), time.time() - t0, len(leaky_cols))

    print("  DA Price History …")
    t0 = time.time()
    global_model = fit_lgbm(
        X_da[fit_idx], y[fit_idx], X_da[val_idx], y[val_idx],
        save_path=PROCESSED_DIR.parent.parent / "models" / "day_ahead_global.pkl",
    )
    global_pred = global_model.predict(X_da[test_idx])
    _score("DA Price History", global_pred, time.time() - t0, len(da_cols))

    print("  DA Hourly (24 experts) …")
    t0 = time.time()
    hourly = HourlyExpertLGBM()
    hourly.fit(
        X_da[fit_idx], y[fit_idx], hours[fit_idx],
        X_da[val_idx], y[val_idx], hours[val_idx],
    )
    hourly_pred = hourly.predict(X_da[test_idx], hours[test_idx])
    _score("DA Hourly", hourly_pred, time.time() - t0, len(da_cols))

    print("  DA Spike (weights on y<0 and y≥100) …")
    t0 = time.time()
    spike = fit_lgbm(
        X_da[fit_idx], y[fit_idx], X_da[val_idx], y[val_idx],
        sample_weight=_spike_weights(y[fit_idx]),
        save_path=PROCESSED_DIR.parent.parent / "models" / "day_ahead_spike.pkl",
    )
    _score("DA Spike", spike.predict(X_da[test_idx]), time.time() - t0, len(da_cols))

    print("  DA Ensemble …")
    ensemble_pred = EnsembleAverage().predict(global_pred, hourly_pred)
    # No extra fit. Report the combined training time of the two members.
    member_time = sum(
        row["training_time_s"] for row in reports
        if row["model"] in {"DA Price History", "DA Hourly"}
    )
    _score("DA Ensemble", ensemble_pred, member_time, len(da_cols))

    print("  Post-gate Oracle (sensitivity only) …")
    t0 = time.time()
    oracle_model = fit_lgbm(
        X_oracle[fit_idx], y[fit_idx], X_oracle[val_idx], y[val_idx],
        save_path=PROCESSED_DIR.parent.parent / "models" / "post_gate_oracle.pkl",
    )
    _score(
        "Post-gate Oracle",
        oracle_model.predict(X_oracle[test_idx]),
        time.time() - t0,
        len(oracle_cols),
    )

    leaderboard = pd.DataFrame(reports)
    leaderboard["information_set"] = "primary"
    leaderboard["rankable"] = True
    leaderboard.loc[
        leaderboard["model"].str.startswith("Naive"), "information_set"
    ] = "primary baseline"
    leaderboard.loc[
        leaderboard["model"] == "Leaky LGBM", ["information_set", "rankable"]
    ] = ["same-auction leakage", False]
    leaderboard.loc[
        leaderboard["model"] == "Post-gate Oracle", ["information_set", "rankable"]
    ] = ["post-gate oracle", False]
    leaderboard = leaderboard.sort_values(
        ["rankable", "MAE"], ascending=[False, True]
    ).reset_index(drop=True)
    metrics_dir = PROCESSED_DIR.parent.parent / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(metrics_dir / "day_ahead_leaderboard.csv", index=False)
    leaderboard[leaderboard["rankable"]].to_csv(
        metrics_dir / "day_ahead_primary_leaderboard.csv", index=False
    )
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            da[["price_eur_mwh", *da_cols]], index=True
        ).to_numpy().tobytes()
    ).hexdigest()
    protocol = load_protocol()
    manifest = {
        "protocol_version": protocol.protocol_version,
        "forecast_origin": protocol.forecast_origin,
        "market": protocol.market,
        "matrix_sha256": digest,
        "rows": len(da),
        "first_delivery_utc": da.index.min().isoformat(),
        "last_delivery_utc": da.index.max().isoformat(),
        "primary_features": da_cols,
        "split_local_date": split_date.isoformat(),
        "random_seed": 42,
    }
    (metrics_dir / "day_ahead_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    by_hour_frames = []
    for name, pred, pred_hours in specs:
        if name in {"Leaky LGBM", "Post-gate Oracle"}:
            continue
        table = mae_by_hour(y[test_idx], pred, pred_hours)
        table.insert(0, "model", name)
        by_hour_frames.append(table)
    by_hour = pd.concat(by_hour_frames, ignore_index=True)
    by_hour.to_csv(metrics_dir / "day_ahead_by_hour.csv", index=False)

    save_frame = da.copy()
    save_frame["local_date"] = pd.to_datetime(save_frame["local_date"])
    save_frame.to_parquet(PROCESSED_DIR / "features_day_ahead.parquet")
    oracle_save = oracle.copy()
    oracle_save["local_date"] = pd.to_datetime(oracle_save["local_date"])
    oracle_save.to_parquet(PROCESSED_DIR / "features_post_gate_oracle.parquet")

    importance = pd.DataFrame({
        "feature": da_cols,
        "importance": global_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(metrics_dir / "day_ahead_importance.csv", index=False)
    top = ", ".join(importance["feature"].head(5).tolist())
    print(f"  Top DA Price History features: {top}")

    plot_day_ahead_by_hour(by_hour)
    plot_day_ahead_regimes(leaderboard)
    return leaderboard


def _spike_weights(y: np.ndarray, neg_weight: float = 4.0, high_weight: float = 3.0) -> np.ndarray:
    """Upweight negative hours and hours at or above 100 €/MWh.

    Early stopping still watches unweighted MAE on the validation set, so the
    extra weight can move the fit toward the tails without choosing the
    iteration that minimises the weighted loss.
    """
    weights = np.ones(len(y), dtype=float)
    weights[y < 0] = neg_weight
    weights[y >= 100.0] = high_weight
    return weights
