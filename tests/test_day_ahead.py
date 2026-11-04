"""Auction-information-set checks. Run: python tests/test_day_ahead.py"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.day_ahead import (  # noqa: E402
    PRICE_FEATURE_COLS,
    assert_auction_information_set,
    build_day_ahead_features,
    feature_columns,
)
from src.protocol import ORACLE_FEATURES, assert_primary_features, load_protocol  # noqa: E402


def _synthetic(days: int = 40) -> pd.DataFrame:
    # Build the grid on the auction clock so every local day has 24 hours.
    # January–February 2023 is CET the whole way, with no DST transition.
    local = pd.date_range("2023-01-10", periods=days * 24, freq="h", tz="Europe/Berlin")
    index = local.tz_convert("UTC").tz_localize(None)
    # Identifiable price: 100 * day-index + local hour.
    day0 = local[0].date()
    price = np.array([
        (ts.date() - day0).days * 100 + ts.hour for ts in local
    ], dtype=float)
    solar = np.array([
        50.0 if 8 <= ts.hour <= 16 else 0.0 for ts in local
    ], dtype=float)
    return pd.DataFrame(
        {
            "price_eur_mwh": price,
            "solar_mw": solar,
            "wind_onshore_mw": 10.0,
            "wind_offshore_mw": 5.0,
            "wind_total_mw": 15.0,
            "gen_forecast_mw": 100.0,
        },
        index=index,
    )


def test_lag_is_same_local_hour_yesterday() -> None:
    raw = _synthetic()
    da = build_day_ahead_features(raw)
    local = raw.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    lookup = {(ts.date(), ts.hour): price for ts, price in zip(local, raw["price_eur_mwh"])}

    checked = 0
    for ts, row in da.iterrows():
        berlin = ts.tz_localize("UTC").tz_convert("Europe/Berlin")
        expected = lookup[(berlin.date() - dt.timedelta(days=1), berlin.hour)]
        assert row["price_lag_24h"] == expected
        expected_week = lookup[(berlin.date() - dt.timedelta(days=7), berlin.hour)]
        assert row["price_lag_168h"] == expected_week
        checked += 1
    assert checked > 24 * 20


def test_same_day_price_is_not_a_feature() -> None:
    raw = _synthetic()
    da = build_day_ahead_features(raw)
    assert "price_lag_1h" not in da.columns
    assert "price_lag_1h" not in feature_columns(da)
    for col in PRICE_FEATURE_COLS:
        assert col in da.columns
    assert_auction_information_set(raw)


def test_solar_day_sum_uses_the_delivery_day_forecast() -> None:
    raw = _synthetic()
    da = build_day_ahead_features(raw, include_oracle_forecasts=True)
    # January is CET (UTC+1). A local day is 23:00 UTC previous day through 22:00 UTC.
    # solar is 50 for local hours 8..16 inclusive = 9 hours.
    assert np.allclose(da["solar_day_sum"].to_numpy(), 9 * 50.0)
    assert np.allclose(da["solar_day_max"].to_numpy(), 50.0)


def test_model_matrix_excludes_clock_helpers() -> None:
    da = build_day_ahead_features(_synthetic())
    cols = feature_columns(da)
    assert "local_date" not in cols
    assert "local_hour" not in cols
    assert "price_eur_mwh" not in cols
    assert "hour" in cols
    assert "thermal_proxy_mw" not in cols


def test_primary_matrix_excludes_unverifiable_forecasts() -> None:
    primary = build_day_ahead_features(_synthetic())
    cols = feature_columns(primary)
    assert not (set(cols) & ORACLE_FEATURES)
    assert_primary_features(cols)
    oracle = build_day_ahead_features(_synthetic(), include_oracle_forecasts=True)
    assert "solar_mw" in feature_columns(oracle)


def test_protocol_contract_loads() -> None:
    protocol = load_protocol()
    assert protocol.timezone == "Europe/Berlin"
    assert "12:00" in protocol.forecast_origin


def test_autumn_dst_products_are_not_averaged() -> None:
    index = pd.date_range("2023-10-01", "2023-11-20", freq="h", tz="UTC", inclusive="left")
    raw = pd.DataFrame(
        {
            "price_eur_mwh": np.arange(len(index), dtype=float),
            "solar_mw": 0.0,
            "wind_onshore_mw": 10.0,
            "wind_offshore_mw": 5.0,
            "wind_total_mw": 15.0,
            "gen_forecast_mw": 100.0,
        },
        index=index.tz_localize(None),
    )
    da = build_day_ahead_features(raw)
    fallback = da[da["local_date"] == dt.date(2023, 10, 29)]
    repeated = fallback[fallback["local_hour"] == 2]
    assert len(fallback) == 25
    assert fallback["product_position"].tolist() == list(range(1, 26))
    assert repeated["local_fold"].tolist() == [0, 1]
    assert repeated["price_eur_mwh"].nunique() == 2


if __name__ == "__main__":
    test_lag_is_same_local_hour_yesterday()
    test_same_day_price_is_not_a_feature()
    test_solar_day_sum_uses_the_delivery_day_forecast()
    test_model_matrix_excludes_clock_helpers()
    test_primary_matrix_excludes_unverifiable_forecasts()
    test_protocol_contract_loads()
    test_autumn_dst_products_are_not_averaged()
    print("day-ahead information set: 7 checks passed")
