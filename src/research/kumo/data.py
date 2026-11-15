"""Issuance-safe windows for day-ahead price foundation models.

Prices are indexed by delivery time, but a day-ahead curve is published before
delivery. For the canonical EPEX-DE benchmark, every price for D-1 is therefore
eligible at the noon D-1 forecast origin for D. Targets always start at the
first product of D. Training/validation windows require all 72 labels to remain
inside their registered split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IssuanceWindow:
    delivery_date: pd.Timestamp
    forecast_origin: pd.Timestamp
    context_start: int
    context_stop: int
    target_start: int
    target_stop: int
    scored_products: int
    context_last_delivery: pd.Timestamp
    context_last_available_at: pd.Timestamp


@dataclass(frozen=True)
class PriceStandardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "PriceStandardizer":
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std())
        return cls(mean=mean, std=std if std >= 1e-8 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


def validate_canonical_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    if "price_eur_mwh" not in frame:
        raise KeyError("price_eur_mwh is required")
    ordered = frame[["price_eur_mwh"]].sort_index().copy()
    if not isinstance(ordered.index, pd.DatetimeIndex):
        raise TypeError("price index must be a DatetimeIndex")
    if ordered.index.tz is not None:
        raise ValueError("canonical EPEX-DE data must use its naive 24-product index")
    if ordered.index.has_duplicates or not ordered.index.is_monotonic_increasing:
        raise ValueError("price timestamps must be unique and increasing")
    if ordered["price_eur_mwh"].isna().any():
        raise ValueError("missing prices cannot be fabricated")
    expected = pd.date_range(ordered.index[0], ordered.index[-1], freq="h")
    if not ordered.index.equals(expected):
        raise ValueError("canonical price series must be hourly and continuous")
    counts = ordered.groupby(ordered.index.normalize()).size()
    if not (counts == 24).all():
        raise ValueError("canonical benchmark must contain exactly 24 products per day")
    return ordered


def price_available_at(delivery_timestamp: pd.Timestamp) -> pd.Timestamp:
    """Publication time implied by a day-ahead auction curve.

    A price for local delivery day T is available from noon on T-1. This is
    conservative for using D-1 curve prices at the noon D-1 issuance for D.
    """
    day = pd.Timestamp(delivery_timestamp).normalize()
    return (day - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).tz_localize("Europe/Berlin")


def build_origins(
    frame: pd.DataFrame,
    *,
    target_start: str | pd.Timestamp,
    target_end_exclusive: str | pd.Timestamp,
    seq_len: int = 512,
    forecast_horizon: int = 72,
    scored_products: int = 24,
    require_full_target: bool = True,
    stride_days: int = 1,
) -> list[IssuanceWindow]:
    ordered = validate_canonical_hourly(frame)
    start = pd.Timestamp(target_start).normalize()
    stop = pd.Timestamp(target_end_exclusive).normalize()
    positions = pd.Series(np.arange(len(ordered)), index=ordered.index)
    origins: list[IssuanceWindow] = []
    for delivery_date in pd.date_range(start, stop, freq=f"{stride_days}D", inclusive="left"):
        if delivery_date not in positions.index:
            continue
        target_i = int(positions.loc[delivery_date])
        context_start = target_i - seq_len
        required = forecast_horizon if require_full_target else scored_products
        if context_start < 0 or target_i + required > len(ordered):
            continue
        target_stop = target_i + forecast_horizon
        if require_full_target:
            last_target = ordered.index[target_stop - 1]
            if last_target >= stop:
                continue
        context_last = ordered.index[target_i - 1]
        origin = (delivery_date - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).tz_localize("Europe/Berlin")
        available = price_available_at(context_last)
        if available > origin:
            raise AssertionError("context contains a price unavailable at issuance")
        origins.append(
            IssuanceWindow(
                delivery_date=delivery_date,
                forecast_origin=origin,
                context_start=context_start,
                context_stop=target_i,
                target_start=target_i,
                target_stop=target_stop,
                scored_products=scored_products,
                context_last_delivery=context_last,
                context_last_available_at=available,
            )
        )
    return origins


class KumoPriceDataset:
    """Torch-compatible dataset without importing torch at module import time."""

    def __init__(
        self,
        frame: pd.DataFrame,
        origins: Sequence[IssuanceWindow],
        standardizer: PriceStandardizer,
        *,
        forecast_horizon: int = 72,
    ) -> None:
        self.frame = validate_canonical_hourly(frame)
        self.values = self.frame["price_eur_mwh"].to_numpy(dtype=np.float32)
        self.origins = list(origins)
        self.standardizer = standardizer
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int):
        import torch

        item = self.origins[index]
        context = self.values[item.context_start:item.context_stop]
        target = self.values[item.target_start:item.target_stop]
        if len(target) != self.forecast_horizon:
            raise ValueError("training item lacks a complete 72-step target")
        x = torch.from_numpy(self.standardizer.transform(context)[None, :].copy())
        y = torch.from_numpy(self.standardizer.transform(target)[None, :].copy())
        mask = torch.ones(x.shape[-1], dtype=torch.bool)
        return x, y, mask


def training_standardizer(
    frame: pd.DataFrame,
    *,
    train_start: str | pd.Timestamp,
    train_end_exclusive: str | pd.Timestamp,
) -> PriceStandardizer:
    ordered = validate_canonical_hourly(frame)
    mask = (ordered.index >= pd.Timestamp(train_start)) & (ordered.index < pd.Timestamp(train_end_exclusive))
    values = ordered.loc[mask, "price_eur_mwh"].to_numpy()
    if not len(values):
        raise ValueError("normalization interval is empty")
    return PriceStandardizer.fit(values)


def origin_manifest(origins: Sequence[IssuanceWindow], split: Literal["train", "validation", "test"]) -> dict:
    if not origins:
        return {"split": split, "windows": 0}
    return {
        "split": split,
        "windows": len(origins),
        "first_delivery_date": origins[0].delivery_date.date().isoformat(),
        "last_delivery_date": origins[-1].delivery_date.date().isoformat(),
        "first_forecast_origin": origins[0].forecast_origin.isoformat(),
        "last_forecast_origin": origins[-1].forecast_origin.isoformat(),
        "input_length": origins[0].context_stop - origins[0].context_start,
        "target_horizon": origins[0].target_stop - origins[0].target_start,
        "scored_products": origins[0].scored_products,
    }


def build_utc_origins(
    frame: pd.DataFrame,
    *,
    target_start: str | pd.Timestamp,
    target_end_exclusive: str | pd.Timestamp,
    seq_len: int = 512,
    forecast_horizon: int = 72,
) -> list[IssuanceWindow]:
    """Build physical-hour origins for UTC-indexed ENTSO-E data.

    Unlike the canonical historical benchmark, the scored product count is 23,
    24, or 25 according to the Europe/Berlin delivery day. No physical hour is
    discarded or duplicated.
    """
    if "price_eur_mwh" not in frame:
        raise KeyError("price_eur_mwh is required")
    ordered = frame[["price_eur_mwh"]].sort_index().copy()
    if ordered.index.tz is None:
        utc = ordered.index.tz_localize("UTC")
    else:
        utc = ordered.index.tz_convert("UTC")
    if utc.has_duplicates or not utc.is_monotonic_increasing:
        raise ValueError("UTC delivery timestamps must be unique and increasing")
    expected = pd.date_range(utc[0], utc[-1], freq="h")
    if not utc.equals(expected):
        raise ValueError("UTC price history must preserve every physical hour")
    if ordered["price_eur_mwh"].isna().any():
        raise ValueError("missing prices cannot be fabricated")
    local = utc.tz_convert("Europe/Berlin")
    local_dates = pd.Index([stamp.date() for stamp in local])
    start = pd.Timestamp(target_start).date()
    stop = pd.Timestamp(target_end_exclusive).date()
    origins: list[IssuanceWindow] = []
    for date in pd.date_range(start, stop, freq="D", inclusive="left"):
        positions = np.flatnonzero(local_dates == date.date())
        if not len(positions):
            continue
        target_i = int(positions[0])
        context_start = target_i - seq_len
        if context_start < 0 or target_i + forecast_horizon > len(ordered):
            continue
        last_context_local = local[target_i - 1]
        origin = (date - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).tz_localize("Europe/Berlin")
        available = (
            pd.Timestamp(last_context_local.date()) - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
        ).tz_localize("Europe/Berlin")
        if available > origin:
            raise AssertionError("context contains a price unavailable at issuance")
        origins.append(
            IssuanceWindow(
                delivery_date=date.tz_localize(None),
                forecast_origin=origin,
                context_start=context_start,
                context_stop=target_i,
                target_start=target_i,
                target_stop=target_i + forecast_horizon,
                scored_products=len(positions),
                context_last_delivery=utc[target_i - 1],
                context_last_available_at=available,
            )
        )
    return origins
