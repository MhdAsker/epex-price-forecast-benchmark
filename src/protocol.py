"""Forecast-origin contract and feature-availability checks.

The benchmark predicts the complete DE-LU delivery day at noon on D-1.  A
feature belongs to the primary information set only when it is both known at
that instant and recoverable historically with the vintage that was known at
that instant.  Series that fail either condition are oracle/sensitivity inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "config" / "forecast_protocol.json"

ORACLE_FEATURES = frozenset(
    {
        "wind_onshore_mw",
        "wind_offshore_mw",
        "wind_total_mw",
        "solar_mw",
        "gen_forecast_mw",
        "solar_day_sum",
        "solar_day_max",
        "wind_day_mean",
        "thermal_proxy_mw",
    }
)


@dataclass(frozen=True)
class ForecastProtocol:
    protocol_version: str
    market: str
    timezone: str
    forecast_origin: str
    target: str


def load_protocol(path: Path = CONTRACT_PATH) -> ForecastProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ForecastProtocol(
        protocol_version=payload["protocol_version"],
        market=payload["market"],
        timezone=payload["timezone"],
        forecast_origin=payload["forecast_origin"],
        target=payload["target"],
    )


def assert_primary_features(feature_names: Iterable[str]) -> None:
    """Reject unverifiable post-gate/revised inputs from a primary model."""
    leaked = sorted(set(feature_names) & ORACLE_FEATURES)
    if leaked:
        raise AssertionError(
            "Primary information set contains post-gate/oracle features: "
            + ", ".join(leaked)
        )
