from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.decision import BatterySpec, optimize_day, realized_value  # noqa: E402
from src.research.features import build_daily_dataset  # noqa: E402
from src.research.generalization import split_zone  # noqa: E402
from src.research.shadow import build_index, write_packet  # noqa: E402


def _hourly(start: str, days: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=days * 24, freq="h")
    prices = 40 + 15 * np.sin(2 * np.pi * np.tile(np.arange(24), days) / 24)
    return pd.DataFrame({"price_eur_mwh": prices}, index=index)


def test_incomplete_test_year_is_rejected() -> None:
    data = build_daily_dataset(_hourly("2012-01-01", 365 * 5 - 10))
    with pytest.raises(ValueError, match="not a complete test year"):
        split_zone(data, "XX", 2016, 4)


def test_battery_dispatch_is_feasible_and_non_simultaneous() -> None:
    prices = np.r_[np.full(8, 10.0), np.full(8, 100.0), np.full(8, 20.0)]
    spec = BatterySpec()
    charge, discharge, soc = optimize_day(prices, spec)
    assert np.all(charge * discharge < 1e-8)
    assert np.all((soc >= -1e-8) & (soc <= spec.energy_mwh + 1e-8))
    assert soc[0] == pytest.approx(soc[-1])
    assert realized_value(prices, charge, discharge, spec) > 0


def test_shadow_packets_are_hash_verified_and_append_only(tmp_path: Path) -> None:
    packets = tmp_path / "packets"
    packet = {
        "forecast_origin": "2017-01-01T12:00:00+01:00",
        "delivery_date": "2017-01-02", "model": "test",
        "predictions": [1.0] * 24, "source_checksums": {"prices": "abc"},
        "simulated_historical": True,
    }
    path = write_packet(packets, packet)
    index = build_index(packets, tmp_path / "index.json")
    assert len(index["entries"]) == 1
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["predictions"][0] = 999
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_index(packets, tmp_path / "index2.json")
