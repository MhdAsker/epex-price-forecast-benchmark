from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.kumo.data import build_utc_origins


def test_utc_sampler_preserves_23_and_25_hour_delivery_days() -> None:
    utc = pd.date_range("2023-01-01", "2024-01-15", freq="h", tz="UTC", inclusive="left")
    frame = pd.DataFrame({"price_eur_mwh": np.arange(len(utc), dtype=float)}, index=utc.tz_localize(None))
    spring = build_utc_origins(frame, target_start="2023-03-26", target_end_exclusive="2023-03-27")
    autumn = build_utc_origins(frame, target_start="2023-10-29", target_end_exclusive="2023-10-30")
    assert spring[0].scored_products == 23
    assert autumn[0].scored_products == 25
    local = utc.tz_convert("Europe/Berlin")
    repeated = local[(local.date == pd.Timestamp("2023-10-29").date()) & (local.hour == 2)]
    assert len(repeated) == 2
    assert repeated[0].utcoffset() != repeated[1].utcoffset()
