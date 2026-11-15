from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.kumo.data import KumoPriceDataset, build_origins, training_standardizer


def _canonical(days: int = 80) -> pd.DataFrame:
    index = pd.date_range("2012-01-01", periods=days * 24, freq="h")
    return pd.DataFrame({"price_eur_mwh": np.arange(len(index), dtype=float)}, index=index)


def test_window_has_512_context_and_72_future_steps() -> None:
    frame = _canonical()
    origins = build_origins(
        frame, target_start="2012-02-01", target_end_exclusive="2012-03-01",
        seq_len=512, forecast_horizon=72,
    )
    assert origins
    first = origins[0]
    assert first.context_stop - first.context_start == 512
    assert first.target_stop - first.target_start == 72
    assert first.context_last_delivery == pd.Timestamp("2012-01-31 23:00")
    assert first.delivery_date == pd.Timestamp("2012-02-01")
    standardizer = training_standardizer(frame, train_start="2012-01-01", train_end_exclusive="2012-02-01")
    x, y, mask = KumoPriceDataset(frame, origins, standardizer)[0]
    assert x.shape == (1, 512)
    assert y.shape == (1, 72)
    assert mask.shape == (512,)


def test_target_never_crosses_registered_split_boundary() -> None:
    frame = _canonical(400)
    origins = build_origins(frame, target_start="2012-02-01", target_end_exclusive="2013-01-01")
    assert origins[-1].delivery_date == pd.Timestamp("2012-12-29")
    assert frame.index[origins[-1].target_stop - 1] == pd.Timestamp("2012-12-31 23:00")
