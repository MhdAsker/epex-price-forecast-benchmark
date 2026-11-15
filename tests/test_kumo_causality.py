from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.kumo.data import KumoPriceDataset, build_origins, training_standardizer


def test_delivery_day_prices_are_labels_not_inputs() -> None:
    index = pd.date_range("2012-01-01", periods=100 * 24, freq="h")
    original = pd.DataFrame({"price_eur_mwh": np.sin(np.arange(len(index)))}, index=index)
    origins = build_origins(original, target_start="2012-02-01", target_end_exclusive="2012-02-10")
    standardizer = training_standardizer(original, train_start="2012-01-01", train_end_exclusive="2012-02-01")
    x_before, y_before, _ = KumoPriceDataset(original, origins, standardizer)[0]
    changed = original.copy()
    day = origins[0].delivery_date
    changed.loc[(changed.index >= day) & (changed.index < day + pd.Timedelta(days=3)), "price_eur_mwh"] += 1000
    x_after, y_after, _ = KumoPriceDataset(changed, origins, standardizer)[0]
    assert np.allclose(x_before.numpy(), x_after.numpy())
    assert not np.allclose(y_before.numpy(), y_after.numpy())


def test_context_prices_were_published_before_issuance() -> None:
    index = pd.date_range("2012-01-01", periods=100 * 24, freq="h")
    frame = pd.DataFrame({"price_eur_mwh": np.arange(len(index))}, index=index)
    origins = build_origins(frame, target_start="2012-02-01", target_end_exclusive="2012-02-10")
    assert all(item.context_last_available_at <= item.forecast_origin for item in origins)
