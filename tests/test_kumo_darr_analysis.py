from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.kumo.darr_analysis import (
    daily_darr_diagnostics,
    reconstruct_darr_components,
)


def test_reconstructs_retrieval_component_from_registered_blend() -> None:
    alpha = 0.01
    direct = np.arange(24, dtype=float)
    retrieval = direct * 2 - 3
    actual = direct + 0.5
    blended = alpha * direct + (1 - alpha) * retrieval
    rows = []
    for hour in range(24):
        common = {"delivery_date": "2016-01-01", "hour": hour, "actual": actual[hour]}
        rows.append({**common, "model": "Kumo fine-tuned", "prediction": direct[hour]})
        rows.append({**common, "model": "Kumo fine-tuned + DARR", "prediction": blended[hour]})
    reconstructed = reconstruct_darr_components(pd.DataFrame(rows), alpha=alpha)
    np.testing.assert_allclose(reconstructed.retrieval_prediction, retrieval, rtol=0, atol=1e-12)


def test_daily_diagnostics_use_identical_targets_and_hours() -> None:
    direct = np.arange(24, dtype=float)
    retrieval = direct[::-1]
    actual = direct.copy()
    blended = 0.01 * direct + 0.99 * retrieval
    components = pd.DataFrame({
        "delivery_date": "2017-01-01", "hour": np.arange(24), "actual": actual,
        "direct_prediction": direct, "retrieval_prediction": retrieval,
        "blend_prediction": blended, "test_year": 2017,
    })
    daily = daily_darr_diagnostics(components).iloc[0]
    assert daily.direct_mae == 0
    assert daily.direct_low_correct
    assert daily.direct_high_correct
    assert daily.retrieval_low_hour == 23
    assert daily.retrieval_high_hour == 0
