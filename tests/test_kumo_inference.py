from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.research.kumo.data import PriceStandardizer, build_origins
from src.research.kumo.inference import predict_origins


class _Output:
    def __init__(self, forecast):
        self.forecast = forecast


class _RepeatModel(torch.nn.Module):
    def forward(self, x_enc, input_mask):
        return _Output(x_enc[:, :, -1:].repeat(1, 1, 72))


def test_inference_scores_first_24_native_steps() -> None:
    index = pd.date_range("2012-01-01", periods=80 * 24, freq="h")
    frame = pd.DataFrame({"price_eur_mwh": np.arange(len(index), dtype=float)}, index=index)
    origins = build_origins(
        frame, target_start="2012-02-01", target_end_exclusive="2012-02-03",
        require_full_target=False,
    )
    panel = predict_origins(_RepeatModel(), frame, origins, PriceStandardizer(0, 1), model_name="test", batch_size=2)
    assert len(panel) == 48
    assert panel.groupby("delivery_date").size().eq(24).all()
    assert panel["native_horizon_step"].min() == 1
    assert panel["native_horizon_step"].max() == 24
