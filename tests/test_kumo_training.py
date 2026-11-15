from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.research.kumo.data import KumoPriceDataset, build_origins, training_standardizer
from src.research.kumo.train import head_change, head_state, train_head


class _Output:
    def __init__(self, forecast):
        self.forecast = forecast


class _TinyHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(512, 8)
        self.head = torch.nn.Linear(8, 72)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, x_enc, input_mask):
        return _Output(self.head(self.encoder(x_enc)))


def test_head_only_training_changes_head_not_encoder(tmp_path) -> None:
    index = pd.date_range("2012-01-01", periods=100 * 24, freq="h")
    price = 30 + 10 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
    frame = pd.DataFrame({"price_eur_mwh": price}, index=index)
    origins = build_origins(frame, target_start="2012-02-01", target_end_exclusive="2012-03-15")
    standardizer = training_standardizer(frame, train_start="2012-01-01", train_end_exclusive="2012-02-01")
    dataset = KumoPriceDataset(frame, origins, standardizer)
    model = _TinyHeadModel()
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    head_before = head_state(model)
    result = train_head(
        model, dataset, dataset, epochs=1, learning_rate=1e-3, batch_size=4,
        seed=13, output_dir=tmp_path, max_train_windows=8, max_validation_windows=4,
    )
    assert result["best_epoch"] == 1
    assert any(row["l2_delta"] > 0 for row in head_change(head_before, head_state(model)).values())
    assert all(torch.equal(value, model.encoder.state_dict()[key]) for key, value in encoder_before.items())
