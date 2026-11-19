from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.research.kumo.data import KumoPriceDataset, PriceStandardizer, build_origins
from src.research.kumo.train import train_head


class _Output:
    def __init__(self, forecast):
        self.forecast = forecast


class _TinyHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(512, 4)
        self.head = torch.nn.Linear(4, 72)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, x_enc, input_mask):
        return _Output(self.head(self.encoder(x_enc)))


class _TinyDropoutHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(512, 4)
        self.dropout = torch.nn.Dropout(p=0.35)
        self.head = torch.nn.Linear(4, 72)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, x_enc, input_mask):
        return _Output(self.head(self.dropout(self.encoder(x_enc))))


def _frame() -> pd.DataFrame:
    index = pd.date_range("2012-01-01", periods=110 * 24, freq="h")
    values = 40 + 12 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
    return pd.DataFrame({"price_eur_mwh": values}, index=index)


def test_scaling_roundtrip_and_targets_are_invariant() -> None:
    frame = _frame()
    origins = build_origins(frame, target_start="2012-02-01", target_end_exclusive="2012-03-01")
    first = PriceStandardizer(mean=-5.0, std=8.0)
    second = PriceStandardizer(mean=40.0, std=12.0)
    _, y_first, _ = KumoPriceDataset(frame, origins, first)[0]
    _, y_second, _ = KumoPriceDataset(frame, origins, second)[0]
    restored_first = first.inverse(y_first.numpy())
    restored_second = second.inverse(y_second.numpy())
    np.testing.assert_allclose(restored_first, restored_second, atol=1e-5)
    np.testing.assert_allclose(first.inverse(first.transform(restored_first)), restored_first, atol=1e-5)


def test_training_resumes_optimizer_and_epoch_state(tmp_path) -> None:
    frame = _frame()
    origins = build_origins(frame, target_start="2012-02-01", target_end_exclusive="2012-03-01")
    dataset = KumoPriceDataset(frame, origins, PriceStandardizer(40.0, 12.0))
    model = _TinyHeadModel()
    first = train_head(
        model, dataset, dataset, epochs=1, learning_rate=1e-3, batch_size=4,
        seed=13, output_dir=tmp_path, max_train_windows=8, max_validation_windows=4,
        resume=True,
    )
    second = train_head(
        model, dataset, dataset, epochs=2, learning_rate=1e-3, batch_size=4,
        seed=13, output_dir=tmp_path, max_train_windows=8, max_validation_windows=4,
        resume=True,
    )
    assert first["epochs_completed"] == 1
    assert second["epochs_completed"] == 2
    assert second["optimizer_steps"] == 4
    assert (tmp_path / "last_training_state.pt").exists()


def test_interrupted_dropout_training_is_exactly_resumable(tmp_path) -> None:
    frame = _frame()
    origins = build_origins(frame, target_start="2012-02-01", target_end_exclusive="2012-03-01")
    dataset = KumoPriceDataset(frame, origins, PriceStandardizer(40.0, 12.0))
    torch.manual_seed(101)
    initial = _TinyDropoutHeadModel().state_dict()

    uninterrupted = _TinyDropoutHeadModel()
    uninterrupted.load_state_dict(initial)
    full = train_head(
        uninterrupted, dataset, dataset, epochs=2, learning_rate=1e-3, batch_size=4,
        seed=13, output_dir=tmp_path / "full", max_train_windows=8,
        max_validation_windows=4, resume=True,
    )

    interrupted = _TinyDropoutHeadModel()
    interrupted.load_state_dict(initial)
    train_head(
        interrupted, dataset, dataset, epochs=1, learning_rate=1e-3, batch_size=4,
        seed=13, output_dir=tmp_path / "resumed", max_train_windows=8,
        max_validation_windows=4, resume=True,
    )
    resumed_model = _TinyDropoutHeadModel()
    resumed_model.load_state_dict(initial)
    resumed = train_head(
        resumed_model, dataset, dataset, epochs=2, learning_rate=1e-3, batch_size=4,
        seed=13, output_dir=tmp_path / "resumed", max_train_windows=8,
        max_validation_windows=4, resume=True,
    )

    assert full["history"] == resumed["history"]
    full_head = torch.load(tmp_path / "full" / "best_head.pt", weights_only=True)
    resumed_head = torch.load(tmp_path / "resumed" / "best_head.pt", weights_only=True)
    assert full_head.keys() == resumed_head.keys()
    for name in full_head:
        torch.testing.assert_close(full_head[name], resumed_head[name], rtol=0, atol=0)
