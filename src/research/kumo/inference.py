"""Batched Kumo inference aligned to complete delivery-day curves."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.research.kumo.data import IssuanceWindow, PriceStandardizer, validate_canonical_hourly


def predict_origins(
    model: Any,
    frame: pd.DataFrame,
    origins: Sequence[IssuanceWindow],
    standardizer: PriceStandardizer,
    *,
    model_name: str,
    batch_size: int = 4,
) -> pd.DataFrame:
    import torch

    ordered = validate_canonical_hourly(frame)
    values = ordered["price_eur_mwh"].to_numpy(dtype=np.float32)
    rows: list[dict] = []
    model.eval()
    first_parameter = next(model.parameters(), None)
    device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    for batch_start in range(0, len(origins), batch_size):
        batch = origins[batch_start:batch_start + batch_size]
        x = np.stack([standardizer.transform(values[o.context_start:o.context_stop]) for o in batch])[:, None, :]
        mask = torch.ones((len(batch), x.shape[-1]), dtype=torch.bool, device=device)
        started = time.perf_counter()
        with torch.no_grad():
            pred = model(
                x_enc=torch.from_numpy(x).float().to(device), input_mask=mask
            ).forecast.detach().cpu().numpy()[:, 0, :]
        elapsed = time.perf_counter() - started
        pred = standardizer.inverse(pred)
        for i, origin in enumerate(batch):
            actual = values[origin.target_start:origin.target_start + origin.scored_products]
            for hour in range(origin.scored_products):
                rows.append({
                    "delivery_date": origin.delivery_date.date().isoformat(), "hour": hour,
                    "actual": float(actual[hour]), "prediction": float(pred[i, hour]),
                    "model": model_name, "seed": 13,
                    "forecast_origin": origin.forecast_origin.isoformat(),
                    "native_horizon_step": hour + 1,
                    "inference_seconds_per_origin": elapsed / len(batch),
                })
    return pd.DataFrame(rows)


def load_head(model: Any, path) -> None:
    import torch

    state = torch.load(path, map_location="cpu")
    unexpected = [key for key in state if not key.startswith("head.")]
    if unexpected:
        raise ValueError(f"head checkpoint contains non-head keys: {unexpected}")
    result = model.load_state_dict(state, strict=False)
    missing_non_head = [key for key in result.missing_keys if key.startswith("head.")]
    if missing_non_head:
        raise RuntimeError(f"head checkpoint is incomplete: {missing_non_head}")
