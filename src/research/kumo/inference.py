"""Batched Kumo inference aligned to complete delivery-day curves."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.research.kumo.data import (
    IssuanceWindow,
    KumoPriceDataset,
    PriceStandardizer,
    validate_canonical_hourly,
)


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


def predict_origins_darr(
    model: Any,
    frame: pd.DataFrame,
    origins: Sequence[IssuanceWindow],
    context_origins: Sequence[IssuanceWindow],
    standardizer: PriceStandardizer,
    direct_points: pd.DataFrame,
    *,
    model_name: str = "Kumo fine-tuned + DARR",
    alpha: float = 0.01,
    k: int = 64,
    temperature: float = 0.05,
    batch_size: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Apply NVIDIA's native DARR retrieval to a fixed direct checkpoint.

    The caller controls the historical memory explicitly. Every context window
    must have a complete continuation ending before the first query target, so
    validation or retrospective labels cannot leak into retrieval.
    """
    import torch
    from torch.utils.data import DataLoader

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("DARR alpha must be in [0, 1]")
    if not context_origins or not origins:
        raise ValueError("DARR requires non-empty context and query origins")
    latest_context_label = max(origin.target_stop for origin in context_origins)
    earliest_query_label = min(origin.target_start for origin in origins)
    if latest_context_label > earliest_query_label:
        raise ValueError("DARR context continuations overlap query targets")

    # Imported from the pinned NVIDIA checkout placed on sys.path by
    # build_verified_model. This reuses the official embedding normalization,
    # similarity weighting, and NaN-safe continuation averaging.
    from sdk.forecasting import build_context_memory, embed_batch, knn_forecast, l2_normalize  # type: ignore

    ordered = validate_canonical_hourly(frame)
    values = ordered["price_eur_mwh"].to_numpy(dtype=np.float32)
    first_parameter = next(model.parameters(), None)
    device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    model.eval()
    started = time.perf_counter()
    loader = DataLoader(
        KumoPriceDataset(frame, context_origins, standardizer),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    db_embeddings, db_targets = build_context_memory(model, loader, device, cosine=True)
    memory_seconds = time.perf_counter() - started

    query_started = time.perf_counter()
    query_embeddings = []
    for batch_start in range(0, len(origins), batch_size):
        batch = origins[batch_start:batch_start + batch_size]
        x = np.stack([
            standardizer.transform(values[origin.context_start:origin.context_stop])
            for origin in batch
        ])[:, None, :]
        mask = torch.ones((len(batch), x.shape[-1]), dtype=torch.bool, device=device)
        with torch.no_grad():
            embedding = embed_batch(model, torch.from_numpy(x).float().to(device), mask)
        query_embeddings.append(embedding.detach().cpu().numpy().astype(np.float32))
    query_embeddings_array = l2_normalize(np.nan_to_num(
        np.concatenate(query_embeddings), nan=0.0, posinf=0.0, neginf=0.0, copy=False,
    ))
    retrieved = knn_forecast(
        db_embeddings, db_targets, query_embeddings_array, k=k, temperature=temperature,
    )[:, 0, :]
    retrieved = standardizer.inverse(retrieved)
    query_seconds = time.perf_counter() - query_started

    direct = direct_points.copy()
    direct["delivery_date"] = direct.delivery_date.astype(str)
    direct_lookup = direct.set_index(["delivery_date", "hour"])["prediction"]
    rows: list[dict] = []
    for query_index, origin in enumerate(origins):
        date = origin.delivery_date.date().isoformat()
        actual = values[origin.target_start:origin.target_start + origin.scored_products]
        for hour in range(origin.scored_products):
            direct_prediction = float(direct_lookup.loc[(date, hour)])
            retrieval_prediction = float(retrieved[query_index, hour])
            rows.append({
                "delivery_date": date, "hour": hour, "actual": float(actual[hour]),
                "prediction": alpha * direct_prediction + (1.0 - alpha) * retrieval_prediction,
                "direct_prediction": direct_prediction,
                "retrieval_prediction": retrieval_prediction,
                "model": model_name, "seed": 13,
                "forecast_origin": origin.forecast_origin.isoformat(),
                "native_horizon_step": hour + 1,
                "inference_seconds_per_origin": query_seconds / len(origins),
            })
    diagnostics = {
        "implementation": "NVIDIA sdk.forecasting build_context_memory + knn_forecast",
        "alpha": alpha, "k": min(k, len(context_origins)), "temperature": temperature,
        "context_windows": len(context_origins), "query_windows": len(origins),
        "context_memory_seconds": memory_seconds, "query_retrieval_seconds": query_seconds,
        "context_last_target_timestamp": ordered.index[latest_context_label - 1].isoformat(),
        "first_query_target_timestamp": ordered.index[earliest_query_label].isoformat(),
    }
    return pd.DataFrame(rows), diagnostics


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
