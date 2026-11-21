"""Turn candidate scores into stable retrieval forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def retrieval_forecasts(
    scored_pairs: pd.DataFrame,
    candidate_library: pd.DataFrame,
    query_features: pd.DataFrame,
    *,
    score_column: str,
    k: int,
    direct_weight: float,
) -> dict[str, np.ndarray]:
    library = candidate_library.set_index("candidate__id")
    direct = query_features.set_index("query__id")["query__direct_forecast_24"]
    forecasts: dict[str, np.ndarray] = {}
    for query_id, group in scored_pairs.groupby("query__id", sort=False):
        selected = group.sort_values(
            [score_column, "candidate__id"], ascending=[False, True]
        ).head(k)
        curves = np.stack([
            np.asarray(library.loc[candidate_id, "candidate__continuation_24"], dtype=float)
            for candidate_id in selected["candidate__id"]
        ])
        ranks = np.arange(1, len(curves) + 1, dtype=float)
        weights = 1.0 / np.log2(ranks + 1.0)
        weights /= weights.sum()
        retrieved = np.sum(curves * weights[:, None], axis=0)
        forecast = direct_weight * np.asarray(direct.loc[query_id], dtype=float) + (1 - direct_weight) * retrieved
        if forecast.shape != (24,) or not np.isfinite(forecast).all():
            raise ValueError(f"invalid forecast for {query_id}")
        forecasts[query_id] = forecast
    return forecasts


def heuristic_scores(pairs: pd.DataFrame) -> np.ndarray:
    """Fixed causal score; no fitted weights."""
    return -(
        0.50 * pairs["pair__embedding_rank"].to_numpy(float)
        + 0.30 * pairs["pair__pattern_rank"].to_numpy(float)
        + 0.20 * pairs["pair__calendar_rank"].to_numpy(float)
    )
