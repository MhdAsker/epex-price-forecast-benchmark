"""Frozen, causal expanding-memory inference for Project A Phase 3."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.research.decision_aware_darr.causal import (
    calendar_distance, eligible_candidate_indices, native_similarity_order,
    pattern_distance, rank_positions, window_id,
)
from src.research.decision_aware_darr.dataset import context_descriptor
from src.research.kumo.data import IssuanceWindow, validate_canonical_hourly


def minimal_library(frame: pd.DataFrame, origins: Sequence[IssuanceWindow]) -> pd.DataFrame:
    ordered = validate_canonical_hourly(frame)
    values = ordered.price_eur_mwh.to_numpy(np.float32)
    rows = []
    for origin in origins:
        descriptor = context_descriptor(values[origin.context_start:origin.context_stop])
        row = {
            "candidate__id": window_id(origin),
            "candidate__delivery_date": origin.delivery_date.date().isoformat(),
            "candidate__continuation_last_delivery": ordered.index[origin.target_stop - 1].isoformat(),
            "candidate__continuation_24": values[origin.target_start:origin.target_start + 24].astype(float).tolist(),
        }
        row.update({f"candidate__context_{key}": value for key, value in descriptor.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def query_table(
    frame: pd.DataFrame, origins: Sequence[IssuanceWindow], direct: dict[str, np.ndarray],
) -> pd.DataFrame:
    values = validate_canonical_hourly(frame).price_eur_mwh.to_numpy(np.float32)
    rows = []
    for origin in origins:
        date = origin.delivery_date.date().isoformat()
        descriptor = context_descriptor(values[origin.context_start:origin.context_stop])
        row = {
            "query__id": window_id(origin), "partition": "retrospective",
            "query__delivery_date": date, "query__forecast_origin": origin.forecast_origin.isoformat(),
            "query__month": int(origin.delivery_date.month), "query__weekday": int(origin.delivery_date.weekday()),
            "query__weekend": bool(origin.delivery_date.weekday() >= 5),
            "query__direct_forecast_24": np.asarray(direct[date], float).tolist(),
        }
        row.update({f"query__context_{key}": value for key, value in descriptor.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_pairs(
    *, frame: pd.DataFrame, memory: Sequence[IssuanceWindow], queries: Sequence[IssuanceWindow],
    embedding_by_id: dict[str, np.ndarray], pattern_by_id: dict[str, np.ndarray],
    checkpoint_sha256: str, top_k: int = 64, temperature: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = validate_canonical_hourly(frame)
    rows, audit = [], []
    for query in queries:
        query_id = window_id(query)
        eligible = eligible_candidate_indices(memory, query, ordered.index)
        eligible_origins = [memory[index] for index in eligible]
        candidate_embeddings = np.stack([embedding_by_id[window_id(item)] for item in eligible_origins])
        order, similarities = native_similarity_order(candidate_embeddings, embedding_by_id[query_id])
        embedding_ranks = rank_positions(similarities, descending=True)
        calendar_scores = calendar_distance(query, eligible_origins)
        calendar_ranks = rank_positions(calendar_scores, descending=False)
        patterns = np.stack([pattern_by_id[window_id(item)] for item in eligible_origins])
        pattern_scores = pattern_distance(pattern_by_id[query_id], patterns)
        pattern_ranks = rank_positions(pattern_scores, descending=False)
        embedding_top = order[:top_k]
        calendar_top = np.argsort(calendar_scores, kind="stable")[:top_k]
        pattern_top = np.argsort(pattern_scores, kind="stable")[:top_k]
        selected = np.unique(np.concatenate([embedding_top, calendar_top, pattern_top]))
        logits = similarities[embedding_top] / max(temperature, 1e-12)
        logits -= logits.max(); weights = np.exp(logits); weights /= weights.sum()
        weight_lookup = dict(zip(embedding_top.tolist(), weights.tolist()))
        embedding_set, calendar_set, pattern_set = map(set, (embedding_top.tolist(), calendar_top.tolist(), pattern_top.tolist()))
        for local in selected:
            candidate = eligible_origins[int(local)]
            rows.append({
                "query__id": query_id, "candidate__id": window_id(candidate),
                "pair__embedding_similarity": float(similarities[local]),
                "pair__embedding_rank": int(embedding_ranks[local]),
                "pair__calendar_distance": float(calendar_scores[local]),
                "pair__calendar_rank": int(calendar_ranks[local]),
                "pair__pattern_distance": float(pattern_scores[local]),
                "pair__pattern_rank": int(pattern_ranks[local]),
                "pair__selected_by_embedding_top64": bool(local in embedding_set),
                "pair__selected_by_calendar_top64": bool(local in calendar_set),
                "pair__selected_by_pattern_top64": bool(local in pattern_set),
                "pair__native_darr_weight": float(weight_lookup.get(int(local), 0.0)),
                "pair__candidate_age_days": int((query.delivery_date - candidate.delivery_date).days),
                "source__checkpoint_sha256": checkpoint_sha256,
            })
        last = max(memory[index].target_stop for index in eligible)
        audit.append({
            "query__id": query_id, "delivery_date": query.delivery_date.date().isoformat(),
            "forecast_origin": query.forecast_origin.isoformat(), "eligible_candidates": len(eligible),
            "selected_candidates": len(selected),
            "latest_candidate_last_delivery": ordered.index[last - 1].isoformat(),
            "strictly_before_issuance": bool(ordered.index[last - 1].tz_localize("Europe/Berlin") < query.forecast_origin),
        })
    return pd.DataFrame(rows), pd.DataFrame(audit)
