"""Causal window identities, eligibility, and deterministic ranking features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.research.kumo.data import IssuanceWindow, PriceStandardizer


def window_id(window: IssuanceWindow) -> str:
    return f"epex_de__{window.delivery_date.strftime('%Y%m%d')}__h72"


def strict_candidate_is_eligible(
    candidate: IssuanceWindow,
    query: IssuanceWindow,
    index: pd.DatetimeIndex,
) -> bool:
    """Require the full candidate continuation to finish before issuance."""
    last_delivery = index[candidate.target_stop - 1]
    issuance_naive = query.forecast_origin.tz_localize(None)
    return bool(last_delivery < issuance_naive)


def eligible_candidate_indices(
    candidates: Sequence[IssuanceWindow],
    query: IssuanceWindow,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    result = np.fromiter(
        (strict_candidate_is_eligible(candidate, query, index) for candidate in candidates),
        dtype=bool,
        count=len(candidates),
    )
    return np.flatnonzero(result)


def assert_candidate_eligible(
    candidate: IssuanceWindow,
    query: IssuanceWindow,
    index: pd.DatetimeIndex,
) -> None:
    if not strict_candidate_is_eligible(candidate, query, index):
        raise ValueError(
            f"illegal candidate {window_id(candidate)} for query {window_id(query)}: "
            "continuation has not completed before issuance"
        )
    if candidate.target_stop > query.target_start:
        raise ValueError("candidate continuation overlaps query target")


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, eps)


def native_similarity_order(
    database_embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    database = l2_normalize(database_embeddings)
    query = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
    similarities = database @ query
    order = np.argsort(-similarities, kind="stable")
    return order, similarities


def native_weighted_forecast(
    continuations: np.ndarray,
    similarities: np.ndarray,
    order: np.ndarray,
    *,
    k: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = order[: min(k, len(order))]
    logits = similarities[selected] / max(float(temperature), 1e-12)
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights = weights / weights.sum()
    forecast = np.sum(continuations[selected] * weights[:, None], axis=0)
    return forecast.astype(np.float32), selected, weights.astype(np.float32)


def rank_positions(values: np.ndarray, *, descending: bool) -> np.ndarray:
    order = np.argsort(-values if descending else values, kind="stable")
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return ranks


def calendar_distance(query: IssuanceWindow, candidates: Sequence[IssuanceWindow]) -> np.ndarray:
    query_month = query.delivery_date.month
    candidate_month = np.array([item.delivery_date.month for item in candidates])
    raw_month = np.abs(candidate_month - query_month)
    circular_month = np.minimum(raw_month, 12 - raw_month) / 6.0
    query_weekday = query.delivery_date.weekday()
    candidate_weekday = np.array([item.delivery_date.weekday() for item in candidates])
    weekday_mismatch = (candidate_weekday != query_weekday).astype(float) * 0.5
    weekend_mismatch = (
        (candidate_weekday >= 5) != (query_weekday >= 5)
    ).astype(float) * 0.5
    return circular_month + weekday_mismatch + weekend_mismatch


def standardized_pattern(values: np.ndarray, standardizer: PriceStandardizer | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if standardizer is not None:
        array = standardizer.transform(array)
    mean = array.mean(axis=-1, keepdims=True)
    std = array.std(axis=-1, keepdims=True)
    return (array - mean) / np.maximum(std, 1e-6)


def pattern_distance(query_pattern: np.ndarray, candidate_patterns: np.ndarray) -> np.ndarray:
    difference = candidate_patterns - np.asarray(query_pattern, dtype=np.float32)
    return np.sqrt(np.mean(np.square(difference), axis=1))


@dataclass(frozen=True)
class SplitPlan:
    train: tuple[IssuanceWindow, ...]
    validation: tuple[IssuanceWindow, ...]

    def assert_no_label_overlap(self) -> None:
        if not self.train or not self.validation:
            raise ValueError("both chronological partitions must be non-empty")
        if max(item.target_stop for item in self.train) > min(item.target_start for item in self.validation):
            raise ValueError("training and validation labels overlap")
