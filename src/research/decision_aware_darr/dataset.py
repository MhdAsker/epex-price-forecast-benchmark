"""Feature/label construction for causal historical retrieval candidates."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research.decision import BatterySpec, optimize_day, realized_value
from src.research.decision_aware_darr.causal import (
    assert_candidate_eligible,
    calendar_distance,
    eligible_candidate_indices,
    native_similarity_order,
    pattern_distance,
    rank_positions,
    standardized_pattern,
    window_id,
)
from src.research.kumo.data import IssuanceWindow, PriceStandardizer, validate_canonical_hourly


def battery_spec_from_config(config: dict) -> BatterySpec:
    return BatterySpec(**config["battery"])


def context_descriptor(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    last24 = array[-24:]
    last168 = array[-168:]
    return {
        "last_price": float(array[-1]),
        "mean_24h": float(last24.mean()),
        "std_24h": float(last24.std()),
        "min_24h": float(last24.min()),
        "max_24h": float(last24.max()),
        "spread_24h": float(last24.max() - last24.min()),
        "mean_168h": float(last168.mean()),
        "std_168h": float(last168.std()),
        "min_168h": float(last168.min()),
        "max_168h": float(last168.max()),
        "spread_168h": float(last168.max() - last168.min()),
        "trend_24h": float(last24[-1] - last24[0]),
    }


def context_pattern_matrix(
    frame: pd.DataFrame,
    origins: Sequence[IssuanceWindow],
    standardizer: PriceStandardizer,
    hours: int,
) -> np.ndarray:
    values = validate_canonical_hourly(frame).price_eur_mwh.to_numpy(dtype=np.float32)
    raw = np.stack([values[item.context_stop - hours:item.context_stop] for item in origins])
    return standardized_pattern(raw, standardizer)


def prepare_candidate_library(
    frame: pd.DataFrame,
    origins: Sequence[IssuanceWindow],
    spec: BatterySpec,
) -> pd.DataFrame:
    ordered = validate_canonical_hourly(frame)
    values = ordered.price_eur_mwh.to_numpy(dtype=np.float32)
    rows = []
    for item in origins:
        context = values[item.context_start:item.context_stop]
        continuation = values[item.target_start:item.target_stop]
        charge, discharge, _ = optimize_day(continuation[:24], spec)
        descriptor = context_descriptor(context)
        row = {
            "candidate__id": window_id(item),
            "candidate__delivery_date": item.delivery_date.date().isoformat(),
            "candidate__forecast_origin": item.forecast_origin.isoformat(),
            "candidate__continuation_last_delivery": ordered.index[item.target_stop - 1].isoformat(),
            "candidate__continuation_24": continuation[:24].astype(float).tolist(),
            "candidate__continuation_72": continuation.astype(float).tolist(),
            "candidate__dispatch_charge_24": charge.astype(float).tolist(),
            "candidate__dispatch_discharge_24": discharge.astype(float).tolist(),
            "candidate__dispatch_throughput_mwh": float(np.sum(charge + discharge)),
        }
        row.update({f"candidate__context_{key}": value for key, value in descriptor.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def direct_prediction_lookup(points: pd.DataFrame, model_name: str) -> dict[str, np.ndarray]:
    selected = points[points.model == model_name].copy()
    selected["delivery_date"] = selected.delivery_date.astype(str)
    lookup = {}
    for date, group in selected.groupby("delivery_date"):
        ordered = group.sort_values("hour")
        if len(ordered) != 24:
            raise ValueError(f"direct prediction {date} does not contain 24 products")
        lookup[date] = ordered.prediction.to_numpy(dtype=np.float32)
    return lookup


def _rank_correlation(actual: np.ndarray, prediction: np.ndarray) -> float:
    value = float(spearmanr(actual, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def build_candidate_batch(
    *,
    frame: pd.DataFrame,
    library_origins: Sequence[IssuanceWindow],
    query_origins: Sequence[IssuanceWindow],
    embeddings: np.ndarray,
    patterns: np.ndarray,
    direct_predictions: dict[str, np.ndarray],
    candidate_library: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = validate_canonical_hourly(frame)
    prices = ordered.price_eur_mwh.to_numpy(dtype=np.float32)
    k = int(config["candidate_library"]["top_k_per_ordering"])
    temperature = float(config["candidate_library"]["darr_temperature"])
    spec = battery_spec_from_config(config)
    index_by_id = {window_id(item): position for position, item in enumerate(library_origins)}
    library_lookup = candidate_library.set_index("candidate__id")
    query_features: list[dict] = []
    query_labels: list[dict] = []
    candidate_features: list[dict] = []
    candidate_labels: list[dict] = []

    for query in query_origins:
        query_id = window_id(query)
        query_index = index_by_id[query_id]
        eligible = eligible_candidate_indices(library_origins, query, ordered.index)
        eligible_origins = [library_origins[position] for position in eligible]
        embedding_order_local, similarities = native_similarity_order(
            embeddings[eligible], embeddings[query_index]
        )
        embedding_ranks = rank_positions(similarities, descending=True)
        calendar_scores = calendar_distance(query, eligible_origins)
        calendar_ranks = rank_positions(calendar_scores, descending=False)
        pattern_scores = pattern_distance(patterns[query_index], patterns[eligible])
        pattern_ranks = rank_positions(pattern_scores, descending=False)
        embedding_top = embedding_order_local[:k]
        calendar_top = np.argsort(calendar_scores, kind="stable")[:k]
        pattern_top = np.argsort(pattern_scores, kind="stable")[:k]
        selected_local = np.unique(np.concatenate([embedding_top, calendar_top, pattern_top]))

        native_logits = similarities[embedding_top] / max(temperature, 1e-12)
        native_logits -= native_logits.max()
        native_weights = np.exp(native_logits); native_weights /= native_weights.sum()
        native_weight_lookup = {
            int(local): float(weight) for local, weight in zip(embedding_top, native_weights)
        }
        embedding_top_set = set(embedding_top.tolist())
        calendar_top_set = set(calendar_top.tolist())
        pattern_top_set = set(pattern_top.tolist())

        context = prices[query.context_start:query.context_stop]
        actual = prices[query.target_start:query.target_start + 24]
        date = query.delivery_date.date().isoformat()
        if date not in direct_predictions:
            raise KeyError(f"missing frozen direct forecast for {date}")
        direct = direct_predictions[date]
        direct_charge, direct_discharge, _ = optimize_day(direct, spec)
        direct_revenue = realized_value(actual, direct_charge, direct_discharge, spec)
        oracle_charge, oracle_discharge, _ = optimize_day(actual, spec)
        oracle_revenue = realized_value(actual, oracle_charge, oracle_discharge, spec)
        descriptor = context_descriptor(context)
        query_row = {
            "query__id": query_id,
            "partition": "train" if query.delivery_date < pd.Timestamp(config["queries"]["train_end_exclusive"]) else "validation",
            "query__delivery_date": date,
            "query__forecast_origin": query.forecast_origin.isoformat(),
            "query__month": int(query.delivery_date.month),
            "query__weekday": int(query.delivery_date.weekday()),
            "query__weekend": bool(query.delivery_date.weekday() >= 5),
            "query__direct_forecast_24": direct.astype(float).tolist(),
            "query__eligible_candidate_count": int(len(eligible)),
        }
        query_row.update({f"query__context_{key}": value for key, value in descriptor.items()})
        query_features.append(query_row)
        query_labels.append({
            "query__id": query_id,
            "label__actual_24": actual.astype(float).tolist(),
            "label__actual_spread_eur_mwh": float(actual.max() - actual.min()),
            "label__actual_std_eur_mwh": float(actual.std()),
            "label__negative_price_hours": int(np.sum(actual < 0)),
            "label__spike_hours_ge_100": int(np.sum(actual >= 100)),
            "label__direct_mae_eur_mwh": float(np.mean(np.abs(direct - actual))),
            "label__direct_revenue_eur": direct_revenue,
            "label__oracle_revenue_eur": oracle_revenue,
        })

        for local in selected_local:
            global_index = int(eligible[local])
            candidate = library_origins[global_index]
            assert_candidate_eligible(candidate, query, ordered.index)
            candidate_id = window_id(candidate)
            library_row = library_lookup.loc[candidate_id]
            continuation = np.asarray(library_row["candidate__continuation_24"], dtype=float)
            charge = np.asarray(library_row["candidate__dispatch_charge_24"], dtype=float)
            discharge = np.asarray(library_row["candidate__dispatch_discharge_24"], dtype=float)
            revenue = realized_value(actual, charge, discharge, spec)
            charge_den = max(float(oracle_charge.sum()), 1e-12)
            discharge_den = max(float(oracle_discharge.sum()), 1e-12)
            error = continuation - actual
            candidate_features.append({
                "query__id": query_id,
                "candidate__id": candidate_id,
                "pair__embedding_similarity": float(similarities[local]),
                "pair__embedding_rank": int(embedding_ranks[local]),
                "pair__calendar_distance": float(calendar_scores[local]),
                "pair__calendar_rank": int(calendar_ranks[local]),
                "pair__pattern_distance": float(pattern_scores[local]),
                "pair__pattern_rank": int(pattern_ranks[local]),
                "pair__selected_by_embedding_top64": bool(local in embedding_top_set),
                "pair__selected_by_calendar_top64": bool(local in calendar_top_set),
                "pair__selected_by_pattern_top64": bool(local in pattern_top_set),
                "pair__native_darr_weight": native_weight_lookup.get(int(local), 0.0),
                "pair__candidate_age_days": int((query.delivery_date - candidate.delivery_date).days),
                "source__checkpoint_sha256": config["model"]["head_checkpoint_sha256"],
            })
            candidate_labels.append({
                "query__id": query_id,
                "candidate__id": candidate_id,
                "label__continuation_mae_eur_mwh": float(np.mean(np.abs(error))),
                "label__continuation_rmse_eur_mwh": float(np.sqrt(np.mean(np.square(error)))),
                "label__continuation_bias_eur_mwh": float(np.mean(error)),
                "label__daily_rank_correlation": _rank_correlation(actual, continuation),
                "label__spread_error_eur_mwh": float(
                    (continuation.max() - continuation.min()) - (actual.max() - actual.min())
                ),
                "label__lowest_hour_correct": bool(np.argmin(continuation) == np.argmin(actual)),
                "label__highest_hour_correct": bool(np.argmax(continuation) == np.argmax(actual)),
                "label__charge_timing_overlap": float(np.minimum(charge, oracle_charge).sum() / charge_den),
                "label__discharge_timing_overlap": float(
                    np.minimum(discharge, oracle_discharge).sum() / discharge_den
                ),
                "label__candidate_revenue_eur": revenue,
                "label__candidate_throughput_mwh": float(np.sum(charge + discharge)),
                "label__candidate_regret_eur": oracle_revenue - revenue,
                "label__incremental_revenue_vs_direct_eur": revenue - direct_revenue,
            })
    return (
        pd.DataFrame(query_features),
        pd.DataFrame(query_labels),
        pd.DataFrame(candidate_features),
        pd.DataFrame(candidate_labels),
    )


def validate_feature_label_separation(features: pd.DataFrame, labels: pd.DataFrame) -> None:
    illegal_feature_columns = [column for column in features if column.startswith("label__")]
    illegal_label_columns = [
        column for column in labels
        if not column.startswith("label__") and column not in {"query__id", "candidate__id"}
    ]
    if illegal_feature_columns:
        raise ValueError(f"ex-post labels present in feature table: {illegal_feature_columns}")
    if illegal_label_columns:
        raise ValueError(f"non-label columns present in label table: {illegal_label_columns}")


def validate_unique_pairs(features: pd.DataFrame) -> None:
    if features.duplicated(["query__id", "candidate__id"]).any():
        raise ValueError("duplicate query/candidate pairs detected")
