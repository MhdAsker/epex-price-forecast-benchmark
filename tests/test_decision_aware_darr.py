from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.decision_aware_darr.causal import (
    SplitPlan,
    assert_candidate_eligible,
    eligible_candidate_indices,
    native_similarity_order,
    native_weighted_forecast,
    window_id,
)
from src.research.decision_aware_darr.dataset import validate_feature_label_separation
from src.research.decision_aware_darr.storage import (
    atomic_json,
    atomic_parquet,
    canonical_hash,
    output_record,
    verify_batch_manifest,
)
from src.research.kumo.data import KumoPriceDataset, PriceStandardizer, build_origins


ROOT = Path(__file__).resolve().parents[1]


def test_status_reports_verified_final_manifest_complete() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "run_decision_aware_darr.py", "status"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = json.loads(completed.stdout)
    assert status["status"] == "complete"
    assert status["embedding_batches_complete"] == status["embedding_batches_total"]
    assert status["candidate_batches_complete"] == status["candidate_batches_total"]


def _frame(days: int = 500) -> pd.DataFrame:
    index = pd.date_range("2012-01-01", periods=days * 24, freq="h")
    values = 40 + 10 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
    return pd.DataFrame({"price_eur_mwh": values}, index=index)


def test_future_candidate_is_rejected_before_query_issuance() -> None:
    frame = _frame()
    origins = build_origins(
        frame, target_start="2012-02-01", target_end_exclusive="2013-04-01"
    )
    query = next(item for item in origins if item.delivery_date == pd.Timestamp("2012-06-01"))
    illegal = next(item for item in origins if item.delivery_date == pd.Timestamp("2012-05-30"))
    with pytest.raises(ValueError, match="continuation has not completed"):
        assert_candidate_eligible(illegal, query, frame.index)
    eligible = eligible_candidate_indices(origins, query, frame.index)
    assert illegal not in [origins[position] for position in eligible]


def test_strict_candidate_boundary_accepts_only_physically_completed_continuation() -> None:
    frame = _frame()
    origins = build_origins(
        frame, target_start="2012-02-01", target_end_exclusive="2013-04-01"
    )
    query = next(item for item in origins if item.delivery_date == pd.Timestamp("2012-06-01"))
    accepted = next(item for item in origins if item.delivery_date == pd.Timestamp("2012-05-28"))
    assert_candidate_eligible(accepted, query, frame.index)


def test_chronological_split_has_no_72_step_label_overlap() -> None:
    frame = _frame(days=1500)
    train = build_origins(
        frame, target_start="2015-01-01", target_end_exclusive="2015-09-01"
    )
    validation = build_origins(
        frame, target_start="2015-09-01", target_end_exclusive="2016-01-01"
    )
    plan = SplitPlan(tuple(train), tuple(validation))
    plan.assert_no_label_overlap()
    assert train[-1].delivery_date == pd.Timestamp("2015-08-29")
    assert validation[0].delivery_date == pd.Timestamp("2015-09-01")


def test_native_similarity_and_weighted_forecast_are_deterministic() -> None:
    database = np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    continuations = np.array([[1, 2], [3, 4], [9, 9]], dtype=np.float32)
    first_order, first_similarity = native_similarity_order(database, query)
    second_order, second_similarity = native_similarity_order(database, query)
    np.testing.assert_array_equal(first_order, second_order)
    np.testing.assert_allclose(first_similarity, second_similarity)
    forecast, selected, weights = native_weighted_forecast(
        continuations, first_similarity, first_order, k=2, temperature=0.05
    )
    expected = np.sum(continuations[selected] * weights[:, None], axis=0)
    np.testing.assert_allclose(forecast, expected, rtol=0, atol=1e-7)
    assert weights.sum() == pytest.approx(1.0)


def test_feature_and_label_tables_are_explicitly_separated() -> None:
    features = pd.DataFrame({"query__id": ["q"], "pair__similarity": [0.9]})
    labels = pd.DataFrame({"query__id": ["q"], "label__mae": [1.0]})
    validate_feature_label_separation(features, labels)
    with pytest.raises(ValueError, match="ex-post labels"):
        validate_feature_label_separation(features.assign(label__future=1), labels)


def test_batch_manifest_requires_matching_hashes_and_atomic_output(tmp_path) -> None:
    root = tmp_path
    output = root / "artifact.parquet"
    atomic_parquet(output, pd.DataFrame({"x": [1, 2]}))
    config = {"version": 1}; config_hash = canonical_hash(config)
    manifest = root / "batch.json"
    atomic_json(manifest, {
        "status": "complete", "config_sha256": config_hash, "source_sha256": "source",
        "outputs": [output_record(output, root, 2)],
    })
    valid, reason = verify_batch_manifest(manifest, root, config_hash, "source")
    assert valid, reason
    valid, reason = verify_batch_manifest(manifest, root, "different", "source")
    assert not valid and "configuration hash conflict" in reason
    output.write_bytes(b"corrupt")
    valid, reason = verify_batch_manifest(manifest, root, config_hash, "source")
    assert not valid and ("size mismatch" in reason or "checksum mismatch" in reason)


def test_standardized_embedding_input_matches_dataset_normalization() -> None:
    frame = _frame()
    origins = build_origins(
        frame, target_start="2012-03-01", target_end_exclusive="2012-04-01"
    )
    standardizer = PriceStandardizer.fit(frame.iloc[: 60 * 24].price_eur_mwh.to_numpy())
    dataset = KumoPriceDataset(frame, origins, standardizer)
    x, _, _ = dataset[0]
    item = origins[0]
    expected = standardizer.transform(
        frame.price_eur_mwh.to_numpy()[item.context_start:item.context_stop]
    )
    np.testing.assert_allclose(x.numpy()[0], expected, rtol=0, atol=0)
    assert window_id(item) == window_id(item)
