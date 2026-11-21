from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_decision_aware_darr_phase2 as runner
from src.research.decision_aware_darr.forecast import retrieval_forecasts
from src.research.decision_aware_darr.reranker import (
    Experiment, FEATURE_SETS, build_pair_table, fit_experiment, load_checkpoint, predict, save_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "results" / "decision_aware_darr" / "phase1"


def test_phase1_inputs_verify_without_regeneration() -> None:
    manifest, digest = runner.verify_phase1()
    assert manifest["status"] == "complete"
    assert len(manifest["outputs"]) == 18
    assert len(digest) == 64


def test_feature_allowlist_is_explicit_and_label_free() -> None:
    table = build_pair_table(PHASE1)
    allowed = FEATURE_SETS["combined"]
    assert len(allowed) == len(set(allowed))
    assert all(feature in table for feature in allowed)
    assert not any("label__" in feature for feature in allowed)
    assert not any(token in feature.lower() for feature in allowed for token in ("actual", "future", "revenue", "regime"))


def test_inner_folds_are_chronological_and_purged() -> None:
    config, _ = runner.load_config()
    table = build_pair_table(PHASE1)
    for fold in config["inner_folds"]:
        fit, score = runner.fold_masks(table, fold)
        fit_dates = pd.to_datetime(table.loc[fit, "query__delivery_date"])
        score_dates = pd.to_datetime(table.loc[score, "query__delivery_date"])
        assert fit_dates.max() < score_dates.min() - pd.Timedelta(days=2)
        assert set(table.loc[fit, "query__id"]).isdisjoint(table.loc[score, "query__id"])


def test_ridge_is_deterministic_and_checkpoint_reload_consistent(tmp_path: Path) -> None:
    table = build_pair_table(PHASE1)
    sample_queries = sorted(table[table.partition == "train"].query__id.unique())[:20]
    sample = table[table.query__id.isin(sample_queries)]
    experiment = Experiment("test", "ridge", "forecast", "combined", {"alpha": 100.0})
    first = fit_experiment(experiment, sample, seed=13)
    second = fit_experiment(experiment, sample, seed=13)
    expected = predict(first, experiment, sample)
    np.testing.assert_allclose(expected, predict(second, experiment, sample), rtol=0, atol=1e-12)
    checkpoint = tmp_path / "model.joblib"
    save_checkpoint(checkpoint, first, experiment, {"test": True})
    reloaded = load_checkpoint(checkpoint)
    np.testing.assert_allclose(expected, predict(reloaded["model"], reloaded["experiment"], sample), rtol=0, atol=1e-12)


def test_retrieval_blending_is_aligned_and_finite() -> None:
    query = pd.read_parquet(PHASE1 / "query_features.parquet").head(2)
    pairs = pd.read_parquet(PHASE1 / "candidate_features.parquet")
    pairs = pairs[pairs.query__id.isin(query.query__id)].copy()
    pairs["score"] = -pairs.pair__embedding_rank.to_numpy(float)
    library = pd.read_parquet(PHASE1 / "candidate_library.parquet")
    forecasts = retrieval_forecasts(pairs, library, query, score_column="score", k=4, direct_weight=0.5)
    assert set(forecasts) == set(query.query__id)
    assert all(value.shape == (24,) and np.isfinite(value).all() for value in forecasts.values())


def test_validation_queries_never_enter_model_selection_folds() -> None:
    config, _ = runner.load_config()
    table = build_pair_table(PHASE1)
    validation = set(table[table.partition == "validation"].query__id)
    for fold in config["inner_folds"]:
        fit, score = runner.fold_masks(table, fold)
        assert validation.isdisjoint(table.loc[fit | score, "query__id"])


def test_phase2_status_verifies_experiments_and_final_outputs(capsys) -> None:
    runner.run_status()
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "complete"
    assert status["experiments_complete"] == status["experiments_total"] == 28
    assert status["final_manifest_verified"] is True
