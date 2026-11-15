"""Small zero-compute guardrail tests for the extended Kumo experiment."""
import numpy as np
import pandas as pd
import pytest

from src.research.kumo.extended import (
    assert_identical_model_panel,
    choose_validation_candidate,
    config_digest,
    history_standardizer,
    history_candidates,
    validation_seasons,
)


def test_history_is_strictly_chronological_and_bounded():
    origins = list(range(1000))
    result = history_candidates(origins, [365, 730, "all_available"])
    assert [name for name, _ in result] == ["365", "730", "all_available"]
    assert result[0][1][0] == 635
    assert result[-1][1][-1] == 999
    with pytest.raises(ValueError):
        history_candidates(origins, [1065])


def test_selection_rejects_test_outcomes_and_missing_candidates():
    rows = [
        {"history_label": "365", "validation_operational_mae_eur_mwh": 7.3},
        {"history_label": "730", "validation_operational_mae_eur_mwh": 7.1},
    ]
    assert choose_validation_candidate(rows, ["365", "730"])["history_label"] == "730"
    with pytest.raises(ValueError):
        choose_validation_candidate(rows[:1], ["365", "730"])
    with pytest.raises(ValueError):
        choose_validation_candidate([{**rows[0], "test_mae": 1}], ["365"])


def test_panel_requires_common_targets_and_timestamps():
    a = pd.DataFrame({
        "delivery_date": ["2016-01-01"] * 2, "hour": [0, 1],
        "actual": [10.0, 12.0], "prediction": [9.5, 12.5], "model": ["A", "A"],
    })
    b = a.copy()
    b["model"] = "B"
    assert_identical_model_panel(pd.concat([a, b]), ["A", "B"])
    b.loc[1, "actual"] = 14.0
    with pytest.raises(ValueError):
        assert_identical_model_panel(pd.concat([a, b]), ["A", "B"])


def test_seasons_and_config_are_deterministic():
    p = pd.DataFrame({
        "delivery_date": ["2015-01-01", "2015-07-01"],
        "hour": [0, 0], "model": ["A", "A"], "actual": [1., 2.], "prediction": [2., 1.],
    })
    seasons = validation_seasons(p)
    assert set(seasons.season) == {"winter", "summer"}
    assert np.allclose(seasons.mae_eur_mwh, 1.)
    assert config_digest({"b": 2, "a": 1}) == config_digest({"a": 1, "b": 2})


def test_training_history_scaler_is_candidate_specific():
    from types import SimpleNamespace
    index = pd.date_range("2014-01-01", periods=150, freq="h")
    frame = pd.DataFrame({"price_eur_mwh": np.r_[np.zeros(75), np.full(75, 100.0)]}, index=index)
    older = [SimpleNamespace(context_start=0, target_stop=75)]
    newer = [SimpleNamespace(context_start=75, target_stop=150)]
    assert history_standardizer(frame, older).mean == 0.0
    assert history_standardizer(frame, newer).mean == 100.0
