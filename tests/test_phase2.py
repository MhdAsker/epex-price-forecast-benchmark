from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.evaluation import (  # noqa: E402
    diebold_mariano,
    moving_block_ci,
    point_metrics,
    probabilistic_metrics,
)
from src.research.features import build_daily_dataset  # noqa: E402
from src.research.models import JointCurveModel, LEARModel  # noqa: E402
from src.research.probabilistic import sequential_probabilistic_forecasts  # noqa: E402


def _hourly(days: int = 30) -> pd.DataFrame:
    index = pd.date_range("2012-01-01", periods=days * 24, freq="h")
    day = np.repeat(np.arange(days), 24)
    hour = np.tile(np.arange(24), days)
    price = 20 + day + 5 * np.sin(2 * np.pi * hour / 24)
    return pd.DataFrame({"price_eur_mwh": price}, index=index)


def test_daily_features_use_only_prior_curves() -> None:
    raw = _hourly()
    original = build_daily_dataset(raw)
    changed = raw.copy()
    target_day = pd.Timestamp("2012-01-20")
    changed.loc[changed.index.normalize() == target_day, "price_eur_mwh"] += 1000
    rebuilt = build_daily_dataset(changed)
    row = np.flatnonzero(original.dates == target_day)[0]
    assert np.allclose(original.X[row], rebuilt.X[row])
    assert not np.allclose(original.y[row], rebuilt.y[row])
    assert not np.allclose(original.X[row + 1], rebuilt.X[row + 1])


def test_daily_models_return_complete_curve() -> None:
    data = build_daily_dataset(_hourly(80))
    for model in (LEARModel(alpha=0.1), JointCurveModel(alpha=0.05)):
        model.fit(data.X[:-5], data.y[:-5])
        prediction = model.predict(data.X[-5:])
        assert prediction.shape == (5, 24)
        assert np.isfinite(prediction).all()


def _point_predictions(days: int = 75) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(4)
    for day in pd.date_range("2016-01-01", periods=days, freq="D"):
        for hour in range(24):
            actual = 30 + 5 * np.sin(2 * np.pi * hour / 24) + rng.normal()
            for model, offset in (("Naive D-1", 3.0), ("LightGBM", 0.5), ("LEAR", -0.2)):
                rows.append(
                    {
                        "delivery_date": day.date().isoformat(), "hour": hour,
                        "actual": actual, "prediction": actual + offset + rng.normal(scale=0.5),
                        "model": model, "seed": 42,
                    }
                )
    return pd.DataFrame(rows)


def test_point_and_probabilistic_metrics_are_finite() -> None:
    points = _point_predictions()
    point = point_metrics(points)
    assert set(point["model"]) == {"Naive D-1", "LightGBM", "LEAR"}
    assert np.isfinite(point["MAE"]).all()
    quantiles = sequential_probabilistic_forecasts(points, min_days=60, refit_days=7)
    assert not quantiles.empty
    assert set(quantiles["method"]) == {"QRA", "Conformal-LightGBM"}
    scores = probabilistic_metrics(quantiles.assign(seed=42))
    assert np.isfinite(scores["CRPS_approx"]).all()
    assert ((scores["coverage_90"] >= 0) & (scores["coverage_90"] <= 1)).all()


def test_dependence_aware_statistics() -> None:
    rng = np.random.default_rng(9)
    differential = rng.normal(loc=-1.0, scale=0.5, size=365)
    low, high = moving_block_ci(differential, replications=300)
    statistic, pvalue = diebold_mariano(differential)
    assert low < high < 0
    assert statistic < 0
    assert pvalue < 0.05
