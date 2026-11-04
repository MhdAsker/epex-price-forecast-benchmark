"""Finalize metrics, inference, probabilistic forecasts, and manifest from a point panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.research.data import EPF_DE_MANIFEST_PATH, fetch_epftoolbox_de
from src.research.evaluation import pairwise_tests, point_metrics, probabilistic_metrics
from src.research.probabilistic import sequential_probabilistic_forecasts
from src.research.visualize import plot_dm_matrix, plot_point_results, plot_probabilistic_calibration

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "results" / "predictions"
METRICS = ROOT / "results" / "metrics"


def main() -> None:
    point_path = PREDICTIONS / "phase2_point_predictions.parquet"
    points = pd.read_parquet(point_path)
    point_table = point_metrics(points)
    point_table.to_csv(METRICS / "phase2_point_metrics.csv", index=False)
    tests = pd.concat(
        [pairwise_tests(group, seed=int(seed)).assign(seed=int(seed))
         for seed, group in points.groupby("seed")],
        ignore_index=True,
    )
    tests.to_csv(METRICS / "phase2_pairwise_tests.csv", index=False)
    quantile_frames = []
    for seed, group in points.groupby("seed"):
        print(f"Probabilistic calibration for seed {seed}", flush=True)
        quantile_frames.append(
            sequential_probabilistic_forecasts(group, refit_days=28).assign(seed=int(seed))
        )
    quantiles = pd.concat(quantile_frames, ignore_index=True)
    quantile_path = PREDICTIONS / "phase2_quantile_predictions.parquet"
    quantiles.to_parquet(quantile_path, index=False)
    probability_table = probabilistic_metrics(quantiles)
    probability_table.to_csv(METRICS / "phase2_probabilistic_metrics.csv", index=False)
    plot_point_results(point_table)
    plot_probabilistic_calibration(probability_table)
    plot_dm_matrix(tests)
    source = fetch_epftoolbox_de()
    source_manifest = json.loads(EPF_DE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(source, index=True).to_numpy().tobytes()
    ).hexdigest()
    manifest = {
        "experiment_version": "2.0.0",
        "status": "complete weekly-refit full panel",
        "source_manifest": source_manifest,
        "source_matrix_sha256": source_hash,
        "point_predictions": str(point_path.relative_to(ROOT)),
        "quantile_predictions": str(quantile_path.relative_to(ROOT)),
        "models": sorted(points["model"].unique().tolist()),
        "test_years": sorted(pd.to_datetime(points["delivery_date"]).dt.year.unique().tolist()),
        "seeds": sorted(points["seed"].unique().astype(int).tolist()),
        "calibration_days": sorted(points["calibration_days"].unique().astype(int).tolist()),
        "retrain_every_days": sorted(points["retrain_every_days"].unique().astype(int).tolist()),
        "qra_refit_days": 28,
        "online_probabilistic_burn_in_days": 60,
        "tuning_log": "results/predictions/phase2_seed42_r7_w1461_tuning.json",
    }
    (METRICS / "phase2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(pd.read_csv(METRICS / "phase2_point_metrics.csv").to_string(index=False))
    print(pd.read_csv(METRICS / "phase2_probabilistic_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
