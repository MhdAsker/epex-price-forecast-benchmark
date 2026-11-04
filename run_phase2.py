"""Run the Phase 2 rolling-origin point and probabilistic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.research.ablations import run_ablation_suite
from src.research.backtest import BacktestSpec, run_point_backtest, save_backtest
from src.research.data import EPF_DE_MANIFEST_PATH, fetch_epftoolbox_de
from src.research.evaluation import pairwise_tests, point_metrics, probabilistic_metrics
from src.research.features import build_daily_dataset
from src.research.probabilistic import sequential_probabilistic_forecasts

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "results" / "metrics"
PREDICTIONS = ROOT / "results" / "predictions"
DEFAULT_MODELS = ("LEAR", "LightGBM", "XGBoost", "CatBoost", "JointCurve", "JointMLP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--test-years", nargs="+", type=int, default=[2016, 2017])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 91])
    parser.add_argument("--calibration-days", type=int, default=1461)
    parser.add_argument("--validation-days", type=int, default=182)
    parser.add_argument("--retrain-every-days", type=int, default=1)
    parser.add_argument("--hyperparameter-refresh-days", type=int, default=28)
    parser.add_argument("--max-test-days", type=int)
    parser.add_argument("--ablations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = fetch_epftoolbox_de()
    data = build_daily_dataset(frame)
    METRICS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    all_points: list[pd.DataFrame] = []
    tuning_paths: list[str] = []
    for seed in args.seeds:
        spec = BacktestSpec(
            test_years=tuple(args.test_years),
            calibration_days=args.calibration_days,
            validation_days=args.validation_days,
            retrain_every_days=args.retrain_every_days,
            hyperparameter_refresh_days=args.hyperparameter_refresh_days,
            seed=seed,
        )
        points, tuning = run_point_backtest(
            data, tuple(args.models), spec, max_test_days=args.max_test_days
        )
        label = f"phase2_seed{seed}_r{args.retrain_every_days}_w{args.calibration_days}"
        pred_path, tune_path = save_backtest(points, tuning, label)
        print(f"Saved {pred_path}")
        tuning_paths.append(str(tune_path.relative_to(ROOT)))
        all_points.append(points)
    combined = pd.concat(all_points, ignore_index=True)
    combined_path = PREDICTIONS / "phase2_point_predictions.parquet"
    combined.to_parquet(combined_path, index=False)
    point_table = point_metrics(combined)
    point_table.to_csv(METRICS / "phase2_point_metrics.csv", index=False)

    # Statistical tests are seed-specific so repeated stochastic forecasts are
    # never treated as independent delivery days.
    tests = pd.concat(
        [pairwise_tests(group, seed=int(seed)).assign(seed=int(seed))
         for seed, group in combined.groupby("seed")],
        ignore_index=True,
    )
    tests.to_csv(METRICS / "phase2_pairwise_tests.csv", index=False)

    probabilistic_frames: list[pd.DataFrame] = []
    if "LightGBM" in args.models and (args.max_test_days is None or args.max_test_days > 60):
        for seed, group in combined.groupby("seed"):
            probabilistic_frames.append(
                sequential_probabilistic_forecasts(group).assign(seed=int(seed))
            )
    if probabilistic_frames:
        quantiles = pd.concat(probabilistic_frames, ignore_index=True)
        quantiles.to_parquet(PREDICTIONS / "phase2_quantile_predictions.parquet", index=False)
        probabilistic_metrics(quantiles).to_csv(
            METRICS / "phase2_probabilistic_metrics.csv", index=False
        )

    ablation_path = None
    if args.ablations:
        ablations = run_ablation_suite(
            data,
            base_spec=BacktestSpec(test_years=tuple(args.test_years), seed=args.seeds[0]),
            max_test_days=args.max_test_days,
        )
        ablation_path = PREDICTIONS / "phase2_ablations.parquet"
        ablations.to_parquet(ablation_path, index=False)

    source_manifest = json.loads(EPF_DE_MANIFEST_PATH.read_text(encoding="utf-8"))
    matrix_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    ).hexdigest()
    manifest = {
        "experiment_version": "2.0.0",
        "source_manifest": source_manifest,
        "matrix_sha256": matrix_hash,
        "models": args.models,
        "test_years": args.test_years,
        "seeds": args.seeds,
        "calibration_days": args.calibration_days,
        "validation_days": args.validation_days,
        "retrain_every_days": args.retrain_every_days,
        "hyperparameter_refresh_days": args.hyperparameter_refresh_days,
        "max_test_days": args.max_test_days,
        "point_predictions": str(combined_path.relative_to(ROOT)),
        "tuning_logs": tuning_paths,
        "ablations": None if ablation_path is None else str(ablation_path.relative_to(ROOT)),
    }
    (METRICS / "phase2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(point_table.to_string(index=False))


if __name__ == "__main__":
    main()
