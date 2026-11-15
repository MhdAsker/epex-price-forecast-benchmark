"""Execute the pre-registered extended EPEX Kumo head-only study.

Examples (from the repository root, in the pinned Kumo environment):
    python run_kumo_extended.py plan
    python run_kumo_extended.py validate --device cpu
    python run_kumo_extended.py retrospective --device cpu

The retrospective command refuses to run without a frozen 2015-only selection.
Training/checkpoints are never committed; metrics are not invented on failure.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.research.decision import BatterySpec, evaluate_dispatch
from src.research.evaluation import pairwise_tests
from src.research.kumo.data import (
    KumoPriceDataset, build_origins, origin_manifest, training_standardizer,
)
from src.research.kumo.evaluation import add_persistence, market_metrics, regime_metrics
from src.research.kumo.extended import (
    assert_identical_model_panel, choose_validation_candidate, config_digest, history_standardizer,
    ensure_new_immutable_selection, history_candidates, validation_seasons,
)
from src.research.kumo.inference import load_head, predict_origins
from src.research.kumo.train import (
    build_verified_model, load_official_standardizer, sha256_file, train_head,
)

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "config" / "kumo_extended_study.json"
OUT = ROOT / "results" / "kumo_finetuning" / "extended"
CHECKPOINTS = ROOT / "artifacts" / "kumo_finetuning" / "extended"
RAW = ROOT / "data" / "raw" / "epftoolbox_epex_de.parquet"
BASELINES = ("Kumo zero-shot", "Kumo fine-tuned", "Naive D-1",
             "LEAR", "JointCurve", "LightGBM", "XGBoost", "CatBoost")


def read_config() -> dict:
    config = json.loads(CFG.read_text(encoding="utf-8"))
    if config["status"] != "pre_registered_not_run":
        raise ValueError("Unexpected experiment configuration status")
    return config


def data_inputs(config: dict):
    if not RAW.exists():
        raise FileNotFoundError(
            "Historical EPEX price parquet is missing. Run the repo's existing "
            "fetch_epftoolbox_de acquisition; never fabricate prices."
        )
    frame = pd.read_parquet(RAW)
    d = config["data"]
    standardizer = training_standardizer(
        frame, train_start=d["training_target_start"],
        train_end_exclusive=d["training_target_end_exclusive"],
    )
    train = build_origins(
        frame, target_start=d["training_target_start"],
        target_end_exclusive=d["training_target_end_exclusive"],
    )
    validation = build_origins(
        frame, target_start=d["validation_target_start"],
        target_end_exclusive=d["validation_target_end_exclusive"],
    )
    histories = history_candidates(train, config["training"]["history_windows_days"])
    if not validation:
        raise ValueError("No issuance-safe 2015 validation origins")
    if any(x.delivery_date.year != 2015 for x in validation):
        raise AssertionError("Selection must use 2015 only")
    return frame, standardizer, train, validation, histories


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no supported GPU is visible to PyTorch")
    return "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"


def plan(config: dict):
    frame, _, train, validation, histories = data_inputs(config)
    summary = {
        "status": "PLAN_ONLY_NO_TRAINING",
        "config_digest": config_digest(config),
        "source_sha256": sha256_file(RAW),
        "training": origin_manifest(train, "train"),
        "validation": origin_manifest(validation, "validation"),
        "histories": [
            {"label": name, "n_windows": len(xs),
             "first_target": xs[0].delivery_date.date().isoformat(),
             "last_target": xs[-1].delivery_date.date().isoformat()}
            for name, xs in histories
        ],
        "source_rows": len(frame),
        "note": "2016-2017 data are reserved for retrospective reporting only; "
                "history all_available is not literally 36 months.",
    }
    print(json.dumps(summary, indent=2))


def train_and_select(config: dict, device: str) -> None:
    frame, _, train, val, histories = data_inputs(config)
    source_hash = sha256_file(RAW)
    cfg_hash = config_digest(config)
    tr = config["training"]
    expected = [label for label, _ in histories]
    if (OUT / "selection.json").exists():
        raise RuntimeError("Selection already frozen; do not silently retrain/change it")
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for label, subset in histories:
        candidate_dir = OUT / "candidates" / label
        checkpoint_dir = CHECKPOINTS / label
        standardizer = history_standardizer(frame, subset)
        metadata_path = candidate_dir / "record.json"
        checkpoint_path = checkpoint_dir / "best_head.pt"
        if metadata_path.exists():
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            if record["config_digest"] != cfg_hash or record["source_sha256"] != source_hash:
                raise RuntimeError(f"Stale candidate cache: {label}")
            if record["normalization"] != {"mean": standardizer.mean, "std": standardizer.std}:
                raise RuntimeError(f"Stale candidate cache: {label}")
            if not checkpoint_path.exists() or sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
                raise RuntimeError(f"Missing or changed candidate checkpoint: {label}")
            if not (candidate_dir / "validation_predictions.parquet").exists():
                raise RuntimeError(f"Missing validation outputs for {label}")
            if sha256_file(candidate_dir / "validation_predictions.parquet") != record["validation_predictions_sha256"]:
                raise RuntimeError(f"Changed validation prediction cache: {label}")
        else:
            model, load_report = build_verified_model(
                json.loads((ROOT / "config" / "kumo_finetuning.json").read_text(encoding="utf-8")),
                device=device,
            )
            training = train_head(
                model, KumoPriceDataset(frame, subset, standardizer),
                KumoPriceDataset(frame, val, standardizer),
                epochs=int(tr["epochs"]), learning_rate=float(tr["learning_rate"]),
                batch_size=int(tr["batch_size"]), seed=int(tr["seed"]),
                output_dir=checkpoint_dir,
            )
            val_panel = predict_origins(
                model, frame, val, standardizer,
                model_name="Kumo fine-tuned", batch_size=int(tr["batch_size"]),
            )
            if len(val_panel) != len(val) * 24:
                raise AssertionError(f"Validation coverage incomplete for {label}")
            candidate_dir.mkdir(parents=True, exist_ok=True)
            pred_path = candidate_dir / "validation_predictions.parquet"
            val_panel.to_parquet(pred_path, index=False)
            metrics = market_metrics(val_panel.assign(test_year=2015))
            seasons = validation_seasons(val_panel)
            seasons.to_csv(candidate_dir / "validation_seasons.csv", index=False)
            metrics.to_csv(candidate_dir / "validation_metrics.csv", index=False)
            score = float(metrics.iloc[0]["MAE"])
            record = {
                "history_label": label,
                "validation_operational_mae_eur_mwh": score,
                "best_epoch_selected_by_native_72_step_mae": training["best_epoch"],
                "native_72_step_validation_mae_eur_mwh": training["best_validation_mae_eur_mwh"],
                "train_windows": len(subset),
                "normalization": {"mean": standardizer.mean, "std": standardizer.std},
                "validation_windows": len(val),
                "first_train_target": subset[0].delivery_date.date().isoformat(),
                "last_train_target": subset[-1].delivery_date.date().isoformat(),
                "source_sha256": source_hash,
                "config_digest": cfg_hash,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "validation_predictions_sha256": sha256_file(pred_path),
                "training_seconds": training["runtime_seconds"],
                "device": device,
                "checkpoint_load_report": load_report.__dict__,
            }
            metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            del model
            gc.collect()
        records.append(record)
        print(f"Validated {label}: {record['validation_operational_mae_eur_mwh']:.4f} EUR/MWh", flush=True)

    chosen = choose_validation_candidate(records, expected)
    selection = {
        "status": "FROZEN_VALIDATION_SELECTED_NO_RETROSPECTIVE_OUTCOMES",
        "config_digest": cfg_hash, "source_sha256": source_hash,
        "history_candidates": records, "selected_history_label": chosen["history_label"],
        "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
        "selection_rule": "lowest eligible 2015 operational 24-product MAE",
        "within_candidate_epoch_selection": "native 72-step MAE, fixed before retrospective",
        "retrospective_not_untouched": True,
    }
    ensure_new_immutable_selection(OUT / "selection.json", selection)
    pd.DataFrame(records).sort_values("validation_operational_mae_eur_mwh").to_csv(
        OUT / "validation_comparison.csv", index=False
    )
    print("Frozen 2015-only selection:", chosen["history_label"], flush=True)


def retrospective(config: dict, device: str) -> None:
    selection_path = OUT / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("First complete the 2015 validation-only selection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["status"] != "FROZEN_VALIDATION_SELECTED_NO_RETROSPECTIVE_OUTCOMES":
        raise RuntimeError("Invalid selection state")
    if selection["config_digest"] != config_digest(config):
        raise RuntimeError("Config changed since selection freeze")
    if selection["source_sha256"] != sha256_file(RAW):
        raise RuntimeError("Data changed since selection freeze")
    result_dir = OUT / "retrospective"
    if result_dir.exists():
        raise RuntimeError("Retrospective results already exist; do not overwrite")
    selected = selection["selected_history_label"]
    checkpoint = CHECKPOINTS / selected / "best_head.pt"
    if not checkpoint.exists() or sha256_file(checkpoint) != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint missing or altered")
    frame, _, _, _, _ = data_inputs(config)
    from src.research.kumo.data import PriceStandardizer
    chosen = next(x for x in selection["history_candidates"]
                  if x["history_label"] == selected)
    standardizer = PriceStandardizer(**chosen["normalization"])
    d = config["data"]
    origins = build_origins(
        frame, target_start=d["retrospective_start"],
        target_end_exclusive=d["retrospective_end_exclusive"],
        require_full_target=False,
    )
    if not origins:
        raise RuntimeError("No eligible retrospective forecast origins")
    pinned = json.loads((ROOT / "config" / "kumo_finetuning.json").read_text(encoding="utf-8"))
    model, report = build_verified_model(pinned, device=device)
    zero = predict_origins(
        model, frame, origins, load_official_standardizer(pinned),
        model_name="Kumo zero-shot", batch_size=8,
    )
    del model
    gc.collect()
    model, _ = build_verified_model(pinned, device=device)
    load_head(model, checkpoint)
    fine = predict_origins(
        model, frame, origins, standardizer,
        model_name="Kumo fine-tuned", batch_size=8,
    )
    del model
    gc.collect()
    if len(zero) != len(origins)*24 or len(fine) != len(origins)*24:
        raise AssertionError("Incomplete Kumo retrospective product coverage")
    panel = add_persistence(frame, pd.concat([zero, fine], ignore_index=True))
    from run_kumo_finetuning import _add_frozen_conventional_models
    panel = _add_frozen_conventional_models(panel)
    assert_identical_model_panel(panel, list(BASELINES))
    panel["test_year"] = pd.to_datetime(panel.delivery_date).dt.year
    panel["experiment_scope"] = "frozen_historical_retrospective_not_untouched"
    result_dir.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(result_dir / "predictions.parquet", index=False)
    market_metrics(panel).to_csv(result_dir / "metrics.csv", index=False)
    regime_metrics(panel).to_csv(result_dir / "regimes.csv", index=False)
    pairwise_tests(panel.assign(fit_seconds=0.0), seed=13).to_csv(
        result_dir / "pairwise_tests.csv", index=False
    )
    daily, battery = evaluate_dispatch(panel, BatterySpec())
    daily.to_parquet(result_dir / "battery_dispatch_daily.parquet", index=False)
    battery.to_csv(result_dir / "battery_metrics.csv", index=False)
    (result_dir / "manifest.json").write_text(json.dumps({
        "status": "EXECUTED_RETROSPECTIVE",
        "selection": selection,
        "origins": origin_manifest(origins, "test"),
        "checkpoint_load_report": report.__dict__,
        "prediction_sha256": sha256_file(result_dir / "predictions.parquet"),
        "qualification": "Historical EPEX 2016-2017 outcomes already seen in earlier studies; retrospective only",
    }, indent=2) + "\n", encoding="utf-8")
    print("Retrospective results saved:", result_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "validate", "retrospective"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = read_config()
    if args.command == "plan":
        plan(config)
    elif args.command == "validate":
        train_and_select(config, resolve_device(args.device))
    else:
        retrospective(config, resolve_device(args.device))


if __name__ == "__main__":
    main()
