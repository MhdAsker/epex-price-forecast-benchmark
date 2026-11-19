"""Run normalization-controlled Kumo domain-adaptation experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch  # Load pinned Windows DLLs before NumPy/Pandas/SciPy.
import numpy as np
import pandas as pd

from src.research.decision import BatterySpec
from src.research.evaluation import pairwise_tests
from src.research.kumo.ablation import (
    audit_standardizer,
    extreme_metrics,
    factorial_effects,
    sha256_file,
)
from src.research.kumo.data import KumoPriceDataset, build_origins, origin_manifest, training_standardizer
from src.research.kumo.darr_analysis import (
    daily_darr_diagnostics,
    neighbor_quality_diagnostics,
    reconstruct_darr_components,
    revenue_gain_concentration,
    summarize_darr,
)
from src.research.kumo.diagnostics import resource_report, write_json
from src.research.kumo.economic import (
    dispatch_diagnostics,
    pairwise_revenue_intervals,
    summarize_dispatch_daily,
)
from src.research.kumo.evaluation import add_persistence, market_metrics, regime_metrics
from src.research.kumo.inference import load_head, predict_origins
from src.research.kumo.train import build_verified_model, head_state, load_official_standardizer, train_head


ROOT = Path(__file__).resolve().parent
MAIN_CONFIG = ROOT / "config" / "kumo_finetuning.json"
ABLATION_CONFIG = ROOT / "config" / "kumo_adaptation_ablation.json"
FRAME_PATH = ROOT / "data" / "raw" / "epftoolbox_epex_de.parquet"
OUT = ROOT / "results" / "kumo_adaptation_ablation" / "phase1"
ARTIFACTS = ROOT / "artifacts" / "kumo_adaptation_ablation" / "phase1_official_scaler"
PHASE2_OUT = ROOT / "results" / "kumo_adaptation_ablation" / "phase2"
PHASE2_ARTIFACTS = ROOT / "artifacts" / "kumo_adaptation_ablation" / "phase2"
PHASE3_OUT = ROOT / "results" / "kumo_adaptation_ablation" / "phase3"
PHASE3_ARTIFACTS = ROOT / "artifacts" / "kumo_adaptation_ablation" / "phase3"
SELECTION_PATH = ROOT / "results" / "kumo_adaptation_ablation" / "selection.json"
DARR_OUT = ROOT / "results" / "kumo_adaptation_ablation" / "darr_analysis"


def _configs() -> tuple[dict, dict]:
    return (
        json.loads(MAIN_CONFIG.read_text(encoding="utf-8")),
        json.loads(ABLATION_CONFIG.read_text(encoding="utf-8")),
    )


def _verify_registered_hash(path: Path, expected: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"hash mismatch for {path}: {observed} != {expected}")
    return observed


def _inputs(main: dict, registered: dict):
    from src.research.kumo.train import resolve_pinned_paths

    paths = resolve_pinned_paths(main)
    hashes = registered["hashes"]
    _verify_registered_hash(FRAME_PATH, hashes["source_dataset_sha256"])
    _verify_registered_hash(paths.checkpoint, hashes["nvidia_checkpoint_sha256"])
    _verify_registered_hash(paths.standardizer, hashes["nvidia_standardizer_sha256"])
    existing_head = ROOT / "artifacts" / "kumo_finetuning" / "extended_365_e10" / "best_head.pt"
    _verify_registered_hash(existing_head, hashes["existing_german_head_sha256"])
    frame = pd.read_parquet(FRAME_PATH)
    german = training_standardizer(
        frame, train_start=main["data"]["train_target_start"],
        train_end_exclusive=main["data"]["train_target_end_exclusive"],
    )
    nvidia = load_official_standardizer(main)
    train = build_origins(
        frame, target_start=main["data"]["train_target_start"],
        target_end_exclusive=main["data"]["train_target_end_exclusive"],
    )
    validation = build_origins(
        frame, target_start=main["data"]["validation_target_start"],
        target_end_exclusive=main["data"]["validation_target_end_exclusive"],
    )
    return frame, german, nvidia, train, validation, existing_head, paths


def run_audit() -> None:
    main, registered = _configs()
    frame, german, nvidia, train, validation, existing_head, paths = _inputs(main, registered)
    model_1, report_1 = build_verified_model(main, device="cpu")
    model_2, report_2 = build_verified_model(main, device="cpu")
    first = validation[:1]
    pred_1 = predict_origins(model_1, frame, first, nvidia, model_name="load_1")
    pred_2 = predict_origins(model_2, frame, first, nvidia, model_name="load_2")
    head_1, head_2 = head_state(model_1), head_state(model_2)
    audit = {
        "status": "complete normalization and checkpoint audit before phase1 outcomes",
        "source": {"path": str(FRAME_PATH), "sha256": sha256_file(FRAME_PATH), "rows": len(frame)},
        "nvidia_checkpoint": {"path": str(paths.checkpoint), "sha256": sha256_file(paths.checkpoint)},
        "existing_german_head": {"path": str(existing_head), "sha256": sha256_file(existing_head)},
        "standardizers": [
            audit_standardizer(
                frame, nvidia, name="NVIDIA published", fit_start=None, fit_end_exclusive=None,
            ),
            audit_standardizer(
                frame, german, name="German training-only",
                fit_start=main["data"]["train_target_start"],
                fit_end_exclusive=main["data"]["train_target_end_exclusive"],
            ),
        ],
        "load_report_1": asdict(report_1), "load_report_2": asdict(report_2),
        "repeat_load_max_abs_head_difference": max(
            float(torch.max(torch.abs(head_1[key] - head_2[key]))) for key in head_1
        ),
        "repeat_load_max_abs_prediction_difference_eur_mwh": float(
            np.max(np.abs(pred_1.prediction.to_numpy() - pred_2.prediction.to_numpy()))
        ),
        "splits": [origin_manifest(train, "train"), origin_manifest(validation, "validation")],
        "no_validation_or_test_in_german_scaler": True,
        "resources": resource_report(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


def _cell_predictions(
    main: dict, registered: dict, frame: pd.DataFrame, german, nvidia,
    all_train, validation, existing_head: Path,
) -> tuple[pd.DataFrame, dict]:
    phase = registered["phase1"]
    train = all_train[-int(phase["training_history_days"]):]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained, report_a = build_verified_model(main, device=device)
    a = predict_origins(
        pretrained, frame, validation, nvidia,
        model_name="A pretrained + NVIDIA normalization", batch_size=int(phase["batch_size"]),
    )
    b = predict_origins(
        pretrained, frame, validation, german,
        model_name="B pretrained + German normalization", batch_size=int(phase["batch_size"]),
    )

    fine_german, report_c = build_verified_model(main, device=device)
    load_head(fine_german, existing_head)
    c = predict_origins(
        fine_german, frame, validation, german,
        model_name="C fine-tuned + German normalization", batch_size=int(phase["batch_size"]),
    )

    fine_nvidia, report_d = build_verified_model(main, device=device)
    metadata_path = ARTIFACTS / "training_metadata.json"
    if metadata_path.exists():
        training_d = json.loads(metadata_path.read_text(encoding="utf-8"))
        load_head(fine_nvidia, ARTIFACTS / "best_head.pt")
    else:
        training_d = train_head(
            fine_nvidia, KumoPriceDataset(frame, train, nvidia),
            KumoPriceDataset(frame, validation, nvidia),
            epochs=int(phase["epochs"]), learning_rate=float(phase["learning_rate"]),
            batch_size=int(phase["batch_size"]), seed=int(registered["protocol"]["seed"]),
            output_dir=ARTIFACTS, early_stopping_patience=int(phase["early_stopping_patience"]),
            resume=True,
        )
    d = predict_origins(
        fine_nvidia, frame, validation, nvidia,
        model_name="D fine-tuned + NVIDIA normalization", batch_size=int(phase["batch_size"]),
    )
    panel = pd.concat([a, b, c, d], ignore_index=True)
    panel = add_persistence(frame, panel)
    panel["test_year"] = 2015
    metadata = {
        "device": device, "train": origin_manifest(train, "train"),
        "validation": origin_manifest(validation, "validation"),
        "load_reports": {"A_B": asdict(report_a), "C": asdict(report_c), "D": asdict(report_d)},
        "D_training": training_d,
        "C_checkpoint_sha256": sha256_file(existing_head),
        "D_checkpoint_sha256": sha256_file(ARTIFACTS / "best_head.pt"),
    }
    return panel, metadata


def run_phase1() -> None:
    main, registered = _configs()
    frame, german, nvidia, train, validation, existing_head, _ = _inputs(main, registered)
    if not (OUT / "audit.json").exists():
        run_audit()
    panel, metadata = _cell_predictions(
        main, registered, frame, german, nvidia, train, validation, existing_head,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "predictions.parquet"
    panel.to_parquet(prediction_path, index=False)
    metrics = market_metrics(panel)
    cell_map = {
        "A pretrained + NVIDIA normalization": "A",
        "B pretrained + German normalization": "B",
        "C fine-tuned + German normalization": "C",
        "D fine-tuned + NVIDIA normalization": "D",
    }
    metrics["cell"] = metrics.model.map(cell_map)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    factor_metrics = metrics.dropna(subset=["cell"]).copy()
    factorial_effects(factor_metrics).to_csv(OUT / "factorial_effects.csv", index=False)
    pairwise_tests(panel, seed=int(registered["protocol"]["seed"])).to_csv(
        OUT / "pairwise_tests.csv", index=False,
    )
    regime_metrics(panel).to_csv(OUT / "regime_metrics.csv", index=False)
    training_prices = frame.loc[
        (frame.index >= pd.Timestamp(main["data"]["train_target_start"]))
        & (frame.index < pd.Timestamp(main["data"]["train_target_end_exclusive"])),
        "price_eur_mwh",
    ].to_numpy()
    extreme_metrics(panel, training_prices).to_csv(OUT / "extreme_metrics.csv", index=False)
    battery_daily, battery_summary = dispatch_diagnostics(panel, BatterySpec())
    battery_daily.to_parquet(OUT / "battery_daily.parquet", index=False)
    battery_summary.to_csv(OUT / "battery_metrics.csv", index=False)
    pairwise_revenue_intervals(battery_daily).to_csv(OUT / "battery_pairwise_intervals.csv", index=False)

    history_c = pd.read_csv(ROOT / "results" / "kumo_finetuning" / "extended" / "training_history.csv")
    history_d = pd.DataFrame(metadata["D_training"]["history"])
    history_c.assign(cell="C").to_csv(OUT / "training_history_C.csv", index=False)
    history_d.assign(cell="D").to_csv(OUT / "training_history_D.csv", index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ordered = factor_metrics.set_index("cell").loc[list("ABCD")]
    axes[0].bar(ordered.index, ordered.MAE)
    axes[0].set(xlabel="Registered cell", ylabel="2015 operational MAE (EUR/MWh)")
    axes[1].plot(history_c.epoch, history_c.validation_mae_eur_mwh, marker="o", label="C German")
    axes[1].plot(history_d.epoch, history_d.validation_mae_eur_mwh, marker="o", label="D NVIDIA")
    axes[1].set(xlabel="Epoch", ylabel="72-step validation MAE (EUR/MWh)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "normalization_ablation.png", dpi=180); plt.close(fig)

    shutil.copy2(ABLATION_CONFIG, OUT / "config_frozen.json")
    artifacts = []
    for path in sorted(OUT.glob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({
                "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    write_json(OUT / "manifest.json", {
        "status": "complete 2015-only normalization-controlled phase1",
        "registration": registered, "metadata": metadata,
        "source_dataset_sha256": sha256_file(FRAME_PATH), "artifacts": artifacts,
        "test_outcomes_used_for_selection": False,
    })
    print(metrics.to_string(index=False))
    print(factorial_effects(factor_metrics).to_string(index=False))
    print(battery_summary.to_string(index=False))


def _phase2_candidate(
    *, name: str, frame: pd.DataFrame, origins, validation, standardizer,
    main: dict, registered: dict, epochs: int, max_optimizer_steps: int | None,
) -> tuple[pd.DataFrame, dict]:
    output_dir = PHASE2_ARTIFACTS / name
    metadata_path = output_dir / "training_metadata.json"
    model, load_report = build_verified_model(main, device="cpu")
    if metadata_path.exists():
        training = json.loads(metadata_path.read_text(encoding="utf-8"))
        load_head(model, output_dir / "best_head.pt")
    else:
        training = train_head(
            model, KumoPriceDataset(frame, origins, standardizer),
            KumoPriceDataset(frame, validation, standardizer),
            epochs=epochs, learning_rate=float(registered["phase2"]["learning_rate"]),
            batch_size=int(registered["phase2"]["batch_size"]),
            seed=int(registered["protocol"]["seed"]), output_dir=output_dir,
            resume=True, max_optimizer_steps=max_optimizer_steps,
        )
    prediction = predict_origins(
        model, frame, validation, standardizer, model_name=name,
        batch_size=int(registered["phase2"]["batch_size"]),
    )
    record = {
        "candidate": name, "train_windows": len(origins),
        "first_training_target": origins[0].delivery_date.date().isoformat(),
        "last_training_target": origins[-1].delivery_date.date().isoformat(),
        "epochs_requested": epochs, "optimizer_step_cap": max_optimizer_steps,
        "training": training, "load_report": asdict(load_report),
        "checkpoint_sha256": sha256_file(output_dir / "best_head.pt"),
    }
    return prediction, record


def run_phase2() -> None:
    main, registered = _configs()
    frame, german, _, all_train, validation, existing_head, _ = _inputs(main, registered)
    phase = registered["phase2"]
    histories = {"365": all_train[-365:], "730": all_train[-730:], "1063_all": all_train}
    predictions = []
    records = []

    # The established German checkpoint's selected epoch is exactly epoch 6,
    # corresponding to 276 optimizer steps. Later epochs did not modify the
    # saved epoch-6 state, so it is reusable for both 365-window budgets.
    established_model, established_report = build_verified_model(main, device="cpu")
    load_head(established_model, existing_head)
    established_prediction = predict_origins(
        established_model, frame, validation, german,
        model_name="fixed_epochs_365", batch_size=int(phase["batch_size"]),
    )
    established_metadata = json.loads(
        (ROOT / "artifacts" / "kumo_finetuning" / "extended_365_e10" / "training_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    for name in ("fixed_epochs_365", "matched_steps_365"):
        copy = established_prediction.copy(); copy["model"] = name; predictions.append(copy)
        records.append({
            "candidate": name, "train_windows": 365,
            "first_training_target": histories["365"][0].delivery_date.date().isoformat(),
            "last_training_target": histories["365"][-1].delivery_date.date().isoformat(),
            "training": established_metadata, "load_report": asdict(established_report),
            "checkpoint_sha256": sha256_file(existing_head),
            "reuse_qualification": "selected epoch-6 state equals fixed-six-epoch/276-step endpoint",
        })

    for history_name in ("730", "1063_all"):
        pred, record = _phase2_candidate(
            name=f"fixed_epochs_{history_name}", frame=frame, origins=histories[history_name],
            validation=validation, standardizer=german, main=main, registered=registered,
            epochs=int(phase["fixed_epoch_budget"]), max_optimizer_steps=None,
        )
        predictions.append(pred); records.append(record)
    matched_steps = int(phase["matched_optimizer_steps"])
    for history_name in ("730", "1063_all"):
        pred, record = _phase2_candidate(
            name=f"matched_steps_{history_name}", frame=frame, origins=histories[history_name],
            validation=validation, standardizer=german, main=main, registered=registered,
            epochs=10, max_optimizer_steps=matched_steps,
        )
        predictions.append(pred); records.append(record)

    panel = pd.concat(predictions, ignore_index=True)
    panel = add_persistence(frame, panel); panel["test_year"] = 2015
    PHASE2_OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PHASE2_OUT / "predictions.parquet", index=False)
    metrics = market_metrics(panel); metrics.to_csv(PHASE2_OUT / "metrics.csv", index=False)
    pairwise_tests(panel, seed=int(registered["protocol"]["seed"])).to_csv(
        PHASE2_OUT / "pairwise_tests.csv", index=False,
    )
    regime_metrics(panel).to_csv(PHASE2_OUT / "regime_metrics.csv", index=False)
    daily, battery = dispatch_diagnostics(panel, BatterySpec())
    daily.to_parquet(PHASE2_OUT / "battery_daily.parquet", index=False)
    battery.to_csv(PHASE2_OUT / "battery_metrics.csv", index=False)
    pairwise_revenue_intervals(daily).to_csv(PHASE2_OUT / "battery_pairwise_intervals.csv", index=False)
    summary_rows = []
    metric_lookup = metrics.set_index("model")
    battery_lookup = battery.set_index("model")
    for record in records:
        name = record["candidate"]
        training = record["training"]
        summary_rows.append({
            "candidate": name, "train_windows": record["train_windows"],
            "best_epoch": training["best_epoch"], "optimizer_steps": training.get("optimizer_steps", 276),
            "validation_72step_MAE": training["best_validation_mae_eur_mwh"],
            "operational_24step_MAE": metric_lookup.loc[name, "MAE"],
            "RMSE": metric_lookup.loc[name, "RMSE"], "bias": metric_lookup.loc[name, "bias"],
            "daily_rank_correlation": metric_lookup.loc[name, "daily_rank_correlation"],
            "negative_recall": metric_lookup.loc[name, "negative_recall"],
            "battery_revenue_eur": battery_lookup.loc[name, "revenue_eur"],
            "training_runtime_seconds": training["runtime_seconds"],
            "checkpoint_sha256": record["checkpoint_sha256"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("operational_24step_MAE")
    summary.to_csv(PHASE2_OUT / "summary.csv", index=False)
    write_json(PHASE2_OUT / "candidate_metadata.json", {"candidates": records})
    shutil.copy2(ABLATION_CONFIG, PHASE2_OUT / "config_frozen.json")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for prefix, label in (("fixed_epochs", "fixed 6 epochs"), ("matched_steps", "matched 276 steps")):
        subset = summary[summary.candidate.str.startswith(prefix)].sort_values("train_windows")
        axes[0].plot(subset.train_windows, subset.operational_24step_MAE, marker="o", label=label)
        axes[1].plot(subset.train_windows, subset.battery_revenue_eur, marker="o", label=label)
    axes[0].set(xlabel="Eligible training windows", ylabel="2015 MAE (EUR/MWh)"); axes[0].legend()
    axes[1].set(xlabel="Eligible training windows", ylabel="2015 battery revenue (EUR)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(PHASE2_OUT / "history_budget_ablation.png", dpi=180); plt.close(fig)
    artifacts = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(PHASE2_OUT.glob("*")) if path.is_file() and path.name != "manifest.json"
    ]
    write_json(PHASE2_OUT / "manifest.json", {
        "status": "complete 2015-only history and training-budget ablation",
        "registration": registered, "source_dataset_sha256": sha256_file(FRAME_PATH),
        "test_outcomes_used_for_selection": False, "artifacts": artifacts,
    })
    print(summary.to_string(index=False))


def run_freeze() -> None:
    _, registered = _configs()
    summary_path = PHASE2_OUT / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Phase 2 summary is required before freezing")
    summary = pd.read_csv(summary_path)
    eligible = summary[summary.candidate != "Naive D-1"].sort_values("validation_72step_MAE")
    selected = eligible.iloc[0]
    expected = registered["phase3"]["selected_checkpoint_sha256"]
    if selected.checkpoint_sha256 != expected:
        raise ValueError(f"registered selection hash {expected} != observed {selected.checkpoint_sha256}")
    checkpoint = PHASE2_ARTIFACTS / "matched_steps_730" / "best_head.pt"
    _verify_registered_hash(checkpoint, expected)
    tied = eligible[np.isclose(eligible.validation_72step_MAE, selected.validation_72step_MAE)]
    selection = {
        "status": "frozen from 2015 development evidence before new retrospective predictions",
        "selection_metric": "72-step validation MAE",
        "selected_candidate": "matched_steps_730",
        "selected_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": expected,
        "selected_72step_MAE": float(selected.validation_72step_MAE),
        "selected_24step_MAE": float(selected.operational_24step_MAE),
        "tied_identical_candidates": tied.candidate.tolist(),
        "test_outcomes_used": False,
        "qualification": "exploratory selection; 730-vs-365 paired block interval includes zero",
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(SELECTION_PATH, selection)
    print(json.dumps(selection, indent=2))


def run_phase3() -> None:
    main, registered = _configs()
    if not SELECTION_PATH.exists():
        run_freeze()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    frame, german, _, _, _, _, _ = _inputs(main, registered)
    selected_checkpoint = ROOT / selection["selected_checkpoint"]
    _verify_registered_hash(selected_checkpoint, selection["checkpoint_sha256"])
    test = build_origins(
        frame, target_start="2016-01-01", target_end_exclusive="2018-01-01",
        require_full_target=False,
    )
    fixed_model, fixed_report = build_verified_model(main, device="cpu")
    load_head(fixed_model, selected_checkpoint)
    fixed = predict_origins(
        fixed_model, frame, test, german, model_name="Kumo adapted 730 fixed",
        batch_size=int(registered["phase2"]["batch_size"]),
    )

    periodic = registered["phase3"]["periodic_update"]
    update_train = build_origins(
        frame, target_start=periodic["update_training_target_start"],
        target_end_exclusive=periodic["update_training_target_end_exclusive"],
    )
    update_monitor = build_origins(
        frame, target_start=periodic["update_monitor_start"],
        target_end_exclusive=periodic["update_monitor_end_exclusive"],
    )
    update_model, update_report = build_verified_model(main, device="cpu")
    load_head(update_model, selected_checkpoint)
    update_dir = PHASE3_ARTIFACTS / "annual_update_2017"
    if (update_dir / "training_metadata.json").exists():
        update_training = json.loads((update_dir / "training_metadata.json").read_text(encoding="utf-8"))
        load_head(update_model, update_dir / "best_head.pt")
    else:
        update_training = train_head(
            update_model, KumoPriceDataset(frame, update_train, german),
            KumoPriceDataset(frame, update_monitor, german),
            epochs=int(periodic["epochs"]), learning_rate=float(periodic["learning_rate"]),
            batch_size=int(periodic["batch_size"]), seed=int(registered["protocol"]["seed"]),
            output_dir=update_dir, resume=True, checkpoint_selection="final_epoch",
        )
    test_2017 = [origin for origin in test if origin.delivery_date.year == 2017]
    updated_2017 = predict_origins(
        update_model, frame, test_2017, german, model_name="Kumo annual head update",
        batch_size=int(periodic["batch_size"]),
    )
    periodic_panel = pd.concat([
        fixed[pd.to_datetime(fixed.delivery_date).dt.year == 2016].assign(model="Kumo annual head update"),
        updated_2017,
    ], ignore_index=True)

    frozen_reference = pd.read_parquet(
        ROOT / "results" / "kumo_finetuning" / "retrospective" / "predictions.parquet"
    )
    panel = pd.concat([
        frozen_reference, fixed.reindex(columns=frozen_reference.columns),
        periodic_panel.reindex(columns=frozen_reference.columns),
    ], ignore_index=True)
    panel["test_year"] = pd.to_datetime(panel.delivery_date).dt.year
    PHASE3_OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PHASE3_OUT / "predictions.parquet", index=False)
    metrics = market_metrics(panel); metrics.to_csv(PHASE3_OUT / "metrics.csv", index=False)
    regimes = regime_metrics(panel); regimes.to_csv(PHASE3_OUT / "regime_metrics.csv", index=False)
    pairwise = [pairwise_tests(panel, seed=13).assign(test_year="combined")]
    for year, group in panel.groupby("test_year"):
        pairwise.append(pairwise_tests(group, seed=13).assign(test_year=int(year)))
    pd.concat(pairwise, ignore_index=True).to_csv(PHASE3_OUT / "pairwise_tests.csv", index=False)
    train_prices = frame.loc[
        (frame.index >= pd.Timestamp(main["data"]["train_target_start"]))
        & (frame.index < pd.Timestamp(main["data"]["train_target_end_exclusive"])), "price_eur_mwh",
    ].to_numpy()
    extreme_metrics(panel, train_prices).to_csv(PHASE3_OUT / "extreme_metrics.csv", index=False)
    daily, battery = dispatch_diagnostics(panel, BatterySpec())
    daily["test_year"] = pd.to_datetime(daily.delivery_date).dt.year
    daily.to_parquet(PHASE3_OUT / "battery_daily.parquet", index=False)
    battery.to_csv(PHASE3_OUT / "battery_metrics.csv", index=False)
    pairwise_revenue_intervals(daily).to_csv(PHASE3_OUT / "battery_pairwise_intervals.csv", index=False)
    by_year = []
    revenue_intervals = []
    for year, group in daily.groupby("test_year"):
        by_year.append(summarize_dispatch_daily(group).assign(test_year=int(year)))
        revenue_intervals.append(pairwise_revenue_intervals(group).assign(test_year=int(year)))
    pd.concat(by_year, ignore_index=True).to_csv(PHASE3_OUT / "battery_metrics_by_year.csv", index=False)
    pd.concat(revenue_intervals, ignore_index=True).to_csv(
        PHASE3_OUT / "battery_pairwise_intervals_by_year.csv", index=False
    )

    import matplotlib.pyplot as plt
    focus = metrics[metrics.model.isin([
        "JointCurve", "LEAR", "LightGBM", "Kumo fine-tuned",
        "Kumo adapted 730 fixed", "Kumo annual head update",
    ])]
    fig, ax = plt.subplots(figsize=(11, 5))
    focus.pivot(index="model", columns="test_year", values="MAE").plot.bar(ax=ax)
    ax.set_ylabel("MAE (EUR/MWh)"); ax.set_title("Previously observed retrospective comparison")
    fig.tight_layout(); fig.savefig(PHASE3_OUT / "retrospective_mae.png", dpi=180); plt.close(fig)
    shutil.copy2(ABLATION_CONFIG, PHASE3_OUT / "config_frozen.json")
    write_json(PHASE3_OUT / "update_metadata.json", {
        "selection": selection, "fixed_load_report": asdict(fixed_report),
        "update_load_report": asdict(update_report), "update_training": update_training,
        "update_train": origin_manifest(update_train, "train"),
        "update_monitor": origin_manifest(update_monitor, "validation"),
        "checkpoint_sha256": sha256_file(update_dir / "best_head.pt"),
        "qualification": "annual pseudo-live proof of concept; final epoch fixed before 2016 monitoring outcomes",
    })
    artifacts = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(PHASE3_OUT.glob("*")) if path.is_file() and path.name != "manifest.json"
    ]
    write_json(PHASE3_OUT / "manifest.json", {
        "status": "complete retrospective and annual-update evaluation",
        "selection": selection, "registration": registered,
        "retrospective_not_untouched": True, "test_outcomes_used_for_selection": False,
        "source_dataset_sha256": sha256_file(FRAME_PATH), "artifacts": artifacts,
    })
    print(metrics[metrics.model.str.contains("Kumo|JointCurve|LEAR")].to_string(index=False))
    print(battery[battery.model.str.contains("Kumo|JointCurve|LEAR")].to_string(index=False))


def run_darr_analysis() -> None:
    source = ROOT / "results" / "kumo_finetuning" / "darr" / "predictions_with_darr.parquet"
    panel = pd.read_parquet(source)
    components = reconstruct_darr_components(panel, alpha=0.01)
    daily = daily_darr_diagnostics(components)
    summary = summarize_darr(daily)
    DARR_OUT.mkdir(parents=True, exist_ok=True)
    components.to_parquet(DARR_OUT / "reconstructed_components.parquet", index=False)
    daily.to_parquet(DARR_OUT / "daily_forecast_diagnostics.parquet", index=False)
    summary.to_csv(DARR_OUT / "forecast_summary.csv", index=False)

    common = components[["delivery_date", "hour", "actual"]]
    dispatch_panel = pd.concat([
        common.assign(prediction=components.direct_prediction, model="Kumo fine-tuned"),
        common.assign(prediction=components.retrieval_prediction, model="DARR retrieval only"),
        common.assign(prediction=components.blend_prediction, model="Kumo fine-tuned + DARR"),
    ], ignore_index=True)
    battery_daily_path = DARR_OUT / "battery_daily.parquet"
    if battery_daily_path.exists():
        battery_daily = pd.read_parquet(battery_daily_path)
        battery_summary = summarize_dispatch_daily(battery_daily)
    else:
        battery_daily, battery_summary = dispatch_diagnostics(dispatch_panel, BatterySpec())
        battery_daily["test_year"] = pd.to_datetime(battery_daily.delivery_date).dt.year
        battery_daily.to_parquet(battery_daily_path, index=False)
        battery_summary.to_csv(DARR_OUT / "battery_summary.csv", index=False)
        intervals = [pairwise_revenue_intervals(battery_daily).assign(period="combined")]
        yearly_summaries = []
        for year, group in battery_daily.groupby("test_year"):
            intervals.append(pairwise_revenue_intervals(group).assign(period=str(year)))
            yearly_summaries.append(summarize_dispatch_daily(group).assign(period=str(year)))
        pd.concat(intervals, ignore_index=True).to_csv(
            DARR_OUT / "battery_pairwise_intervals.csv", index=False
        )
        pd.concat(yearly_summaries, ignore_index=True).to_csv(
            DARR_OUT / "battery_summary_by_year.csv", index=False
        )
        revenue_gain_concentration(battery_daily).to_csv(
            DARR_OUT / "revenue_gain_concentration.csv", index=False
        )

    neighbor_path = DARR_OUT / "neighbor_quality_daily.parquet"
    if neighbor_path.exists():
        neighbors = pd.read_parquet(neighbor_path)
    else:
        main, registered = _configs()
        frame, german, _, _, _, existing_head, _ = _inputs(main, registered)
        model, _ = build_verified_model(main, device="cpu")
        load_head(model, existing_head)
        context_origins = build_origins(
            frame, target_start="2012-02-01", target_end_exclusive="2015-01-01",
            stride_days=3,
        )
        query_origins = build_origins(
            frame, target_start="2016-01-01", target_end_exclusive="2018-01-01",
            require_full_target=False,
        )
        neighbors = neighbor_quality_diagnostics(
            model, frame, context_origins, query_origins, german,
            k=64, temperature=0.05, batch_size=8,
        )
        neighbors.to_parquet(neighbor_path, index=False)
    neighbor_summaries = []
    for period, group in [
        ("combined", neighbors),
        *[(str(year), part) for year, part in neighbors.groupby("test_year")],
    ]:
        neighbor_summaries.append({
            "period": period,
            "mean_top1_similarity": float(group.top1_similarity.mean()),
            "mean_topk_similarity": float(group.mean_topk_similarity.mean()),
            "mean_effective_neighbor_count": float(group.effective_neighbor_count.mean()),
            "mean_top1_weight": float(group.top1_weight.mean()),
            "mean_top1_target_mae": float(group.top1_target_mae.mean()),
            "mean_top1_target_bias": float(group.top1_target_bias.mean()),
            "same_calendar_month_rate": float(group.top1_same_calendar_month.mean()),
            "similarity_target_mae_correlation": float(
                group.top1_similarity.corr(group.top1_target_mae)
            ),
            "n_queries": int(len(group)),
        })
    pd.DataFrame(neighbor_summaries).to_csv(DARR_OUT / "neighbor_quality_summary.csv", index=False)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    combined = summary[summary.period == "combined"].set_index("component")
    combined[["MAE", "mean_predicted_spread"]].plot.bar(ax=axes[0])
    axes[0].set_ylabel("EUR/MWh"); axes[0].set_title("Accuracy and spread amplification")
    revenue = battery_summary.set_index("model").revenue_eur.sort_values()
    revenue.plot.barh(ax=axes[1]); axes[1].set_xlabel("Realized revenue (EUR)")
    axes[1].set_title("Same physical battery constraints")
    fig.tight_layout(); fig.savefig(DARR_OUT / "darr_accuracy_value.png", dpi=180); plt.close(fig)

    artifacts = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(DARR_OUT.glob("*")) if path.is_file() and path.name != "manifest.json"
    ]
    write_json(DARR_OUT / "manifest.json", {
        "status": "complete reconstruction and decision diagnostics for frozen native-DARR outputs",
        "source": source.relative_to(ROOT).as_posix(), "source_sha256": sha256_file(source),
        "alpha": 0.01, "configuration_tuned_on_retrospective": False,
        "reconstruction_identity": "retrieval = (blend - alpha * direct) / (1 - alpha)",
        "neighbor_similarity": "recomputed from the unchanged archived checkpoint and frozen memory",
        "artifacts": artifacts,
    })
    print(summary.to_string(index=False))
    print(battery_summary.to_string(index=False))
    print(revenue_gain_concentration(battery_daily).to_string(index=False))


def run_phase3_postprocess() -> None:
    """Add year-specific economic uncertainty without repeating model inference."""
    daily_path = PHASE3_OUT / "battery_daily.parquet"
    manifest_path = PHASE3_OUT / "manifest.json"
    if not daily_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("completed Phase 3 outputs are required")
    daily = pd.read_parquet(daily_path)
    if "test_year" not in daily:
        daily["test_year"] = pd.to_datetime(daily.delivery_date).dt.year
    summaries = []
    intervals = []
    for year, group in daily.groupby("test_year"):
        summaries.append(summarize_dispatch_daily(group).assign(test_year=int(year)))
        intervals.append(pairwise_revenue_intervals(group).assign(test_year=int(year)))
    pd.concat(summaries, ignore_index=True).to_csv(PHASE3_OUT / "battery_metrics_by_year.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(
        PHASE3_OUT / "battery_pairwise_intervals_by_year.csv", index=False
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(PHASE3_OUT.glob("*")) if path.is_file() and path.name != "manifest.json"
    ]
    manifest["economic_uncertainty_periods"] = [2016, 2017, "combined"]
    write_json(manifest_path, manifest)
    print(pd.concat(summaries, ignore_index=True).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=[
            "audit", "phase1", "phase2", "freeze", "phase3", "phase3-postprocess", "darr-analysis"
        ]
    )
    args = parser.parse_args()
    if args.command == "audit":
        run_audit()
    elif args.command == "phase1":
        run_phase1()
    elif args.command == "phase2":
        run_phase2()
    elif args.command == "freeze":
        run_freeze()
    elif args.command == "phase3":
        run_phase3()
    elif args.command == "phase3-postprocess":
        run_phase3_postprocess()
    else:
        run_darr_analysis()


if __name__ == "__main__":
    main()
