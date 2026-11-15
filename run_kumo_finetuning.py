"""Run the registered Kumo CPU smoke study or a GPU grid extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch  # Load the pinned runtime before SciPy/Matplotlib DLLs on Windows.
import pandas as pd

from src.research.decision import BatterySpec, evaluate_dispatch
from src.research.evaluation import pairwise_tests
from src.research.kumo.data import (
    KumoPriceDataset,
    build_origins,
    origin_manifest,
    training_standardizer,
)
from src.research.kumo.diagnostics import resource_report, write_json
from src.research.kumo.evaluation import add_persistence, market_metrics, regime_metrics
from src.research.kumo.inference import load_head, predict_origins, predict_origins_darr
from src.research.kumo.train import (
    build_verified_model,
    load_official_standardizer,
    save_standardizer,
    train_head,
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "kumo_finetuning.json"
OUT = ROOT / "results" / "kumo_finetuning" / "development"
CHECKPOINT_OUT = ROOT / "artifacts" / "kumo_finetuning" / "cpu_smoke"
SELECTION_PATH = ROOT / "results" / "kumo_finetuning" / "selection.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_records(directory: Path) -> list[dict]:
    return [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def run_artifact_audit() -> None:
    """Attach immutable checksums to completed Kumo result directories."""
    source = ROOT / "data" / "raw" / "epftoolbox_epex_de.parquet"
    source_record = {"path": source.relative_to(ROOT).as_posix(), "sha256": file_hash(source), "bytes": source.stat().st_size}
    base = ROOT / "results" / "kumo_finetuning"
    for name in ("extended", "retrospective", "darr"):
        directory = base / name
        if not directory.exists():
            continue
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
            "status": "completed validation-selected extended training",
            "selection": json.loads((directory / "selection.json").read_text(encoding="utf-8")),
        }
        manifest["source_dataset"] = source_record
        manifest["artifacts"] = _artifact_records(directory)
        write_json(manifest_path, manifest)
        print(f"audited {name}: {len(manifest['artifacts'])} artifacts")


def _plot(training: dict, panel: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plots = OUT / "plots"; plots.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(training["history"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.epoch, history.train_mse_standardized, marker="o", label="train MSE")
    ax.plot(history.epoch, history.validation_mse_standardized, marker="o", label="validation MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Standardized MSE"); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "cpu_smoke_loss_curve.png", dpi=180); plt.close(fig)
    sample_date = sorted(panel.delivery_date.unique())[0]
    sample = panel[panel.delivery_date == sample_date]
    fig, ax = plt.subplots(figsize=(9, 4))
    actual = sample.drop_duplicates("hour").sort_values("hour")
    ax.plot(actual.hour, actual.actual, color="black", marker="o", label="actual")
    for model in ("Kumo zero-shot", "Kumo fine-tuned", "Naive D-1"):
        curve = sample[sample.model == model].sort_values("hour")
        ax.plot(curve.hour, curve.prediction, marker=".", label=model)
    ax.set_title(f"Development forecast: {sample_date}"); ax.set_xlabel("Product hour")
    ax.set_ylabel("EUR/MWh"); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "cpu_smoke_forecast_example.png", dpi=180); plt.close(fig)


def _add_frozen_conventional_models(panel: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(ROOT / "results" / "predictions" / "phase2_point_predictions.parquet")
    dates = set(panel.delivery_date.unique())
    models = {"LEAR", "LightGBM", "XGBoost", "CatBoost", "JointCurve"}
    frozen = frozen[
        frozen.delivery_date.isin(dates) & frozen.model.isin(models) & (frozen.seed == 42)
    ].copy()
    frozen["forecast_origin"] = None
    frozen["native_horizon_step"] = frozen["hour"] + 1
    frozen["inference_seconds_per_origin"] = float("nan")
    frozen["experiment_scope"] = "frozen_phase2_weekly_refit_reference"
    return pd.concat([panel, frozen.reindex(columns=panel.columns)], ignore_index=True)


def run_ablation() -> None:
    """Run the CPU-feasible 3/6/12-month registered history ablation."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame = pd.read_parquet(ROOT / "data" / "raw" / "epftoolbox_epex_de.parquet")
    standardizer = training_standardizer(
        frame, train_start=config["data"]["train_target_start"],
        train_end_exclusive=config["data"]["train_target_end_exclusive"],
    )
    all_train = build_origins(
        frame, target_start=config["data"]["train_target_start"],
        target_end_exclusive=config["data"]["train_target_end_exclusive"],
    )
    validation = build_origins(
        frame, target_start=config["data"]["validation_target_start"],
        target_end_exclusive=config["data"]["validation_target_end_exclusive"],
    )[:30]
    dev_start = pd.Timestamp(config["data"]["development_smoke_start"])
    development = build_origins(
        frame, target_start=dev_start, target_end_exclusive=dev_start + pd.Timedelta(days=30),
        require_full_target=False,
    )
    rows = []
    ablation_dir = OUT / "history_ablation"; ablation_dir.mkdir(parents=True, exist_ok=True)
    for history_days in (92, 183, 365):
        record_path = ablation_dir / f"history_{history_days}.json"
        if record_path.exists():
            rows.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        origins = all_train[-min(history_days, len(all_train)):]
        model, _ = build_verified_model(config, device="cpu")
        training = train_head(
            model, KumoPriceDataset(frame, origins, standardizer),
            KumoPriceDataset(frame, validation, standardizer),
            epochs=3, learning_rate=1e-4, batch_size=8, seed=13,
            output_dir=ROOT / "artifacts" / "kumo_finetuning" / f"history_{history_days}",
        )
        prediction = predict_origins(
            model, frame, development, standardizer,
            model_name="Kumo fine-tuned", batch_size=8,
        )
        panel = add_persistence(frame, prediction)
        panel["test_year"] = 2016
        metric = market_metrics(panel).set_index("model").loc["Kumo fine-tuned"].to_dict()
        _, battery = evaluate_dispatch(panel, BatterySpec())
        battery_row = battery.set_index("model").loc["Kumo fine-tuned"].to_dict()
        record = {
            "history_days": history_days, "train_windows": len(origins),
            "first_training_target": origins[0].delivery_date.date().isoformat(),
            "last_training_target": origins[-1].delivery_date.date().isoformat(),
            "best_epoch": training["best_epoch"],
            "validation_mae_eur_mwh": training["best_validation_mae_eur_mwh"],
            "development_mae_eur_mwh": metric["MAE"],
            "development_rank_correlation": metric["daily_rank_correlation"],
            "development_battery_revenue_eur": battery_row["revenue_eur"],
            "training_runtime_seconds": training["runtime_seconds"],
            "checkpoint_sha256": training["checkpoint_sha256"],
            "qualification": "30-day CPU development ablation; not full retrospective evidence",
        }
        write_json(record_path, record); rows.append(record)
        pd.DataFrame(rows).sort_values("history_days").to_csv(ablation_dir / "metrics.csv", index=False)
        print(json.dumps(record, indent=2), flush=True)

    table = pd.DataFrame(rows).sort_values("history_days")
    table.to_csv(ablation_dir / "metrics.csv", index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(table.history_days, table.development_mae_eur_mwh, marker="o")
    axes[0].set(xlabel="Training history (days)", ylabel="Development MAE (EUR/MWh)")
    axes[1].plot(table.history_days, table.development_battery_revenue_eur, marker="o")
    axes[1].set(xlabel="Training history (days)", ylabel="30-day battery value (EUR)")
    fig.tight_layout(); fig.savefig(ablation_dir / "history_ablation.png", dpi=180); plt.close(fig)
    ablation_artifacts = [
        ablation_dir / "metrics.csv", ablation_dir / "history_ablation.png",
        *sorted(ablation_dir.glob("history_*.json")),
    ]
    write_json(ablation_dir / "manifest.json", {
        "status": "completed CPU-feasible 3/6/12-month development ablation",
        "unexecuted_registered_history_days": [730, 1065],
        "artifacts": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in ablation_artifacts
        ],
    })


def _registered_datasets(config: dict):
    frame = pd.read_parquet(ROOT / "data" / "raw" / "epftoolbox_epex_de.parquet")
    standardizer = training_standardizer(
        frame, train_start=config["data"]["train_target_start"],
        train_end_exclusive=config["data"]["train_target_end_exclusive"],
    )
    train = build_origins(
        frame, target_start=config["data"]["train_target_start"],
        target_end_exclusive=config["data"]["train_target_end_exclusive"],
    )
    validation = build_origins(
        frame, target_start=config["data"]["validation_target_start"],
        target_end_exclusive=config["data"]["validation_target_end_exclusive"],
    )
    return frame, standardizer, train, validation


def run_grid() -> None:
    """Registered full grid. Intended for CUDA >=8 GB; resumable by candidate."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame, standardizer, train_origins, validation_origins = _registered_datasets(config)
    train_data = KumoPriceDataset(frame, train_origins, standardizer)
    validation_data = KumoPriceDataset(frame, validation_origins, standardizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_dir = ROOT / "results" / "kumo_finetuning" / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for epochs in config["registered_grid"]["epochs"]:
        for learning_rate in config["registered_grid"]["learning_rates"]:
            for batch_size in config["registered_grid"]["batch_sizes"]:
                label = f"e{epochs}_lr{learning_rate:g}_b{batch_size}"
                artifact_dir = ROOT / "artifacts" / "kumo_finetuning" / "grid" / label
                metadata_path = artifact_dir / "training_metadata.json"
                if metadata_path.exists():
                    result = json.loads(metadata_path.read_text(encoding="utf-8"))
                else:
                    model, _ = build_verified_model(config, device=device)
                    result = train_head(
                        model, train_data, validation_data, epochs=int(epochs),
                        learning_rate=float(learning_rate), batch_size=int(batch_size),
                        seed=int(config["registered_grid"]["seed"]), output_dir=artifact_dir,
                    )
                rows.append({
                    "candidate": label, "epochs_requested": epochs,
                    "learning_rate": learning_rate, "batch_size": batch_size,
                    "best_epoch": result["best_epoch"],
                    "validation_mae_eur_mwh": result["best_validation_mae_eur_mwh"],
                    "runtime_seconds": result["runtime_seconds"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                })
                pd.DataFrame(rows).to_csv(grid_dir / "grid_metrics.csv", index=False)
    table = pd.DataFrame(rows).sort_values("validation_mae_eur_mwh").reset_index(drop=True)
    best = table.iloc[0].to_dict()
    selection = {
        "selection_rule": "minimum validation MAE; 2016-2017 outcomes not read",
        "device": device, "train_windows": len(train_origins),
        "validation_windows": len(validation_origins), "selected": best,
        "checkpoint": f"artifacts/kumo_finetuning/grid/{best['candidate']}/best_head.pt",
    }
    write_json(grid_dir / "selection.json", selection)
    write_json(SELECTION_PATH, selection)
    shutil.copy2(CONFIG_PATH, grid_dir / "config_frozen.json")
    print(table.to_string(index=False))


def run_extended() -> None:
    """Adaptive longer CPU run; selects only on the complete 2015 validation panel."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame, standardizer, all_train, validation = _registered_datasets(config)
    candidate = config["extended_cpu_candidate"]
    train_origins = all_train[-int(candidate["training_history_days"]):]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact_dir = ROOT / "artifacts" / "kumo_finetuning" / "extended_365_e10"
    metadata_path = artifact_dir / "training_metadata.json"
    if metadata_path.exists():
        training = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_selection = ROOT / "results" / "kumo_finetuning" / "extended" / "selection.json"
        load_report_payload = (
            json.loads(previous_selection.read_text(encoding="utf-8")).get("load_report")
            if previous_selection.exists() else None
        )
    else:
        model, load_report = build_verified_model(config, device=device)
        load_report_payload = load_report.__dict__
        training = train_head(
            model, KumoPriceDataset(frame, train_origins, standardizer),
            KumoPriceDataset(frame, validation, standardizer),
            epochs=int(candidate["epochs"]), learning_rate=float(candidate["learning_rate"]),
            batch_size=int(candidate["batch_size"]), seed=int(config["registered_grid"]["seed"]),
            output_dir=artifact_dir,
            early_stopping_patience=int(candidate["early_stopping_patience"]),
        )
    selection = {
        "selection_rule": "minimum MAE on all issuance-safe 2015 validation origins",
        "selection_type": "adaptive extended candidate; not the complete registered grid",
        "motivation": "the 12-month three-epoch ablation improved through its final epoch",
        "test_outcomes_read_for_selection": False,
        "device": device, "train_windows": len(train_origins),
        "validation_windows": len(validation),
        "training_first_target": train_origins[0].delivery_date.date().isoformat(),
        "training_last_target": train_origins[-1].delivery_date.date().isoformat(),
        "best_epoch": training["best_epoch"],
        "best_validation_mae_eur_mwh": training["best_validation_mae_eur_mwh"],
        "checkpoint_sha256": training["checkpoint_sha256"],
        "checkpoint": "artifacts/kumo_finetuning/extended_365_e10/best_head.pt",
        "load_report": load_report_payload,
    }
    write_json(SELECTION_PATH, selection)
    out = ROOT / "results" / "kumo_finetuning" / "extended"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "selection.json", selection)
    shutil.copy2(CONFIG_PATH, out / "config_frozen.json")
    history = pd.DataFrame(training["history"])
    history.to_csv(out / "training_history.csv", index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epoch, history.train_mse_standardized, marker="o", label="train")
    axes[0].plot(history.epoch, history.validation_mse_standardized, marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Standardized MSE"); axes[0].legend()
    axes[1].plot(history.epoch, history.validation_mae_eur_mwh, marker="o")
    axes[1].axvline(training["best_epoch"], color="black", linestyle="--", label="selected")
    axes[1].set(xlabel="Epoch", ylabel="Validation MAE (EUR/MWh)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "extended_training_curve.png", dpi=180); plt.close(fig)
    print(json.dumps(selection, indent=2))


def run_retrospective() -> None:
    """Evaluate only the validation-selected grid checkpoint on 2016-2017."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not SELECTION_PATH.exists():
        raise FileNotFoundError("Run a validation-only selection command before retrospective evaluation")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    frame, standardizer, _, _ = _registered_datasets(config)
    origins = build_origins(
        frame, target_start="2016-01-01", target_end_exclusive="2018-01-01",
        require_full_target=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zero_model, load_report = build_verified_model(config, device=device)
    official = load_official_standardizer(config)
    zero = predict_origins(zero_model, frame, origins, official, model_name="Kumo zero-shot", batch_size=8)
    fine_model, _ = build_verified_model(config, device=device)
    load_head(fine_model, ROOT / selection["checkpoint"])
    fine = predict_origins(fine_model, frame, origins, standardizer, model_name="Kumo fine-tuned", batch_size=8)
    panel = add_persistence(frame, pd.concat([zero, fine], ignore_index=True))
    panel = _add_frozen_conventional_models(panel)
    panel["test_year"] = pd.to_datetime(panel.delivery_date).dt.year
    panel["experiment_scope"] = "frozen_retrospective"
    panel["comparison_protocol"] = panel.model.map(
        lambda name: "weekly-refit frozen Phase 2 reference"
        if name in {"LEAR", "LightGBM", "XGBoost", "CatBoost", "JointCurve"}
        else "fixed Kumo checkpoint or persistence"
    )
    out = ROOT / "results" / "kumo_finetuning" / "retrospective"
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "predictions.parquet", index=False)
    market_metrics(panel).to_csv(out / "metrics.csv", index=False)
    regime_metrics(panel).to_csv(out / "regime_metrics.csv", index=False)
    pairwise = [pairwise_tests(panel, seed=13).assign(test_year="combined")]
    for year, year_panel in panel.groupby("test_year"):
        pairwise.append(pairwise_tests(year_panel, seed=13).assign(test_year=int(year)))
    pd.concat(pairwise, ignore_index=True).to_csv(out / "pairwise_tests.csv", index=False)
    daily, battery = evaluate_dispatch(panel, BatterySpec())
    daily.to_parquet(out / "battery_dispatch_daily.parquet", index=False)
    battery.to_csv(out / "battery_metrics.csv", index=False)
    import matplotlib.pyplot as plt
    metrics = pd.read_csv(out / "metrics.csv")
    kumo_metrics = metrics[metrics.model.isin(["Kumo zero-shot", "Kumo fine-tuned", "Naive D-1", "JointCurve", "LightGBM"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    kumo_metrics.pivot(index="model", columns="test_year", values="MAE").plot.bar(ax=ax)
    ax.set_ylabel("MAE (EUR/MWh)"); ax.set_title("Frozen full-year retrospective accuracy")
    fig.tight_layout(); fig.savefig(out / "retrospective_mae.png", dpi=180); plt.close(fig)
    write_json(out / "manifest.json", {
        "status": "complete frozen retrospective evaluation",
        "selection": selection, "load_report": load_report.__dict__,
        "test_outcomes_used_for_selection": False, "origins": origin_manifest(origins, "test"),
        "resources": resource_report(),
        "qualification": "retrospective extension on previously examined 2016-2017 benchmark years",
    })


def run_darr() -> None:
    """Evaluate native NVIDIA DARR as an incremental frozen-checkpoint extension."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    direct_path = ROOT / "results" / "kumo_finetuning" / "retrospective" / "predictions.parquet"
    if not direct_path.exists():
        raise FileNotFoundError("Run the frozen direct retrospective before the DARR extension")
    frame, standardizer, _, validation = _registered_datasets(config)
    darr = config["darr_extension"]
    stride_days = int(darr["context_stride_hours"]) // 24
    context = build_origins(
        frame, target_start=darr["context_target_start"],
        target_end_exclusive=darr["context_target_end_exclusive"], stride_days=stride_days,
    )
    test = build_origins(
        frame, target_start="2016-01-01", target_end_exclusive="2018-01-01",
        require_full_target=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, load_report = build_verified_model(config, device=device)
    load_head(model, ROOT / selection["checkpoint"])

    direct_validation = predict_origins(
        model, frame, validation, standardizer,
        model_name="Kumo fine-tuned", batch_size=8,
    )
    darr_validation, validation_diagnostics = predict_origins_darr(
        model, frame, validation, context, standardizer, direct_validation,
        alpha=float(darr["alpha"]), k=int(darr["k"]),
        temperature=float(darr["temperature"]), batch_size=8,
    )
    validation_panel = pd.concat([direct_validation, darr_validation], ignore_index=True)

    retrospective_panel = pd.read_parquet(direct_path)
    direct_test = retrospective_panel[retrospective_panel.model == "Kumo fine-tuned"].copy()
    darr_test, test_diagnostics = predict_origins_darr(
        model, frame, test, context, standardizer, direct_test,
        alpha=float(darr["alpha"]), k=int(darr["k"]),
        temperature=float(darr["temperature"]), batch_size=8,
    )
    darr_test["test_year"] = pd.to_datetime(darr_test.delivery_date).dt.year
    darr_test["experiment_scope"] = "post-retrospective registered native-DARR extension"
    darr_test["comparison_protocol"] = "fixed fine-tuned checkpoint plus fixed 2012-2014 memory"
    panel = pd.concat([retrospective_panel, darr_test.reindex(columns=retrospective_panel.columns)], ignore_index=True)

    out = ROOT / "results" / "kumo_finetuning" / "darr"
    out.mkdir(parents=True, exist_ok=True)
    validation_panel.to_parquet(out / "validation_predictions.parquet", index=False)
    market_metrics(validation_panel).to_csv(out / "validation_metrics.csv", index=False)
    panel.to_parquet(out / "predictions_with_darr.parquet", index=False)
    market_metrics(panel).to_csv(out / "metrics.csv", index=False)
    regime_metrics(panel).to_csv(out / "regime_metrics.csv", index=False)
    pairwise = [pairwise_tests(panel, seed=13).assign(test_year="combined")]
    for year, year_panel in panel.groupby("test_year"):
        pairwise.append(pairwise_tests(year_panel, seed=13).assign(test_year=int(year)))
    pd.concat(pairwise, ignore_index=True).to_csv(out / "pairwise_tests.csv", index=False)
    daily, battery = evaluate_dispatch(panel, BatterySpec())
    daily.to_parquet(out / "battery_dispatch_daily.parquet", index=False)
    battery.to_csv(out / "battery_metrics.csv", index=False)
    write_json(out / "manifest.json", {
        "status": "complete native-DARR incremental experiment",
        "registration_timing": darr["status"], "selection": selection,
        "configuration": darr, "validation_diagnostics": validation_diagnostics,
        "test_diagnostics": test_diagnostics, "load_report": load_report.__dict__,
        "context": origin_manifest(context, "train"),
        "validation": origin_manifest(validation, "validation"),
        "test": origin_manifest(test, "test"),
        "qualification": "DARR was registered after direct retrospective results; defaults were not tuned on test outcomes",
    })
    print(pd.read_csv(out / "validation_metrics.csv").to_string(index=False))
    print(pd.read_csv(out / "metrics.csv").query("model.str.contains('Kumo')", engine="python").to_string(index=False))


def run_smoke() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame_path = ROOT / "data" / "raw" / "epftoolbox_epex_de.parquet"
    frame = pd.read_parquet(frame_path)
    standardizer = training_standardizer(
        frame, train_start=config["data"]["train_target_start"],
        train_end_exclusive=config["data"]["train_target_end_exclusive"],
    )
    train_origins = build_origins(
        frame, target_start=config["data"]["train_target_start"],
        target_end_exclusive=config["data"]["train_target_end_exclusive"],
    )
    validation_origins = build_origins(
        frame, target_start=config["data"]["validation_target_start"],
        target_end_exclusive=config["data"]["validation_target_end_exclusive"],
    )
    smoke = config["cpu_smoke"]
    train_origins = train_origins[-int(smoke["max_train_windows"]):]
    validation_origins = validation_origins[:int(smoke["max_validation_windows"])]
    dev_start = pd.Timestamp(config["data"]["development_smoke_start"])
    dev_end = dev_start + pd.Timedelta(days=int(smoke["max_development_origins"]))
    development_origins = build_origins(
        frame, target_start=dev_start, target_end_exclusive=dev_end,
        require_full_target=False,
    )
    train_data = KumoPriceDataset(frame, train_origins, standardizer)
    validation_data = KumoPriceDataset(frame, validation_origins, standardizer)
    model, load_report = build_verified_model(config, device="cpu")
    official_standardizer = load_official_standardizer(config)
    zero = predict_origins(
        model, frame, development_origins, official_standardizer,
        model_name="Kumo zero-shot", batch_size=int(smoke["batch_size"]),
    )
    training = train_head(
        model, train_data, validation_data, epochs=int(smoke["epochs"]),
        learning_rate=float(smoke["learning_rate"]), batch_size=int(smoke["batch_size"]),
        seed=int(config["registered_grid"]["seed"]), output_dir=CHECKPOINT_OUT,
    )
    fine = predict_origins(
        model, frame, development_origins, standardizer,
        model_name="Kumo fine-tuned", batch_size=int(smoke["batch_size"]),
    )
    panel = add_persistence(frame, pd.concat([zero, fine], ignore_index=True))
    panel = _add_frozen_conventional_models(panel)
    panel["test_year"] = pd.to_datetime(panel.delivery_date).dt.year
    panel["experiment_scope"] = "cpu_development_smoke_not_full_retrospective"
    panel["comparison_protocol"] = panel.model.map(
        lambda name: "weekly-refit frozen Phase 2 reference"
        if name in {"LEAR", "LightGBM", "XGBoost", "CatBoost", "JointCurve"}
        else "fixed Kumo checkpoint or persistence"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "predictions.parquet"
    panel.to_parquet(prediction_path, index=False)
    metrics = market_metrics(panel)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    regime_metrics(panel).to_csv(OUT / "regime_metrics.csv", index=False)
    tests = pairwise_tests(panel.assign(fit_seconds=0.0), seed=13)
    tests.to_csv(OUT / "pairwise_tests.csv", index=False)
    dispatch_daily, dispatch = evaluate_dispatch(panel, BatterySpec())
    dispatch_daily.to_parquet(OUT / "battery_dispatch_daily.parquet", index=False)
    dispatch.to_csv(OUT / "battery_metrics.csv", index=False)
    save_standardizer(standardizer, OUT)
    save_standardizer(official_standardizer, OUT / "nvidia_zero_shot")
    _plot(training, panel)
    source_manifest_path = ROOT / "data" / "raw" / "epftoolbox_epex_de.manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    shutil.copy2(CONFIG_PATH, OUT / "config_frozen.json")
    artifact_paths = [
        prediction_path, OUT / "metrics.csv", OUT / "regime_metrics.csv",
        OUT / "pairwise_tests.csv", OUT / "battery_metrics.csv",
        OUT / "battery_dispatch_daily.parquet", OUT / "price_standardizer.json",
        OUT / "nvidia_zero_shot" / "price_standardizer.json",
        OUT / "plots" / "cpu_smoke_loss_curve.png",
        OUT / "plots" / "cpu_smoke_forecast_example.png", OUT / "config_frozen.json",
    ]
    manifest = {
        "status": "completed genuine CPU smoke; full registered grid not executed",
        "configuration": config, "load_report": load_report.__dict__,
        "resources": resource_report(), "source_manifest": source_manifest,
        "normalization_audit": {
            "zero_shot": "NVIDIA published standardizer, checksum verified",
            "fine_tuned": "scalar statistics fitted only on registered 2012-2014 German training prices",
            "german_training_mean": standardizer.mean,
            "german_training_std": standardizer.std,
            "nvidia_mean": official_standardizer.mean,
            "nvidia_std": official_standardizer.std
        },
        "source_parquet_sha256": file_hash(frame_path),
        "splits": [
            origin_manifest(train_origins, "train"),
            origin_manifest(validation_origins, "validation"),
            origin_manifest(development_origins, "test"),
        ],
        "training": training,
        "prediction_sha256": file_hash(prediction_path),
        "artifacts": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        ],
        "qualification": (
            "The checkpoint contains a real updated pretrained head, but the limited CPU smoke "
            "is not evidence for the registered full-grid or 2016-2017 retrospective claims."
        ),
    }
    # The large checkpoint stays outside Git; its hash remains auditable.
    manifest["training"]["checkpoint"] = "artifacts/kumo_finetuning/cpu_smoke/best_head.pt (gitignored)"
    write_json(OUT / "manifest.json", manifest)
    print(metrics.to_string(index=False))
    print(dispatch.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "ablation", "extended", "grid", "retrospective", "darr", "audit"])
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke()
    elif args.command == "ablation":
        run_ablation()
    elif args.command == "extended":
        run_extended()
    elif args.command == "grid":
        run_grid()
    elif args.command == "retrospective":
        run_retrospective()
    elif args.command == "darr":
        run_darr()
    elif args.command == "audit":
        run_artifact_audit()


if __name__ == "__main__":
    main()
