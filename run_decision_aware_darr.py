"""Build a causal, resumable historical candidate dataset for Prompt 1 only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch  # Load Windows DLLs before NumPy/Pandas/SciPy.
import numpy as np
import pandas as pd

from src.research.decision_aware_darr.causal import (
    SplitPlan,
    eligible_candidate_indices,
    native_similarity_order,
    native_weighted_forecast,
    window_id,
)
from src.research.decision_aware_darr.dataset import (
    battery_spec_from_config,
    build_candidate_batch,
    context_pattern_matrix,
    direct_prediction_lookup,
    prepare_candidate_library,
    validate_feature_label_separation,
    validate_unique_pairs,
)
from src.research.decision_aware_darr.embeddings import (
    embedding_ids,
    extract_embeddings,
    load_embedding_store,
)
from src.research.decision_aware_darr.storage import (
    atomic_json,
    atomic_npz,
    atomic_parquet,
    canonical_hash,
    output_record,
    sha256_file,
    verify_batch_manifest,
    verify_output,
)
from src.research.kumo.data import build_origins, training_standardizer, validate_canonical_hourly
from src.research.kumo.inference import load_head
from src.research.kumo.train import build_verified_model, resolve_pinned_paths


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "decision_aware_darr_phase1.json"
KUMO_CONFIG_PATH = ROOT / "config" / "kumo_finetuning.json"
FRAME_PATH = ROOT / "data" / "raw" / "epftoolbox_epex_de.parquet"
RESULTS = ROOT / "results" / "decision_aware_darr" / "phase1"
ARTIFACTS = ROOT / "artifacts" / "decision_aware_darr" / "phase1"
EMBED_MANIFESTS = RESULTS / "batches" / "embeddings"
CANDIDATE_MANIFESTS = RESULTS / "batches" / "candidates"
STATE_PATH = RESULTS / "progress_state.json"
PROGRESS_PATH = ROOT / "docs" / "DECISION_AWARE_DARR_PROGRESS.md"


def _load_configs() -> tuple[dict, dict, str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    kumo = json.loads(KUMO_CONFIG_PATH.read_text(encoding="utf-8"))
    return config, kumo, canonical_hash(config)


def _verify_file(path: Path, expected_hash: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ValueError(f"checksum conflict for {path}: {observed} != {expected_hash}")


def _inputs() -> tuple[dict, dict, str, pd.DataFrame, list, SplitPlan, object, pd.DataFrame]:
    config, kumo, config_hash = _load_configs()
    _verify_file(FRAME_PATH, config["hashes"]["source_dataset_sha256"])
    head_path = ROOT / config["model"]["head_checkpoint"]
    direct_path = ROOT / config["model"]["direct_predictions"]
    _verify_file(head_path, config["model"]["head_checkpoint_sha256"])
    _verify_file(direct_path, config["model"]["direct_predictions_sha256"])
    paths = resolve_pinned_paths(kumo)
    _verify_file(paths.checkpoint, config["hashes"]["nvidia_checkpoint_sha256"])
    _verify_file(paths.standardizer, config["hashes"]["nvidia_standardizer_sha256"])
    frame = pd.read_parquet(FRAME_PATH)
    library = build_origins(
        frame,
        target_start=config["candidate_library"]["target_start"],
        target_end_exclusive=config["candidate_library"]["target_end_exclusive"],
    )
    train = build_origins(
        frame,
        target_start=config["queries"]["train_start"],
        target_end_exclusive=config["queries"]["train_end_exclusive"],
    )
    validation = build_origins(
        frame,
        target_start=config["queries"]["validation_start"],
        target_end_exclusive=config["queries"]["validation_end_exclusive"],
    )
    split = SplitPlan(tuple(train), tuple(validation)); split.assert_no_label_overlap()
    if len(library) != config["candidate_library"]["expected_windows"]:
        raise ValueError("candidate-library count differs from frozen configuration")
    if len(train) != config["queries"]["expected_train_queries"]:
        raise ValueError("training-query count differs from frozen configuration")
    if len(validation) != config["queries"]["expected_validation_queries"]:
        raise ValueError("validation-query count differs from frozen configuration")
    standardizer = training_standardizer(
        frame, train_start="2012-02-01", train_end_exclusive="2015-01-01"
    )
    direct = pd.read_parquet(direct_path)
    return config, kumo, config_hash, frame, library, split, standardizer, direct


def _expected_batches(total: int, size: int) -> int:
    return int(math.ceil(total / size))


def _batch_status(directory: Path, prefix: str, count: int, config_hash: str, source_hash: str) -> list[dict]:
    rows = []
    for number in range(count):
        path = directory / f"{prefix}_{number:03d}.json"
        valid, reason = verify_batch_manifest(path, ROOT, config_hash, source_hash)
        rows.append({"batch_id": f"{prefix}_{number:03d}", "complete": valid, "reason": reason})
    return rows


def _state_snapshot(stage: str, error: str | None = None) -> dict:
    config, _, config_hash = _load_configs()
    source_hash = config["hashes"]["source_dataset_sha256"]
    embedding_count = _expected_batches(
        config["candidate_library"]["expected_windows"],
        config["batching"]["embedding_windows_per_batch"],
    )
    query_count = config["queries"]["expected_train_queries"] + config["queries"]["expected_validation_queries"]
    candidate_count = _expected_batches(query_count, config["batching"]["queries_per_candidate_batch"])
    embeddings = _batch_status(
        EMBED_MANIFESTS, "embedding", embedding_count, config_hash, source_hash
    )
    candidates = _batch_status(
        CANDIDATE_MANIFESTS, "candidate", candidate_count, config_hash, source_hash
    )
    return {
        "status": "complete" if stage == "complete" else "incomplete",
        "current_stage": stage,
        "config_sha256": config_hash,
        "source_sha256": source_hash,
        "embedding_batches_complete": sum(row["complete"] for row in embeddings),
        "embedding_batches_total": embedding_count,
        "candidate_batches_complete": sum(row["complete"] for row in candidates),
        "candidate_batches_total": candidate_count,
        "last_successful_embedding_batch": next(
            (row["batch_id"] for row in reversed(embeddings) if row["complete"]), None
        ),
        "last_successful_candidate_batch": next(
            (row["batch_id"] for row in reversed(candidates) if row["complete"]), None
        ),
        "error": error,
        "resume_command": "python run_decision_aware_darr.py build --resume",
    }


def _write_progress(stage: str, error: str | None = None) -> dict:
    state = _state_snapshot(stage, error)
    atomic_json(STATE_PATH, state)
    complete_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in RESULTS.glob("*.parquet")
    )
    tests = "not yet executed" if not (RESULTS / "test_status.json").exists() else json.loads(
        (RESULTS / "test_status.json").read_text(encoding="utf-8")
    ).get("summary", "unknown")
    text = f"""# Decision-aware DARR Prompt 1 progress

## Current state

- Latest completed stage: {stage if stage != 'complete' else 'Stage 7 — finalization'}.
- Current stage: {stage}.
- Status: {'COMPLETE' if stage == 'complete' else 'INCOMPLETE'}.
- Branch: `feature/decision-aware-darr-dataset`.
- Parent commit: `7f12d2384afb74bc2d24125be70fd9aaa8cb6616`.
- Configuration SHA-256: `{state['config_sha256']}`.
- Source SHA-256: `{state['source_sha256']}`.

## Verified batch progress

- Embedding batches: {state['embedding_batches_complete']} / {state['embedding_batches_total']} complete.
- Candidate batches: {state['candidate_batches_complete']} / {state['candidate_batches_total']} complete.
- Last successful embedding batch: {state['last_successful_embedding_batch'] or 'none'}.
- Last successful candidate batch: {state['last_successful_candidate_batch'] or 'none'}.

## Completed final files

{chr(10).join(f'- `{path}`' for path in complete_files) if complete_files else '- None yet.'}

## Tests

- {tests}

## Blockers and errors

- {error or 'None.'}

## Pending work

{'- None; Prompt 1 is complete.' if stage == 'complete' else '- Resume the earliest checksum-invalid or missing batch, then audit and analyze.'}

## Exact resume commands

```powershell
python run_decision_aware_darr.py status
python run_decision_aware_darr.py build --resume
python run_decision_aware_darr.py audit
python run_decision_aware_darr.py analyze
```

Reruns skip only batches whose manifest, configuration hash, source hash, byte count, and output checksum all verify. Conflicting configurations fail explicitly.
"""
    PROGRESS_PATH.write_text(text, encoding="utf-8")
    return state


def run_plan() -> None:
    config, kumo, config_hash, frame, library, split, _, direct = _inputs()
    ordered = validate_canonical_hourly(frame)
    queries = list(split.train) + list(split.validation)
    eligible_counts = [len(eligible_candidate_indices(library, query, ordered.index)) for query in queries]
    payload = {
        "status": "frozen causal plan",
        "config_sha256": config_hash,
        "source_sha256": config["hashes"]["source_dataset_sha256"],
        "library_windows": len(library),
        "train_queries": len(split.train),
        "validation_queries": len(split.validation),
        "purged_origin_dates": ["2015-08-30", "2015-08-31"],
        "eligible_candidates_min": min(eligible_counts),
        "eligible_candidates_max": max(eligible_counts),
        "embedding_batches": _expected_batches(
            len(library), config["batching"]["embedding_windows_per_batch"]
        ),
        "candidate_batches": _expected_batches(
            len(queries), config["batching"]["queries_per_candidate_batch"]
        ),
        "direct_prediction_rows": int(len(direct)),
        "nvidia_commit": kumo["model"]["official_commit"],
        "no_reranker_training": True,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_json(RESULTS / "plan.json", payload)
    _write_progress("Stage 1 complete; Stage 2/3 implementation")
    print(json.dumps(payload, indent=2))


def _build_embeddings(
    config: dict, kumo: dict, config_hash: str, frame: pd.DataFrame, library: list,
    standardizer, *, max_batches: int | None,
) -> int:
    source_hash = config["hashes"]["source_dataset_sha256"]
    size = int(config["batching"]["embedding_windows_per_batch"])
    count = _expected_batches(len(library), size)
    pending = []
    for number in range(count):
        manifest_path = EMBED_MANIFESTS / f"embedding_{number:03d}.json"
        valid, reason = verify_batch_manifest(manifest_path, ROOT, config_hash, source_hash)
        if valid:
            continue
        if manifest_path.exists() and "configuration hash conflict" in reason:
            raise ValueError(f"refusing conflicting batch {manifest_path}: {reason}")
        pending.append(number)
    if max_batches is not None:
        pending = pending[:max_batches]
    if not pending:
        return 0
    model, load_report = build_verified_model(kumo, device="cpu")
    load_head(model, ROOT / config["model"]["head_checkpoint"])
    generated = 0
    for number in pending:
        start = number * size; stop = min(len(library), start + size)
        batch = library[start:stop]
        embeddings = extract_embeddings(model, frame, batch, standardizer, batch_size=8)
        output = ARTIFACTS / "embeddings" / f"embedding_{number:03d}.npz"
        atomic_npz(
            output,
            ids=np.asarray(embedding_ids(batch), dtype=f"<U{max(map(len, embedding_ids(batch)))}"),
            embeddings=embeddings,
        )
        manifest = {
            "status": "complete", "stage": "embedding", "batch_number": number,
            "config_sha256": config_hash, "source_sha256": source_hash,
            "first_window_id": window_id(batch[0]), "last_window_id": window_id(batch[-1]),
            "windows": len(batch), "embedding_dimension": int(embeddings.shape[1]),
            "load_report": load_report.__dict__,
            "outputs": [output_record(output, ROOT, len(batch))],
        }
        atomic_json(EMBED_MANIFESTS / f"embedding_{number:03d}.json", manifest)
        generated += 1
        _write_progress("Stage 3 embedding generation")
        print(f"completed embedding batch {number + 1}/{count}", flush=True)
    return generated


def _ensure_static_tables(
    config: dict, frame: pd.DataFrame, library: list,
) -> pd.DataFrame:
    path = RESULTS / "candidate_library.parquet"
    if path.exists():
        return pd.read_parquet(path)
    table = prepare_candidate_library(frame, library, battery_spec_from_config(config))
    atomic_parquet(path, table)
    return table


def _native_equivalence(
    config: dict, frame: pd.DataFrame, library: list, split: SplitPlan,
    embeddings: np.ndarray, standardizer,
) -> None:
    path = RESULTS / "native_equivalence.json"
    if path.exists():
        return
    from sdk.forecasting import knn_forecast  # type: ignore

    ordered = validate_canonical_hourly(frame)
    query = split.train[0]
    query_index = {window_id(item): i for i, item in enumerate(library)}[window_id(query)]
    eligible = eligible_candidate_indices(library, query, ordered.index)
    order, similarities = native_similarity_order(embeddings[eligible], embeddings[query_index])
    raw = ordered.price_eur_mwh.to_numpy(dtype=np.float32)
    continuations = np.stack([
        raw[library[position].target_start:library[position].target_stop] for position in eligible
    ])
    adapter, selected, weights = native_weighted_forecast(
        continuations, similarities, order,
        k=config["candidate_library"]["top_k_per_ordering"],
        temperature=config["candidate_library"]["darr_temperature"],
    )
    official = knn_forecast(
        embeddings[eligible], continuations[:, None, :], embeddings[query_index][None, :],
        k=config["candidate_library"]["top_k_per_ordering"],
        temperature=config["candidate_library"]["darr_temperature"],
    )[0, 0]
    maximum = float(np.max(np.abs(adapter - official)))
    # The official path accumulates float32 continuations through its
    # NaN-safe helper. One raw-price float32 quantization step is 1.53e-5 in
    # this audit; identities, ordering, and weights are otherwise identical.
    tolerance = 2e-5
    if maximum > tolerance:
        raise ValueError(f"custom adapter differs from NVIDIA knn_forecast: {maximum}")
    atomic_json(path, {
        "status": "numerically verified", "query_id": window_id(query),
        "eligible_candidates": len(eligible), "k": len(selected),
        "max_abs_forecast_difference": maximum,
        "absolute_tolerance": tolerance,
        "weight_sum": float(weights.sum()),
        "qualification": "identity/similarity adapter uses the same equations but exposes extra research fields",
    })


def _build_candidates(
    config: dict, config_hash: str, frame: pd.DataFrame, library: list, split: SplitPlan,
    standardizer, direct: pd.DataFrame, *, max_batches: int | None,
) -> int:
    source_hash = config["hashes"]["source_dataset_sha256"]
    embeddings = load_embedding_store(
        root=ROOT, manifest_dir=EMBED_MANIFESTS, expected_ids=embedding_ids(library),
        config_hash=config_hash, source_hash=source_hash,
    )
    _native_equivalence(config, frame, library, split, embeddings, standardizer)
    candidate_library = _ensure_static_tables(config, frame, library)
    patterns = context_pattern_matrix(
        frame, library, standardizer, int(config["candidate_library"]["pattern_hours"])
    )
    direct_lookup = direct_prediction_lookup(direct, config["model"]["direct_model_name"])
    queries = list(split.train) + list(split.validation)
    size = int(config["batching"]["queries_per_candidate_batch"])
    count = _expected_batches(len(queries), size)
    pending = []
    for number in range(count):
        manifest_path = CANDIDATE_MANIFESTS / f"candidate_{number:03d}.json"
        valid, reason = verify_batch_manifest(manifest_path, ROOT, config_hash, source_hash)
        if valid:
            continue
        if manifest_path.exists() and "configuration hash conflict" in reason:
            raise ValueError(f"refusing conflicting batch {manifest_path}: {reason}")
        pending.append(number)
    if max_batches is not None:
        pending = pending[:max_batches]
    generated = 0
    for number in pending:
        start = number * size; stop = min(len(queries), start + size)
        batch = queries[start:stop]
        query_features, query_labels, candidate_features, candidate_labels = build_candidate_batch(
            frame=frame, library_origins=library, query_origins=batch,
            embeddings=embeddings, patterns=patterns, direct_predictions=direct_lookup,
            candidate_library=candidate_library, config=config,
        )
        validate_feature_label_separation(query_features, query_labels)
        validate_feature_label_separation(candidate_features, candidate_labels)
        validate_unique_pairs(candidate_features)
        output_dir = ARTIFACTS / "candidate_batches" / f"candidate_{number:03d}"
        outputs = []
        for name, table in (
            ("query_features", query_features), ("query_labels", query_labels),
            ("candidate_features", candidate_features), ("candidate_labels", candidate_labels),
        ):
            path = output_dir / f"{name}.parquet"
            atomic_parquet(path, table); outputs.append(output_record(path, ROOT, len(table)))
        manifest = {
            "status": "complete", "stage": "candidate", "batch_number": number,
            "config_sha256": config_hash, "source_sha256": source_hash,
            "first_query_id": window_id(batch[0]), "last_query_id": window_id(batch[-1]),
            "queries": len(batch), "candidate_pairs": len(candidate_features),
            "outputs": outputs,
        }
        atomic_json(CANDIDATE_MANIFESTS / f"candidate_{number:03d}.json", manifest)
        generated += 1
        _write_progress("Stage 5 candidate generation")
        print(f"completed candidate batch {number + 1}/{count}", flush=True)
    return generated


def _assemble(config: dict, config_hash: str) -> None:
    source_hash = config["hashes"]["source_dataset_sha256"]
    count = _expected_batches(
        config["queries"]["expected_train_queries"] + config["queries"]["expected_validation_queries"],
        config["batching"]["queries_per_candidate_batch"],
    )
    tables: dict[str, list[pd.DataFrame]] = {name: [] for name in (
        "query_features", "query_labels", "candidate_features", "candidate_labels"
    )}
    for number in range(count):
        manifest_path = CANDIDATE_MANIFESTS / f"candidate_{number:03d}.json"
        valid, reason = verify_batch_manifest(manifest_path, ROOT, config_hash, source_hash)
        if not valid:
            raise RuntimeError(f"cannot assemble: {manifest_path.name} {reason}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["outputs"]:
            name = Path(record["path"]).stem
            tables[name].append(pd.read_parquet(ROOT / record["path"]))
    for name, parts in tables.items():
        frame = pd.concat(parts, ignore_index=True)
        if name.startswith("query_"):
            frame = frame.drop_duplicates("query__id").sort_values("query__id")
        else:
            frame = frame.sort_values(["query__id", "candidate__id"])
        atomic_parquet(RESULTS / f"{name}.parquet", frame.reset_index(drop=True))


def run_build(*, resume: bool, max_batches: int | None) -> None:
    config, kumo, config_hash, frame, library, split, standardizer, direct = _inputs()
    if not resume and (EMBED_MANIFESTS.exists() or CANDIDATE_MANIFESTS.exists()):
        raise RuntimeError("existing batches found; rerun with --resume")
    try:
        generated_embeddings = _build_embeddings(
            config, kumo, config_hash, frame, library, standardizer, max_batches=max_batches,
        )
        remaining_limit = None if max_batches is None else max(0, max_batches - generated_embeddings)
        embedding_total = _expected_batches(
            len(library), config["batching"]["embedding_windows_per_batch"]
        )
        status = _batch_status(
            EMBED_MANIFESTS, "embedding", embedding_total, config_hash,
            config["hashes"]["source_dataset_sha256"],
        )
        if not all(row["complete"] for row in status):
            _write_progress("Stage 3 embedding generation")
            return
        _build_candidates(
            config, config_hash, frame, library, split, standardizer, direct,
            max_batches=remaining_limit,
        )
        candidate_total = _expected_batches(
            len(split.train) + len(split.validation), config["batching"]["queries_per_candidate_batch"]
        )
        candidate_status = _batch_status(
            CANDIDATE_MANIFESTS, "candidate", candidate_total, config_hash,
            config["hashes"]["source_dataset_sha256"],
        )
        if all(row["complete"] for row in candidate_status):
            _assemble(config, config_hash)
            _write_progress("Stage 5 complete; ready for audit")
    except Exception as error:
        _write_progress("blocked during build", str(error))
        raise


def run_status() -> None:
    stage = "status inspection"
    manifest_path = RESULTS / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config, _, config_hash = _load_configs()
            outputs_valid = all(
                verify_output(record, ROOT)[0]
                for record in manifest.get("outputs", [])
            )
            if (
                manifest.get("status") == "complete"
                and manifest.get("config_sha256") == config_hash
                and manifest.get("source_sha256") == config["hashes"]["source_dataset_sha256"]
                and manifest.get("outputs")
                and outputs_valid
            ):
                stage = "complete"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    state = _state_snapshot(stage)
    print(json.dumps(state, indent=2))


def run_audit() -> None:
    from src.research.decision_aware_darr.analysis import audit_final_dataset

    config, _, config_hash, frame, library, split, _, _ = _inputs()
    report = audit_final_dataset(
        root=ROOT, results=RESULTS, config=config, config_hash=config_hash,
        frame=frame, library_origins=library, split=split,
    )
    atomic_json(RESULTS / "audit.json", report)
    _write_progress("Stage 5 audited; ready for Stage 6")
    print(json.dumps(report, indent=2))


def run_analyze() -> None:
    from src.research.decision_aware_darr.analysis import analyze_and_finalize

    config, _, config_hash, frame, library, split, _, _ = _inputs()
    report = analyze_and_finalize(
        root=ROOT, results=RESULTS, config=config, config_hash=config_hash,
        frame=frame, library_origins=library, split=split,
    )
    _write_progress("complete")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "build", "status", "audit", "analyze"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if args.command == "plan": run_plan()
    elif args.command == "build": run_build(resume=args.resume, max_batches=args.max_batches)
    elif args.command == "status": run_status()
    elif args.command == "audit": run_audit()
    else: run_analyze()


if __name__ == "__main__":
    main()
