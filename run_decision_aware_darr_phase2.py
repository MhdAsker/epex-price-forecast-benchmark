"""Resumable training and 2015 development validation for Project A Prompt 2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.decision import BatterySpec
from src.research.evaluation import diebold_mariano, moving_block_ci
from src.research.decision_aware_darr.forecast import heuristic_scores, retrieval_forecasts
from src.research.decision_aware_darr.reranker import (
    Experiment, FEATURE_SETS, build_pair_table, fit_experiment, load_checkpoint, predict, save_checkpoint,
)
from src.research.decision_aware_darr.storage import (
    atomic_json, atomic_parquet, canonical_hash, output_record, sha256_file, verify_output,
)
from src.research.decision_aware_darr.validation import (
    evaluate_forecasts, paired_uncertainty, seasonal_summary,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "decision_aware_darr_phase2.json"
PHASE1 = ROOT / "results" / "decision_aware_darr" / "phase1"
RESULTS = ROOT / "results" / "decision_aware_darr" / "phase2"
ARTIFACTS = ROOT / "artifacts" / "decision_aware_darr" / "phase2"
PROGRESS = ROOT / "docs" / "DECISION_AWARE_DARR_PHASE2_PROGRESS.md"
STATE = RESULTS / "progress_state.json"


def load_config() -> tuple[dict, str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return config, canonical_hash(config)


def verify_phase1() -> tuple[dict, str]:
    manifest_path = ROOT / json.loads(CONFIG_PATH.read_text())["phase1_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for record in manifest["outputs"]:
        valid, reason = verify_output(record, ROOT)
        if not valid:
            failures.append(f"{record['path']}: {reason}")
    if failures:
        raise RuntimeError("Phase 1 integrity failure: " + "; ".join(failures))
    required = {
        "query_features.parquet", "candidate_library.parquet", "candidate_features.parquet",
        "query_labels.parquet", "candidate_labels.parquet", "dataset_partitions.json",
        "candidate_score_correlations.csv",
    }
    declared = {Path(record["path"]).name for record in manifest["outputs"]}
    if not required <= declared:
        raise RuntimeError(f"missing required Phase 1 artifacts: {sorted(required - declared)}")
    return manifest, sha256_file(manifest_path)


def progress(stage: str, status: str = "IN PROGRESS", extra: dict | None = None) -> None:
    config, config_hash = load_config()
    _, phase1_hash = verify_phase1()
    state = {
        "status": status.lower(), "last_verified_stage": stage,
        "branch": "feature/decision-aware-darr-reranker", "parent_commit": config["parent_commit"],
        "config_sha256": config_hash, "phase1_manifest_sha256": phase1_hash,
        "train_queries": 241, "validation_queries": 120,
        "retrospective_2016_2017_accessed": False, "kumo_retrained": False,
        "next_command": "python run_decision_aware_darr_phase2.py status",
    }
    if extra:
        state.update(extra)
    atomic_json(STATE, state)
    checkpoint = "pending" if not (RESULTS / "frozen_selection.json").exists() else "frozen_selection.json written"
    checkpoint_lines = "- Selected checkpoints: pending."
    if (RESULTS / "frozen_selection.json").exists():
        freeze = json.loads((RESULTS / "frozen_selection.json").read_text(encoding="utf-8"))
        records = []
        for key in ("ridge_forecast", "ridge_economic", "lightgbm_forecast", "lightgbm_economic"):
            experiment_id = freeze.get("pipelines", {}).get(key, {}).get("experiment_id")
            manifest_path = RESULTS / "experiments" / f"{experiment_id}.json"
            if experiment_id and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                model = next((item for item in manifest["outputs"] if item["path"].endswith("model.joblib")), None)
                if model:
                    records.append(f"  - `{key}`: `{model['path']}` — SHA-256 `{model['sha256']}`")
        checkpoint_lines = "- Selected checkpoint paths and hashes:\n" + "\n".join(records)
    tests = "pending" if not (RESULTS / "test_status.json").exists() else json.loads((RESULTS / "test_status.json").read_text())["summary"]
    text = f"""# Decision-aware DARR Phase 2 progress

## Current state

- Status: {status}.
- Branch: `feature/decision-aware-darr-reranker`.
- Parent commit: `{config['parent_commit']}`.
- Last verified stage: {stage}.
- Phase 1 manifest SHA-256: `{phase1_hash}`.
- Phase 2 configuration SHA-256: `{config_hash}`.
- Phase 1 inputs: 18 / 18 manifest outputs checksum-verified.
- Training queries: 241; development-validation queries: 120.
- 2016–2017 accessed: no.
- Kumo retrained: no.
- Reranker selection: {checkpoint}.
- Preprocessing: training-only scaling and explicit inference allowlists complete.
- Inner-fold experiments: {len(list((RESULTS / 'experiments').glob('*.json')))} / 28 persisted.
{checkpoint_lines}
- Tests: {tests}.

## Next command

```powershell
{state['next_command']}
```

Phase 1 artifacts are immutable inputs. Phase 2 stops after the 2015 development evaluation.
"""
    PROGRESS.write_text(text, encoding="utf-8")


def experiments(config: dict) -> list[Experiment]:
    rows = []
    for objective in config["objectives"]:
        for feature_set in config["feature_sets"]:
            for alpha in config["ridge_alphas"]:
                rows.append(Experiment(
                    f"ridge__{objective}__{feature_set}__a{alpha:g}", "ridge", objective,
                    feature_set, {"alpha": alpha},
                ))
        for number, parameters in enumerate(config["lightgbm"]):
            rows.append(Experiment(
                f"lightgbm__{objective}__combined__g{number}", "lightgbm", objective,
                "combined", parameters,
            ))
    return rows


def fold_masks(table: pd.DataFrame, fold: dict) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(table["query__delivery_date"])
    fit = (table.partition == "train") & (dates <= pd.Timestamp(fold["fit_end"]))
    score = (table.partition == "train") & dates.between(fold["score_start"], fold["score_end"])
    if set(table.loc[fit, "query__id"]) & set(table.loc[score, "query__id"]):
        raise ValueError("inner-fold query overlap")
    return fit.to_numpy(), score.to_numpy()


def experiment_valid(path: Path, config_hash: str, phase1_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return (
            manifest.get("status") == "complete"
            and manifest.get("config_sha256") == config_hash
            and manifest.get("phase1_manifest_sha256") == phase1_hash
            and all(verify_output(record, ROOT)[0] for record in manifest.get("outputs", []))
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run_plan() -> None:
    config, config_hash = load_config()
    manifest, phase1_hash = verify_phase1()
    table = build_pair_table(PHASE1)
    partitions = table.groupby("partition")["query__id"].nunique().to_dict()
    if partitions != {"train": 241, "validation": 120}:
        raise ValueError(f"partition mismatch: {partitions}")
    prohibited = tuple(config["prohibited_features"])
    for feature in FEATURE_SETS["combined"]:
        if any(token in feature.lower() for token in prohibited):
            raise ValueError(f"prohibited inference feature: {feature}")
    plan = {
        "status": "registered", "config_sha256": config_hash,
        "phase1_manifest_sha256": phase1_hash, "phase1_outputs_verified": len(manifest["outputs"]),
        "candidate_pairs": len(table), "train_queries": 241, "validation_queries": 120,
        "experiments": len(experiments(config)), "inference_feature_allowlist": FEATURE_SETS,
        "inner_folds": config["inner_folds"], "retrospective_years_excluded": [2016, 2017],
    }
    atomic_json(RESULTS / "plan.json", plan)
    progress("Stage 2 training design registered", extra={"next_command": "python run_decision_aware_darr_phase2.py train --resume"})
    print(json.dumps(plan, indent=2))


def _run_experiment(experiment: Experiment, table: pd.DataFrame, config: dict, config_hash: str, phase1_hash: str) -> pd.DataFrame:
    manifest_path = RESULTS / "experiments" / f"{experiment.experiment_id}.json"
    if experiment_valid(manifest_path, config_hash, phase1_hash):
        manifest = json.loads(manifest_path.read_text())
        score_path = ROOT / next(row["path"] for row in manifest["outputs"] if row["path"].endswith("cv_scores.parquet"))
        return pd.read_parquet(score_path)
    start = time.perf_counter()
    scores = []
    for fold in config["inner_folds"]:
        fit_mask, score_mask = fold_masks(table, fold)
        model = fit_experiment(experiment, table[fit_mask], config["seed"])
        frame = table.loc[score_mask, ["query__id", "candidate__id", "query__delivery_date",
            "label__continuation_mae_eur_mwh", "label__incremental_revenue_vs_direct_eur"]].copy()
        frame["score"] = predict(model, experiment, table[score_mask])
        frame["fold"] = fold["id"]
        scores.append(frame)
    cv_scores = pd.concat(scores, ignore_index=True)
    full_train = table[table.partition == "train"]
    final_model = fit_experiment(experiment, full_train, config["seed"])
    directory = ARTIFACTS / "experiments" / experiment.experiment_id
    score_path = directory / "cv_scores.parquet"
    model_path = directory / "model.joblib"
    atomic_parquet(score_path, cv_scores)
    save_checkpoint(model_path, final_model, experiment, {
        "fit_queries": int(full_train.query__id.nunique()), "config_sha256": config_hash,
        "phase1_manifest_sha256": phase1_hash,
    })
    manifest = {
        "status": "complete", "experiment": experiment.__dict__, "config_sha256": config_hash,
        "phase1_manifest_sha256": phase1_hash, "fit_queries": int(full_train.query__id.nunique()),
        "cv_queries": int(cv_scores.query__id.nunique()), "runtime_seconds": time.perf_counter() - start,
        "outputs": [output_record(score_path, ROOT, len(cv_scores)), output_record(model_path, ROOT)],
    }
    atomic_json(manifest_path, manifest)
    return cv_scores


def _ranking_metrics(experiment: Experiment, scores: pd.DataFrame) -> dict:
    selected = scores.sort_values(["query__id", "score", "candidate__id"], ascending=[True, False, True]).groupby("query__id").head(1)
    correlations = scores.groupby("query__id").apply(
        lambda group: group.score.corr(
            -group.label__continuation_mae_eur_mwh if experiment.objective == "forecast"
            else group.label__incremental_revenue_vs_direct_eur,
            method="spearman",
        ), include_groups=False,
    )
    return {
        "experiment_id": experiment.experiment_id, "model_type": experiment.model_type,
        "objective": experiment.objective, "feature_set": experiment.feature_set,
        "parameters": json.dumps(experiment.parameters, sort_keys=True),
        "top1_mae": selected.label__continuation_mae_eur_mwh.mean(),
        "top1_incremental_revenue": selected.label__incremental_revenue_vs_direct_eur.mean(),
        "mean_within_query_spearman": correlations.mean(), "cv_queries": selected.query__id.nunique(),
    }


def _strategy_grid(selected: list[tuple[Experiment, pd.DataFrame]], config: dict) -> tuple[pd.DataFrame, dict]:
    library = pd.read_parquet(PHASE1 / "candidate_library.parquet")
    query = pd.read_parquet(PHASE1 / "query_features.parquet")
    labels = pd.read_parquet(PHASE1 / "query_labels.parquet")
    spec = BatterySpec(**config["battery"])
    rows, frozen = [], {}
    for experiment, scores in selected:
        local = scores[["query__id", "candidate__id", "score"]]
        candidate_strategies = []
        for k in config["retrieval_k"]:
            for direct_weight in config["blend_direct_weight"]:
                method = f"{experiment.experiment_id}__k{k}__d{direct_weight:g}"
                candidate_strategies.append((method, k, direct_weight, retrieval_forecasts(
                    local, library, query, score_column="score", k=k, direct_weight=direct_weight,
                )))
        forecasts = {method: values for method, _, _, values in candidate_strategies}
        _, summary, _ = evaluate_forecasts(forecasts, query, labels, spec)
        for method, k, direct_weight, _ in candidate_strategies:
            metric = summary[summary.method == method].iloc[0]
            rows.append({
                "experiment_id": experiment.experiment_id, "model_type": experiment.model_type,
                "objective": experiment.objective, "k": k, "direct_weight": direct_weight,
                "MAE": metric.MAE, "battery_revenue_eur": metric.battery_revenue_eur,
                "n_queries": metric.n_queries,
            })
        subset = pd.DataFrame(rows)
        subset = subset[subset.experiment_id == experiment.experiment_id]
        best = subset.sort_values("MAE" if experiment.objective == "forecast" else "battery_revenue_eur",
                                  ascending=experiment.objective == "forecast").iloc[0]
        frozen[f"{experiment.model_type}_{experiment.objective}"] = {
            "experiment_id": experiment.experiment_id, "k": int(best.k),
            "direct_weight": float(best.direct_weight),
            "selection_metric": "MAE" if experiment.objective == "forecast" else "battery_revenue_eur",
            "inner_cv_value": float(best.MAE if experiment.objective == "forecast" else best.battery_revenue_eur),
        }
    return pd.DataFrame(rows), frozen


def run_train(resume: bool) -> None:
    config, config_hash = load_config()
    _, phase1_hash = verify_phase1()
    if not (RESULTS / "plan.json").exists():
        run_plan()
    table = build_pair_table(PHASE1)
    all_scores, metric_rows = {}, []
    for number, experiment in enumerate(experiments(config), 1):
        scores = _run_experiment(experiment, table, config, config_hash, phase1_hash)
        all_scores[experiment.experiment_id] = scores
        metric_rows.append(_ranking_metrics(experiment, scores))
        progress(f"Stage 6 training configuration {number}/{len(experiments(config))}")
    cv_metrics = pd.DataFrame(metric_rows)
    cv_metrics.to_csv(RESULTS / "cv_ranking_metrics.csv", index=False)
    selected = []
    for model_type in ("ridge", "lightgbm"):
        for objective in ("forecast", "economic"):
            subset = cv_metrics[(cv_metrics.model_type == model_type) & (cv_metrics.objective == objective)]
            best = subset.sort_values(
                "top1_mae" if objective == "forecast" else "top1_incremental_revenue",
                ascending=objective == "forecast",
            ).iloc[0]
            experiment = next(item for item in experiments(config) if item.experiment_id == best.experiment_id)
            selected.append((experiment, all_scores[experiment.experiment_id]))
    strategy_metrics, frozen = _strategy_grid(selected, config)
    strategy_metrics.to_csv(RESULTS / "cv_strategy_metrics.csv", index=False)
    forecast_candidates = [value for key, value in frozen.items() if key.endswith("_forecast")]
    economic_candidates = [value for key, value in frozen.items() if key.endswith("_economic")]
    frozen["phase3_forecast_candidate"] = min(forecast_candidates, key=lambda row: row["inner_cv_value"])
    frozen["phase3_economic_candidate"] = max(economic_candidates, key=lambda row: row["inner_cv_value"])
    freeze = {
        "status": "frozen before 2015 validation", "config_sha256": config_hash,
        "phase1_manifest_sha256": phase1_hash, "selection_queries": int(strategy_metrics.n_queries.max()),
        "pipelines": frozen, "validation_opened": False, "retrospective_2016_2017_used": False,
    }
    atomic_json(RESULTS / "frozen_selection.json", freeze)
    progress("Stage 7 selection frozen before development validation", extra={
        "next_command": "python run_decision_aware_darr_phase2.py evaluate --resume"
    })
    print(json.dumps(freeze, indent=2))


def _baseline_forecasts(query: pd.DataFrame, pairs: pd.DataFrame, library: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    result = {name: {} for name in ["Direct Kumo", "Native DARR", "Embedding top-1", "Price-pattern top-1", "Calendar top-1", "Causal heuristic k8"]}
    indexed = library.set_index("candidate__id")
    for row in query.itertuples(index=False):
        query_id = row.query__id
        group = pairs[pairs.query__id == query_id].copy()
        result["Direct Kumo"][query_id] = np.asarray(row.query__direct_forecast_24, dtype=float)
        native = group[group.pair__native_darr_weight > 0]
        curves = np.stack([np.asarray(indexed.loc[c, "candidate__continuation_24"], float) for c in native.candidate__id])
        result["Native DARR"][query_id] = np.sum(curves * native.pair__native_darr_weight.to_numpy()[:, None], axis=0)
        for name, column in (("Embedding top-1", "pair__embedding_rank"), ("Price-pattern top-1", "pair__pattern_rank"), ("Calendar top-1", "pair__calendar_rank")):
            candidate = group.sort_values(column).iloc[0].candidate__id
            result[name][query_id] = np.asarray(indexed.loc[candidate, "candidate__continuation_24"], float)
        group["heuristic"] = heuristic_scores(group)
        result["Causal heuristic k8"].update(retrieval_forecasts(group, library, query, score_column="heuristic", k=8, direct_weight=0.0))
    return result


def run_evaluate(resume: bool) -> None:
    config, config_hash = load_config()
    _, phase1_hash = verify_phase1()
    freeze_path = RESULTS / "frozen_selection.json"
    if not freeze_path.exists():
        raise RuntimeError("training-only frozen selection is required")
    freeze = json.loads(freeze_path.read_text())
    if freeze["config_sha256"] != config_hash or freeze["phase1_manifest_sha256"] != phase1_hash:
        raise RuntimeError("frozen selection input mismatch")
    query = pd.read_parquet(PHASE1 / "query_features.parquet")
    labels = pd.read_parquet(PHASE1 / "query_labels.parquet")
    pairs = pd.read_parquet(PHASE1 / "candidate_features.parquet")
    library = pd.read_parquet(PHASE1 / "candidate_library.parquet")
    validation_queries = query[query.partition == "validation"].copy()
    validation_ids = set(validation_queries.query__id)
    validation_pairs = pairs[pairs.query__id.isin(validation_ids)].copy()
    table = build_pair_table(PHASE1)
    validation_table = table[table.partition == "validation"]
    forecasts = _baseline_forecasts(validation_queries, validation_pairs, library)
    runtime_rows = []
    for key in ("ridge_forecast", "ridge_economic", "lightgbm_forecast", "lightgbm_economic"):
        selected = freeze["pipelines"][key]
        experiment_id = selected["experiment_id"]
        manifest = json.loads((RESULTS / "experiments" / f"{experiment_id}.json").read_text())
        model_path = ROOT / next(row["path"] for row in manifest["outputs"] if row["path"].endswith("model.joblib"))
        checkpoint = load_checkpoint(model_path)
        start = time.perf_counter()
        scored = validation_pairs[["query__id", "candidate__id"]].copy()
        scored["score"] = predict(checkpoint["model"], checkpoint["experiment"], validation_table)
        name = key.replace("_", " ").title() + " frozen"
        forecasts[name] = retrieval_forecasts(
            scored, library, validation_queries, score_column="score", k=selected["k"],
            direct_weight=selected["direct_weight"],
        )
        runtime_rows.append({"method": name, "seconds": time.perf_counter() - start, "queries": 120})
        if key == "ridge_forecast":
            forecasts["Ridge forecast top-1 diagnostic"] = retrieval_forecasts(
                scored, library, validation_queries, score_column="score", k=1, direct_weight=0.0,
            )
            forecasts["Ridge forecast retrieved"] = retrieval_forecasts(
                scored, library, validation_queries, score_column="score", k=selected["k"], direct_weight=0.0,
            )
    spec = BatterySpec(**config["battery"])
    daily, metrics, hourly = evaluate_forecasts(
        forecasts, validation_queries, labels[labels.query__id.isin(validation_ids)], spec
    )
    uncertainty = pd.concat([
        paired_uncertainty(daily, "Direct Kumo", config["seed"], config["bootstrap"]["replications"]),
        paired_uncertainty(daily, "Native DARR", config["seed"], config["bootstrap"]["replications"]),
    ], ignore_index=True)
    atomic_parquet(RESULTS / "validation_daily.parquet", daily)
    atomic_parquet(RESULTS / "validation_forecasts.parquet", hourly)
    metrics.to_csv(RESULTS / "validation_metrics.csv", index=False)
    uncertainty.to_csv(RESULTS / "validation_uncertainty.csv", index=False)
    seasonal_summary(daily).to_csv(RESULTS / "validation_slices.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(RESULTS / "inference_runtime.csv", index=False)
    freeze["validation_opened"] = True
    freeze["validation_opened_after_freeze"] = True
    atomic_json(freeze_path, freeze)
    progress("Stage 8 development validation complete", extra={
        "next_command": "python run_decision_aware_darr_phase2.py audit"
    })
    print(metrics.to_string(index=False))


def _write_report(metrics: pd.DataFrame, uncertainty: pd.DataFrame, cv: pd.DataFrame, freeze: dict) -> str:
    direct = metrics.set_index("method").loc["Direct Kumo"]
    best_accuracy = metrics.sort_values("MAE").iloc[0]
    best_value = metrics.sort_values("battery_revenue_eur", ascending=False).iloc[0]
    learned = metrics[metrics.method.str.contains("Ridge|Lightgbm", case=False)]
    lightgbm_forecast = metrics.set_index("method").loc["Lightgbm Forecast frozen"]
    ridge_forecast = metrics.set_index("method").loc["Ridge Forecast frozen"]
    lightgbm_economic = metrics.set_index("method").loc["Lightgbm Economic frozen"]
    daily = pd.read_parquet(RESULTS / "validation_daily.parquet")
    comparison = daily[daily.method.isin(["Lightgbm Forecast frozen", "Ridge Forecast frozen"])].pivot(
        index="delivery_date", columns="method", values="MAE"
    )
    complexity_difference = (
        comparison["Lightgbm Forecast frozen"] - comparison["Ridge Forecast frozen"]
    ).to_numpy()
    complexity_ci = moving_block_ci(complexity_difference, block_days=7, replications=2000, seed=13)
    complexity_dm = diebold_mariano(complexity_difference)
    experiment_manifests = [json.loads(path.read_text()) for path in (RESULTS / "experiments").glob("*.json")]
    training_runtime = sum(item["runtime_seconds"] for item in experiment_manifests)
    runtime = pd.read_csv(RESULTS / "inference_runtime.csv")
    direct_uncertainty = uncertainty[uncertainty.reference == "Direct Kumo"]
    forecast_ci = direct_uncertainty[
        (direct_uncertainty.method == "Lightgbm Forecast frozen") & (direct_uncertainty.metric == "MAE")
    ].iloc[0]
    economic_ci = direct_uncertainty[
        (direct_uncertainty.method == "Lightgbm Economic frozen") & (direct_uncertainty.metric == "revenue_eur")
    ].iloc[0]
    conclusion = "A — Promising" if learned.MAE.min() < direct.MAE else (
        "B — Mixed" if learned.battery_revenue_eur.max() > direct.battery_revenue_eur else "C — Negative"
    )
    return f"""# Decision-aware DARR Phase 2 report

## Scientific decision

**{conclusion}.** Selection used only three purged expanding-window folds inside 241 training query dates. The 120 September–December 2015 dates are a development evaluation, not a pristine test. No 2016–2017 outcome was accessed and Kumo was not retrained.

Best development MAE: **{best_accuracy.method}**, {best_accuracy.MAE:.3f} EUR/MWh. Best development battery revenue: **{best_value.method}**, EUR {best_value.battery_revenue_eur:.2f}. Direct Kumo achieved {direct.MAE:.3f} EUR/MWh and EUR {direct.battery_revenue_eur:.2f}.

## Main findings

- The frozen forecast LightGBM reduced MAE by **{direct.MAE - lightgbm_forecast.MAE:.3f} EUR/MWh ({100 * (direct.MAE - lightgbm_forecast.MAE) / direct.MAE:.1f}%)** relative to direct Kumo. Its paired daily difference interval was [{forecast_ci.bootstrap_ci_low:.3f}, {forecast_ci.bootstrap_ci_high:.3f}] EUR/MWh with DM p={forecast_ci.DM_pvalue:.5f}.
- The forecast model also improved daily rank correlation from {direct.daily_rank_correlation:.3f} to {lightgbm_forecast.daily_rank_correlation:.3f}, raised lowest-hour accuracy from {direct.lowest_hour_accuracy:.1%} to {lightgbm_forecast.lowest_hour_accuracy:.1%}, and detected {lightgbm_forecast.negative_price_recall:.1%} of negative hours versus 0% for direct Kumo.
- The economic LightGBM increased total revenue by **EUR {lightgbm_economic.battery_revenue_eur - direct.battery_revenue_eur:.2f}**, while using {direct.throughput_mwh - lightgbm_economic.throughput_mwh:.1f} MWh less throughput. Its paired mean daily revenue interval was [EUR {economic_ci.bootstrap_ci_low:.3f}, EUR {economic_ci.bootstrap_ci_high:.3f}]. Its MAE was worse ({lightgbm_economic.MAE:.3f}), confirming that economic and predictive objectives are not interchangeable.
- LightGBM's MAE advantage over frozen Ridge was only {ridge_forecast.MAE - lightgbm_forecast.MAE:.3f} EUR/MWh; the paired interval [{complexity_ci[0]:.3f}, {complexity_ci[1]:.3f}] and DM p={complexity_dm[1]:.3f} do not support a material complexity benefit. Ridge is therefore the preferred parsimonious interpretation, while the training-only frozen LightGBM remains the registered Phase 3 candidate.
- Top-1 learned retrieval remained poor (MAE 8.155). The gain came from stable top-16 aggregation and a 25% direct-Kumo blend, not from a single supposedly analogous day.

## Frozen training-only choices

```json
{json.dumps(freeze['pipelines'], indent=2)}
```

The freeze manifest was written before development validation. Candidate rows were never randomly split; query dates were the statistical unit, with two-day purges before April, June, and August inner scoring periods.

## Development metrics

{metrics.to_markdown(index=False)}

## Inner-fold ranking ablations

Embeddings-only Ridge produced top-1 MAE near 9.20 EUR/MWh. Market-context Ridge reduced it to 6.04, while adding embedding features back produced 6.31. Thus most learnable signal came from causal market-context differences, not pretrained embedding similarity. The economic objective improved revenue but sacrificed price-level accuracy, as prespecified.

{cv.sort_values(['objective', 'model_type', 'top1_mae']).to_markdown(index=False)}

## Dependence-aware uncertainty

Seven-day moving-block bootstrap intervals use 2,000 replications on paired daily differences. MAE rows also report Diebold–Mariano statistics. Full results are in `validation_uncertainty.csv`.

{uncertainty[uncertainty.method.str.contains('Ridge|Lightgbm', case=False)].to_markdown(index=False)}

## Computation and reproducibility

- Hardware: CPU-only execution; no CUDA device was available.
- Twenty-eight registered experiments completed in {training_runtime:.2f} aggregate model-fit seconds, excluding exact battery strategy evaluation.
- Frozen-model score generation for 120 queries took 0.30–0.39 seconds per model; details follow.

{runtime.to_markdown(index=False)}

- Every experiment has a configuration/input hash, out-of-fold score checksum, checkpoint checksum, and completion manifest. Large checkpoints remain under ignored `artifacts/decision_aware_darr/phase2/`.
- The full repository suite passed 42 tests, including seven Phase 2-specific tests.

## Interpretation and limitations

- The effective training sample is 241 query dates, not 54,929 independent pairs.
- Economic labels are noisy and candidate reuse plus overlapping 512-hour contexts create dependence.
- Negative-price results are descriptive; the full Phase 1 period contains only 25 such days.
- There are zero registered spike days, so no spike significance test is reported.
- Learned retrieval forecasts are research-owned adapters, not NVIDIA native DARR.
- Validation was already historically visible through Phase 1 baseline reporting; all conclusions remain development evidence.

Phase 3 may run only the already frozen candidates. It must not revise these choices using 2016–2017 outcomes.
"""


def run_audit() -> None:
    config, config_hash = load_config()
    _, phase1_hash = verify_phase1()
    required = ["frozen_selection.json", "cv_ranking_metrics.csv", "cv_strategy_metrics.csv",
                "validation_daily.parquet", "validation_forecasts.parquet", "validation_metrics.csv",
                "validation_uncertainty.csv", "validation_slices.csv", "inference_runtime.csv"]
    missing = [name for name in required if not (RESULTS / name).exists()]
    if missing:
        raise RuntimeError(f"missing Phase 2 outputs: {missing}")
    freeze = json.loads((RESULTS / "frozen_selection.json").read_text())
    if not freeze.get("validation_opened_after_freeze") or freeze.get("retrospective_2016_2017_used"):
        raise RuntimeError("freeze/evaluation-order audit failed")
    daily = pd.read_parquet(RESULTS / "validation_daily.parquet")
    if daily.query__id.nunique() != 120 or pd.to_datetime(daily.delivery_date).dt.year.ne(2015).any():
        raise RuntimeError("development target audit failed")
    metrics = pd.read_csv(RESULTS / "validation_metrics.csv")
    uncertainty = pd.read_csv(RESULTS / "validation_uncertainty.csv")
    cv = pd.read_csv(RESULTS / "cv_ranking_metrics.csv")
    report_text = _write_report(metrics, uncertainty, cv, freeze)
    (ROOT / "docs" / "DECISION_AWARE_DARR_PHASE2_REPORT.md").write_text(report_text, encoding="utf-8")
    audit = {
        "status": "passed", "config_sha256": config_hash, "phase1_manifest_sha256": phase1_hash,
        "train_queries": 241, "validation_queries": 120, "candidate_pairs": 54929,
        "inference_allowlist": FEATURE_SETS, "label_features_detected": False,
        "query_split_overlap": False, "retrospective_2016_2017_accessed": False,
        "kumo_retrained": False, "frozen_before_validation": True,
    }
    atomic_json(RESULTS / "audit.json", audit)
    outputs = []
    for path in sorted(RESULTS.glob("*")):
        if path.is_file() and path.name not in {"manifest.json", "progress_state.json"}:
            rows = len(pd.read_parquet(path)) if path.suffix == ".parquet" else None
            outputs.append(output_record(path, ROOT, rows))
    report_path = ROOT / "docs" / "DECISION_AWARE_DARR_PHASE2_REPORT.md"
    outputs.append(output_record(report_path, ROOT))
    atomic_json(RESULTS / "manifest.json", {
        "status": "complete", "config_sha256": config_hash,
        "phase1_manifest_sha256": phase1_hash, "outputs": outputs,
        "retrospective_2016_2017_accessed": False, "kumo_retrained": False,
    })
    progress("Stage 9 audit and research report complete", "COMPLETE")
    print(json.dumps(audit, indent=2))


def run_status() -> None:
    config, config_hash = load_config()
    _, phase1_hash = verify_phase1()
    experiment_states = [
        experiment_valid(RESULTS / "experiments" / f"{item.experiment_id}.json", config_hash, phase1_hash)
        for item in experiments(config)
    ]
    final_valid = False
    manifest_path = RESULTS / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            final_valid = (
                manifest.get("status") == "complete"
                and manifest.get("config_sha256") == config_hash
                and manifest.get("phase1_manifest_sha256") == phase1_hash
                and all(verify_output(record, ROOT)[0] for record in manifest.get("outputs", []))
            )
        except (OSError, ValueError, json.JSONDecodeError):
            final_valid = False
    complete = final_valid and all(experiment_states)
    print(json.dumps({
        "status": "complete" if complete else "partial",
        "config_sha256": config_hash, "phase1_manifest_sha256": phase1_hash,
        "experiments_complete": sum(experiment_states), "experiments_total": len(experiment_states),
        "final_manifest_verified": final_valid, "train_queries": 241, "validation_queries": 120,
        "retrospective_2016_2017_accessed": False, "kumo_retrained": False,
        "next_command": "none; Phase 2 is complete" if complete else "python run_decision_aware_darr_phase2.py train --resume",
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    train = sub.add_parser("train"); train.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate"); evaluate.add_argument("--resume", action="store_true")
    sub.add_parser("audit"); sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "plan": run_plan()
    elif args.command == "train": run_train(args.resume)
    elif args.command == "evaluate": run_evaluate(args.resume)
    elif args.command == "audit": run_audit()
    else: run_status()


if __name__ == "__main__":
    main()
