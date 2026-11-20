"""Final integrity audit and non-learned retrieval baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research.decision import BatterySpec, optimize_day, realized_value
from src.research.decision_aware_darr.causal import SplitPlan, window_id
from src.research.decision_aware_darr.dataset import (
    validate_feature_label_separation,
    validate_unique_pairs,
)
from src.research.decision_aware_darr.storage import atomic_json, atomic_parquet, output_record
from src.research.kumo.data import IssuanceWindow, validate_canonical_hourly


FINAL_TABLES = (
    "candidate_library", "query_features", "query_labels", "candidate_features", "candidate_labels"
)


def _load_tables(results: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in FINAL_TABLES:
        path = results / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"required final table missing: {path}")
        tables[name] = pd.read_parquet(path)
    return tables


def audit_final_dataset(
    *, root: Path, results: Path, config: dict, config_hash: str,
    frame: pd.DataFrame, library_origins: Sequence[IssuanceWindow], split: SplitPlan,
) -> dict:
    tables = _load_tables(results)
    query_features = tables["query_features"]
    query_labels = tables["query_labels"]
    candidate_features = tables["candidate_features"]
    candidate_labels = tables["candidate_labels"]
    candidate_library = tables["candidate_library"]
    validate_feature_label_separation(query_features, query_labels)
    validate_feature_label_separation(candidate_features, candidate_labels)
    validate_unique_pairs(candidate_features)
    if query_features.query__id.nunique() != 361 or len(query_features) != 361:
        raise ValueError("final query table does not contain exactly 361 unique queries")
    if len(candidate_library) != config["candidate_library"]["expected_windows"]:
        raise ValueError("candidate library size mismatch")
    if set(candidate_features[["query__id", "candidate__id"]].itertuples(index=False, name=None)) != set(
        candidate_labels[["query__id", "candidate__id"]].itertuples(index=False, name=None)
    ):
        raise ValueError("feature and label pair keys differ")
    if set(query_features.query__id) != set(query_labels.query__id):
        raise ValueError("query feature and label keys differ")
    if query_features.duplicated("query__id").any() or query_labels.duplicated("query__id").any():
        raise ValueError("duplicate query records detected")

    joined = candidate_features[["query__id", "candidate__id"]].merge(
        query_features[["query__id", "query__forecast_origin"]], on="query__id", validate="many_to_one"
    ).merge(
        candidate_library[["candidate__id", "candidate__continuation_last_delivery"]],
        on="candidate__id", validate="many_to_one",
    )
    issuance = pd.to_datetime(joined.query__forecast_origin, utc=True)
    continuation_end = pd.to_datetime(joined.candidate__continuation_last_delivery).dt.tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    if not (continuation_end < issuance).all():
        raise ValueError("candidate continuation at or after query issuance detected")
    if pd.to_datetime(query_features.query__delivery_date).dt.year.ne(2015).any():
        raise ValueError("evaluation-year query leaked into historical training dataset")
    split.assert_no_label_overlap()
    counts = candidate_features.groupby("query__id").size()
    partitions = query_features.groupby("partition").size().to_dict()
    if partitions != {"train": 241, "validation": 120}:
        raise ValueError(f"partition counts differ from registration: {partitions}")
    atomic_json(results / "dataset_partitions.json", {
        "train": {
            "queries": 241,
            "first_delivery_date": query_features[query_features.partition == "train"].query__delivery_date.min(),
            "last_delivery_date": query_features[query_features.partition == "train"].query__delivery_date.max(),
        },
        "validation": {
            "queries": 120,
            "first_delivery_date": query_features[query_features.partition == "validation"].query__delivery_date.min(),
            "last_delivery_date": query_features[query_features.partition == "validation"].query__delivery_date.max(),
        },
        "purged_query_dates": ["2015-08-30", "2015-08-31"],
        "target_overlap": False,
        "evaluation_years_excluded": [2016, 2017],
    })
    atomic_json(results / "feature_schema.json", {
        "query_features": list(query_features.columns),
        "candidate_features": list(candidate_features.columns),
        "candidate_library": list(candidate_library.columns),
        "query_labels": list(query_labels.columns),
        "candidate_labels": list(candidate_labels.columns),
        "inference_rule": "query/candidate/pair/source prefixes only; label__ columns prohibited",
    })
    outputs = [output_record(results / f"{name}.parquet", root, len(tables[name])) for name in FINAL_TABLES]
    return {
        "status": "passed",
        "config_sha256": config_hash,
        "source_sha256": config["hashes"]["source_dataset_sha256"],
        "query_rows": len(query_features),
        "train_queries": partitions["train"],
        "validation_queries": partitions["validation"],
        "candidate_library_rows": len(candidate_library),
        "candidate_pair_rows": len(candidate_features),
        "candidate_pairs_min_per_query": int(counts.min()),
        "candidate_pairs_max_per_query": int(counts.max()),
        "candidate_pairs_mean_per_query": float(counts.mean()),
        "strict_causal_pairs_verified": int(len(joined)),
        "duplicate_query_candidate_pairs": 0,
        "feature_label_key_match": True,
        "feature_label_column_separation": True,
        "train_validation_label_overlap": False,
        "evaluation_year_queries_present": False,
        "outputs": outputs,
    }


def _curve_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - actual
    rank = float(spearmanr(actual, prediction).statistic)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "rank_correlation": rank if np.isfinite(rank) else 0.0,
        "spread_error": float(
            (prediction.max() - prediction.min()) - (actual.max() - actual.min())
        ),
        "lowest_hour_correct": bool(np.argmin(prediction) == np.argmin(actual)),
        "highest_hour_correct": bool(np.argmax(prediction) == np.argmax(actual)),
    }


def _baseline_daily(tables: dict[str, pd.DataFrame], spec: BatterySpec) -> pd.DataFrame:
    queries = tables["query_features"].merge(tables["query_labels"], on="query__id", validate="one_to_one")
    features = tables["candidate_features"]
    library = tables["candidate_library"].set_index("candidate__id")
    rows = []
    for query in queries.itertuples(index=False):
        query_id = query.query__id
        actual = np.asarray(query.label__actual_24, dtype=float)
        direct = np.asarray(query.query__direct_forecast_24, dtype=float)
        pairs = features[features.query__id == query_id]
        selected = {}
        selected["Direct Kumo"] = direct
        native = pairs[pairs.pair__native_darr_weight > 0].copy()
        continuations = np.stack([
            np.asarray(library.loc[candidate_id, "candidate__continuation_24"], dtype=float)
            for candidate_id in native.candidate__id
        ])
        selected["Native DARR"] = np.sum(
            continuations * native.pair__native_darr_weight.to_numpy()[:, None], axis=0
        )
        for name, rank_column in (
            ("Embedding top-1", "pair__embedding_rank"),
            ("Calendar top-1", "pair__calendar_rank"),
            ("Price-pattern top-1", "pair__pattern_rank"),
        ):
            candidate_id = pairs.sort_values(rank_column).iloc[0].candidate__id
            selected[name] = np.asarray(
                library.loc[candidate_id, "candidate__continuation_24"], dtype=float
            )
        oracle_charge, oracle_discharge, _ = optimize_day(actual, spec)
        oracle = realized_value(actual, oracle_charge, oracle_discharge, spec)
        for method, prediction in selected.items():
            charge, discharge, _ = optimize_day(prediction, spec)
            revenue = realized_value(actual, charge, discharge, spec)
            row = {
                "query__id": query_id,
                "delivery_date": query.query__delivery_date,
                "partition": query.partition,
                "method": method,
                "revenue_eur": revenue,
                "regret_eur": oracle - revenue,
                "throughput_mwh": float(np.sum(charge + discharge)),
                "actual_negative_hours": query.label__negative_price_hours,
                "actual_spike_hours": query.label__spike_hours_ge_100,
                "actual_std": float(np.std(actual)),
            }
            row.update(_curve_metrics(actual, prediction))
            rows.append(row)
    daily = pd.DataFrame(rows)
    dates = pd.to_datetime(daily.delivery_date)
    daily["season"] = dates.dt.month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"})
    volatility_cutoff = float(daily.drop_duplicates("query__id").actual_std.quantile(0.75))
    daily["high_volatility"] = daily.actual_std >= volatility_cutoff
    return daily


def _baseline_summary(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.groupby(["partition", "method"], as_index=False).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        bias=("bias", "mean"), daily_rank_correlation=("rank_correlation", "mean"),
        mean_spread_error=("spread_error", "mean"),
        lowest_hour_accuracy=("lowest_hour_correct", "mean"),
        highest_hour_accuracy=("highest_hour_correct", "mean"),
        battery_revenue_eur=("revenue_eur", "sum"), regret_eur=("regret_eur", "sum"),
        throughput_mwh=("throughput_mwh", "sum"), n_queries=("query__id", "nunique"),
    ).sort_values(["partition", "MAE"])


def _regime_summary(daily: pd.DataFrame) -> pd.DataFrame:
    groups = []
    regimes = {
        "all": np.ones(len(daily), dtype=bool),
        "negative_price_day": daily.actual_negative_hours > 0,
        "price_spike_day": daily.actual_spike_hours > 0,
        "high_volatility": daily.high_volatility,
        "ordinary": (daily.actual_negative_hours == 0) & (daily.actual_spike_hours == 0) & ~daily.high_volatility,
    }
    for regime, mask in regimes.items():
        subset = daily[mask]
        if subset.empty:
            groups.append(pd.DataFrame([{"regime": regime, "method": "none", "n_queries": 0}]))
            continue
        result = subset.groupby("method", as_index=False).agg(
            MAE=("MAE", "mean"), rank_correlation=("rank_correlation", "mean"),
            spread_error=("spread_error", "mean"), revenue_eur=("revenue_eur", "sum"),
            n_queries=("query__id", "nunique"),
        )
        result.insert(0, "regime", regime); groups.append(result)
    season = daily.groupby(["season", "method"], as_index=False).agg(
        MAE=("MAE", "mean"), rank_correlation=("rank_correlation", "mean"),
        spread_error=("spread_error", "mean"), revenue_eur=("revenue_eur", "sum"),
        n_queries=("query__id", "nunique"),
    ).rename(columns={"season": "regime"})
    return pd.concat([*groups, season], ignore_index=True)


def _candidate_diagnostics(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    merged = tables["candidate_features"].merge(
        tables["candidate_labels"], on=["query__id", "candidate__id"], validate="one_to_one"
    )
    targets = {
        "continuation_mae": "label__continuation_mae_eur_mwh",
        "rank_quality": "label__daily_rank_correlation",
        "absolute_spread_error": "label__spread_error_eur_mwh",
        "incremental_battery_value": "label__incremental_revenue_vs_direct_eur",
    }
    rows = []
    for score_name, score_column, direction in (
        ("embedding_similarity", "pair__embedding_similarity", 1),
        ("calendar_distance", "pair__calendar_distance", -1),
        ("pattern_distance", "pair__pattern_distance", -1),
    ):
        score = direction * merged[score_column]
        for target_name, target_column in targets.items():
            target = merged[target_column].abs() if target_name == "absolute_spread_error" else merged[target_column]
            rho = float(spearmanr(score, target).statistic)
            within = merged.assign(_score=score, _target=target).groupby("query__id").apply(
                lambda group: spearmanr(group._score, group._target).statistic,
                include_groups=False,
            )
            rows.append({
                "retrieval_score": score_name, "outcome": target_name,
                "global_spearman": rho, "mean_within_query_spearman": float(np.nanmean(within)),
                "candidate_pairs": len(merged), "queries": merged.query__id.nunique(),
            })
    correlations = pd.DataFrame(rows)
    quality = merged.groupby("query__id").agg(
        mae_p10=("label__continuation_mae_eur_mwh", lambda x: x.quantile(0.1)),
        mae_p90=("label__continuation_mae_eur_mwh", lambda x: x.quantile(0.9)),
        value_p10=("label__incremental_revenue_vs_direct_eur", lambda x: x.quantile(0.1)),
        value_p90=("label__incremental_revenue_vs_direct_eur", lambda x: x.quantile(0.9)),
    )
    candidate_sets = {
        query: set(group.candidate__id) for query, group in merged.groupby("query__id")
    }
    ordered_queries = sorted(candidate_sets)
    adjacent_jaccard = []
    for first, second in zip(ordered_queries[:-1], ordered_queries[1:]):
        a, b = candidate_sets[first], candidate_sets[second]
        adjacent_jaccard.append(len(a & b) / len(a | b))
    reuse = merged.groupby("candidate__id").size()
    summary = {
        "candidate_pairs": len(merged), "unique_queries": merged.query__id.nunique(),
        "distinct_candidate_histories": merged.candidate__id.nunique(),
        "mean_candidate_reuse": float(reuse.mean()), "max_candidate_reuse": int(reuse.max()),
        "mean_adjacent_query_candidate_jaccard": float(np.mean(adjacent_jaccard)),
        "mean_embedding_similarity": float(merged.pair__embedding_similarity.mean()),
        "std_embedding_similarity": float(merged.pair__embedding_similarity.std()),
        "mean_continuation_mae": float(merged.label__continuation_mae_eur_mwh.mean()),
        "mean_incremental_battery_value": float(merged.label__incremental_revenue_vs_direct_eur.mean()),
        "embedding_similarity_quantiles": {
            str(q): float(merged.pair__embedding_similarity.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "continuation_mae_quantiles": {
            str(q): float(merged.label__continuation_mae_eur_mwh.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "incremental_value_quantiles": {
            str(q): float(merged.label__incremental_revenue_vs_direct_eur.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "queries_with_mae_range_over_1_eur_pct": float(
            np.mean((quality.mae_p90 - quality.mae_p10) > 1.0) * 100
        ),
        "queries_with_value_range_over_5_eur_pct": float(
            np.mean((quality.value_p90 - quality.value_p10) > 5.0) * 100
        ),
    }
    return correlations, summary


def analyze_and_finalize(
    *, root: Path, results: Path, config: dict, config_hash: str,
    frame: pd.DataFrame, library_origins: Sequence[IssuanceWindow], split: SplitPlan,
) -> dict:
    audit_path = results / "audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "passed":
        raise RuntimeError("a passing audit is required before baseline analysis")
    tables = _load_tables(results)
    daily = _baseline_daily(tables, BatterySpec(**config["battery"]))
    summary = _baseline_summary(daily)
    regimes = _regime_summary(daily)
    correlations, quality = _candidate_diagnostics(tables)
    atomic_parquet(results / "baseline_daily.parquet", daily)
    summary.to_csv(results / "baseline_metrics.csv", index=False)
    regimes.to_csv(results / "baseline_regimes.csv", index=False)
    correlations.to_csv(results / "candidate_score_correlations.csv", index=False)
    atomic_json(results / "data_quality.json", quality)
    validation = summary[summary.partition == "validation"].sort_values("MAE")
    report = {
        "status": "complete Prompt 1 dataset and non-learned baseline analysis",
        "config_sha256": config_hash,
        "queries": int(tables["query_features"].query__id.nunique()),
        "candidate_pairs": int(len(tables["candidate_features"])),
        "distinct_candidates": int(tables["candidate_features"].candidate__id.nunique()),
        "validation_best_MAE_method": str(validation.iloc[0].method),
        "validation_best_MAE": float(validation.iloc[0].MAE),
        "quality": quality,
        "prompt2_readiness": "conditional: sufficient pair rows for a constrained reranker, but only 241 unique training query dates",
    }
    atomic_json(results / "analysis_summary.json", report)
    _write_report(results, config, summary, correlations, regimes, quality, report)
    outputs = []
    for path in sorted(results.glob("*")):
        if path.is_file() and path.name not in {"manifest.json", "progress_state.json"}:
            rows = None
            if path.suffix == ".parquet": rows = len(pd.read_parquet(path))
            outputs.append(output_record(path, root, rows))
    manifest = {
        "status": "complete", "config_sha256": config_hash,
        "source_sha256": config["hashes"]["source_dataset_sha256"],
        "checkpoint_sha256": config["model"]["head_checkpoint_sha256"],
        "no_reranker_trained": True, "retrospective_2016_2017_used_as_training_queries": False,
        "outputs": outputs,
    }
    atomic_json(results / "manifest.json", manifest)
    return report


def _write_report(
    results: Path, config: dict, metrics: pd.DataFrame, correlations: pd.DataFrame,
    regimes: pd.DataFrame, quality: dict, report: dict,
) -> None:
    validation = metrics[metrics.partition == "validation"].sort_values("MAE")
    correlation = correlations[
        (correlations.retrieval_score == "embedding_similarity")
        & (correlations.outcome == "continuation_mae")
    ].iloc[0]
    negative = regimes[regimes.regime == "negative_price_day"]
    spike = regimes[regimes.regime == "price_spike_day"]
    text = f"""# Decision-aware DARR Phase 1 report

## Conclusion

Prompt 1 is complete. It constructs a causal, checksum-verified historical candidate dataset and does not train a reranker. The dataset contains **{report['queries']} query dates** and **{report['candidate_pairs']:,} query-candidate pairs** drawn from **{report['distinct_candidates']} distinct historical continuations**.

The sample is large in pair rows but small in independent query dates: only 241 training and 120 validation days. Prompt 2 should therefore use a constrained, strongly regularized reranker and dependence-aware evaluation rather than a flexible high-capacity model.

## Causal protocol

- Forecast origin: noon Europe/Berlin on D-1.
- Query period: 2015 only; 2016–2017 are excluded from training-data construction.
- Training queries: 2015-01-01 through 2015-08-29.
- Validation queries: 2015-09-01 through 2015-12-29.
- The August 30–31 origins are purged so 72-step labels cannot cross the partition boundary.
- Every candidate's final continuation delivery hour is strictly earlier than query issuance.
- Query outcomes and ex-post regimes exist only in label tables.

The adapter reproduces NVIDIA's embedding normalization, cosine ordering, top-k selection, and temperature-softmax forecast. It exposes identities and similarities that `knn_forecast` does not return; numerical equivalence is recorded in `native_equivalence.json`.

## Validation baselines

{validation.to_markdown(index=False)}

No hyperparameter is selected using 2016–2017. These are descriptive non-learned baselines, not reranker results.

## Candidate-quality evidence

- Embedding similarity versus continuation MAE global Spearman: **{correlation.global_spearman:.3f}**.
- Mean within-query Spearman: **{correlation.mean_within_query_spearman:.3f}**.
- Mean adjacent-query candidate-set Jaccard: **{quality['mean_adjacent_query_candidate_jaccard']:.3f}**.
- Queries with a candidate MAE p90-p10 range above 1 EUR/MWh: **{quality['queries_with_mae_range_over_1_eur_pct']:.1f}%**.
- Queries with an incremental-value p90-p10 range above EUR 5: **{quality['queries_with_value_range_over_5_eur_pct']:.1f}%**.

These statistics determine whether candidate choice contains learnable signal. High pair counts do not remove dependence caused by candidate reuse and overlapping 512-hour contexts.

## Difficult regimes

Negative-price-day rows: {int(negative.n_queries.max()) if len(negative) else 0} queries. Spike-day rows: {int(spike.n_queries.max()) if len(spike) and spike.method.ne('none').any() else 0} queries. Ex-post regimes are evaluation-only and never feature columns. Full seasonal and regime tables are in `baseline_regimes.csv`.

## Prompt 2 inputs

- `query_features.parquet`
- `candidate_library.parquet`
- `candidate_features.parquet`
- `query_labels.parquet`
- `candidate_labels.parquet`
- `dataset_partitions.json`
- `candidate_score_correlations.csv`

Prompt 2 must preserve the registered chronological split, fit preprocessing on training queries only, and keep all `label__*` columns out of inference features.

## Limitations

- Only 361 independent 2015 query days exist; candidate rows are not independent.
- Historical EPEX-DE is not pooled with modern DE-LU.
- Sparse or absent price spikes limit spike conclusions.
- Battery labels are tied to the registered 1 MW / 1 MWh specification.
- This phase demonstrates dataset validity and baseline relationships, not learned-reranker performance.
"""
    (results / "PHASE1_REPORT.md").write_text(text, encoding="utf-8")
