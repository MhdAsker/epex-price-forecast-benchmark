"""Small causal rerankers for the registered Phase 2 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KEYS = ["query__id", "candidate__id"]
EMBEDDING_FEATURES = ["pair__embedding_similarity", "derived__log_embedding_rank"]
MARKET_FEATURES = [
    "pair__calendar_distance", "pair__pattern_distance",
    "derived__log_calendar_rank", "derived__log_pattern_rank",
    "derived__log_candidate_age_days", "query__weekend",
    "derived__month_sin", "derived__month_cos",
    "query__context_std_24h", "query__context_spread_24h", "query__context_std_168h",
    "derived__abs_last_price", "derived__abs_mean_24h", "derived__abs_std_24h",
    "derived__abs_spread_24h", "derived__abs_mean_168h", "derived__abs_std_168h",
    "derived__abs_spread_168h", "derived__abs_trend_24h",
]
FEATURE_SETS = {
    "embedding_only": EMBEDDING_FEATURES,
    "market_context": MARKET_FEATURES,
    "combined": EMBEDDING_FEATURES + MARKET_FEATURES,
}


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    model_type: str
    objective: str
    feature_set: str
    parameters: dict[str, Any]


def build_pair_table(phase1: Path) -> pd.DataFrame:
    """Construct only the prespecified issuance-time feature matrix."""
    pair = pd.read_parquet(phase1 / "candidate_features.parquet")
    labels = pd.read_parquet(phase1 / "candidate_labels.parquet")
    query = pd.read_parquet(phase1 / "query_features.parquet")
    library = pd.read_parquet(phase1 / "candidate_library.parquet")
    query_columns = [
        "query__id", "partition", "query__delivery_date", "query__month", "query__weekend",
        "query__context_last_price", "query__context_mean_24h", "query__context_std_24h",
        "query__context_spread_24h", "query__context_mean_168h", "query__context_std_168h",
        "query__context_spread_168h", "query__context_trend_24h",
    ]
    candidate_columns = [
        "candidate__id", "candidate__context_last_price", "candidate__context_mean_24h",
        "candidate__context_std_24h", "candidate__context_spread_24h",
        "candidate__context_mean_168h", "candidate__context_std_168h",
        "candidate__context_spread_168h", "candidate__context_trend_24h",
    ]
    table = pair.merge(labels, on=KEYS, validate="one_to_one").merge(
        query[query_columns], on="query__id", validate="many_to_one"
    ).merge(library[candidate_columns], on="candidate__id", validate="many_to_one")
    table["derived__log_embedding_rank"] = np.log1p(table["pair__embedding_rank"])
    table["derived__log_calendar_rank"] = np.log1p(table["pair__calendar_rank"])
    table["derived__log_pattern_rank"] = np.log1p(table["pair__pattern_rank"])
    table["derived__log_candidate_age_days"] = np.log1p(table["pair__candidate_age_days"])
    angle = 2 * np.pi * (table["query__month"] - 1) / 12
    table["derived__month_sin"] = np.sin(angle)
    table["derived__month_cos"] = np.cos(angle)
    table["query__weekend"] = table["query__weekend"].astype(float)
    for suffix in (
        "last_price", "mean_24h", "std_24h", "spread_24h",
        "mean_168h", "std_168h", "spread_168h", "trend_24h",
    ):
        table[f"derived__abs_{suffix}"] = np.abs(
            table[f"query__context_{suffix}"] - table[f"candidate__context_{suffix}"]
        )
    allowed = set(KEYS + ["partition", "query__delivery_date"])
    allowed.update(FEATURE_SETS["combined"])
    allowed.update([
        "label__continuation_mae_eur_mwh", "label__incremental_revenue_vs_direct_eur",
        "label__candidate_revenue_eur",
    ])
    return table[list(dict.fromkeys(column for column in table.columns if column in allowed))]


def target(table: pd.DataFrame, objective: str) -> np.ndarray:
    if objective == "forecast":
        return -table["label__continuation_mae_eur_mwh"].to_numpy(float)
    if objective == "economic":
        return table["label__incremental_revenue_vs_direct_eur"].to_numpy(float)
    raise ValueError(f"unknown objective: {objective}")


def fit_experiment(experiment: Experiment, train: pd.DataFrame, seed: int) -> Any:
    features = FEATURE_SETS[experiment.feature_set]
    x = train[features].to_numpy(float)
    y = target(train, experiment.objective)
    if experiment.model_type == "ridge":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(experiment.parameters["alpha"]))),
        ])
    elif experiment.model_type == "lightgbm":
        model = LGBMRegressor(
            objective="regression", random_state=seed, deterministic=True,
            force_col_wise=True, verbosity=-1, n_jobs=4, **experiment.parameters,
        )
    else:
        raise ValueError(f"unknown model type: {experiment.model_type}")
    return model.fit(x, y)


def predict(model: Any, experiment: Experiment, table: pd.DataFrame) -> np.ndarray:
    values = model.predict(table[FEATURE_SETS[experiment.feature_set]].to_numpy(float))
    if not np.isfinite(values).all():
        raise ValueError("reranker produced non-finite scores")
    return np.asarray(values, dtype=float)


def save_checkpoint(path: Path, model: Any, experiment: Experiment, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    joblib.dump({"model": model, "experiment": experiment, "metadata": metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    return joblib.load(path)
