"""Online QRA and split-conformal calibration from out-of-sample forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _wide_points(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    keys = ["delivery_date", "hour"]
    wide = points.pivot_table(index=keys, columns="model", values="prediction")
    actual = points.groupby(keys, sort=True)["actual"].first()
    return wide.sort_index(), actual.reindex(wide.index)


def _conformal_radius(errors: np.ndarray, coverage: float) -> float:
    n = len(errors)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(np.abs(errors), level, method="higher"))


def sequential_probabilistic_forecasts(
    points: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    calibration_days: int = 182,
    min_days: int = 60,
    refit_days: int = 28,
    conformal_base: str = "LightGBM",
) -> pd.DataFrame:
    """Calibrate distributions using only earlier outer-test forecasts.

    QRA regresses outcomes on the competing point forecasts separately at each
    quantile. Split conformal uses finite-sample corrected absolute residual
    radii around one declared point model. The first ``min_days`` are a strict
    online calibration burn-in and are not scored probabilistically.
    """
    wide, actual = _wide_points(points)
    if conformal_base not in wide.columns:
        raise KeyError(f"Conformal base {conformal_base!r} not found")
    dates = pd.Index(wide.index.get_level_values("delivery_date").unique())
    rows: list[dict] = []
    qra_models: dict[float, object] = {}
    for date_pos, date in enumerate(dates):
        if date_pos < min_days:
            continue
        prior_dates = dates[max(0, date_pos - calibration_days) : date_pos]
        prior_mask = wide.index.get_level_values("delivery_date").isin(prior_dates)
        current_mask = wide.index.get_level_values("delivery_date") == date
        train_X = wide.loc[prior_mask].to_numpy(dtype=float)
        train_y = actual.loc[prior_mask].to_numpy(dtype=float)
        current_X = wide.loc[current_mask].to_numpy(dtype=float)
        current_actual = actual.loc[current_mask].to_numpy(dtype=float)
        hours = wide.loc[current_mask].index.get_level_values("hour").to_numpy(dtype=int)
        if date_pos % refit_days == min_days % refit_days or not qra_models:
            qra_models = {}
            for quantile in quantiles:
                model = make_pipeline(
                    StandardScaler(),
                    QuantileRegressor(quantile=quantile, alpha=1e-4, solver="highs"),
                )
                model.fit(train_X, train_y)
                qra_models[quantile] = model
        qra_matrix = np.column_stack(
            [qra_models[q].predict(current_X) for q in quantiles]
        )
        qra_matrix.sort(axis=1)

        base_idx = list(wide.columns).index(conformal_base)
        base_current = current_X[:, base_idx]
        base_errors = train_y - train_X[:, base_idx]
        conformal_matrix = np.empty((len(current_X), len(quantiles)), dtype=float)
        for q_idx, quantile in enumerate(quantiles):
            if quantile == 0.5:
                conformal_matrix[:, q_idx] = base_current
            else:
                tail = min(quantile, 1 - quantile)
                radius = _conformal_radius(base_errors, 1 - 2 * tail)
                conformal_matrix[:, q_idx] = base_current + (-radius if quantile < 0.5 else radius)
        for method, matrix in (("QRA", qra_matrix), (f"Conformal-{conformal_base}", conformal_matrix)):
            for row_idx, hour in enumerate(hours):
                for q_idx, quantile in enumerate(quantiles):
                    rows.append(
                        {
                            "delivery_date": date,
                            "hour": int(hour),
                            "actual": float(current_actual[row_idx]),
                            "method": method,
                            "quantile": float(quantile),
                            "prediction": float(matrix[row_idx, q_idx]),
                            "calibration_days": len(prior_dates),
                        }
                    )
    return pd.DataFrame(rows)
