"""
LightGBM variants used by the day-ahead comparison.

The tree hyperparameters match ``LGBMModel`` so a gap against the leaky
reference is a gap in the information set, not a gap in tuning.

Hourly experts are the standard day-ahead design: price formation at hour 03
and hour 12 does not share a leaf structure, and each expert only sees rows
from its own Europe/Berlin hour.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _regressor(num_leaves: int = 63, min_child_samples: int = 20) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def fit_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None = None,
    save_path: Path | None = None,
    num_leaves: int = 63,
    min_child_samples: int = 20,
) -> lgb.LGBMRegressor:
    """Fit one regressor. Early stopping uses unweighted MAE on ``X_val``."""
    model = _regressor(num_leaves=num_leaves, min_child_samples=min_child_samples)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
    return model


class HourlyExpertLGBM:
    """One LightGBM per local hour, with a global model as fallback."""

    def __init__(self) -> None:
        self.models: dict[int, lgb.LGBMRegressor] = {}
        self.fallback: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hours_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hours_val: np.ndarray,
    ) -> None:
        self.fallback = fit_lgbm(X_train, y_train, X_val, y_val)
        self.models = {}
        for hour in range(24):
            train_rows = hours_train == hour
            val_rows = hours_val == hour
            # Each hour has roughly 1/24 of the sample. Smaller leaves keep
            # the expert from memorising a few hundred rows.
            if int(train_rows.sum()) < 80 or int(val_rows.sum()) < 12:
                continue
            self.models[hour] = fit_lgbm(
                X_train[train_rows],
                y_train[train_rows],
                X_val[val_rows],
                y_val[val_rows],
                num_leaves=31,
                min_child_samples=10,
            )
        joblib.dump(
            {"models": self.models, "fallback": self.fallback},
            MODELS_DIR / "day_ahead_hourly.pkl",
        )

    def predict(self, X: np.ndarray, hours: np.ndarray) -> np.ndarray:
        if self.fallback is None:
            raise RuntimeError("Call fit() before predict().")
        preds = np.asarray(self.fallback.predict(X), dtype=float).copy()
        for hour, model in self.models.items():
            rows = hours == hour
            if np.any(rows):
                preds[rows] = model.predict(X[rows])
        return preds


class EnsembleAverage:
    """Equal-weight mean of two prediction vectors. No extra parameters."""

    def predict(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if len(first) != len(second):
            raise ValueError("Ensemble members must cover the same hours")
        return 0.5 * (np.asarray(first, dtype=float) + np.asarray(second, dtype=float))
