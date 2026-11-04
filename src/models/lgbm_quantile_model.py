"""
LightGBM Quantile Regressor.

Fits three LightGBM models at quantiles [q10, q50, q90] to produce an 80%
prediction interval alongside the point forecast.  The q50 model is used
as the probabilistic point forecast; q10/q90 form the interval.

Evaluated with Pinball (Quantile) Loss:
    L_q(y, ŷ) = q * (y − ŷ)   if y ≥ ŷ
              = (1−q) * (ŷ − y) if y < ŷ

A well-calibrated 80% interval should contain ~80% of actual observations
(coverage).  Lower pinball loss = better.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_QUANTILES = [0.10, 0.50, 0.90]


class LGBMQuantileModel:
    """
    Fits one LightGBMRegressor per quantile (objective='quantile').

    Parameters
    ----------
    quantiles : list of floats in (0, 1).  Default [0.10, 0.50, 0.90].
    """

    def __init__(self, quantiles: list[float] | None = None) -> None:
        self.quantiles: list[float] = quantiles or DEFAULT_QUANTILES
        self.models: dict[float, lgb.LGBMRegressor] = {}

    # ── private ──────────────────────────────────────────────────────────────

    def _make_model(self, alpha: float) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            metric="quantile",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    # ── public ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train one model per quantile with early stopping on the validation set."""
        for q in self.quantiles:
            model = self._make_model(q)
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="quantile",
                callbacks=callbacks,
            )
            self.models[q] = model
            logger.info(
                "LGBM quantile q=%.2f — best iter: %d",
                q, model.best_iteration_,
            )

        out_path = MODELS_DIR / "lgbm_quantile.pkl"
        joblib.dump(self.models, out_path)
        logger.info("Saved LGBM quantile models → %s", out_path)

    def predict(self, X: np.ndarray) -> dict[float, np.ndarray]:
        """Return {quantile: predictions} for all fitted quantiles."""
        if not self.models:
            raise RuntimeError("Call fit() before predict().")
        return {q: m.predict(X) for q, m in self.models.items()}

    def predict_interval(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (q10_preds, q50_preds, q90_preds)."""
        preds = self.predict(X)
        return preds[0.10], preds[0.50], preds[0.90]

