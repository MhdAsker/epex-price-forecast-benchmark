"""
XGBoost Quantile Regressor.

Uses the native ``reg:quantileerror`` objective introduced in XGBoost 1.7
(stable in 2.x).  Fits three models at [q10, q50, q90] to produce an 80%
prediction interval.

If the installed XGBoost version does not support ``reg:quantileerror``, an
``ImportError`` is raised with clear upgrade instructions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from packaging.version import Version

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_QUANTILES = [0.10, 0.50, 0.90]

_XGB_MIN_VERSION = "1.7.0"


def _check_xgb_version() -> None:
    if Version(xgb.__version__) < Version(_XGB_MIN_VERSION):
        raise ImportError(
            f"XGBoost {_XGB_MIN_VERSION}+ required for reg:quantileerror objective. "
            f"Installed: {xgb.__version__}.  Run: pip install --upgrade xgboost"
        )


class XGBQuantileModel:
    """
    Fits one XGBRegressor per quantile (objective='reg:quantileerror').

    Parameters
    ----------
    quantiles : list of floats in (0, 1).  Default [0.10, 0.50, 0.90].
    """

    def __init__(self, quantiles: list[float] | None = None) -> None:
        _check_xgb_version()
        self.quantiles: list[float] = quantiles or DEFAULT_QUANTILES
        self.models: dict[float, xgb.XGBRegressor] = {}

    # ── private ──────────────────────────────────────────────────────────────

    def _make_model(self, alpha: float) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="quantile",
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
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.models[q] = model
            logger.info(
                "XGB quantile q=%.2f — best iter: %d",
                q, model.best_iteration,
            )

        out_path = MODELS_DIR / "xgb_quantile.pkl"
        joblib.dump(self.models, out_path)
        logger.info("Saved XGBoost quantile models → %s", out_path)

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

