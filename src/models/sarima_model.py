import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class SARIMAModel:
    """
    SARIMAX(1,0,1)(1,0,1,24) fitted on price_eur_mwh series only.
    No exogenous features — purely autoregressive baseline.
    """

    def __init__(self):
        self.result = None

    def fit(self, train_series: pd.Series) -> None:
        logger.info("Fitting SARIMA(1,0,1)(1,0,1,24) on %d observations...", len(train_series))
        t0 = time.time()

        model = SARIMAX(
            train_series,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = model.fit(disp=False)

        elapsed = time.time() - t0
        logger.info("SARIMA fitted in %.1fs — AIC: %.2f", elapsed, self.result.aic)

        out_path = MODELS_DIR / "sarima.pkl"
        joblib.dump(self.result, out_path)
        logger.info("Saved SARIMA model to %s", out_path)

    def predict(self, n_steps: int) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        forecast = self.result.forecast(steps=n_steps)
        return forecast.to_numpy()

    def fit_predict(self, train: pd.Series, test: pd.Series) -> np.ndarray:
        self.fit(train)
        return self.predict(len(test))
