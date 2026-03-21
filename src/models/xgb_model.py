import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class XGBModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="mae",
        )

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> None:
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = self.model.best_iteration
        val_preds = self.model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        logger.info("XGBoost fitted — best iteration: %d, val MAE: %.4f", best_iter, val_mae)

        out_path = MODELS_DIR / "xgb.pkl"
        joblib.dump(self.model, out_path)
        logger.info("Saved XGBoost model to %s", out_path)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        importances = self.model.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
