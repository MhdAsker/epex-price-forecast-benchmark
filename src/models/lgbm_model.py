import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class LGBMModel:
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> None:
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # suppress per-iteration stdout
        ]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=callbacks,
        )

        best_iter = self.model.best_iteration_
        val_preds = self.model.predict(X_val, num_iteration=best_iter)
        val_mae = mean_absolute_error(y_val, val_preds)
        logger.info("LightGBM fitted — best iteration: %d, val MAE: %.4f", best_iter, val_mae)

        out_path = MODELS_DIR / "lgbm.pkl"
        joblib.dump(self.model, out_path)
        logger.info("Saved LightGBM model to %s", out_path)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        importances = self.model.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
