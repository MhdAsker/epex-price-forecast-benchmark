"""Point models with a common daily-curve interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import lightgbm as lgb
import numpy as np
import torch
from sklearn.linear_model import Lasso, MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.research.features import stack_hourly


class DailyModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DailyModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class LEARModel(DailyModel):
    """Twenty-four hour-specific LASSO equations on the full lagged curves."""

    def __init__(self, alpha: float = 0.1, seed: int = 42):
        self.model = MultiOutputRegressor(
            make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=10000, random_state=seed)),
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LEARModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)


class JointCurveModel(DailyModel):
    """Multi-task elastic net sharing a sparse structure across 24 products."""

    def __init__(self, alpha: float = 0.05, l1_ratio: float = 0.7, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = MultiTaskElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=50000, tol=1e-3, random_state=seed
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JointCurveModel":
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(X)), dtype=float)


class _HourlyStackedModel(DailyModel):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_HourlyStackedModel":
        hourly_X, hourly_y = stack_hourly(X, y)
        self.estimator.fit(hourly_X, hourly_y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        hourly_X, _ = stack_hourly(X)
        return np.asarray(self.estimator.predict(hourly_X), dtype=float).reshape(-1, 24)


class LightGBMModel(_HourlyStackedModel):
    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.05, seed: int = 42):
        super().__init__(
            lgb.LGBMRegressor(
                n_estimators=120, learning_rate=learning_rate, num_leaves=num_leaves,
                min_child_samples=30, colsample_bytree=0.8, subsample=0.8,
                random_state=seed, n_jobs=-1, verbose=-1,
            )
        )


class XGBoostModel(_HourlyStackedModel):
    def __init__(self, max_depth: int = 6, learning_rate: float = 0.05, seed: int = 42):
        super().__init__(
            XGBRegressor(
                n_estimators=120, learning_rate=learning_rate, max_depth=max_depth,
                min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                objective="reg:absoluteerror", random_state=seed, n_jobs=-1,
            )
        )


class CatBoostModel(_HourlyStackedModel):
    def __init__(self, depth: int = 7, learning_rate: float = 0.05, seed: int = 42):
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("Install catboost from requirements.txt") from exc
        super().__init__(
            CatBoostRegressor(
                iterations=120, depth=depth, learning_rate=learning_rate,
                loss_function="MAE", random_seed=seed, verbose=False,
                allow_writing_files=False, thread_count=-1,
            )
        )


class JointMLPModel(DailyModel):
    """Compact 24-output DNN; chosen over recurrent models for simultaneous auctions."""

    def __init__(
        self, hidden: int = 128, dropout: float = 0.1, epochs: int = 40,
        learning_rate: float = 1e-3, seed: int = 42,
    ):
        self.hidden, self.dropout, self.epochs = hidden, dropout, epochs
        self.learning_rate, self.seed = learning_rate, seed
        self.scaler = StandardScaler()
        self.net: torch.nn.Module | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JointMLPModel":
        torch.manual_seed(self.seed)
        scaled = self.scaler.fit_transform(X).astype("float32")
        target = np.asarray(y, dtype="float32")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(scaled.shape[1], self.hidden), torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 24),
        )
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.L1Loss()
        xb, yb = torch.from_numpy(scaled), torch.from_numpy(target)
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.net(xb), yb)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model is not fitted")
        self.net.eval()
        with torch.no_grad():
            pred = self.net(torch.from_numpy(self.scaler.transform(X).astype("float32")))
        return pred.numpy().astype(float)


MODEL_GRIDS: dict[str, list[dict]] = {
    "LEAR": [{"alpha": 0.03}, {"alpha": 0.1}, {"alpha": 0.3}],
    "LightGBM": [{"num_leaves": 15}, {"num_leaves": 31}, {"num_leaves": 63}],
    "XGBoost": [{"max_depth": 4}, {"max_depth": 6}, {"max_depth": 8}],
    "CatBoost": [{"depth": 5}, {"depth": 7}, {"depth": 9}],
    "JointCurve": [{"alpha": 0.02}, {"alpha": 0.05}, {"alpha": 0.1}],
    "JointMLP": [{"hidden": 64}, {"hidden": 128}],
}


MODEL_CLASSES = {
    "LEAR": LEARModel,
    "LightGBM": LightGBMModel,
    "XGBoost": XGBoostModel,
    "CatBoost": CatBoostModel,
    "JointCurve": JointCurveModel,
    "JointMLP": JointMLPModel,
}
