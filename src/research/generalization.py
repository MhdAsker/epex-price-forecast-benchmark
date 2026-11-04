"""Cross-zone local, pooled, and transfer-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler

from src.research.features import DailyDataset


@dataclass(frozen=True)
class ZoneSplit:
    zone: str
    test_year: int
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    test_dates: pd.DatetimeIndex


def split_zone(data: DailyDataset, zone: str, test_year: int, calibration_years: int = 4) -> ZoneSplit:
    train_start = pd.Timestamp(f"{test_year - calibration_years}-01-01")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year + 1}-01-01")
    train = (data.dates >= train_start) & (data.dates < test_start)
    test = (data.dates >= test_start) & (data.dates < test_end)
    expected = 366 if pd.Timestamp(f"{test_year}-12-31").is_leap_year else 365
    if int(test.sum()) != expected:
        raise ValueError(f"{zone} {test_year} is not a complete test year: {int(test.sum())} days")
    # D-7 feature construction legitimately consumes the first seven source
    # days; validate calendar coverage rather than requiring 4*365 usable rows.
    train_dates = data.dates[train]
    if (
        not len(train_dates)
        or train_dates.min() > train_start + pd.Timedelta(days=7)
        or train_dates.max() < test_start - pd.Timedelta(days=1)
    ):
        raise ValueError(f"{zone} lacks four complete calibration years")
    return ZoneSplit(zone, test_year, data.X[train], data.y[train], data.X[test], data.y[test], data.dates[test])


def _zone_features(X: np.ndarray, zone_index: int, n_zones: int) -> np.ndarray:
    identity = np.zeros((len(X), n_zones), dtype=float)
    identity[:, zone_index] = 1.0
    return np.column_stack([X, identity])


def _fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, alpha: float = 0.05) -> np.ndarray:
    scaler = StandardScaler().fit(X_train)
    model = MultiTaskElasticNet(alpha=alpha, l1_ratio=0.7, max_iter=20000, tol=1e-3, random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    return model.predict(scaler.transform(X_test))


def _transfer_predict(
    source_X: np.ndarray,
    source_y: np.ndarray,
    target_X: np.ndarray,
    target_y: np.ndarray,
    test_X: np.ndarray,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    scaler = StandardScaler().fit(np.vstack([source_X, target_X]))
    model = MultiTaskElasticNet(
        alpha=alpha, l1_ratio=0.7, max_iter=20000, tol=1e-3,
        random_state=42, warm_start=True,
    )
    model.fit(scaler.transform(source_X), source_y)
    model.fit(scaler.transform(target_X), target_y)
    return model.predict(scaler.transform(test_X))


def run_generalization(splits: dict[str, ZoneSplit], finetune_days: int = 365) -> pd.DataFrame:
    zones = tuple(sorted(splits))
    rows: list[dict] = []
    for target_index, zone in enumerate(zones):
        print(f"Cross-zone fits for {zone}", flush=True)
        split = splits[zone]
        local_X = _zone_features(split.train_X, target_index, len(zones))
        test_X = _zone_features(split.test_X, target_index, len(zones))
        source_X_parts, source_y_parts = [], []
        for source_index, source_zone in enumerate(zones):
            source = splits[source_zone]
            source_X_parts.append(_zone_features(source.train_X, source_index, len(zones)))
            source_y_parts.append(source.train_y)
        pooled_X, pooled_y = np.vstack(source_X_parts), np.vstack(source_y_parts)
        source_X = np.vstack([x for i, x in enumerate(source_X_parts) if i != target_index])
        source_y = np.vstack([y for i, y in enumerate(source_y_parts) if i != target_index])
        predictions = {
            "Naive D-1": split.test_X[:, :24],
            "Local JointCurve": _fit_predict(local_X, split.train_y, test_X),
            "Pooled JointCurve": _fit_predict(pooled_X, pooled_y, test_X),
            "Transfer JointCurve": _transfer_predict(
                source_X, source_y, local_X[-finetune_days:], split.train_y[-finetune_days:], test_X
            ),
        }
        for model, pred in predictions.items():
            for day, date in enumerate(split.test_dates):
                for hour in range(24):
                    rows.append({
                        "zone": zone, "test_year": split.test_year,
                        "delivery_date": date.date().isoformat(), "hour": hour,
                        "actual": float(split.test_y[day, hour]),
                        "prediction": float(pred[day, hour]), "model": model,
                    })
    return pd.DataFrame(rows)


def generalization_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in predictions.groupby(["zone", "test_year", "model"], sort=False):
        error = group["prediction"].to_numpy() - group["actual"].to_numpy()
        naive = predictions[
            (predictions["zone"] == keys[0])
            & (predictions["test_year"] == keys[1])
            & (predictions["model"] == "Naive D-1")
        ]
        records.append({
            "zone": keys[0], "test_year": keys[1], "model": keys[2],
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "rMAE_D1": float(np.mean(np.abs(error)) / np.mean(np.abs(naive["prediction"] - naive["actual"]))),
            "n_hours": len(group),
        })
    return pd.DataFrame(records).sort_values(["zone", "MAE"]).reset_index(drop=True)
