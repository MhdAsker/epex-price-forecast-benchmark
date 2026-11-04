"""Non-anticipative battery dispatch and economic-value evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import spearmanr


@dataclass(frozen=True)
class BatterySpec:
    energy_mwh: float = 1.0
    power_mw: float = 1.0
    roundtrip_efficiency: float = 0.9
    transaction_cost_eur_mwh: float = 0.5
    degradation_cost_eur_mwh: float = 2.0
    initial_soc_fraction: float = 0.5
    terminal_soc_fraction: float = 0.5


def optimize_day(prices: np.ndarray, spec: BatterySpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a 24-hour mixed-integer schedule with no simultaneous charge/discharge."""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    eta = np.sqrt(spec.roundtrip_efficiency)
    # Variables: charge[n], discharge[n], soc[n+1], charge_on[n].
    size = 4 * n + 1
    c = np.zeros(size)
    c[:n] = prices + spec.transaction_cost_eur_mwh + spec.degradation_cost_eur_mwh
    c[n:2*n] = -prices + spec.transaction_cost_eur_mwh + spec.degradation_cost_eur_mwh
    lower = np.zeros(size)
    upper = np.r_[
        np.full(n, spec.power_mw), np.full(n, spec.power_mw),
        np.full(n + 1, spec.energy_mwh), np.ones(n),
    ]
    equality = np.zeros((n + 2, size))
    rhs = np.zeros(n + 2)
    for hour in range(n):
        equality[hour, hour] = -eta
        equality[hour, n + hour] = 1 / eta
        equality[hour, 2 * n + hour] = -1
        equality[hour, 2 * n + hour + 1] = 1
    initial = spec.initial_soc_fraction * spec.energy_mwh
    terminal = spec.terminal_soc_fraction * spec.energy_mwh
    equality[n, 2 * n] = 1
    equality[n + 1, 3 * n] = 1
    rhs[n], rhs[n + 1] = initial, terminal
    inequality = np.zeros((2 * n, size))
    upper_ineq = np.zeros(2 * n)
    for hour in range(n):
        binary = 3 * n + 1 + hour
        inequality[hour, hour] = 1
        inequality[hour, binary] = -spec.power_mw
        inequality[n + hour, n + hour] = 1
        inequality[n + hour, binary] = spec.power_mw
        upper_ineq[n + hour] = spec.power_mw
    constraints = [
        LinearConstraint(equality, rhs, rhs),
        LinearConstraint(inequality, np.full(2 * n, -np.inf), upper_ineq),
    ]
    integrality = np.r_[np.zeros(3 * n + 1), np.ones(n)]
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Battery optimization failed: {result.message}")
    return result.x[:n], result.x[n:2*n], result.x[2*n:3*n+1]


def realized_value(actual: np.ndarray, charge: np.ndarray, discharge: np.ndarray, spec: BatterySpec) -> float:
    throughput_cost = spec.transaction_cost_eur_mwh + spec.degradation_cost_eur_mwh
    return float(np.sum(actual * (discharge - charge) - throughput_cost * (charge + discharge)))


def evaluate_dispatch(predictions: pd.DataFrame, spec: BatterySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict] = []
    key_cols = ["zone", "test_year"] if "zone" in predictions else ["test_year"]
    for keys, panel in predictions.groupby(key_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for date, day in panel.groupby("delivery_date", sort=True):
            actual = day.drop_duplicates("hour").sort_values("hour")["actual"].to_numpy()
            oracle_charge, oracle_discharge, _ = optimize_day(actual, spec)
            oracle = realized_value(actual, oracle_charge, oracle_discharge, spec)
            for model, model_day in day.groupby("model", sort=False):
                forecast = model_day.sort_values("hour")["prediction"].to_numpy()
                charge, discharge, soc = optimize_day(forecast, spec)
                revenue = realized_value(actual, charge, discharge, spec)
                row = {col: value for col, value in zip(key_cols, keys)}
                row.update({
                    "delivery_date": date, "model": model, "revenue_eur": revenue,
                    "oracle_eur": oracle, "regret_eur": oracle - revenue,
                    "throughput_mwh": float(np.sum(charge + discharge)),
                    "terminal_soc_mwh": float(soc[-1]),
                })
                daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    summary = daily.groupby(key_cols + ["model"], as_index=False).agg(
        revenue_eur=("revenue_eur", "sum"), oracle_eur=("oracle_eur", "sum"),
        regret_eur=("regret_eur", "sum"), throughput_mwh=("throughput_mwh", "sum"),
        profitable_days=("revenue_eur", lambda x: int((x > 0).sum())), n_days=("delivery_date", "nunique"),
    )
    mae = predictions.assign(abs_error=lambda x: (x.actual - x.prediction).abs()).groupby(
        key_cols + ["model"], as_index=False
    )["abs_error"].mean().rename(columns={"abs_error": "MAE"})
    summary = summary.merge(mae, on=key_cols + ["model"])
    summary["accuracy_rank"] = summary.groupby(key_cols)["MAE"].rank(method="min")
    summary["value_rank"] = summary.groupby(key_cols)["revenue_eur"].rank(method="min", ascending=False)
    rank_rows = []
    for keys, group in summary.groupby(key_cols):
        rho, pvalue = spearmanr(group["accuracy_rank"], group["value_rank"])
        key_values = keys if isinstance(keys, tuple) else (keys,)
        rank_rows.append({**dict(zip(key_cols, key_values)), "spearman_rank_correlation": rho, "pvalue": pvalue})
    summary = summary.merge(pd.DataFrame(rank_rows), on=key_cols)
    return daily, summary.sort_values(key_cols + ["value_rank"]).reset_index(drop=True)
