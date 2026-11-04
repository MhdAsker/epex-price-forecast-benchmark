# EPEX DE-LU Day-Ahead Price Forecast Benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen)](tests/)
[![Market: EPEX SPOT DE-LU](https://img.shields.io/badge/Market-EPEX%20SPOT%20DE--LU-orange)](https://www.epexspot.com/)
[![Benchmark: Lago et al.](https://img.shields.io/badge/Benchmark-epftoolbox-purple)](https://github.com/jeslago/epftoolbox)

An audit-grade, leak-free benchmark for day-ahead electricity price forecasting in the German-Luxembourg (**DE-LU**) bidding zone. 

Unlike standard time-series regression tasks, day-ahead power markets operate under strict auction timing: all 24 hourly products for delivery day $D$ clear simultaneously at **12:00 Europe/Berlin on day $D-1$**. This repository documents the widespread "same-auction information leakage" trap, enforces formal protocol guardrails, and provides state-of-the-art point and probabilistic forecasting benchmarks across both recent post-crisis data (2022–2024) and standardized multi-year academic testbeds (2012–2017).

---

## 📑 Table of Contents

- [Executive Summary & The Information Leakage Trap](#-executive-summary--the-information-leakage-trap)
- [Benchmark Results](#-benchmark-results)
  - [Phase 1: Real-World 2024 DE-LU Day-Ahead Auction](#phase-1-real-world-2024-de-lu-day-ahead-auction-20222024)
  - [Phase 2: Standardized Multi-Year Rolling-Origin Benchmark](#phase-2-standardized-multi-year-rolling-origin-benchmark-20162017)
  - [Statistical Significance (Diebold-Mariano Tests)](#statistical-significance-diebold-mariano-tests)
  - [Probabilistic & Quantile Calibration](#probabilistic--quantile-calibration)
- [Formal Auction Protocol & Guardrails](#-formal-auction-protocol--guardrails)
- [Diagnostic Plots & Visualizations](#-diagnostic-plots--visualizations)
- [Repository Architecture](#-repository-architecture)
- [Reproducibility & Quickstart](#-reproducibility--quickstart)
- [Methodological Key Insights](#-methodological-key-insights)

---

## ⚠️ Executive Summary & The Information Leakage Trap

In European day-ahead power markets (EPEX SPOT / SDAC), market participants submit order curves before the gate closure at **12:00 on $D-1$**. The auction clears simultaneously for all 24 hours of day $D$.

### The Classic Leakage Trap:
A frequent error in published machine learning benchmarks is conditioning next-hour forecasts on `price_lag_1h`. In an operational day-ahead setting at 12:00 on $D-1$:
- For hour 00 of day $D$, the actual price of hour 23 on day $D-1$ is **unknown** (it hasn't delivered yet).
- For hours 01–23 of day $D$, using `price_lag_1h` directly leaks prices established in the **very same auction block**.

```
Gate Closure: 12:00 (D-1)
                     |
                     v
[Delivery Day D-1]   |   [Delivery Day D: 24 Hourly Products Clear Simultaneously]
... 10  11  12  13 ...   [00:00] [01:00] ... [12:00] ... [22:00] [23:00]
                     |      ^       ^
                     |      |_______|--- Using D[h-1] to predict D[h] leaks prices
                     |                   from the exact same auction!
```

This benchmark explicitly isolates this effect:
- **Leaky LightGBM** (`price_lag_1h` included) reports an artificial **8.66 €/MWh MAE**.
- **Audited Auction-Safe Ensemble** achieves **19.64 €/MWh MAE** (a realistic 25.1% improvement over persistence).

---

## 📊 Benchmark Results

### Phase 1: Real-World 2024 DE-LU Day-Ahead Auction (2022–2024)

- **Training Period:** 2022-01-01 to 2023-12-31 (17,185 hourly observations).
- **Test Period:** 2024-01-01 to 2024-11-30 (8,040 hourly observations).
- **Forecast Origin:** 12:00 Europe/Berlin on $D-1$.
- **Negative-price hours in test:** 449 hours (5.6%).

| Model | Information Set | Status | MAE (€/MWh) | RMSE (€/MWh) | MAE Negative | Neg Recall | Neg Precision |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **DA Ensemble** | Auction-safe ($D-1$) | **Primary Winner** | **19.64** | **29.59** | 37.87 | 17.2% | **82.8%** |
| **DA Price History** | Auction-safe ($D-1$) | Primary | 20.32 | 29.78 | 31.54 | 30.5% | 73.3% |
| **DA Spike** | Auction-safe ($D-1$) | Primary | 20.66 | 30.46 | **30.27** | **33.6%** | 64.0% |
| **DA Hourly** | 24 distinct hourly estimators | Primary | 21.70 | 32.44 | 47.46 | 7.1% | 94.1% |
| **Naive D-1** | Lag 24h persistence | Baseline | 26.21 | 39.26 | 37.61 | 36.8% | 36.7% |
| **Naive D-7** | Lag 168h persistence | Baseline | 28.98 | 43.34 | 43.64 | 34.1% | 32.0% |
| *Post-gate Oracle* | Exogenous wind/solar (18:00 D-1) | Sensitivity | 19.46 | 26.62 | 20.97 | 41.0% | 92.5% |
| *Leaky LightGBM* | `price_lag_1h` (intra-auction leak) | Invalid diagnostic | 8.66 | 14.92 | 16.84 | 73.3% | 78.0% |

> **Key takeaway:** The primary auction-safe ensemble delivers a **25.1% MAE reduction** over the D-1 naive baseline without relying on post-gate exogenous forecast revisions.

---

### Phase 2: Standardized Multi-Year Rolling-Origin Benchmark (2016–2017)

To ensure comparability with international academic literature, Phase 2 implements a continuous **weekly-refit rolling calibration** over the standardized Lago et al. (*epftoolbox*) German EPEX dataset (4-year calibration window: 1,461 days, rolling forward weekly).

#### Test Year 2016 (8,784 hours):
| Model | MAE (€/MWh) | RMSE (€/MWh) | rMAE vs D-1 | Bias (€) | MAE Negative | Neg Recall | Spike MAE | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **4.28** | 7.10 | **0.661** | +0.45 | 30.14 | 2.1% | 48.93 | 151s |
| **JointCurve** | 4.32 | **6.92** | 0.667 | +0.11 | **24.35** | 24.7% | **41.19** | 31s |
| **LEAR** | 4.37 | 6.99 | 0.675 | +0.11 | 24.37 | 25.8% | 42.36 | **6s** |
| **CatBoost** | 4.45 | 7.18 | 0.688 | +0.38 | 30.65 | 4.1% | 48.50 | 280s |
| **LightGBM** | 4.60 | 7.60 | 0.711 | **+0.07** | 30.14 | **37.1%** | 45.31 | 60s |
| *Naive D-1* | 6.47 | 10.21 | 1.000 | -0.04 | 28.80 | 19.6% | 35.13 | 0s |
| *Naive D-7* | 6.80 | 11.44 | 1.051 | -0.03 | 47.33 | 10.3% | 57.45 | 0s |
| *JointMLP* | 15.45 | 19.61 | 2.387 | -6.52 | 53.42 | 0.0% | 92.30 | 14s |

#### Test Year 2017 (8,760 hours):
| Model | MAE (€/MWh) | RMSE (€/MWh) | rMAE vs D-1 | Bias (€) | MAE Negative | Neg Recall | Spike MAE | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **JointCurve** | **6.59** | **10.47** | **0.666** | -0.17 | **29.37** | **40.7%** | **35.42** | 20s |
| **LightGBM** | 6.61 | 10.93 | 0.668 | -0.38 | 33.69 | 31.7% | 38.12 | 63s |
| **LEAR** | 6.62 | 10.53 | 0.669 | -0.24 | 30.94 | 39.3% | 36.00 | **6s** |
| **CatBoost** | 6.65 | 11.49 | 0.672 | +0.10 | 40.41 | 9.7% | 55.44 | 313s |
| **XGBoost** | 6.68 | 11.60 | 0.675 | **-0.08** | 39.13 | 13.1% | 57.87 | 118s |
| *Naive D-1* | 9.89 | 15.58 | 1.000 | +0.09 | 39.32 | 24.8% | 32.81 | 0s |
| *Naive D-7* | 11.44 | 18.24 | 1.157 | +0.14 | 52.95 | 6.9% | 33.87 | 0s |
| *JointMLP* | 21.20 | 26.25 | 2.143 | -10.80 | 57.21 | 0.0% | 43.49 | 15s |

---

### Statistical Significance (Diebold-Mariano Tests)

Pairwise Diebold-Mariano (DM) tests with multi-horizon adjustments and bootstrap confidence intervals evaluated over 731 out-of-sample days:
- **JointCurve, LEAR, and Gradient Boosting (XGBoost/LightGBM/CatBoost)** form a statistical equivalence tier ($p > 0.05$), each significantly outperforming persistence baselines ($p < 10^{-16}$).
- **LEAR (Lasso Estimated AutoRegressive)** trains in **6 seconds** while matching 280-second CatBoost models within 0.03–0.08 €/MWh across both years.
- **JointMLP** fails catastrophically under extreme price shifts and regime changes, showing that unregularized deep networks overfit non-stationary electricity auctions.

---

### Probabilistic & Quantile Calibration

Day-ahead risk management and battery arbitrage require reliable predictive densities. We evaluated **Quantile Regression Averaging (QRA)** and **Conformalized Quantile Predictions** across 50%, 80%, and 90% target prediction intervals:

| Method | Test Year | Mean Pinball Loss | CRPS (approx) | 90% Coverage (Width) | 80% Coverage (Width) | 50% Coverage (Width) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **QRA** | 2016 | **1.330** | **2.660** | 85.7% (15.58 €) | 74.9% (11.05 €) | 47.4% (5.23 €) |
| **Conformal-LGBM** | 2016 | 1.467 | 2.933 | **88.2%** (17.42 €) | **78.4%** (12.60 €) | **50.2%** (5.89 €) |
| **QRA** | 2017 | **2.136** | **4.272** | 85.0% (24.54 €) | 74.1% (16.95 €) | 46.7% (7.37 €) |
| **Conformal-LGBM** | 2017 | 2.223 | 4.446 | **86.8%** (26.93 €) | **76.5%** (17.40 €) | **48.6%** (7.63 €) |

---

## 🛡️ Formal Auction Protocol & Guardrails

The machine-readable forecast contract is formalized in [`config/forecast_protocol.json`](config/forecast_protocol.json) and enforced at runtime by [`src/protocol.py`](src/protocol.py):

1. **Cutoff Rule:** Strict 12:00 Europe/Berlin gate on $D-1$.
2. **Lag Availability:** Only $D-1$, $D-2$, $D-7$, and $D-14$ full-curve vectors and profile statistics are admitted.
3. **Autumn Daylight Saving Time (DST):** On 25-hour transition days in October, the two distinct 02:00 UTC/local hours are retained as separate products using `local_fold` and `product_position` instead of being averaged or dropped.
4. **Automated Leakage Testing:** PyTest suite scrambles same-day future hours to verify that estimator predictions remain strictly unchanged.

---

## 📈 Diagnostic Plots & Visualizations

All diagnostic figures are generated to `results/plots/`:

| Diagnostic Plot | Description |
|:---|:---|
| [`phase2_point_mae_by_year.png`](results/plots/phase2_point_mae_by_year.png) | Annual MAE comparison across all Phase 2 models vs Naive persistence. |
| [`phase2_probabilistic_calibration.png`](results/plots/phase2_probabilistic_calibration.png) | Reliability diagram and prediction interval width for QRA and Conformal LGBM. |
| [`phase2_dm_matrix.png`](results/plots/phase2_dm_matrix.png) | Heatmap of pairwise Diebold-Mariano test p-values across all models. |
| [`day_ahead_by_hour.png`](results/plots/day_ahead_by_hour.png) | Model-by-model MAE curve broken down by delivery hour (00:00 to 23:00). |
| [`day_ahead_regimes.png`](results/plots/day_ahead_regimes.png) | Performance under extreme price volatility, negative prices, and peak hours. |
| [`actual_vs_predicted.png`](results/plots/actual_vs_predicted.png) | 30-day time-series overlay of ground truth prices against model predictions. |
| [`error_distribution.png`](results/plots/error_distribution.png) | Residual error distributions showing skewness and heavy-tail behavior. |
| [`negative_price_analysis.png`](results/plots/negative_price_analysis.png) | Scatter analysis of negative pricing events colored by hour of day. |
| [`feature_importance.png`](results/plots/feature_importance.png) | Top predictors identified by split-count and gain importance. |

---

## 📂 Repository Architecture

```text
epex-price-forecast-benchmark/
├── config/
│   └── forecast_protocol.json          # Machine-readable gate closure & availability contract
├── data/
│   ├── raw/                            # Raw ENTSO-E / epftoolbox parquets (gitignored)
│   └── processed/                      # Engineered feature matrices (gitignored)
├── results/
│   ├── metrics/                        # Leaderboards, DM tests, and manifest JSON files
│   ├── plots/                          # PNG diagnostic and calibration figures
│   └── predictions/                    # Out-of-sample prediction parquets
├── src/
│   ├── day_ahead.py                    # Phase 1 auction-safe feature pipeline
│   ├── protocol.py                     # Information boundary validation and guardrails
│   ├── cv.py                           # Expanding-window walk-forward cross validation
│   ├── evaluate.py                     # Metric calculations (MAE, RMSE, Pinball, MAPE)
│   ├── fetch.py                        # ENTSO-E transparency API ingestion
│   ├── features.py                     # Base feature engineering
│   ├── visualize.py                    # Phase 1 visualization functions
│   ├── models/
│   │   ├── day_ahead_models.py         # Phase 1 DA estimators (Ensemble, Spike, Hourly)
│   │   ├── lgbm_quantile_model.py      # Multi-quantile LightGBM regressor
│   │   ├── xgb_quantile_model.py       # Native quantile error XGBoost regressor
│   │   ├── lgbm_model.py               # Standard LightGBM point estimator
│   │   ├── xgb_model.py                # Standard XGBoost point estimator
│   │   ├── lstm_model.py               # PyTorch LSTM recurrent model
│   │   └── sarima_model.py             # Seasonal ARIMA baseline
│   └── research/                       # Phase 2 epftoolbox rolling benchmark engine
│       ├── backtest.py                 # Multi-year rolling origin backtester
│       ├── evaluation.py               # DM test matrix and statistical evaluation
│       ├── features.py                 # Standardized daily matrix builder
│       ├── probabilistic.py            # QRA and Conformal prediction implementations
│       └── visualize.py                # Phase 2 visualization suite
├── tests/
│   ├── test_day_ahead.py               # Leakage guardrails and DST transition tests
│   └── test_phase2.py                  # Rolling backtest and evaluation unit tests
├── finalize_phase2.py                  # Generate Phase 2 tables, tests, and plots
├── run_day_ahead.py                    # Primary Phase 1 benchmark runner
├── run_phase2.py                       # Phase 2 rolling backtest orchestrator
├── run_benchmark.py                    # Diagnostic legacy benchmark runner
├── requirements.txt                    # Python package dependencies
└── check_status.py                     # Pipeline environment & artifact checker
```

---

## 🚀 Reproducibility & Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/MhdAsker/epex-price-forecast-benchmark.git
cd epex-price-forecast-benchmark

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Tests & Leakage Guardrails

```bash
python -m pytest -v
```
All 11 tests check for:
- Temporal order preservation (no future observation shuffling).
- Strict information availability at 12:00 on $D-1$.
- 25-hour Autumn DST clock change preservation.
- Diebold-Mariano test consistency.

### 3. Run Benchmarks

```bash
# Run primary Phase 1 Auction Benchmark (2024 test period)
python run_day_ahead.py

# Finalize Phase 2 rolling benchmark metrics & calibration plots
python finalize_phase2.py

# Run pipeline health check
python check_status.py
```

*(Optional: Set `ENTSOE_API_KEY` in `.env` if re-fetching raw ENTSO-E data via `python src/fetch.py`).*

---

## 💡 Methodological Key Insights

1. **Information Leakage Invalidation:** Machine learning papers claiming sub-10 €/MWh MAE on day-ahead EPEX without hourly decomposition almost invariably leak same-auction prices via 1-hour lags. Enforcing an audited D-1 12:00 cutoff sets the true competitive benchmark at **~19.6 €/MWh** on 2024 DE-LU data.
2. **Linear Parsimony vs Gradient Boosting:** In standardized rolling benchmarks (Phase 2), regularized linear multi-output models (**LEAR** and **JointCurve**) perform within 1% of heavy gradient boosting trees (XGBoost/CatBoost) while training in a fraction of the time (6s vs 280s).
3. **Deep Learning Fragility in Extreme Regimes:** Multilayer Perceptrons and recurrent networks struggled with the heavy tails and negative-price shocks of German electricity prices, yielding severe negative bias and high ramp errors.
4. **Quantile Calibration:** Conformalized quantile prediction achieves near-perfect empirical coverage (88.2% on 90% nominal interval) even during renewable supply shocks, making it suitable for practical risk-constrained battery dispatch.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. Market data originates from the ENTSO-E Transparency Platform and epftoolbox/Zenodo.
