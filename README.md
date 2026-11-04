# EPEX day-ahead price forecasting benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.3.0-purple)](CITATION.cff)

A leakage-audited research benchmark for point forecasts, predictive intervals, cross-market generalization, and storage decision value in European day-ahead electricity markets.

The central design constraint is operational: all 24 hourly products for delivery day **D** are forecast together at **12:00 Europe/Berlin on D-1**. A model may use only information available at that origin. This invalidates the common practice of using the preceding delivery hour to predict the next hour inside the same auction curve.

The repository develops the benchmark in three layers:

1. **Protocol validity:** formal forecast-time information boundaries and leakage tests on recent DE-LU data.
2. **Forecast rigor:** multi-year rolling point and probabilistic evaluation with nested temporal tuning, dependence-aware inference, and ablations.
3. **Generalization and value:** four European zones, local/pooled/transfer learning, non-anticipative battery dispatch, immutable artifacts, and a reproducibility release.

## Research status

| Capability | Status | Evidence |
|---|---|---|
| Noon D-1 information contract | Complete | [`config/forecast_protocol.json`](config/forecast_protocol.json) |
| Leakage and DST guardrails | Complete | 14 automated tests |
| Recent DE-LU holdout | Complete | 2024, 8,040 hourly products |
| Full-year rolling benchmark | Complete | 2016 and 2017, 17,544 products |
| Point models | Complete | naive, LEAR, LightGBM, XGBoost, CatBoost, JointCurve, JointMLP |
| Probabilistic forecasts | Complete | QRA and conformal LightGBM, seven quantiles |
| Dependence-aware inference | Complete | daily losses, moving-block bootstrap, Diebold-Mariano tests |
| Cross-zone generalization | Complete | BE, DE, FR, Nordic; 140,352 predictions |
| Decision value | Complete | mixed-integer 1 MW / 1 MWh battery dispatch |
| Immutable release artifacts | Complete | checksums, manifests, cards, release bundle command |
| Audited exogenous vintages | Not yet available | excluded rather than treated as known |
| DOI | Pending external deposit | Zenodo metadata is prepared; no DOI is claimed |

## Main findings

- The valid 2024 DE-LU ensemble reaches **19.64 EUR/MWh MAE**, 25.1% below D-1 persistence. A deliberately leaky one-hour-lag diagnostic reaches 8.66 EUR/MWh and is **not rankable**.
- On the standardized German benchmark, the best MAE is **4.28** in 2016 (XGBoost) and **6.59** in 2017 (JointCurve). The leading classical and tree models are statistically difficult to separate, while all materially outperform persistence.
- Probabilistic forecasts remain under-covered: QRA obtains 85.7% and 85.0% empirical coverage for nominal 90% intervals in 2016 and 2017. Conformal calibration improves this to 88.2% and 86.8%, at the cost of wider intervals.
- Transfer is not uniformly beneficial. It is best by MAE in France, local training is best in Belgium, Germany, and Nordic, and pooled training is worse than persistence in Nordic.
- Accuracy and economic value do not have identical rankings. In Belgium and France, pooled models produce the greatest battery value even though another model has lower MAE.

These are benchmark results, not claims of universal superiority. Market, period, feature vintage, calibration design, and decision objective all affect rankings.

## Phase 1: forecast-time validity on DE-LU 2024

**Data:** 2022-01-01 through 2024-11-30. 2022-2023 train the estimators; 2024 is held out. The primary information set uses complete price curves and features admitted by the noon D-1 protocol. Exogenous revisions published after the gate are isolated as a sensitivity analysis.

| Model | Status | MAE | RMSE | Negative-price MAE | Negative recall |
|---|---:|---:|---:|---:|---:|
| **DA Ensemble** | primary winner | **19.64** | **29.59** | 37.87 | 17.2% |
| DA Price History | primary | 20.32 | 29.78 | 31.54 | 30.5% |
| DA Spike | primary | 20.66 | 30.46 | **30.27** | **33.6%** |
| DA Hourly | primary | 21.70 | 32.44 | 47.46 | 7.1% |
| Naive D-1 | baseline | 26.21 | 39.26 | 37.61 | 36.8% |
| Naive D-7 | baseline | 28.98 | 43.34 | 43.64 | 34.1% |

The post-gate oracle and leaky LightGBM diagnostic are stored separately and cannot win the primary leaderboard. This distinction matters more than small differences among valid model specifications.

### Information boundary

At the forecast origin, prices from delivery day D are unknown because all products clear together. Therefore `price_lag_1h` is not a valid recursive signal for hours 01-23 of D.

```text
11:00 D-1     12:00 D-1                         Delivery day D
known data ─── forecast origin ───────────────── 00 01 ... 22 23
                    │                             └── one joint auction curve ──┘
                    └── every model input must be available here
```

The protocol implementation also retains both products in the autumn 25-hour transition using product position and local-fold identity; it does not average away the duplicated local hour.

## Phase 2: rigorous point and probabilistic benchmark

**Source:** the open EPEX-Germany file distributed with the [epftoolbox benchmark](https://github.com/jeslago/epftoolbox), associated with Lago et al. (2021), Zenodo record 4624805. The raw series spans 2012-01-09 to 2017-12-31. Only prices enter the primary design matrix because the publication vintages of bundled exogenous columns have not been independently verified.

**Executed experiment:** four-year rolling calibration window, full test years 2016 and 2017, seed 42, weekly parameter refits, blocked 182-day inner validation, and hyperparameter refresh every 28 days. Forecast origins remain daily. The daily-refit JointCurve sensitivity uses a resumable checkpoint. The larger three-seed daily configuration in [`config/phase2.json`](config/phase2.json) is the research target; it is not misrepresented as an already executed artifact.

### Point accuracy

| Model | 2016 MAE | 2016 RMSE | 2017 MAE | 2017 RMSE | Total recorded fit time |
|---|---:|---:|---:|---:|---:|
| JointCurve | 4.32 | **6.92** | **6.59** | **10.47** | 51.5 s |
| LEAR | 4.37 | 6.99 | 6.62 | 10.53 | **12.2 s** |
| LightGBM | 4.60 | 7.60 | 6.61 | 10.93 | 123.1 s |
| XGBoost | **4.28** | 7.10 | 6.68 | 11.60 | 269.5 s |
| CatBoost | 4.45 | 7.18 | 6.65 | 11.49 | 593.5 s |
| Naive D-1 | 6.47 | 10.21 | 9.89 | 15.58 | 0 s |
| Naive D-7 | 6.80 | 11.44 | 11.44 | 18.24 | 0 s |
| JointMLP | 15.45 | 19.61 | 21.20 | 26.25 | 29.3 s |

The compact deep model is a negative result: it is substantially worse than persistence under the chosen architecture and optimization budget. The repository reports it rather than hiding it.

The recovered **daily-refit JointCurve** sensitivity is now complete: MAE is 4.327 in 2016 and 6.588 in 2017, versus 4.317 and 6.591 for weekly refitting. Daily parameter estimation therefore adds roughly 324 seconds of recorded fit time without a meaningful accuracy gain in this specification. The panel and summary are stored in [`phase2_daily_jointcurve.parquet`](results/predictions/phase2_daily_jointcurve.parquet) and [`phase2_daily_jointcurve_metrics.csv`](results/metrics/phase2_daily_jointcurve_metrics.csv).

### Statistical inference

Tests operate on one aggregate absolute-error loss per delivery day, not on 24 hourly rows treated as independent. Confidence intervals use a seven-day moving-block bootstrap with 2,000 replications. Diebold-Mariano tests use heteroskedasticity/autocorrelation-consistent variance.

For the combined 731-day panel, JointCurve versus LEAR is borderline but not conventionally significant (`p = 0.0518`); JointCurve versus LightGBM (`p = 0.0962`) and versus XGBoost (`p = 0.8333`) are also not significant at 5%. Each leading model is decisively better than D-1 persistence in the reported panel. See [`phase2_pairwise_tests.csv`](results/metrics/phase2_pairwise_tests.csv) for every pair rather than relying on a winner-only summary.

### Probabilistic performance

Seven quantiles (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95) are produced sequentially after a 60-day burn-in.

| Method | Year | Pinball | Approx. CRPS | 90% coverage / width | 80% coverage / width | 50% coverage / width |
|---|---:|---:|---:|---:|---:|---:|
| QRA | 2016 | **1.330** | **2.660** | 85.7% / 15.58 | 74.9% / 11.05 | 47.4% / 5.23 |
| Conformal-LightGBM | 2016 | 1.467 | 2.933 | **88.2%** / 17.42 | **78.4%** / 12.60 | **50.2%** / 5.89 |
| QRA | 2017 | **2.136** | **4.272** | 85.0% / 24.54 | 74.1% / 16.95 | 46.7% / 7.37 |
| Conformal-LightGBM | 2017 | 2.223 | 4.446 | **86.8%** / 26.93 | **76.5%** / 17.40 | **48.6%** / 7.63 |

Coverage is close to, but below, nominal levels. The correct interpretation is “improved calibration with residual under-coverage,” not perfect calibration.

## Phase 3: generalization and decision value

Phase 3 freezes the Phase 2 JointCurve specification and evaluates three training strategies:

- **Local:** four calendar years from the target zone.
- **Pooled:** the union of all four zone calibration panels plus zone identity.
- **Transfer:** pretrain on three source zones, then warm-start fine-tuning on the final 365 target-zone calibration days.

The Nordic source ends on 2018-12-24. To preserve the complete-year rule, 2017—not partial 2018—is the Nordic test year.

### Cross-zone forecast results

| Zone / test | Best strategy | MAE | RMSE | rMAE vs D-1 | D-1 MAE |
|---|---|---:|---:|---:|---:|
| Belgium / 2016 | Local | **6.13** | 14.55 | 0.738 | 8.30 |
| Germany / 2017 | Local | **6.62** | 10.48 | 0.670 | 9.89 |
| France / 2016 | Transfer | **5.14** | 15.68 | 0.713 | 7.21 |
| Nordic / 2017 | Local | **1.71** | 3.08 | 0.743 | 2.31 |

Important failure cases remain visible: France local has 8.60 MAE and a 55.49 RMSE because extreme errors dominate; Nordic pooled has 2.79 MAE and is worse than persistence (`rMAE = 1.208`). Geographic pooling is therefore a hypothesis to test, not a guaranteed improvement.

### Battery decision value

Each daily schedule is chosen from that model's 24-hour forecast before delivery and settled against actual prices. The optimization enforces:

- 1 MW charge/discharge power and 1 MWh energy capacity;
- 90% round-trip efficiency;
- 0.50 EUR/MWh transaction and 2.00 EUR/MWh degradation cost;
- no simultaneous charge and discharge through binary constraints;
- 50% initial and terminal state of charge each day.

| Zone | Highest-value forecast | Annual realized value | Accuracy rank | Value rank | Perfect-foresight bound |
|---|---|---:|---:|---:|---:|
| Belgium | Pooled | **EUR 6,520** | 2 | 1 | EUR 9,512 |
| Germany | Local | **EUR 5,150** | 1 | 1 | EUR 7,370 |
| France | Pooled | **EUR 6,526** | 2 | 1 | EUR 8,444 |
| Nordic | Local | **EUR 227** | 1 | 1 | EUR 731 |

In France, transfer has the lowest MAE but pooled earns EUR 558 more. In Belgium, local has the lowest MAE but pooled earns EUR 147 more. The perfect-foresight result is a non-rankable upper bound, not a deployable strategy.

### Regimes and limits

Conditional metrics are generated for negative prices, top-5% absolute-price spikes, peak hours, weekends, and fixed-date holidays. These are ex-post labels and never model inputs.

This release does **not** claim audited high-renewable, extreme-weather, or energy-crisis regime results for the historical cross-zone panel. Those labels require traceable realized series and definitions. Likewise, load, weather, outages, flows, fuel, and carbon features remain excluded until their forecast-time vintages can be reproduced. This is an intentional validity boundary.

## Reproduce the work

### Install

```bash
git clone https://github.com/MhdAsker/epex-price-forecast-benchmark.git
cd epex-price-forecast-benchmark
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

Python dependencies are pinned in [`requirements.txt`](requirements.txt). Raw data and large model files are ignored; acquisition functions recreate checksum manifests.

### Validate

```bash
python -m pytest -q
```

The 14 tests cover information isolation, DST handling, daily-curve construction, estimator dimensions, probabilistic scores, dependence-aware inference, incomplete-year rejection, battery feasibility/non-simultaneity, and shadow-packet checksum detection.

### Run experiments

```bash
# Phase 1: recent DE-LU protocol benchmark
python run_day_ahead.py

# Phase 2: full research target (expensive)
python run_phase2.py --checkpoint

# Rebuild Phase 2 metrics and plots from saved predictions
python finalize_phase2.py

# Phase 3: fetch four open price files, fit, evaluate, and report
python run_phase3.py

# Recreate paper tables/figures from immutable Phase 3 predictions
# and package dist/epex-benchmark-0.3.0.zip plus its checksum
python release.py
```

`run_phase2.py --checkpoint` writes an idempotent per-model panel so an interruption can resume without duplicating delivery dates. ENTSO-E acquisition requires `ENTSOE_API_KEY`; the open epftoolbox files do not.

## Artifact map

```text
config/
  forecast_protocol.json       noon D-1 contract
  phase2.json                  target rolling experiment
  phase3.json                  zones and decision specification
src/research/
  backtest.py                  rolling origin + temporal tuning + checkpoints
  probabilistic.py             sequential QRA and conformal calibration
  evaluation.py               point/probabilistic metrics, bootstrap, DM
  generalization.py            local, pooled, and transfer experiments
  decision.py                  non-anticipative battery MILP
  regimes.py                   ex-post conditional evaluation
  shadow.py                    append-only checksum and hash-chain packets
results/
  metrics/                     machine-readable tables and manifests
  predictions/                 out-of-sample panels
  plots/                       generated figures
reports/
  PHASE3_RESEARCH_REPORT.md    concise paper-style report
  DATA_CARD.md                 provenance and exclusions
  MODEL_CARDS.md               intended use and model limitations
release.py                     one-command reproducibility bundle
```

The Phase 3 manifest records a SHA-256 checksum and byte count for every research artifact. Historical shadow packets are explicitly marked as simulated; they must never be mixed with genuinely submitted live forecasts.

## Reproducibility and audit policy

- Raw sources receive retrieval metadata and content checksums.
- Prediction panels contain forecast date, product hour, actual, prediction, model, and experiment identity.
- Test outcomes never select hyperparameters.
- Same-auction leakage tests mutate future target values and require predictions to remain invariant.
- Statistical tests aggregate to the delivery-day level.
- Post-gate or perfect-foresight quantities are labeled non-rankable.
- Release metadata is available in [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json). A DOI will be added only after an authorized Zenodo deposit.

## Known limitations and next work

1. The complete multi-model Phase 2 artifact currently contains seed 42 with weekly parameter refits; daily refitting has been completed for JointCurve only. The configured three-seed daily panel should be completed before making seed-stability claims.
2. QRA and conformal intervals are still under-covered in volatile periods.
3. Cross-zone tests cover one full test year per zone; additional normal, crisis, and recent negative-price years require new immutable inputs.
4. The transfer design is coefficient warm-starting, not a domain-adversarial or hierarchical model.
5. Battery value is a deterministic day-ahead schedule. Intraday recourse, imbalance settlement, market impact, and reserve co-optimization are outside the present scope.
6. The repository prepares, but cannot independently mint, a DOI without release-account authorization.

## Citation

If you use the repository, cite the archived software release once a DOI is available and cite the benchmark methodology:

> Lago, J., Marcjasz, G., De Schutter, B., and Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy*, 293, 116983.

See [`CITATION.cff`](CITATION.cff) for machine-readable software metadata.

## License

MIT. See [`LICENSE`](LICENSE).
