# Head-only domain adaptation of NVIDIA Kumo-Forecast for EPEX-DE

## Executive conclusion

This branch implements an issuance-safe NVIDIA Kumo-Forecast extension and executes a **real, limited CPU fine-tuning study**. It does not execute the registered 12-candidate grid or the full 2016–2017 retrospective experiment because the available PyTorch environment is CPU-only and the installed GPU has 4 GB VRAM.

On 30 fixed January 2016 development days, the genuine head update reduced Kumo MAE from **10.14 to 6.13 EUR/MWh**, improved daily rank correlation from **−0.108 to 0.838**, and changed battery value from **−EUR 133 to EUR 400**. Fine-tuned Kumo beat D-1 persistence (7.10 MAE; EUR 269), but remained behind the frozen conventional models (5.16–5.84 MAE; EUR 401–438). The 30-day block-bootstrap interval supports an improvement over zero-shot and persistence, but comparisons with the conventional models are not significant at 5% and are too small for a research-grade superiority claim.

The defensible answer is therefore:

> Domain-specific head adaptation is promising and clearly repairs the poor zero-shot fit on this development slice, but the available experiment does **not** establish that Kumo exceeds strong conventional EPEX models or that it improves full-year battery value. A validation-selected full grid and frozen 2016–2017 run are still required.

## Registered status

| Stage | Status |
|---|---|
| Repository/protocol audit | Complete |
| Official NVIDIA implementation audit | Complete |
| Exact code/checkpoint pinning | Complete |
| Noon D-1 causal sampler | Complete and tested |
| 72-step checkpoint compatibility | Complete: no skipped/missing keys |
| Zero-shot development benchmark | Complete, 30 days |
| Genuine head-only CPU fine-tuning | Complete, 90 windows / 3 epochs |
| 3/6/12-month data ablation | Complete, development only |
| 24-month and longer ablation | Registered, not executed locally |
| Full 3×2×2 hyperparameter grid | Implemented/resumable, not executed |
| Frozen 2016–2017 retrospective | Implemented, blocked on grid selection |
| Full-year statistical/battery conclusions | Not available |
| DARR or cross-market extension | Not started by design |

## Official implementation and immutable pins

The extension imports NVIDIA’s model builder and checkpoint format; it does not recreate Kumo’s neural architecture.

- Official source: [NVIDIA/Kumo-TS](https://github.com/NVIDIA/Kumo-TS)
- Fine-tuning reference: [`finetune_example.py`](https://github.com/NVIDIA/Kumo-TS/blob/ad459475d1302fc8365e7117e39837a8ecb6a7e3/Kumo-Forecast/examples/finetune_example.py)
- Kumo-TS commit: `ad459475d1302fc8365e7117e39837a8ecb6a7e3`
- Package version: `1.0.1`
- Checkpoint repository: [`nvidia/Kumo-Forecast`](https://huggingface.co/nvidia/Kumo-Forecast)
- Checkpoint revision: `402caa11b7ff585141e912c6743026141f648892`
- Base checkpoint: `moment_head_512_6hr.pt`, 1,404,422,626 bytes
- Checkpoint SHA-256: `04778bee8196c19af3a3e8dbd93e467a85a25208c63285917185347d9ec71743`
- MOMENT backbone: `AutonLab/MOMENT-1-large`
- Backbone revision: `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`
- Input length: 512 hourly prices
- Native output horizon: 72 hourly steps

The official environment is reproduced from NVIDIA’s `uv.lock`; `pyarrow==17.0.0` is the only project-specific addition needed to read the existing immutable parquet data.

### Checkpoint compatibility evidence

| Check | Result |
|---|---:|
| Checkpoint tensors | 224 |
| Compatible tensors loaded | 224 |
| Skipped tensors | 0 |
| Missing tensors | 0 |
| Unexpected tensors | 0 |
| Forecast-head tensors | `head.linear.weight`, `head.linear.bias` |
| Compatible head tensors | 2 of 2 |
| Trainable parameters | 4,718,664 |
| Frozen parameters | 341,240,320 |
| Total parameters | 345,958,984 |

This experiment genuinely adapts the published 72-step forecasting head. It is not a newly initialized 24-step head mislabeled as fine-tuning.

## Operational data design

### Historical market

The experiment uses only the price column from the Lago et al. historical EPEX-Germany benchmark. Bundled exogenous columns remain excluded because their publication vintages have not been independently audited.

- Registered training targets: 2012-02-01 through 2014-12-29.
- Registered validation targets: 2015 only; 72-step targets cannot cross into 2016.
- Retrospective test registration: 2016–2017.
- Executed development slice: 2016-01-01 through 2016-01-30.

The test years were already examined by this repository, so any future result is explicitly a retrospective extension, not a pristine holdout.

### Issuance mapping

For delivery day D:

1. Forecast issuance is recorded as 12:00 Europe/Berlin on D-1.
2. The 512-price context ends with product 23 of D-1. Those prices are eligible because the complete D-1 auction curve was published on D-2.
3. The native target contains D, D+1, and D+2 (72 canonical products).
4. Operational scoring extracts native steps 1–24 for delivery day D.
5. Training and validation require all 72 labels to remain inside their registered split.

The official generic CSV sampler was not used unchanged because generic sliding windows do not encode the auction issuance time. The custom sampler records the issuance time, context interval, target interval, and price availability check for every origin.

For modern ENTSO-E UTC data, a separate sampler preserves every physical hour and records 23, 24, or 25 scored delivery products. Automated tests cover both 2023 DST transitions. Historical EPEX-DE and modern DE-LU are never pooled.

### Normalization

- Unchanged zero-shot Kumo uses NVIDIA’s checksum-verified published standardizer: mean −5.6712, standard deviation 8.6933.
- Fine-tuned Kumo uses statistics fitted only on registered 2012–2014 German prices: mean 37.6540, standard deviation 16.6665.

No validation or development price enters either German training statistic.

## Execution environment

| Resource | Observed value |
|---|---|
| OS | Windows 11 |
| CPU RAM | 16 GB |
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 4,096 MiB |
| Driver | 610.62 |
| Official locked PyTorch | 2.12.1+cpu |
| CUDA available to PyTorch | No |
| Free disk before artifact downloads | 23.8 GiB |

The pinned checkpoint and backbone consume about 2.8 GB. CPU batch benchmarking measured approximately 0.39 s/origin at batch 1 and 0.19 s/origin at batch 8 for forward inference. The full 12-candidate grid was not launched because it would require many hours of repeated frozen-backbone passes and the 4 GB GPU does not meet the registered ≥8 GB preference.

## Executed fine-tuning

The smallest genuine study used the final 90 eligible training origins of 2014, the first 30 validation origins of 2015, batch size 4, learning rate 1e-4, seed 13, and three epochs. Checkpoint selection used validation MAE only.

| Epoch | Train MSE (standardized) | Validation MSE | Validation MAE (EUR/MWh) |
|---:|---:|---:|---:|
| 1 | 0.589 | 0.763 | 11.86 |
| 2 | 0.377 | 0.692 | 11.15 |
| **3** | **0.313** | **0.628** | **10.63** |

- Training runtime: 110.6 seconds on CPU.
- Best epoch: 3.
- Head-weight L2 change: 1.640 (maximum element change 0.00417).
- Head-bias L2 change: 0.00165.
- Saved head SHA-256: `a8964b6560f2e1b5639ad3369fef032856047908fd69cae019cfcdcabab28015`.
- The large head artifact is intentionally gitignored; its checksum and training history are committed.

The decreasing training and validation curves show no observed overfitting within three epochs, but three epochs and 30 validation origins are insufficient to rule it out.

## Development results: identical 30 delivery days

The conventional predictions are frozen Phase 2 artifacts using weekly refitting and four-year rolling calibration. Kumo uses a fixed checkpoint. The dates/products match, but training protocols are not fully equivalent.

| Model | MAE | RMSE | Bias | Daily rank ρ | Lowest-hour hit | Highest-hour hit | Spread MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| JointCurve | **5.16** | **7.20** | 0.10 | **0.865** | **33.3%** | 46.7% | 7.46 |
| LEAR | 5.18 | 7.27 | 0.21 | 0.858 | **33.3%** | 43.3% | 7.17 |
| XGBoost | 5.54 | 7.75 | −1.19 | 0.851 | 23.3% | **56.7%** | **7.12** |
| CatBoost | 5.74 | 7.82 | −1.37 | 0.835 | 26.7% | 33.3% | 7.14 |
| LightGBM | 5.84 | 7.97 | −1.46 | 0.855 | 30.0% | 53.3% | 7.65 |
| **Kumo fine-tuned** | **6.13** | **8.11** | −1.27 | **0.838** | 16.7% | 43.3% | 7.82 |
| Naive D-1 | 7.10 | 9.84 | 0.33 | 0.707 | 13.3% | 26.7% | 8.81 |
| Kumo zero-shot | 10.14 | 12.38 | 2.07 | −0.108 | 3.3% | 0.0% | 19.82 |

Fine-tuning appears to repair the daily profile and ranking much more than it merely shifts the mean. Bias changes from +2.07 to −1.27 EUR/MWh, while daily rank correlation changes from negative to 0.838 and spread error falls by 12 EUR/MWh.

### Dependence-aware inference

Daily absolute losses were compared using seven-day moving blocks with 2,000 replications and the repository’s HAC Diebold–Mariano implementation.

| Comparison (fine-tuned minus comparator) | Mean daily loss difference | 95% block CI | DM p-value |
|---|---:|---:|---:|
| Kumo zero-shot | **−4.01** | **[−5.11, −3.24]** | <1e-18 |
| Naive D-1 | **−0.97** | **[−1.92, −0.19]** | 0.034 |
| JointCurve | +0.97 | [−0.26, 1.75] | 0.062 |
| LEAR | +0.95 | [−0.26, 1.71] | 0.062 |
| LightGBM | +0.30 | [−0.87, 1.04] | 0.545 |
| XGBoost | +0.59 | [−0.50, 1.23] | 0.205 |
| CatBoost | +0.39 | [−0.87, 1.32] | 0.473 |

These p-values are exploratory because there are only 30 dependent daily observations. Statistical significance on this slice is not evidence of full-year generalization.

### Negative prices, volatility, and unusual events

The slice contains only four negative-price products across two days and zero prices at or above the registered EUR 100/MWh spike threshold. Fine-tuned Kumo predicts none of the four negative products, so negative-price precision/F1 are undefined and recall is zero. No spike-detection conclusion is possible.

On the eight top-quartile daily-volatility days, MAE is 8.26 for fine-tuned Kumo, 13.26 for zero-shot, 6.31 for JointCurve, and 8.12 for LightGBM. Fine-tuning helps substantially relative to zero-shot but does not lead the conventional models.

Seasonal conclusions are also unavailable because the executed development period covers January only.

## Training-history ablation

The CPU-feasible ablation fixes three epochs, learning rate 1e-4, batch size 8, the same first 30 validation origins, and the same 30 development days.

| History | Windows | Best epoch | Validation MAE | Development MAE | Rank ρ | Battery value | Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 92 days | 92 | 1 | 10.36 | 6.78 | 0.786 | EUR 361 | 108 s |
| 183 days | 183 | 2 | 9.88 | 6.30 | 0.811 | EUR 396 | 194 s |
| 365 days | 365 | 3 | **8.84** | **5.94** | **0.824** | EUR 389 | 374 s |

Within this limited design, more history monotonically improves validation and forecast MAE through 12 months. Battery value is not monotonic: six months exceeds twelve months. The 3-month batch-8 result also differs from the batch-4 smoke, showing that history length cannot be separated from optimization until the registered grid is complete. The requested 24-month and longer runs remain unexecuted.

## Battery decision value

Every strategy uses the existing 1 MW / 1 MWh mixed-integer optimizer with 90% round-trip efficiency, EUR 0.50/MWh transaction cost, EUR 2/MWh degradation cost, no simultaneous charging/discharging, and equal 50% initial/terminal SOC.

| Forecast | 30-day realized value | Perfect-foresight regret | Throughput |
|---|---:|---:|---:|
| LightGBM | **EUR 438.09** | EUR 120.21 | 60.08 MWh |
| JointCurve | EUR 437.04 | EUR 121.26 | 67.79 MWh |
| LEAR | EUR 435.62 | EUR 122.68 | 69.79 MWh |
| XGBoost | EUR 429.43 | EUR 128.87 | 59.98 MWh |
| CatBoost | EUR 400.88 | EUR 157.42 | 60.08 MWh |
| **Kumo fine-tuned** | **EUR 400.38** | **EUR 157.91** | 62.78 MWh |
| Naive D-1 | EUR 269.39 | EUR 288.91 | 84.71 MWh |
| Kumo zero-shot | **−EUR 133.02** | EUR 691.32 | 28.73 MWh |
| Perfect foresight | EUR 558.30 | 0 | non-deployable bound |

Fine-tuning adds EUR 533 relative to zero-shot and EUR 131 relative to persistence on this slice, but it does not exceed the best conventional battery value. Lower MAE and decision value remain separate outcomes.

## Answers to the research questions

1. **RQ1 – zero-shot improvement:** Yes on the 30-day development slice; the effect is large and the paired interval excludes zero. Full-year confirmation is pending.
2. **RQ2 – conventional competition:** Fine-tuned Kumo is competitive with persistence but trails every frozen conventional model by point MAE. Differences versus conventional models are not significant on 30 days.
3. **RQ3 – difficult regimes:** Fine-tuning improves high-volatility MAE versus zero-shot, but negative/spike samples are insufficient and negative recall remains zero.
4. **RQ4 – price ranking:** Fine-tuning raises daily rank correlation from −0.108 to 0.838 and improves extreme-hour identification, but JointCurve/LEAR remain stronger overall.
5. **RQ5 – battery value:** Fine-tuning converts a loss-making zero-shot schedule into positive value, but does not beat LightGBM/JointCurve.
6. **RQ6 – data requirement:** 3→6→12 months improves MAE in the fixed batch-8 ablation. The benefit threshold, 24-month behavior, and longer-history optimum remain unresolved.
7. **RQ7 – general degradation:** Not answered. Only the German head was evaluated; no external general-forecasting suite was run. Head parameter changes are not evidence of economic understanding or preserved generality.

## Reproduce

```powershell
# From repository root: clone exact NVIDIA commit, install its frozen lock,
# and download checksum-pinned checkpoint/backbone artifacts.
python scripts/setup_kumo.py

# Causality and interface tests (lightweight project environment).
python -m pytest tests/test_kumo_*.py -q

# Genuine CPU smoke and locally feasible history ablation.
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py smoke
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py ablation

# On a compatible CUDA environment (preferably >=8 GB VRAM): resumable full grid.
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py grid

# Only after validation-only grid selection exists:
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py retrospective
```

The grid writes one candidate checkpoint at a time and resumes completed candidates. The retrospective command refuses to run without `grid/selection.json`, preventing test outcomes from selecting the checkpoint.

## Artifact map

```text
config/kumo_finetuning.json                 preregistered protocol/grid
src/research/kumo/data.py                   issuance-safe canonical and UTC samplers
src/research/kumo/train.py                  official model load verification + head training
src/research/kumo/inference.py              first-24-step operational inference
src/research/kumo/evaluation.py             ranking/negative/spike/regime metrics
src/research/kumo/diagnostics.py            environment metadata
tests/test_kumo_*.py                        causality, DST, training, inference tests
results/kumo_finetuning/development/        committed smoke predictions/metrics/plots
artifacts/kumo_finetuning/                  gitignored head checkpoints
```

## Remaining limitations

- No full hyperparameter grid, full-year retrospective, 24-month/longer ablation, multi-seed analysis, or general-domain degradation suite has been executed.
- The development slice is only 30 days and was part of the repository’s previously examined 2016 test year.
- Conventional models use weekly rolling refits, while Kumo uses a fixed fine-tuned checkpoint.
- The historical dataset has canonical 24-product days and cannot test DST directly; DST safety is tested using physical UTC ENTSO-E-style synthetic data.
- Modern DE-LU data end on 2024-11-30, so 2024 is an incomplete retrospective period and was not mixed with historical EPEX-DE.
- DARR and cross-market Kumo adaptation are intentionally deferred until head-only value is confirmed on a full registered evaluation.
