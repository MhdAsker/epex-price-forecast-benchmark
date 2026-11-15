# Head-only domain adaptation of NVIDIA Kumo-Forecast for EPEX-DE

## Executive conclusion

This branch implements and executes an issuance-safe, head-only adaptation of NVIDIA Kumo-Forecast on the historical German EPEX benchmark. The encoder and embedder remain frozen; all 224 published checkpoint tensors, including both 72-step forecasting-head tensors, are loaded before training. The experiment uses a 512-hour price context and scores native forecast steps 1–24 at the noon D-1 operational origin.

The principal result is a qualified positive result for domain adaptation and a negative result for model superiority:

> Fine-tuning reduces two-year Kumo MAE from **9.57 to 6.56 EUR/MWh** and converts a loss-making zero-shot battery policy into **EUR 6,681** of realized value. The improvement over zero-shot is large and statistically supported, but fine-tuned Kumo remains significantly worse than JointCurve, LEAR, XGBoost, LightGBM, and CatBoost under the repository's frozen retrospective comparison.

Native NVIDIA DARR is also evaluated only after the direct checkpoint is frozen. With untuned official defaults and a strictly pre-2015 retrieval memory, DARR worsens two-year MAE to **10.10 EUR/MWh** and creates a +4.53 EUR/MWh bias. It nevertheless raises battery value to **EUR 7,491**, demonstrating that accuracy and economic rankings can diverge. DARR is not a forecast-accuracy improvement in this experiment.

The 2016–2017 years were already studied in this repository. These are retrospective extension results, not a newly untouched holdout. Conventional models also use weekly rolling refits and four-year calibration, whereas Kumo uses one fixed checkpoint; identical dates and products do not make the training protocols identical.

## Study status

| Stage | Status |
|---|---|
| Repository and protocol audit | Complete |
| NVIDIA source/checkpoint pinning | Complete |
| Noon D-1 causal sampler and DST tests | Complete |
| Published 72-step head compatibility | Complete: 224/224 tensors loaded |
| Genuine CPU head-only fine-tuning | Complete |
| 3/6/12-month development ablation | Complete |
| Full-2015 validation coverage | Complete: 363 origins |
| Monitored 10-epoch extension | Complete: early stopped after epoch 9; epoch 6 selected |
| Frozen 2016–2017 retrospective | Complete: 731 days / 17,544 products per model |
| Dependence-aware inference | Complete: 2,000 seven-day block-bootstrap replications and DM tests |
| Battery decision-value evaluation | Complete |
| Native DARR extension | Complete; negative accuracy result |
| Full 12-candidate registered grid | Implemented but not executed |
| 24-month/longer and multi-seed ablations | Not executed |
| General-domain degradation suite | Not executed |

## Immutable implementation pins

The extension imports NVIDIA's implementation; it does not recreate Kumo's architecture.

- Official source: [NVIDIA/Kumo-TS](https://github.com/NVIDIA/Kumo-TS)
- Fine-tuning reference: [`finetune_example.py`](https://github.com/NVIDIA/Kumo-TS/blob/ad459475d1302fc8365e7117e39837a8ecb6a7e3/Kumo-Forecast/examples/finetune_example.py)
- Kumo-TS commit: `ad459475d1302fc8365e7117e39837a8ecb6a7e3`
- Package version: `1.0.1`
- Kumo checkpoint revision: `402caa11b7ff585141e912c6743026141f648892`
- Checkpoint: `moment_head_512_6hr.pt`
- Checkpoint SHA-256: `04778bee8196c19af3a3e8dbd93e467a85a25208c63285917185347d9ec71743`
- MOMENT backbone revision: `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc`
- Selected fine-tuned head SHA-256: `bcbb8c66ccafdfd2f89b9dead859592a3bbd3a53e7a735eac5f075f8d2627c0c`

| Compatibility check | Result |
|---|---:|
| Checkpoint tensors | 224 |
| Compatible tensors loaded | 224 |
| Skipped / missing / unexpected | 0 / 0 / 0 |
| Compatible head tensors | 2 of 2 |
| Trainable head parameters | 4,718,664 |
| Frozen parameters | 341,240,320 |

This is adaptation of the pretrained forecasting head, not training a newly initialized 24-step output layer.

## Operational protocol

For delivery day D, issuance is 12:00 Europe/Berlin on D-1. Each example contains the last 512 eligible hourly prices, ending at product 23 of D-1, and a 72-step historical training target. Operational evaluation extracts steps 1–24 for D. Every window records the issuance time, context and target indices, last price availability time, and scored-product count.

Training targets span 2012-02-01 through 2014-12-29. The extended candidate uses the final 365 eligible training windows. Validation contains all 363 issuance-safe 2015 origins whose complete 72-step labels remain in 2015. The frozen retrospective contains all 366 days of 2016 and 365 days of 2017; only the first 24 labels are required for end-of-series scoring.

The canonical Lago et al. EPEX-DE series has 24 products per day. A separate UTC sampler and tests preserve 23/24/25 physical products for modern ENTSO-E data; historical EPEX-DE and DE-LU are not pooled. Unvintaged exogenous variables remain excluded.

Normalization is fit only on 2012–2014 German prices for the adapted model. Zero-shot inference uses NVIDIA's checksum-verified published standardizer. No validation or retrospective target enters training statistics.

## Compute environment

| Resource | Observed value |
|---|---|
| OS | Windows 11 |
| RAM | 16 GB |
| GPU | NVIDIA GeForce GTX 1650, 4 GB |
| Locked PyTorch | 2.12.1+cpu |
| CUDA available to PyTorch | No |
| Fine-tuning device | CPU |

The selected adaptive run trained for nine epochs in **1,719 seconds**. Inference averaged about **0.19–0.20 seconds per daily origin** at batch size 8. Large checkpoints remain outside Git; hashes and metadata are committed.

## Validation-selected longer training

The earlier 12-month development ablation improved through its third and final epoch, motivating a monitored extension to at most ten epochs. This extension was declared before reading the new retrospective outcomes, used all of 2015 for validation, and stopped after three consecutive non-improving epochs. It is an adaptive follow-up, not the complete registered 3×2×2 grid.

| Epoch | Train MSE | Validation MSE | 72-step validation MAE |
|---:|---:|---:|---:|
| 1 | 0.4141 | 0.3325 | 7.136 |
| 2 | 0.2954 | 0.3019 | 6.779 |
| 3 | 0.2720 | 0.2878 | 6.614 |
| 4 | 0.2479 | 0.3012 | 6.735 |
| 5 | 0.2345 | 0.2885 | 6.680 |
| **6** | **0.2197** | **0.2833** | **6.607** |
| 7 | 0.2140 | 0.2858 | 6.625 |
| 8 | 0.2011 | 0.2955 | 6.756 |
| 9 | 0.2023 | 0.2931 | 6.817 |

Epoch 6 is selected mechanically by minimum full-2015 validation MAE. Falling training loss with deteriorating validation performance after epoch 6 is direct evidence of overfitting pressure. The 24-step operational validation MAE for the selected model is 6.02 EUR/MWh; this differs from the selection metric because selection averages all 72 native steps.

## Full-year retrospective accuracy

Every model is scored on the same 8,784 products in 2016 and 8,760 products in 2017 with no missing predictions.

| Model | 2016 MAE | 2017 MAE | Combined MAE | Combined daily rank rho |
|---|---:|---:|---:|---:|
| JointCurve | 4.32 | **6.59** | **5.45** | **0.847** |
| XGBoost | **4.28** | 6.68 | 5.48 | 0.833 |
| LEAR | 4.37 | 6.62 | 5.49 | 0.842 |
| CatBoost | 4.45 | 6.65 | 5.55 | 0.836 |
| LightGBM | 4.60 | 6.61 | 5.60 | 0.833 |
| **Kumo fine-tuned** | **5.00** | **8.13** | **6.56** | **0.791** |
| Naive D-1 | 6.47 | 9.89 | 8.18 | 0.683 |
| Kumo zero-shot | 7.97 | 11.17 | 9.57 | -0.025 |

Fine-tuning improves Kumo MAE by 2.97 EUR/MWh in 2016 and 3.04 EUR/MWh in 2017. It also changes daily rank correlation from approximately zero to 0.818 and 0.764. Fine-tuned Kumo beats persistence in both years, but it is not competitive with the strongest rolling conventional methods under point accuracy.

### Dependence-aware comparisons

Loss differences are daily MAE for the first-listed model minus the comparator. Confidence intervals use 2,000 seven-day moving-block bootstrap replications.

| Comparison | Combined difference | 95% block CI | DM p-value |
|---|---:|---:|---:|
| Fine-tuned - zero-shot | **-3.006** | **[-3.282, -2.729]** | <1e-100 |
| Fine-tuned - D-1 | **-1.616** | **[-1.952, -1.272]** | <1e-18 |
| Fine-tuned - JointCurve | +1.110 | [0.835, 1.398] | <1e-13 |
| Fine-tuned - LEAR | +1.071 | [0.792, 1.353] | <1e-12 |

The fine-tuning benefit over zero-shot is statistically supported in each year and in the combined panel. The disadvantage versus JointCurve, LEAR, LightGBM, XGBoost, and CatBoost is also statistically supported. Statistical significance here does not remove the protocol mismatch between fixed Kumo and weekly-refit conventional models.

## Seasons and difficult regimes

Fine-tuning improves over zero-shot in every season and both years. It remains behind JointCurve and LEAR in the principal difficult regimes.

| Regime | Year | Fine-tuned | Zero-shot | JointCurve | LEAR | Events |
|---|---:|---:|---:|---:|---:|---:|
| Summer | 2016 | 3.25 | 5.89 | 2.97 | 3.04 | 2,208 h |
| Summer | 2017 | 5.03 | 7.44 | 4.14 | 4.19 | 2,208 h |
| Winter | 2016 | 6.70 | 10.54 | 5.52 | 5.60 | 2,184 h |
| Winter | 2017 | 11.71 | 15.64 | 8.84 | 8.94 | 2,160 h |
| Negative price | 2016 | 30.97 | 36.68 | 24.35 | 24.37 | 97 h |
| Negative price | 2017 | 37.42 | 42.15 | 29.37 | 30.94 | 145 h |
| High-volatility days | 2016 | 10.06 | 14.27 | 8.43 | 8.49 | 57 d |
| High-volatility days | 2017 | 12.81 | 17.24 | 10.30 | 10.42 | 126 d |

Fine-tuned negative-price recall is 12.4% in 2016 and 24.1% in 2017, with F1 of 0.182 and 0.347. Only one >=100 EUR/MWh spike occurs in 2016, so no meaningful 2016 spike conclusion is reported. There are 61 such hours across 12 days in 2017; fine-tuned Kumo detects none, as do several conventional methods. This is a material weakness rather than evidence of robust event forecasting.

## Battery decision value

The existing optimizer uses 1 MW charge/discharge power, 1 MWh capacity, 90% round-trip efficiency, EUR 0.50/MWh transaction cost, EUR 2/MWh degradation cost, no simultaneous charge/discharge, and 50% initial/terminal daily SOC.

| Forecast | 2016 revenue | 2017 revenue | Two-year revenue | Two-year regret |
|---|---:|---:|---:|---:|
| JointCurve | 3,843 | **5,215** | **9,058** | 3,513 |
| LEAR | **3,844** | 5,147 | 8,991 | 3,581 |
| LightGBM | 3,370 | 4,905 | 8,274 | 4,297 |
| XGBoost | 3,244 | 4,451 | 7,695 | 4,876 |
| Kumo fine-tuned + DARR | 3,029 | 4,462 | 7,491 | 5,080 |
| CatBoost | 2,983 | 4,471 | 7,454 | 5,117 |
| **Kumo fine-tuned** | **3,010** | **3,671** | **6,681** | **5,890** |
| Naive D-1 | 2,525 | 3,554 | 6,078 | 6,493 |
| Kumo zero-shot | -1,046 | -2,949 | -3,995 | 16,566 |

Fine-tuning adds EUR 10,676 relative to zero-shot and EUR 603 relative to persistence over two years, but remains EUR 2,377 below JointCurve. Accuracy and value rankings differ: XGBoost has the best 2016 MAE but only fourth-highest 2016 revenue; DARR has the worst combined point MAE yet improves fine-tuned Kumo's battery revenue by EUR 810.

## Native DARR extension

DARR uses NVIDIA's pinned `build_context_memory` and `knn_forecast` functions. Memory consists of 355 complete 2012–2014 windows at the official 72-hour stride. Every retrieved 72-step continuation ends by 2014-12-31 23:00, before validation and retrospective targets. The official defaults are fixed without test tuning: `alpha=0.01`, `k=64`, temperature `0.05`.

| Model | 2015 validation MAE | 2016 MAE | 2017 MAE | Combined bias |
|---|---:|---:|---:|---:|
| Fine-tuned direct | **6.02** | **5.00** | **8.13** | -0.62 |
| Fine-tuned + DARR | 8.54 | 10.41 | 9.79 | +4.53 |

DARR is significantly worse than the direct checkpoint on combined daily loss: direct minus DARR is -3.538 EUR/MWh, with 95% block CI [-4.223, -2.818] and DM p<1e-21. It is also worse in each year. The battery improvement appears to come from changed ranking and more aggressive spread capture, not calibrated prices: DARR increases throughput and improves some extreme-hour hit rates while severely degrading level accuracy. This result does not rule out validation-tuned blending or rolling causal memory, but such experiments must be registered separately.

## Training-history ablation

The earlier development-only ablation fixes three epochs, learning rate 1e-4, batch size 8, 30 validation origins, and 30 January 2016 development days.

| History | Windows | Best epoch | Validation MAE | Development MAE | Battery value |
|---:|---:|---:|---:|---:|---:|
| 3 months | 92 | 1 | 10.36 | 6.78 | EUR 361 |
| 6 months | 183 | 2 | 9.88 | 6.30 | EUR 396 |
| 12 months | 365 | 3 | **8.84** | **5.94** | EUR 389 |

More history improves MAE through 12 months in this limited design, while battery value is non-monotonic. Because batch/epoch settings and the small development period are fixed, this does not establish a universal data threshold. The requested 24-month and longer histories remain unexecuted.

## Research-question answers

1. **RQ1:** Yes. Head-only adaptation materially and significantly improves zero-shot Kumo on both complete retrospective years.
2. **RQ2:** No under this comparison. Fine-tuned Kumo beats persistence but remains significantly behind all five conventional learned models by MAE.
3. **RQ3:** Fine-tuning improves every examined difficult regime versus zero-shot, but remains weak on negative prices and spikes and trails JointCurve/LEAR.
4. **RQ4:** Fine-tuning repairs daily ranking and hour selection; combined rank correlation rises from -0.025 to 0.791.
5. **RQ5:** Yes versus zero-shot and persistence; no versus the strongest conventional strategies. DARR further shows that worse MAE can still raise dispatch value.
6. **RQ6:** Three to twelve months show increasing point benefit, but the minimum sufficient history and longer-history optimum remain unresolved.
7. **RQ7:** German validation overfitting appears after epoch 6. General forecasting degradation is unanswered because no external-domain suite was executed.

## Reproduce

```powershell
python scripts/setup_kumo.py

$kumoTests = rg --files tests | Where-Object { $_ -match 'test_kumo_.*\.py$' }
python -m pytest @kumoTests -q

.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py smoke
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py ablation
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py extended
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py retrospective
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_finetuning.py darr
python run_kumo_finetuning.py audit
```

`grid` remains available for the complete 12-candidate validation-only search on stronger hardware. `extended` is resumable and writes progress/checkpoint metadata after every epoch. Retrospective and DARR outputs contain predictions, metrics, regime tables, pairwise tests, battery dispatch, manifests, and hashes. Model weights stay gitignored under `artifacts/kumo_finetuning/`.

## Limitations

- The extended checkpoint is adaptively motivated and is not the winner of the unexecuted full registered grid.
- The retrospective years were previously examined in the parent benchmark.
- Fixed-checkpoint Kumo and weekly-refit conventional models have different training schedules.
- Only the historical canonical EPEX-DE experiment is complete; modern DE-LU fine-tuning is not executed.
- Only one optimization seed, one extended learning rate, and up to 12 months of data are evaluated.
- The full checkpoint is large and CPU training limits the feasible search.
- DARR uses fixed pre-2015 memory and official defaults. Rolling causal memory and validation-tuned blending are new hypotheses, not conclusions from this study.
- External-domain evaluation is required before claiming that head adaptation preserves Kumo's general forecasting capability.
