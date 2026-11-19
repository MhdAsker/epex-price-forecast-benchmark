# Explaining Kumo domain adaptation for German EPEX prices

## Executive conclusion

This study isolates the source of the previously reported improvement from NVIDIA Kumo-Forecast zero-shot inference (9.57 EUR/MWh) to head-only German adaptation (6.56 EUR/MWh) on the already-observed 2016–2017 EPEX-DE retrospective.

The central result is unambiguous on the eligible 2015 development period: **the improvement comes from forecasting-head weight updates, not from substituting German normalization statistics**. Changing only the external scaler changes MAE by 0.000011 EUR/MWh, while updating the published 72-step head reduces MAE by about 3.61 EUR/MWh under either normalization convention. Kumo's internal RevIN makes affine changes to the external input convention almost exactly cancel after the matching inverse transformation.

A 24-month history is the best development configuration, but its 0.068 EUR/MWh advantage over the 12-month head is not statistically supported by the seven-day block interval. Using every eligible pre-2015 window (1,063 windows, about 35 months) is worse than 24 months. More history is therefore not monotonically beneficial, and processing more epochs does not rescue the longer histories: the selected checkpoints occur within the common 276-step budget.

The study is deliberately conservative. All new choices are made on 2015. The 2016–2017 results remain a retrospective extension, not an untouched test. Conventional models are rolling-refit systems whereas the primary Kumo comparison uses a fixed checkpoint; identical targets do not make those adaptation schedules equivalent.

## Research question and pre-specified contrasts

The primary question is how much of Kumo's electricity-market improvement is attributable to:

1. a German training-only affine standardizer;
2. genuine changes to the pretrained forecasting-head weights; and
3. exposure to additional historical market regimes.

Phase 1 uses a 2 x 2 design:

| Cell | Weights | External normalization | Training |
|---|---|---|---|
| A | Published NVIDIA checkpoint | Published NVIDIA scaler | None |
| B | Published NVIDIA checkpoint | German 2012–2014 scaler | None |
| C | German-adapted head | German 2012–2014 scaler | Separate training |
| D | NVIDIA-normalized adapted head | Published NVIDIA scaler | Separate training |

Cell D is trained independently. It is not Cell C with a mismatched inference scaler. The planned effects are B-A (normalization without learning), C-B and D-A (weight adaptation under each scaler), C-D (adapted-normalization comparison), and the difference of the two learning effects (interaction).

## Immutable protocol and provenance

- Parent branch commit: `330df1537ca67d6b18fe7bb26ccadf61d127d58d`
- Official implementation: [NVIDIA/Kumo-TS](https://github.com/NVIDIA/Kumo-TS), specifically the pinned [`finetune_example.py`](https://github.com/NVIDIA/Kumo-TS/blob/ad459475d1302fc8365e7117e39837a8ecb6a7e3/Kumo-Forecast/examples/finetune_example.py)
- NVIDIA Kumo-TS commit: `ad459475d1302fc8365e7117e39837a8ecb6a7e3`
- NVIDIA forecast checkpoint revision: `402caa11b7ff585141e912c6743026141f648892`
- Checkpoint SHA-256: `04778bee8196c19af3a3e8dbd93e467a85a25208c63285917185347d9ec71743`
- Source EPEX-DE SHA-256: `1cae127d6647c0e8fa37c3b9726b1ecb8bf3676b1707aec62a2c707639bac7fb`
- Forecast origin: 12:00 Europe/Berlin on D-1
- Context / native target / score: 512 / 72 / first 24 hours
- Random seed: 13
- Development-selection boundary: 2015 only
- Dependence-aware uncertainty: paired daily losses, seven-day moving-block bootstrap, 2,000 replications, and HAC Diebold–Mariano tests

All 224 published tensors load compatibly, including both forecasting-head tensors. The model has 4,718,664 trainable head parameters and 341,240,320 frozen parameters. The MOMENT encoder and embedding layers remain frozen. No 24-step replacement head is introduced.

The sampler records the issuance time and explicit context and target indices. Training windows cannot cross validation boundaries. The historical Lago-style data retain their canonical 24-product daily-curve convention; the repository's separate timestamped sampler retains all 23/25 physical products on DST days.

## Normalization and checkpoint audit

| Standardizer | Mean | Population standard deviation | Fit observations | Last fit timestamp |
|---|---:|---:|---:|---|
| NVIDIA published | -5.6712 | 8.6933 | published artifact | published artifact |
| German training-only | 37.6540 | 16.6665 | 25,560 | 2014-12-31 23:00 |

Both forward/inverse transformations round-trip to at most 1.53e-5 EUR/MWh. The German statistics are exactly recomputed from the registered pre-2015 interval; neither 2015 nor 2016–2017 affects them. Two clean checkpoint loads have zero maximum head-tensor difference and zero prediction difference.

The pinned NVIDIA backbone applies per-window reversible instance normalization before the encoder and reverses it after the head. Consequently, a consistently applied external affine standardizer is theoretically redundant up to floating-point effects. Phase 1 tests rather than assumes this implication.

## Phase 1 results: normalization versus learned weights

### Full-2015 operational scores

| Cell | MAE | RMSE | Bias | Daily rank rho | Battery revenue |
|---|---:|---:|---:|---:|---:|
| A: pretrained + NVIDIA | 9.6415 | 12.2906 | +0.4445 | -0.0356 | -EUR 1,341 |
| B: pretrained + German | 9.6415 | 12.2906 | +0.4446 | -0.0356 | -EUR 1,341 |
| C: adapted + German | **6.0243** | **8.0973** | -0.8555 | **0.8040** | **EUR 4,677** |
| D: adapted + NVIDIA | 6.0310 | 8.2595 | +0.4916 | 0.7954 | EUR 4,617 |
| Naive D-1 | 8.0901 | 11.3598 | +0.0037 | 0.6716 | EUR 3,752 |

### Factorial effects on MAE

| Contrast | MAE difference (EUR/MWh) | Interpretation |
|---|---:|---|
| B-A | +0.000011 | Scaling alone has no practical effect |
| C-B | -3.617195 | German-normalized learned-weight effect |
| D-A | -3.610477 | NVIDIA-normalized learned-weight effect |
| C-D | -0.006707 | No supported advantage between adapted conventions |
| Interaction | -0.006718 | No material normalization/learning interaction |

For C versus D, the daily-loss difference is -0.0067 EUR/MWh, the 95% block interval is [-0.1353, 0.1262], and the DM p-value is 0.922. In contrast, both learned-weight effects are about 3.61 EUR/MWh and decisively supported. The tiny A/B numerical difference is statistically detectable only because it is nearly constant; its scale is eleven millionths of one EUR/MWh and is scientifically irrelevant.

Cell D's selected epoch is epoch 3 after 276 updates and 1,129 CPU seconds. Relative to the published head, its weight matrix changes by L2 norm 2.357 (maximum elementwise change 0.00670) and its bias changes by L2 norm 0.00155. The selected German-normalized 12-month head changes the weight matrix by L2 norm 3.280. These measurements verify that parameters changed; they are not interpreted as evidence that individual parameters encode market economics.

**Phase 1 decision:** proceed to longer-history experiments because genuine head adaptation—not scaling—accounts for essentially all of the observed improvement.

## Phase 2 results: history length and optimizer budget

Eligible windows are the last 365, last 730, or all 1,063 pre-2015 windows. The last set is described as “all eligible” or about 35 months, not as a fabricated complete 36-month sample.

Two budgets separate diversity from additional computation:

- fixed six epochs, which processes more updates when history is longer;
- a matched cap of 276 optimizer steps, using epoch-seeded permutations without replacement.

| History | Budget | Best epoch | Steps processed | 72-step validation MAE | Operational MAE | Battery revenue | CPU training time |
|---|---|---:|---:|---:|---:|---:|---:|
| 12 months / 365 | fixed and matched endpoint | 6 | 276 | 6.6073 | 6.0243 | EUR 4,677 | 1,719 s |
| 24 months / 730 | fixed 6 epochs | 3 | 552 | **6.4936** | **5.9559** | EUR 4,644 | 1,817 s |
| 24 months / 730 | matched | 3 | 276 | **6.4936** | **5.9559** | EUR 4,644 | 911 s |
| all eligible / 1,063 | fixed 6 epochs | 2 | 798 | 6.6331 | 6.1369 | EUR 4,541 | 1,644 s |
| all eligible / 1,063 | matched | 2 | 276 | 6.6331 | 6.1369 | EUR 4,541 | 333 s |

The fixed and matched variants select identical checkpoints for each history. Additional updates after the selected epoch lower neither the registered selection metric nor the saved model. The 24-month head improves operational MAE over 12 months by 0.0684 EUR/MWh, but its paired block interval [-0.0507, 0.2123] contains zero and the DM p-value is 0.295. Against all eligible history, the 24-month improvement is 0.1810 EUR/MWh; the DM p-value is 0.043, while the bootstrap interval [-0.00004, 0.3610] narrowly includes zero. These mixed uncertainty summaries do not justify a strong superiority claim.

The 24-month candidate is frozen because it minimizes the pre-specified 72-step 2015 validation MAE. Its checkpoint SHA-256 is `12a020a5b46924a92fbff0c4e244a4130c477bfcf5a0d39a28c3c3562191e9b3`. Selection is exploratory and explicitly qualified by the overlapping 12-versus-24-month uncertainty interval.

The aggregate gain is not uniform. Relative to the 12-month head, the 24-month candidate improves the 2015 high-volatility-day MAE from 8.020 to 7.665 and summer MAE from 5.146 to 4.956, but worsens negative-price-hour MAE from 21.303 to 22.692, holiday MAE from 8.206 to 9.225, and weekend MAE from 6.075 to 6.215. It predicts no negative prices in 2015 (zero recall). The selected head is therefore an average-loss compromise, not a solution to Kumo's tail weakness.

Longer history does not monotonically improve adaptation. Likely mechanisms include regime mismatch and the very small trainable output head having to compromise across older market distributions. This is an empirical association, not proof of a causal market mechanism.

## Interrupted-training integrity

During the all-history experiment, an interrupted run exposed that optimizer state alone was insufficient for exact continuation when dropout was active. Training now persists and restores Python, NumPy, PyTorch, and CUDA RNG states, along with the optimizer, current head, initial head, best head, epoch, and step count. Epoch-specific data-loader generators provide deterministic sample order.

The non-exact interrupted artifact was moved to an ignored archive and excluded from every table. The valid fixed and matched all-history candidates share a clean common state and checksum. A regression test now compares uninterrupted dropout training with a process-style resume and requires bitwise-identical saved head tensors and identical histories.

## Phase 3: frozen retrospective and adaptation schedule

The frozen 24-month head is evaluated on exactly the same 17,544 products as the archived reference panel. The annual strategy uses the frozen head in 2016, then starts from it and processes labels from 2015 through September 2016 for three pre-registered epochs. October–December 2016 is monitoring-only. The final epoch is fixed in advance so those already-observed outcomes cannot select a checkpoint.

| Model | 2016 MAE | 2017 MAE | Combined MAE | Combined battery revenue |
|---|---:|---:|---:|---:|
| JointCurve | **4.317** | **6.591** | **5.452** | **EUR 9,058** |
| LEAR | 4.370 | 6.615 | 5.491 | EUR 8,991 |
| Original 12-month Kumo | 5.000 | 8.130 | 6.563 | EUR 6,681 |
| Frozen 24-month Kumo | 4.836 | 8.076 | 6.454 | **EUR 6,942** |
| Annual-update Kumo | 4.836 | **7.906** | **6.369** | EUR 6,892 |
| Naive D-1 | 6.470 | 9.892 | 8.179 | EUR 6,078 |

The frozen 24-month head improves the original Kumo by 0.164 EUR/MWh in 2016 and 0.054 in 2017. Its combined daily-loss improvement is 0.109, but the block interval [-0.013, 0.225] for original minus frozen includes zero (`p = 0.068`). This matches the weak development evidence: 24 months is a plausible small improvement, not a robustly established one.

The annual head update improves 2017 MAE by 0.171 relative to the frozen 24-month head, but its interval [-0.037, 0.363] also includes zero (`p = 0.083`). The three monitoring MAEs are 9.233, 8.560, and 8.808 EUR/MWh; the fixed final-epoch rule retains epoch 3 even though epoch 2 looks better. This is direct evidence that more adaptation can overfit and why retrospective epoch selection would be invalid.

Both Kumo variants remain materially behind JointCurve. JointCurve beats the annual strategy by 0.916 EUR/MWh combined, with interval [0.655, 1.203] and `p < 1e-10`. The annual strategy remains significantly better than D-1 persistence by 1.810 EUR/MWh. Thus, adaptation narrows but does not close the specialized-model gap.

### Regimes and extremes

The annual update's 2017 gain is concentrated in high-volatility days (MAE 12.62 to 12.29), summer (4.93 to 4.66), autumn (9.66 to 9.31), holidays (12.89 to 12.28), and weekdays (7.86 to 7.64). Winter slightly worsens (11.63 to 11.68), and weekend accuracy remains weak (8.56). Negative-price-hour MAE improves from 39.61 to 37.32 and recall from 19.3% to 28.3%, but remains far behind JointCurve's 29.37 MAE and 40.7% recall.

There is only one >=100 EUR/MWh hour in 2016, so no 2016 spike-detection conclusion is made. Across 61 such hours on 12 days in 2017, annual Kumo detects none and has a -43.0 EUR/MWh spike bias. The frozen 24-month head detects 2 of 61 (3.3%) but still underpredicts the regime by 42.1 EUR/MWh. Tail accuracy therefore remains a decisive disadvantage rather than an incidental aggregate-error effect.

### Decision value and cost

The 24-month head adds EUR 261 of combined revenue over the original head, but the mean daily difference interval [-0.193, 0.971] includes zero. The annual strategy adds EUR 211 over the original, also unsupported [-0.246, 0.913], and earns EUR 50 less than the fixed 24-month head; the fixed-minus-annual interval [-0.443, 0.528] includes zero. Annual adaptation improves point accuracy without demonstrable economic value.

The annual training processes 240 optimizer steps over three CPU epochs in 2,058 seconds before 2017, followed by inference at roughly 0.71 seconds per daily origin. Its final head SHA-256 is `0e2b3b0f6b84ba8f8a13d5b97372e90516d611848060cadcf40971cc0d40252f`. Its higher compute cost and non-significant MAE/value changes provide no evidence that this update frequency is worthwhile in the present environment.

## DARR accuracy/value diagnostic

Native DARR is analyzed only after direct-model development and retrospective evaluation. The configuration remains the archived one: the original 12-month adapted head, 355 strictly pre-2015 context windows, NVIDIA's native embedding/retrieval code, `k=64`, temperature 0.05, and direct-model weight `alpha=0.01`. No 2016–2017 outcome tunes retrieval.

Because the archived panel retains the direct and blended forecasts, the retrieval-only forecast is recovered exactly as `(blend - 0.01 * direct) / 0.99` before a neighbor-level replay with the unchanged checkpoint.

| Forecast component | MAE | Bias | Daily rank rho | Mean predicted spread | Battery revenue |
|---|---:|---:|---:|---:|---:|
| Direct adapted Kumo | **6.563** | -0.616 | **0.791** | 21.80 | EUR 6,681 |
| Retrieval only | 10.173 | +4.584 | 0.772 | **27.45** | EUR 7,466 |
| 99% retrieval blend | 10.100 | +4.532 | 0.773 | 27.34 | **EUR 7,491** |

DARR degrades general forecasting by replacing almost all of the direct forecast with a biased retrieval estimate. Its MAE is 5.51 points worse than direct in 2016 and 1.66 worse in 2017. Ranking also falls slightly. However, it nearly removes the direct model's systematic spread compression: mean spread error changes from -5.10 to +0.44 EUR/MWh. Low-hour identification rises from 26.5% to 32.3%, while high-hour accuracy is broadly unchanged (32.1% to 31.1%).

The wider curves change dispatch. DARR raises throughput from 2,176.6 to 2,316.6 MWh and degradation cost from EUR 4,353 to EUR 4,633, but raises revenue by EUR 810. The paired daily gain is EUR 1.108 with block interval [0.480, 1.716]. This is almost entirely a 2017 result: +EUR 791 with a supported daily interval, versus only +EUR 19 in 2016 with an interval spanning zero. Across both years, 416 days improve and 297 worsen; positive gains total EUR 1,977 and downside totals -EUR 1,167. The top 10 days account for 15.3% of positive gains, so the effect is not solely one extreme event, though the largest daily gain/loss are +EUR 57.83 and -EUR 29.49.

The frozen neighbor replay explains why embedding similarity does not imply forecast quality. Mean top-1 cosine similarity is 0.981 and mean top-64 similarity is 0.965, yet the top neighbor's 24-hour continuation has 13.42 EUR/MWh MAE. The correlation between top-1 similarity and continuation MAE is only -0.055. Softmax weights are diffuse: the average top-neighbor weight is 2.16% and the effective neighbor count is 62.1 of 64. Only 32.7% of top neighbors share the query's calendar month. Thus, the pretrained embedding regards many contexts as geometrically close without identifying a sharply predictive electricity-market analogue. Averaging those neighbors reduces top-1 noise but imports level bias and amplified spreads.

The correct interpretation is a decision-ranking reversal. Retrieval amplifies economically useful spreads and changes selected hours, but it is not a better general price forecast. Its extra revenue is supported in this historical battery simulation, while the accompanying bias, error, higher cycling, and period instability make it unsuitable as an accuracy improvement claim.

## Compute and practical importance

The measured environment is Windows 11 with 16 GB system RAM, an NVIDIA GTX 1650 (4 GB), and PyTorch 2.12.1 CPU-only. CUDA is not available to the locked environment. Head training and inference are therefore genuine but CPU-bound. The official checkpoint is loaded for every independently trained cell; large checkpoints and resumable optimizer bundles remain outside Git, while hashes and metadata are committed.

Computational cost matters because the strongest conventional models already use rolling refits. A fixed Kumo checkpoint is cheaper online but adapts less often. The registered annual update is a proof of concept, not a claim of equal retraining frequency.

## Threats to validity

1. The 2016–2017 periods and their conventional benchmark outcomes were already observed before this ablation. They are retrospective evaluation, not a newly untouched holdout.
2. Kumo's fixed or annual head schedule differs from the conventional models' rolling calibration/refit schedules.
3. The canonical historical EPEX-DE series is not interchangeable with modern DE-LU. No cross-zone splice is made.
4. Negative-price and spike samples are sparse in some development slices. Absence of detections is reported, but sparse samples do not support sweeping event-detection conclusions.
5. No general-domain external suite is available here, so the study cannot establish whether German adaptation degrades Kumo on unrelated forecasting domains.
6. Only one random seed is executed for these expensive CPU experiments. Dependence-aware date uncertainty is reported, but seed stability remains untested.
7. DARR's archived forecast output does not retain neighbor identities or similarity values. Retrieval-only forecasts can be reconstructed exactly from the registered linear blend, but neighbor-level interpretation requires a separately registered rerun.

## Reproduction

From the repository root, after installing the pinned NVIDIA checkout and environment documented in `docs/KUMO_FINETUNING_RESEARCH.md`:

```powershell
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_adaptation_ablation.py audit
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_adaptation_ablation.py phase1
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_adaptation_ablation.py phase2
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_adaptation_ablation.py freeze
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_adaptation_ablation.py phase3
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_adaptation_ablation.py phase3-postprocess
.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe run_kumo_adaptation_ablation.py darr-analysis
```

Each phase verifies registered dataset/checkpoint hashes, writes a frozen configuration and artifact manifest, and fails rather than substituting synthetic forecasts. Large model weights remain ignored under `artifacts/kumo_adaptation_ablation/`.

The completed repository suite reports **27 passed tests**. The four committed adaptation-result manifests enumerate 49 artifacts; an independent checksum audit reports zero missing, size-mismatched, or hash-mismatched files.

## Answer to the primary research question

The existing Kumo improvement is a real domain-adaptation effect in the pretrained 72-step forecasting head. It is not explained by German-market normalization. Increasing eligible history from 12 to 24 months yields a small, uncertain development gain, while using all available history degrades validation accuracy. An annual update improves point accuracy slightly but does not add supported battery value. Native DARR earns more by amplifying spreads, not by retrieving futures whose quality tracks embedding similarity. Head-only adaptation materially rescues Kumo from an unsuitable zero-shot forecast but does not close the accuracy or economic-value gap to specialized rolling electricity-market models.
