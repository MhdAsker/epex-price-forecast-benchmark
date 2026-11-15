# EPEX Kumo: longer-history and multi-season evaluation

**Status: reproducible experiment implemented; no new fine-tuning or retrospective run has been executed by this branch.** The existing 30-day CPU result and 3/6/12-month ablations remain unchanged in `docs/KUMO_FINETUNING_RESEARCH.md`.

## Research question

Does head-only adaptation of NVIDIA's 512-context / 72-step Kumo Forecast on longer historical German price data improve *operational* noon-D-1 delivery-day forecasting, seasonal robustness, and realized BESS value beyond pretrained Kumo and strong conventional predictors?

The model checkpoint, frozen encoder, target and publication-information protocol are unchanged. Retrieval (DARR) and cross-market adaptation are explicitly out of scope.

## Registered design

Configuration: `config/kumo_extended_study.json`.

- **Market and input:** canonical historical Lago et al. EPEX-DE price-only benchmark. No weather, wind, load, or other unaudited future-vintage covariates.
- **Issue time:** noon Europe/Berlin on delivery day D-1. Use 512 historically published prices as context and a 72-step output head. Score only D's 24 canonical hourly products. Physical 23/25-hour ENTSO-E data is a *different* dataset/protocol and must not be silently substituted.
- **Training:** issuance-safe target windows from February 2012 through December 2014, ending early enough to keep all 72 labels within that period.
- **History candidates:** latest 365 daily training origins; latest 730 daily training origins; **all available eligible training origins**. The last is roughly 35 months of target dates, not a guaranteed 36-month sample; the runner reports the actual dates/counts.
- **Normalization:** recompute each candidate's price mean and standard deviation *only* from its own chronological context-to-target training interval. Do not use the 2015 validation or 2016–17 retrospective prices to calculate these statistics.
- **Training budget:** fixed 3 epochs, learning rate `1e-4`, batch 8, seed 13, head-only weights. This fixes epochs and hyperparameters, but **does not equalize gradient-update counts** across history lengths. Any conclusion about data quantity alone must acknowledge that limitation.
- **Checkpoint selection within each candidate:** the existing `train_head` chooses the best epoch by validation MAE over the native 72-step horizon. Cross-history model selection then uses **2015 24-product operational validation MAE**. This scoring distinction is registered and should be evaluated rather than hidden.
- **Validation:** all eligible issuance-safe 2015 days, with winter/spring/summer/autumn performance reported. Record samples and evidence of overfitting. The 2015 labels are used for selecting model/epoch, not for weight updates.
- **Retrospective:** 2016–2017, with an immutable 2015 selection file required *before* running. These years were already seen in previous published EPEX experiments, so this is an explicitly **retrospective** comparison, not a newly untouched holdout.
- **Baselines:** Kumo zero-shot, D-1 persistence, LEAR, JointCurve, LightGBM, XGBoost and CatBoost on identical delivery dates/products. Existing baselines use weekly refitting and four-year rolling windows, whereas Kumo uses one fixed head; report the training-protocol difference.
- **Battery value:** reuse the repository's **1 MW / 1 MWh** battery, fixed daily terminal SOC, efficiency, transaction and degradation assumptions. Do not copy the different battery parameters from the separate German BESS trading engine.
- **Statistical analysis:** reuse the existing daily paired tests and seven-day moving-block bootstrap. Report coverage and uncertainty; do not overinterpret negative-price/spike categories containing few events.

The experiment explicitly **does not select** the best head according to 2016 or 2017 predictions. If results are viewed and additional hyperparameters are then proposed, those constitute exploratory follow-up, not a fresh independent test.

## Execution on an environment with the historical data and pinned NVIDIA model

From the repository root on a compatible Windows machine (the existing `scripts/setup_kumo.py` currently assumes Windows-style `.venv/Scripts/python.exe`):

```powershell
python scripts/setup_kumo.py
.\.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe -m pytest tests/test_kumo_extended.py -q
```

Fetch the existing dataset using the repository's published acquisition routine if the Parquet file is missing:

```powershell
python -c "from src.research.data import fetch_epftoolbox_de; fetch_epftoolbox_de()"
```

Use the *same pinned Kumo virtual environment* for all following commands. For example:

```powershell
$py = ".\.research_deps\Kumo-TS\Kumo-Forecast\.venv\Scripts\python.exe"
& $py run_kumo_extended.py plan
& $py run_kumo_extended.py validate --device cpu
& $py run_kumo_extended.py retrospective --device cpu
```

Where a compatible CUDA environment exists and is verified, substitute `--device cuda`. The original recorded machine has a GTX 1650 with 4 GB VRAM and PyTorch without CUDA; **CPU is the cautious default**, and the longer-history runs may take significantly longer than the earlier limited smoke test.

**Data and dependency prerequisites:** the runner requires `data/raw/epftoolbox_epex_de.parquet`, `results/predictions/phase2_point_predictions.parquet`, the existing SciPy optimization dependencies, the pinned NVIDIA source tree and trusted downloaded model artifacts. Missing prerequisites must yield a failure, not fabricated training results. Prefer using the already working pinned Kumo environment and installing missing requirements in it rather than installing a conflicting second PyTorch version.

## What the commands do

`plan`: reads and validates the existing dataset and prints exact counts, chronological origin ranges, and checksums. **No training.**

`validate`: runs *genuine* head-only training for the 365, 730, and all-available history candidates. Caches checkpoints outside Git; stores candidate-specific 2015 forecasts, seasonal metrics, checkpoint hashes, and the validation-selected **immutable** `selection.json`. Resuming is supported by candidate metadata and checksums; suspicious or incomplete caches fail.

`retrospective`: will *not run* without the frozen selection. Recreates both Kumo point-forecast variants on 2016–2017, joins identical-date conventional baselines, checks price/coverage equality, and measures MAE, RMSE, regimes, paired losses, battery revenue, throughput and regret. Writes a qualified retrospective manifest.

No full-grid search, DARR, or model-weight fine-tuning beyond the registered head-only runs is automatically launched.

## Remaining rigor gates

1. Run the tests in a real environment, including the repository's full test suite, and correct issues before training.
2. Execute `plan` and verify source hashes, origin coverage, and 2015 seasons.
3. Actually train and complete all three validation candidates. Keep large checkpoints and any proprietary credentials outside Git.
4. Freeze the 2015-selected checkpoint; do not edit it after examining 2016–2017.
5. Execute the retrospective evaluation. Inspect 2016 and 2017 individually, not just the combined average.
6. Publish paired comparisons, battery economics, and limitations alongside resource measurements.
7. On an independent later dataset, repeat the winning configuration before claiming generalization to modern German electricity markets.

**No new accuracy, revenue, significance, or fine-tuning-completion claims are supported until the runner has been executed and its outputs independently inspected.**
