# Phase 3 research report

## Scope

This release evaluates local, pooled, and transfer 24-product curve models on four European price zones. Each result uses four prior calendar years for calibration and one complete held-out calendar year. Only lagged prices and calendar variables enter models because the bundled exogenous variables do not carry independently audited publication vintages.

## Generalization results

| zone   | model               |     MAE |   rMAE_D1 |
|:-------|:--------------------|--------:|----------:|
| BE     | Local JointCurve    | 6.12654 |  0.73831  |
| DE     | Local JointCurve    | 6.62405 |  0.669626 |
| FR     | Transfer JointCurve | 5.14288 |  0.713319 |
| NP     | Local JointCurve    | 1.71273 |  0.742693 |

## Decision-value results

| zone   | model             |   revenue_eur |   regret_eur |
|:-------|:------------------|--------------:|-------------:|
| BE     | Pooled JointCurve |      6520.48  |     2991.64  |
| DE     | Local JointCurve  |      5150.07  |     2219.68  |
| FR     | Pooled JointCurve |      6526.32  |     1917.68  |
| NP     | Local JointCurve  |       226.515 |      504.308 |

Battery schedules are fixed from the forecast before observing delivery prices. Realized revenue includes 90% round-trip efficiency, EUR 0.50/MWh transaction costs, EUR 2.00/MWh degradation, 1 MW/1 MWh limits, and equal initial/terminal state of charge. The perfect-foresight schedule is used only as a non-rankable regret bound.

## Regime analysis

Regimes are ex-post evaluation labels, never model inputs. This price-only release reports negative prices, top-5% absolute-price spikes, peak hours, weekends, and fixed-date holidays. Energy-crisis, high-renewable, and extreme-weather labels are **not claimed** for these 2016/2017 tests because auditable realized weather and renewable series were not bundled. Adding unverifiable labels would weaken the forecast-time audit.

## Limitations

- Cross-zone experiments freeze the Phase 2 JointCurve specification instead of retuning on test outcomes.
- Pooled and transfer experiments establish geographic robustness, not causal transfer.
- Historical files are immutable checksum-pinned snapshots, but they are retrospective simulations, not live submissions.
- A DOI is not minted automatically: Zenodo account authorization is an external release step.
