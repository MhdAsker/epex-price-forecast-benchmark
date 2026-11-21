# Decision-aware DARR Phase 3 frozen retrospective report

## Classification

**B — Mixed generalization.** This is a frozen retrospective on previously examined 2016–2017 years, not an untouched test. No Kumo or reranker weight, feature, k, blend, or threshold was changed after viewing these outcomes.

## Primary findings

- Frozen LightGBM forecast MAE: **5.678 EUR/MWh**, versus 6.563 for direct Kumo and 10.100 for native DARR.
- Frozen economic retrieval revenue: **EUR 8579.13**, versus EUR 6680.96 for direct Kumo and EUR 9058.08 for JointCurve.
- Forecast retrieval does not beat JointCurve accuracy (5.452).
- Economic retrieval does improve direct-Kumo revenue. Costs, throughput, regret, losing days, drawdown, and timing overlaps are included in `summary_metrics.csv`.
- The frozen Ridge forecast achieved 5.521 MAE, **0.157 better than LightGBM**. The paired Ridge-minus-LightGBM interval was [-0.237, -0.078] with DM p=0.00011; the Phase 2 tree advantage did not generalize.
- Economic LightGBM added EUR 1898.18 while using 41.6 MWh less throughput, so its gain is not explained by more cycling. However, the forecast LightGBM earned even more (EUR 8686.05), and JointCurve earned EUR 9058.08.
- Negative-price recall was 21.1% for forecast LightGBM, 24.0% for Ridge, 19.4% for direct Kumo, and 34.3% for JointCurve across 242 negative hours.
- Only 62 hours exceeded the predeclared EUR 100/MWh threshold. Forecast LightGBM detected 0.0%, Ridge 1.6%, and JointCurve 12.9%; these rare-event findings are descriptive.

## Annual and combined results

| period   | model                    |      MAE |   battery_revenue_eur |   n_days |
|:---------|:-------------------------|---------:|----------------------:|---------:|
| combined | Frozen LightGBM economic |  8.66117 |               8579.13 |      731 |
| combined | Frozen LightGBM forecast |  5.6778  |               8686.05 |      731 |
| combined | Frozen Ridge economic    |  7.31494 |               8899.14 |      731 |
| combined | Frozen Ridge forecast    |  5.52114 |               8491.19 |      731 |
| combined | JointCurve               |  5.45224 |               9058.08 |      731 |
| combined | Kumo fine-tuned          |  6.56266 |               6680.96 |      731 |
| combined | Kumo fine-tuned + DARR   | 10.1003  |               7490.94 |      731 |
| combined | LEAR                     |  5.49118 |               8990.53 |      731 |
| 2016     | Frozen LightGBM economic |  7.44288 |               3584.05 |      366 |
| 2016     | Frozen LightGBM forecast |  4.40386 |               3598.49 |      366 |
| 2016     | Frozen Ridge economic    |  6.2099  |               3752.35 |      366 |
| 2016     | Frozen Ridge forecast    |  4.27648 |               3524.04 |      366 |
| 2016     | JointCurve               |  4.31697 |               3843.45 |      366 |
| 2016     | Kumo fine-tuned          |  4.99957 |               3010.04 |      366 |
| 2016     | Kumo fine-tuned + DARR   | 10.4136  |               3028.95 |      366 |
| 2016     | LEAR                     |  4.37002 |               3843.99 |      366 |
| 2017     | Frozen LightGBM economic |  9.8828  |               4995.08 |      365 |
| 2017     | Frozen LightGBM forecast |  6.95523 |               5087.56 |      365 |
| 2017     | Frozen Ridge economic    |  8.423   |               5146.78 |      365 |
| 2017     | Frozen Ridge forecast    |  6.76921 |               4967.15 |      365 |
| 2017     | JointCurve               |  6.59061 |               5214.63 |      365 |
| 2017     | Kumo fine-tuned          |  8.13002 |               3670.92 |      365 |
| 2017     | Kumo fine-tuned + DARR   |  9.78616 |               4461.99 |      365 |
| 2017     | LEAR                     |  6.61542 |               5146.54 |      365 |

## Primary paired uncertainty

| period   | model_1                  | model_2         | role             | metric      |   mean_daily_difference |   total_difference |    ci_low |   ci_high |   DM_statistic |     DM_pvalue |   n_days |
|:---------|:-------------------------|:----------------|:-----------------|:------------|------------------------:|-------------------:|----------:|----------:|---------------:|--------------:|---------:|
| combined | Frozen LightGBM forecast | Kumo fine-tuned | forecast_primary | MAE         |               -0.884852 |           -646.826 | -1.10998  | -0.661537 |       -7.54793 |   4.42241e-14 |      731 |
| combined | Frozen LightGBM forecast | Kumo fine-tuned | forecast_primary | revenue_eur |                2.74295  |           2005.1   |  2.05523  |  3.48338  |      nan       | nan           |      731 |
| combined | Frozen LightGBM economic | Kumo fine-tuned | economic_primary | MAE         |                2.09851  |           1534.01  |  1.47471  |  2.74231  |        6.26333 |   3.76844e-10 |      731 |
| combined | Frozen LightGBM economic | Kumo fine-tuned | economic_primary | revenue_eur |                2.59669  |           1898.18  |  1.94986  |  3.29706  |      nan       | nan           |      731 |
| 2016     | Frozen LightGBM forecast | Kumo fine-tuned | forecast_primary | MAE         |               -0.595704 |           -218.028 | -0.837459 | -0.365044 |       -4.68924 |   2.74227e-06 |      366 |
| 2016     | Frozen LightGBM forecast | Kumo fine-tuned | forecast_primary | revenue_eur |                1.6078   |            588.455 |  0.733509 |  2.39907  |      nan       | nan           |      366 |
| 2016     | Frozen LightGBM economic | Kumo fine-tuned | economic_primary | MAE         |                2.44331  |            894.251 |  1.62593  |  3.26242  |        5.64359 |   1.66543e-08 |      366 |
| 2016     | Frozen LightGBM economic | Kumo fine-tuned | economic_primary | revenue_eur |                1.56835  |            574.017 |  0.772313 |  2.26084  |      nan       | nan           |      366 |
| 2017     | Frozen LightGBM forecast | Kumo fine-tuned | forecast_primary | MAE         |               -1.17479  |           -428.799 | -1.51571  | -0.800073 |       -6.28056 |   3.37357e-10 |      365 |
| 2017     | Frozen LightGBM forecast | Kumo fine-tuned | forecast_primary | revenue_eur |                3.88121  |           1416.64  |  2.85632  |  4.89366  |      nan       | nan           |      365 |
| 2017     | Frozen LightGBM economic | Kumo fine-tuned | economic_primary | MAE         |                1.75277  |            639.763 |  0.866953 |  2.76121  |        3.44738 |   0.000566047 |      365 |
| 2017     | Frozen LightGBM economic | Kumo fine-tuned | economic_primary | revenue_eur |                3.62784  |           1324.16  |  2.59205  |  4.65865  |      nan       | nan           |      365 |

## Scientific answers

1. Learned forecast retrieval improved combined direct-Kumo MAE.
2. It outperformed native NVIDIA DARR.
3. Annual stability is shown above; effects are not assumed stable when signs differ.
4. It competed with LEAR/JointCurve accuracy.
5. Economic-focused retrieval increased net battery revenue versus direct Kumo.
6. Revenue uncertainty is reported at the paired daily level; gross point gains alone are not treated as evidence.
7. Mean spread error and throughput diagnose whether aggression explains value.
8. Negative/spike results are descriptive with explicit event counts; no rare-event significance is claimed.
9. Ridge/LightGBM differences are exploratory and paired in `paired_tests.csv`.
10. Continuation is warranted only if the frozen gains are stable enough for the intended objective.

## Protocol and limitations

The retrieval memory expands causally using the Phase 1/2 eligibility rule: every candidate's complete 72-hour continuation ends before query issuance. Conventional models use the repository's frozen weekly-refit forecasts, whereas Kumo and reranker parameters are fixed. This asymmetry is retained and disclosed. The €100/MWh spike threshold and volatility threshold were registered before Phase 3 scoring. No spread-amplification control was selected because no lambda was frozen on prior data.
