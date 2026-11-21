# Decision-aware DARR Phase 2 report

## Scientific decision

**A — Promising.** Selection used only three purged expanding-window folds inside 241 training query dates. The 120 September–December 2015 dates are a development evaluation, not a pristine test. No 2016–2017 outcome was accessed and Kumo was not retrained.

Best development MAE: **Lightgbm Forecast frozen**, 5.489 EUR/MWh. Best development battery revenue: **Lightgbm Economic frozen**, EUR 2023.91. Direct Kumo achieved 6.287 EUR/MWh and EUR 1882.19.

## Main findings

- The frozen forecast LightGBM reduced MAE by **0.798 EUR/MWh (12.7%)** relative to direct Kumo. Its paired daily difference interval was [-1.257, -0.437] EUR/MWh with DM p=0.00015.
- The forecast model also improved daily rank correlation from 0.838 to 0.856, raised lowest-hour accuracy from 20.8% to 40.0%, and detected 28.6% of negative hours versus 0% for direct Kumo.
- The economic LightGBM increased total revenue by **EUR 141.72**, while using 30.1 MWh less throughput. Its paired mean daily revenue interval was [EUR 0.127, EUR 2.263]. Its MAE was worse (6.881), confirming that economic and predictive objectives are not interchangeable.
- LightGBM's MAE advantage over frozen Ridge was only 0.010 EUR/MWh; the paired interval [-0.210, 0.140] and DM p=0.910 do not support a material complexity benefit. Ridge is therefore the preferred parsimonious interpretation, while the training-only frozen LightGBM remains the registered Phase 3 candidate.
- Top-1 learned retrieval remained poor (MAE 8.155). The gain came from stable top-16 aggregation and a 25% direct-Kumo blend, not from a single supposedly analogous day.

## Frozen training-only choices

```json
{
  "ridge_forecast": {
    "experiment_id": "ridge__forecast__market_context__a1",
    "k": 16,
    "direct_weight": 0.25,
    "selection_metric": "MAE",
    "inner_cv_value": 4.201395640284868
  },
  "ridge_economic": {
    "experiment_id": "ridge__economic__market_context__a10",
    "k": 16,
    "direct_weight": 0.0,
    "selection_metric": "battery_revenue_eur",
    "inner_cv_value": 939.7713080273674
  },
  "lightgbm_forecast": {
    "experiment_id": "lightgbm__forecast__combined__g0",
    "k": 16,
    "direct_weight": 0.25,
    "selection_metric": "MAE",
    "inner_cv_value": 4.167689126599982
  },
  "lightgbm_economic": {
    "experiment_id": "lightgbm__economic__combined__g1",
    "k": 8,
    "direct_weight": 0.0,
    "selection_metric": "battery_revenue_eur",
    "inner_cv_value": 947.7299656291677
  },
  "phase3_forecast_candidate": {
    "experiment_id": "lightgbm__forecast__combined__g0",
    "k": 16,
    "direct_weight": 0.25,
    "selection_metric": "MAE",
    "inner_cv_value": 4.167689126599982
  },
  "phase3_economic_candidate": {
    "experiment_id": "lightgbm__economic__combined__g1",
    "k": 8,
    "direct_weight": 0.0,
    "selection_metric": "battery_revenue_eur",
    "inner_cv_value": 947.7299656291677
  }
}
```

The freeze manifest was written before development validation. Candidate rows were never randomly split; query dates were the statistical unit, with two-day purges before April, June, and August inner scoring periods.

## Development metrics

| method                          |      MAE |     RMSE |       bias |   daily_rank_correlation |   lowest_hour_accuracy |   highest_hour_accuracy |   mean_spread_error |   negative_price_recall |   battery_revenue_eur |   revenue_per_day_eur |   throughput_mwh |   revenue_per_throughput_eur_mwh |   degradation_eur |   transaction_eur |   regret_eur |   charge_timing_overlap |   discharge_timing_overlap |   losing_days |   worst_day_revenue_eur |   maximum_drawdown_eur |   top_10pct_positive_revenue_share |   n_queries |
|:--------------------------------|---------:|---------:|-----------:|-------------------------:|-----------------------:|------------------------:|--------------------:|------------------------:|----------------------:|----------------------:|-----------------:|---------------------------------:|------------------:|------------------:|-------------:|------------------------:|---------------------------:|--------------:|------------------------:|-----------------------:|-----------------------------------:|------------:|
| Lightgbm Forecast frozen        |  5.48883 |  7.29777 | -0.0742577 |                 0.855988 |               0.4      |                0.408333 |          -4.90092   |               0.285714  |               1900.18 |               15.8349 |          351.666 |                          5.40337 |           703.333 |           175.833 |      748.995 |                0.536077 |                   0.465608 |             2 |                -5.60612 |               -5.60612 |                           0.2211   |         120 |
| Ridge Forecast frozen           |  5.49908 |  7.38068 | -0.015758  |                 0.856351 |               0.416667 |                0.358333 |          -4.48435   |               0.238095  |               1840.22 |               15.3352 |          349.972 |                          5.2582  |           699.944 |           174.986 |      808.955 |                0.538897 |                   0.427935 |             2 |                -1.67579 |               -1.67579 |                           0.224931 |         120 |
| Ridge forecast retrieved        |  5.59781 |  7.62908 |  0.264577  |                 0.846169 |               0.391667 |                0.35     |          -2.82289   |               0.285714  |               1800.68 |               15.0057 |          351.255 |                          5.12642 |           702.51  |           175.628 |      848.497 |                0.531295 |                   0.430341 |             3 |                -1.42539 |               -1.42539 |                           0.226904 |         120 |
| Direct Kumo                     |  6.28707 |  8.36707 | -0.856762  |                 0.838302 |               0.208333 |                0.391667 |          -6.93944   |               0         |               1882.19 |               15.6849 |          375.982 |                          5.00606 |           751.963 |           187.991 |      766.991 |                0.473413 |                   0.512045 |             7 |                -7.38308 |               -7.38308 |                           0.235027 |         120 |
| Ridge Economic frozen           |  6.78529 |  9.17316 |  2.70993   |                 0.849256 |               0.391667 |                0.458333 |          -2.71071   |               0         |               1967.25 |               16.3938 |          363.477 |                          5.4123  |           726.955 |           181.739 |      681.927 |                0.543229 |                   0.538731 |             2 |                -1.47889 |               -1.47889 |                           0.22661  |         120 |
| Lightgbm Economic frozen        |  6.8813  |  9.12437 |  1.19518   |                 0.844005 |               0.35     |                0.458333 |          -0.835084  |               0.0714286 |               2023.91 |               16.8659 |          345.837 |                          5.8522  |           691.674 |           172.919 |      625.268 |                0.522959 |                   0.539103 |             2 |                -1.42587 |               -1.42587 |                           0.22714  |         120 |
| Causal heuristic k8             |  7.55366 | 10.0523  |  3.40197   |                 0.84049  |               0.341667 |                0.433333 |           2.27335   |               0.0952381 |               1839.06 |               15.3255 |          377.882 |                          4.86675 |           755.763 |           188.941 |      810.121 |                0.565536 |                   0.48649  |             4 |                -7.60697 |               -7.60697 |                           0.233274 |         120 |
| Native DARR                     |  7.95707 | 10.6935  |  2.82206   |                 0.811322 |               0.375    |                0.416667 |           0.0835325 |               0         |               1919.88 |               15.999  |          360.268 |                          5.32904 |           720.535 |           180.134 |      729.295 |                0.542244 |                   0.466464 |             2 |                -7.78562 |               -7.78562 |                           0.227831 |         120 |
| Ridge forecast top-1 diagnostic |  8.15511 | 14.7991  | -0.724663  |                 0.737898 |               0.325    |                0.283333 |           3.02567   |               0.309524  |               1319.72 |               10.9977 |          377.676 |                          3.49432 |           755.352 |           188.838 |     1329.46  |                0.475383 |                   0.374978 |             8 |               -13.1603  |              -18.8282  |                           0.278757 |         120 |
| Embedding top-1                 | 10.3459  | 13.8177  |  1.81504   |                 0.699942 |               0.183333 |                0.383333 |           1.80608   |               0         |               1498.84 |               12.4904 |          387.895 |                          3.86404 |           775.791 |           193.948 |     1150.33  |                0.457561 |                   0.439105 |            12 |               -10.8867  |              -11.6155  |                           0.27678  |         120 |
| Price-pattern top-1             | 10.4379  | 16.5186  |  4.07559   |                 0.791284 |               0.3      |                0.275    |           5.19592   |               0.119048  |               1473.26 |               12.2772 |          407.101 |                          3.61891 |           814.202 |           203.551 |     1175.91  |                0.490454 |                   0.336156 |             9 |               -12.842   |              -17.3552  |                           0.250176 |         120 |
| Calendar top-1                  | 13.8146  | 17.0869  | 10.3034    |                 0.752536 |               0.2      |                0.4      |           6.8135    |               0.047619  |               1527.7  |               12.7308 |          387.126 |                          3.94625 |           774.252 |           193.563 |     1121.48  |                0.446801 |                   0.455573 |            13 |               -14.0871  |              -14.0871  |                           0.235859 |         120 |

## Inner-fold ranking ablations

Embeddings-only Ridge produced top-1 MAE near 9.20 EUR/MWh. Market-context Ridge reduced it to 6.04, while adding embedding features back produced 6.31. Thus most learnable signal came from causal market-context differences, not pretrained embedding similarity. The economic objective improved revenue but sacrificed price-level accuracy, as prespecified.

| experiment_id                          | model_type   | objective   | feature_set    | parameters                                                                                                                                    |   top1_mae |   top1_incremental_revenue |   mean_within_query_spearman |   cv_queries |
|:---------------------------------------|:-------------|:------------|:---------------|:----------------------------------------------------------------------------------------------------------------------------------------------|-----------:|---------------------------:|-----------------------------:|-------------:|
| lightgbm__economic__combined__g1       | lightgbm     | economic    | combined       | {"learning_rate": 0.02, "max_depth": 3, "min_child_samples": 100, "n_estimators": 160, "num_leaves": 7, "reg_alpha": 2.0, "reg_lambda": 20.0} |    6.84516 |                 -0.438125  |                   0.271588   |           89 |
| lightgbm__economic__combined__g0       | lightgbm     | economic    | combined       | {"learning_rate": 0.03, "max_depth": 3, "min_child_samples": 80, "n_estimators": 100, "num_leaves": 5, "reg_alpha": 1.0, "reg_lambda": 10.0}  |    7.53929 |                 -0.905327  |                   0.261905   |           89 |
| ridge__economic__market_context__a1000 | ridge        | economic    | market_context | {"alpha": 1000.0}                                                                                                                             |    7.63719 |                 -0.152143  |                   0.233124   |           89 |
| ridge__economic__combined__a1000       | ridge        | economic    | combined       | {"alpha": 1000.0}                                                                                                                             |    7.80523 |                 -1.04219   |                   0.235052   |           89 |
| ridge__economic__market_context__a100  | ridge        | economic    | market_context | {"alpha": 100.0}                                                                                                                              |    7.84056 |                 -0.141111  |                   0.232616   |           89 |
| ridge__economic__market_context__a1    | ridge        | economic    | market_context | {"alpha": 1.0}                                                                                                                                |    7.88977 |                  0.0571233 |                   0.232503   |           89 |
| ridge__economic__market_context__a10   | ridge        | economic    | market_context | {"alpha": 10.0}                                                                                                                               |    7.88977 |                  0.0571233 |                   0.232456   |           89 |
| ridge__economic__combined__a1          | ridge        | economic    | combined       | {"alpha": 1.0}                                                                                                                                |    8.19986 |                 -0.886893  |                   0.235626   |           89 |
| ridge__economic__combined__a10         | ridge        | economic    | combined       | {"alpha": 10.0}                                                                                                                               |    8.19986 |                 -0.886893  |                   0.235581   |           89 |
| ridge__economic__combined__a100        | ridge        | economic    | combined       | {"alpha": 100.0}                                                                                                                              |    8.19986 |                 -0.886893  |                   0.235759   |           89 |
| ridge__economic__embedding_only__a1    | ridge        | economic    | embedding_only | {"alpha": 1.0}                                                                                                                                |    8.93148 |                 -2.62062   |                  -0.00807218 |           89 |
| ridge__economic__embedding_only__a10   | ridge        | economic    | embedding_only | {"alpha": 10.0}                                                                                                                               |    9.02737 |                 -2.52475   |                  -0.00808951 |           89 |
| ridge__economic__embedding_only__a100  | ridge        | economic    | embedding_only | {"alpha": 100.0}                                                                                                                              |    9.10838 |                 -2.65884   |                  -0.00893364 |           89 |
| ridge__economic__embedding_only__a1000 | ridge        | economic    | embedding_only | {"alpha": 1000.0}                                                                                                                             |    9.14609 |                 -2.94172   |                  -0.0163319  |           89 |
| lightgbm__forecast__combined__g0       | lightgbm     | forecast    | combined       | {"learning_rate": 0.03, "max_depth": 3, "min_child_samples": 80, "n_estimators": 100, "num_leaves": 5, "reg_alpha": 1.0, "reg_lambda": 10.0}  |    5.9273  |                 -1.15543   |                   0.581713   |           89 |
| lightgbm__forecast__combined__g1       | lightgbm     | forecast    | combined       | {"learning_rate": 0.02, "max_depth": 3, "min_child_samples": 100, "n_estimators": 160, "num_leaves": 7, "reg_alpha": 2.0, "reg_lambda": 20.0} |    6.0083  |                 -0.773141  |                   0.590161   |           89 |
| ridge__forecast__market_context__a1    | ridge        | forecast    | market_context | {"alpha": 1.0}                                                                                                                                |    6.03642 |                 -0.29906   |                   0.576138   |           89 |
| ridge__forecast__market_context__a10   | ridge        | forecast    | market_context | {"alpha": 10.0}                                                                                                                               |    6.08618 |                 -0.543037  |                   0.576048   |           89 |
| ridge__forecast__market_context__a100  | ridge        | forecast    | market_context | {"alpha": 100.0}                                                                                                                              |    6.16397 |                 -0.852252  |                   0.575899   |           89 |
| ridge__forecast__market_context__a1000 | ridge        | forecast    | market_context | {"alpha": 1000.0}                                                                                                                             |    6.20383 |                 -0.439961  |                   0.573824   |           89 |
| ridge__forecast__combined__a1          | ridge        | forecast    | combined       | {"alpha": 1.0}                                                                                                                                |    6.31404 |                 -0.732865  |                   0.572732   |           89 |
| ridge__forecast__combined__a10         | ridge        | forecast    | combined       | {"alpha": 10.0}                                                                                                                               |    6.31404 |                 -0.732865  |                   0.572704   |           89 |
| ridge__forecast__combined__a1000       | ridge        | forecast    | combined       | {"alpha": 1000.0}                                                                                                                             |    6.35318 |                 -1.03884   |                   0.570331   |           89 |
| ridge__forecast__combined__a100        | ridge        | forecast    | combined       | {"alpha": 100.0}                                                                                                                              |    6.42409 |                 -1.20832   |                   0.57255    |           89 |
| ridge__forecast__embedding_only__a1    | ridge        | forecast    | embedding_only | {"alpha": 1.0}                                                                                                                                |    9.19938 |                 -3.0911    |                   0.0497568  |           89 |
| ridge__forecast__embedding_only__a10   | ridge        | forecast    | embedding_only | {"alpha": 10.0}                                                                                                                               |    9.19938 |                 -3.0911    |                   0.0498006  |           89 |
| ridge__forecast__embedding_only__a100  | ridge        | forecast    | embedding_only | {"alpha": 100.0}                                                                                                                              |    9.19938 |                 -3.0911    |                   0.0502446  |           89 |
| ridge__forecast__embedding_only__a1000 | ridge        | forecast    | embedding_only | {"alpha": 1000.0}                                                                                                                             |    9.20243 |                 -2.61756   |                   0.0542785  |           89 |

## Dependence-aware uncertainty

Seven-day moving-block bootstrap intervals use 2,000 replications on paired daily differences. MAE rows also report Diebold–Mariano statistics. Full results are in `validation_uncertainty.csv`.

| method                          | reference   | metric      |   mean_daily_difference |   bootstrap_ci_low |   bootstrap_ci_high |   DM_statistic |     DM_pvalue |   n_days |
|:--------------------------------|:------------|:------------|------------------------:|-------------------:|--------------------:|---------------:|--------------:|---------:|
| Lightgbm Economic frozen        | Direct Kumo | MAE         |                0.594233 |        -0.259267   |          1.49218    |        1.28583 |   0.198504    |      120 |
| Lightgbm Economic frozen        | Direct Kumo | revenue_eur |                1.18102  |         0.126971   |          2.26285    |      nan       | nan           |      120 |
| Lightgbm Economic frozen        | Direct Kumo | regret_eur  |               -1.18102  |        -2.26285    |         -0.126971   |      nan       | nan           |      120 |
| Lightgbm Forecast frozen        | Direct Kumo | MAE         |               -0.798239 |        -1.25691    |         -0.437042   |       -3.79554 |   0.000147324 |      120 |
| Lightgbm Forecast frozen        | Direct Kumo | revenue_eur |                0.149965 |        -1.10733    |          1.09767    |      nan       | nan           |      120 |
| Lightgbm Forecast frozen        | Direct Kumo | regret_eur  |               -0.149965 |        -1.09767    |          1.10733    |      nan       | nan           |      120 |
| Ridge Economic frozen           | Direct Kumo | MAE         |                0.498225 |        -0.373771   |          1.32612    |        1.03294 |   0.301631    |      120 |
| Ridge Economic frozen           | Direct Kumo | revenue_eur |                0.708867 |        -0.707033   |          1.99843    |      nan       | nan           |      120 |
| Ridge Economic frozen           | Direct Kumo | regret_eur  |               -0.708867 |        -1.99843    |          0.707033   |      nan       | nan           |      120 |
| Ridge Forecast frozen           | Direct Kumo | MAE         |               -0.787992 |        -1.24316    |         -0.384914   |       -3.64048 |   0.00027213  |      120 |
| Ridge Forecast frozen           | Direct Kumo | revenue_eur |               -0.349702 |        -1.5871     |          0.696095   |      nan       | nan           |      120 |
| Ridge Forecast frozen           | Direct Kumo | regret_eur  |                0.349702 |        -0.696095   |          1.5871     |      nan       | nan           |      120 |
| Ridge forecast retrieved        | Direct Kumo | MAE         |               -0.689255 |        -1.27796    |         -0.14309    |       -2.35495 |   0.018525    |      120 |
| Ridge forecast retrieved        | Direct Kumo | revenue_eur |               -0.679216 |        -2.05365    |          0.406387   |      nan       | nan           |      120 |
| Ridge forecast retrieved        | Direct Kumo | regret_eur  |                0.679216 |        -0.406387   |          2.05365    |      nan       | nan           |      120 |
| Ridge forecast top-1 diagnostic | Direct Kumo | MAE         |                1.86804  |         0.654126   |          3.46787    |        2.57874 |   0.00991605  |      120 |
| Ridge forecast top-1 diagnostic | Direct Kumo | revenue_eur |               -4.68721  |        -7.18591    |         -2.9323     |      nan       | nan           |      120 |
| Ridge forecast top-1 diagnostic | Direct Kumo | regret_eur  |                4.68721  |         2.9323     |          7.18591    |      nan       | nan           |      120 |
| Lightgbm Economic frozen        | Native DARR | MAE         |               -1.07577  |        -1.91005    |         -0.267005   |       -2.62752 |   0.00860106  |      120 |
| Lightgbm Economic frozen        | Native DARR | revenue_eur |                0.866891 |         0.00467762 |          1.80692    |      nan       | nan           |      120 |
| Lightgbm Economic frozen        | Native DARR | regret_eur  |               -0.866891 |        -1.80692    |         -0.00467762 |      nan       | nan           |      120 |
| Lightgbm Forecast frozen        | Native DARR | MAE         |               -2.46824  |        -3.40398    |         -1.64802    |       -5.57991 |   2.40644e-08 |      120 |
| Lightgbm Forecast frozen        | Native DARR | revenue_eur |               -0.164169 |        -1.36237    |          0.868476   |      nan       | nan           |      120 |
| Lightgbm Forecast frozen        | Native DARR | regret_eur  |                0.164169 |        -0.868476   |          1.36237    |      nan       | nan           |      120 |
| Ridge Economic frozen           | Native DARR | MAE         |               -1.17178  |        -2.01944    |         -0.551653   |       -3.07095 |   0.00213377  |      120 |
| Ridge Economic frozen           | Native DARR | revenue_eur |                0.394733 |        -0.761851   |          1.4761     |      nan       | nan           |      120 |
| Ridge Economic frozen           | Native DARR | regret_eur  |               -0.394733 |        -1.4761     |          0.761851   |      nan       | nan           |      120 |
| Ridge Forecast frozen           | Native DARR | MAE         |               -2.45799  |        -3.3611     |         -1.67151    |       -5.76609 |   8.11329e-09 |      120 |
| Ridge Forecast frozen           | Native DARR | revenue_eur |               -0.663836 |        -1.76546    |          0.282838   |      nan       | nan           |      120 |
| Ridge Forecast frozen           | Native DARR | regret_eur  |                0.663836 |        -0.282838   |          1.76546    |      nan       | nan           |      120 |
| Ridge forecast retrieved        | Native DARR | MAE         |               -2.35926  |        -3.26491    |         -1.56421    |       -5.58799 |   2.29711e-08 |      120 |
| Ridge forecast retrieved        | Native DARR | revenue_eur |               -0.99335  |        -2.13433    |         -0.0895623  |      nan       | nan           |      120 |
| Ridge forecast retrieved        | Native DARR | regret_eur  |                0.99335  |         0.0895623  |          2.13433    |      nan       | nan           |      120 |
| Ridge forecast top-1 diagnostic | Native DARR | MAE         |                0.198043 |        -0.821722   |          1.56557    |        0.32211 |   0.74737     |      120 |
| Ridge forecast top-1 diagnostic | Native DARR | revenue_eur |               -5.00135  |        -7.66325    |         -3.06078    |      nan       | nan           |      120 |
| Ridge forecast top-1 diagnostic | Native DARR | regret_eur  |                5.00135  |         3.06078    |          7.66325    |      nan       | nan           |      120 |

## Computation and reproducibility

- Hardware: CPU-only execution; no CUDA device was available.
- Twenty-eight registered experiments completed in 15.06 aggregate model-fit seconds, excluding exact battery strategy evaluation.
- Frozen-model score generation for 120 queries took 0.30–0.39 seconds per model; details follow.

| method                   |   seconds |   queries |
|:-------------------------|----------:|----------:|
| Ridge Forecast frozen    |  0.30421  |       120 |
| Ridge Economic frozen    |  0.358501 |       120 |
| Lightgbm Forecast frozen |  0.394175 |       120 |
| Lightgbm Economic frozen |  0.379529 |       120 |

- Every experiment has a configuration/input hash, out-of-fold score checksum, checkpoint checksum, and completion manifest. Large checkpoints remain under ignored `artifacts/decision_aware_darr/phase2/`.
- The full repository suite passed 42 tests, including seven Phase 2-specific tests.

## Interpretation and limitations

- The effective training sample is 241 query dates, not 54,929 independent pairs.
- Economic labels are noisy and candidate reuse plus overlapping 512-hour contexts create dependence.
- Negative-price results are descriptive; the full Phase 1 period contains only 25 such days.
- There are zero registered spike days, so no spike significance test is reported.
- Learned retrieval forecasts are research-owned adapters, not NVIDIA native DARR.
- Validation was already historically visible through Phase 1 baseline reporting; all conclusions remain development evidence.

Phase 3 may run only the already frozen candidates. It must not revise these choices using 2016–2017 outcomes.
