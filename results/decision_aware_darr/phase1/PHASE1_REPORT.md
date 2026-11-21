# Decision-aware DARR Phase 1 report

## Conclusion

Prompt 1 is complete. It constructs a causal, checksum-verified historical candidate dataset and does not train a reranker. The dataset contains **361 query dates** and **54,929 query-candidate pairs** drawn from **1424 distinct historical continuations**.

The sample is large in pair rows but small in independent query dates: only 241 training and 120 validation days. Prompt 2 should therefore use a constrained, strongly regularized reranker and dependence-aware evaluation rather than a flexible high-capacity model.

## Causal protocol

- Forecast origin: noon Europe/Berlin on D-1.
- Query period: 2015 only; 2016–2017 are excluded from training-data construction.
- Training queries: 2015-01-01 through 2015-08-29.
- Validation queries: 2015-09-01 through 2015-12-29.
- The August 30–31 origins are purged so 72-step labels cannot cross the partition boundary.
- Every candidate's final continuation delivery hour is strictly earlier than query issuance.
- Query outcomes and ex-post regimes exist only in label tables.

The adapter reproduces NVIDIA's embedding normalization, cosine ordering, top-k selection, and temperature-softmax forecast. It exposes identities and similarities that `knn_forecast` does not return; numerical equivalence is recorded in `native_equivalence.json`.

## Validation baselines

| partition   | method              |      MAE |     RMSE |      bias |   daily_rank_correlation |   mean_spread_error |   lowest_hour_accuracy |   highest_hour_accuracy |   battery_revenue_eur |   regret_eur |   throughput_mwh |   n_queries |
|:------------|:--------------------|---------:|---------:|----------:|-------------------------:|--------------------:|-----------------------:|------------------------:|----------------------:|-------------:|-----------------:|------------:|
| validation  | Direct Kumo         |  6.28707 |  8.36707 | -0.856762 |                 0.838302 |          -6.93944   |               0.208333 |                0.391667 |               1882.19 |      766.991 |          375.982 |         120 |
| validation  | Native DARR         |  7.95707 | 10.6935  |  2.82206  |                 0.811322 |           0.0835325 |               0.375    |                0.416667 |               1919.88 |      729.295 |          360.268 |         120 |
| validation  | Embedding top-1     | 10.3459  | 13.8177  |  1.81504  |                 0.699942 |           1.80608   |               0.183333 |                0.383333 |               1498.84 |     1150.33  |          387.895 |         120 |
| validation  | Price-pattern top-1 | 10.4379  | 16.5186  |  4.07559  |                 0.791284 |           5.19592   |               0.3      |                0.275    |               1473.26 |     1175.91  |          407.101 |         120 |
| validation  | Calendar top-1      | 13.8146  | 17.0869  | 10.3034   |                 0.752536 |           6.8135    |               0.2      |                0.4      |               1527.7  |     1121.48  |          387.126 |         120 |

No hyperparameter is selected using 2016–2017. These are descriptive non-learned baselines, not reranker results.

## Candidate-quality evidence

- Embedding similarity versus continuation MAE global Spearman: **-0.037**.
- Mean within-query Spearman: **-0.019**.
- Mean adjacent-query candidate-set Jaccard: **0.228**.
- Queries with a candidate MAE p90-p10 range above 1 EUR/MWh: **100.0%**.
- Queries with an incremental-value p90-p10 range above EUR 5: **100.0%**.

These statistics determine whether candidate choice contains learnable signal. High pair counts do not remove dependence caused by candidate reuse and overlapping 512-hour contexts.

## Difficult regimes

Negative-price-day rows: 25 queries. Spike-day rows: 0 queries. Ex-post regimes are evaluation-only and never feature columns. Full seasonal and regime tables are in `baseline_regimes.csv`.

## Prompt 2 inputs

- `query_features.parquet`
- `candidate_library.parquet`
- `candidate_features.parquet`
- `query_labels.parquet`
- `candidate_labels.parquet`
- `dataset_partitions.json`
- `candidate_score_correlations.csv`

Prompt 2 must preserve the registered chronological split, fit preprocessing on training queries only, and keep all `label__*` columns out of inference features.

## Limitations

- Only 361 independent 2015 query days exist; candidate rows are not independent.
- Historical EPEX-DE is not pooled with modern DE-LU.
- Sparse or absent price spikes limit spike conclusions.
- Battery labels are tied to the registered 1 MW / 1 MWh specification.
- This phase demonstrates dataset validity and baseline relationships, not learned-reranker performance.
