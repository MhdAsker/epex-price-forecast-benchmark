# Model cards

## Local JointCurve

Multi-task elastic net trained on four years from the target zone. It predicts all 24 auction products jointly from complete D-1, D-2, and D-7 curves plus lag-curve summaries and calendar cycles.

## Pooled JointCurve

The same estimator trained on the union of all four zones with a zone indicator. It tests whether broader geographic data improves a target zone without accessing its test year.

## Transfer JointCurve

The estimator is pretrained on the three source zones and then fine-tuned on the final 365 target-zone calibration days. Warm-starting transfers coefficients; target test outcomes are never used.

All three are intended for benchmark research, not unattended bidding. They do not ingest load, weather, outages, flows, fuel, or carbon until forecast-time vintages are traceable.
