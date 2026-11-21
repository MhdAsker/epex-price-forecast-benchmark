# Decision-aware DARR Phase 2 progress

## Current state

- Status: COMPLETE.
- Branch: `feature/decision-aware-darr-reranker`.
- Parent commit: `7a3eb78`.
- Last verified stage: Stage 9 audit and research report complete.
- Phase 1 manifest SHA-256: `3b4973f6da581aea0c9d6a609efc1db69cfc73a631e71b6d736b87fe6b68c03d`.
- Phase 2 configuration SHA-256: `737ac5eeb0c427294badfc51279481da42c4e8c019b16c4c5e7b589cc8579d1c`.
- Phase 1 inputs: 18 / 18 manifest outputs checksum-verified.
- Training queries: 241; development-validation queries: 120.
- 2016–2017 accessed: no.
- Kumo retrained: no.
- Reranker selection: frozen_selection.json written.
- Preprocessing: training-only scaling and explicit inference allowlists complete.
- Inner-fold experiments: 28 / 28 persisted.
- Selected checkpoint paths and hashes:
  - `ridge_forecast`: `artifacts/decision_aware_darr/phase2/experiments/ridge__forecast__market_context__a1/model.joblib` — SHA-256 `54062a44b3b698c41f1b0174158d80ba24553a3d7cc1ee920dd88d76c29dadb2`
  - `ridge_economic`: `artifacts/decision_aware_darr/phase2/experiments/ridge__economic__market_context__a10/model.joblib` — SHA-256 `548d504de135a6a0aee15a0c261e7b4a83369b4214e4e02f1b8820dbc8f1696a`
  - `lightgbm_forecast`: `artifacts/decision_aware_darr/phase2/experiments/lightgbm__forecast__combined__g0/model.joblib` — SHA-256 `8fbe9278fae20d3878bd0e316f3a69a87889144ce10f676c8b6155343daef136`
  - `lightgbm_economic`: `artifacts/decision_aware_darr/phase2/experiments/lightgbm__economic__combined__g1/model.joblib` — SHA-256 `cf7ff4b3cac17d00345f89af35de594876c1bd17e5b972bb88e142e8a3508844`
- Tests: 42 passed in 38.69s; includes 7 Phase 2 causality, preprocessing, determinism, checkpoint, alignment, and resume-status tests.

## Next command

```powershell
python run_decision_aware_darr_phase2.py status
```

Phase 1 artifacts are immutable inputs. Phase 2 stops after the 2015 development evaluation.
