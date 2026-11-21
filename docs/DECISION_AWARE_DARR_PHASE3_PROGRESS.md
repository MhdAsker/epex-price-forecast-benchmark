# Decision-aware DARR Phase 3 progress

## Current state

- Status: COMPLETE.
- Branch: `feature/decision-aware-darr-phase3`.
- Parent commit: `93ca62c`.
- Last verified stage: Stage 10 final audit and report complete.
- Phase 2 manifest SHA-256: `19d7cba0471f04146596afe461ebd70c63029bac51ba7cd3fd8273065866975d`.
- Frozen checkpoint hashes:
  - `ridge_forecast`: `54062a44b3b698c41f1b0174158d80ba24553a3d7cc1ee920dd88d76c29dadb2`
  - `ridge_economic`: `548d504de135a6a0aee15a0c261e7b4a83369b4214e4e02f1b8820dbc8f1696a`
  - `lightgbm_forecast`: `8fbe9278fae20d3878bd0e316f3a69a87889144ce10f676c8b6155343daef136`
  - `lightgbm_economic`: `cf7ff4b3cac17d00345f89af35de594876c1bd17e5b972bb88e142e8a3508844`
- Development reproduction: passed.
- Embedding batches complete: 12 / 12.
- Query batches complete: 25 / 25.
- 2016–2017 used for fitting/tuning: no.
- Kumo/rerankers retrained: no.
- Tests: 48 passed in 43.07s; includes 6 Phase 3 checkpoint, reproduction, causality, alignment, and protocol tests.

## Next command
```powershell
python run_decision_aware_darr_phase3.py status
```
