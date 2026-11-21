# Decision-aware DARR Phase 2 progress

## Current state

- Status: IN PROGRESS.
- Branch: `feature/decision-aware-darr-reranker`.
- Parent commit: `7a3eb78`.
- Last verified stage: Stage 7 selection frozen before development validation.
- Phase 1 manifest SHA-256: `3b4973f6da581aea0c9d6a609efc1db69cfc73a631e71b6d736b87fe6b68c03d`.
- Phase 2 configuration SHA-256: `737ac5eeb0c427294badfc51279481da42c4e8c019b16c4c5e7b589cc8579d1c`.
- Phase 1 inputs: 18 / 18 manifest outputs checksum-verified.
- Training queries: 241; development-validation queries: 120.
- 2016–2017 accessed: no.
- Kumo retrained: no.
- Reranker selection: frozen_selection.json written.
- Tests: pending.

## Next command

```powershell
python run_decision_aware_darr_phase2.py evaluate --resume
```

Phase 1 artifacts are immutable inputs. Phase 2 stops after the 2015 development evaluation.
