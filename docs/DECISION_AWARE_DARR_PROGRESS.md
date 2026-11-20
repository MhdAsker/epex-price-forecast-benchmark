# Decision-aware DARR Prompt 1 progress

## Current state

- Latest completed stage: Stage 3 embedding generation.
- Current stage: Stage 3 embedding generation.
- Status: INCOMPLETE.
- Branch: `feature/decision-aware-darr-dataset`.
- Parent commit: `7f12d2384afb74bc2d24125be70fd9aaa8cb6616`.
- Configuration SHA-256: `57205c8cbb295f940985f6633779944fda52ffcef993e40dd67e5cadfc38f987`.
- Source SHA-256: `1cae127d6647c0e8fa37c3b9726b1ecb8bf3676b1707aec62a2c707639bac7fb`.

## Verified batch progress

- Embedding batches: 1 / 23 complete.
- Candidate batches: 0 / 13 complete.
- Last successful embedding batch: embedding_000.
- Last successful candidate batch: none.

## Completed final files

- None yet.

## Tests

- 34 passed in 26.73s; includes 7 decision-aware DARR causality/resume tests

## Blockers and errors

- None.

## Pending work

- Resume the earliest checksum-invalid or missing batch, then audit and analyze.

## Exact resume commands

```powershell
python run_decision_aware_darr.py status
python run_decision_aware_darr.py build --resume
python run_decision_aware_darr.py audit
python run_decision_aware_darr.py analyze
```

Reruns skip only batches whose manifest, configuration hash, source hash, byte count, and output checksum all verify. Conflicting configurations fail explicitly.
