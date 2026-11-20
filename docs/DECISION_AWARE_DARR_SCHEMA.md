# Decision-aware DARR Phase 1 schema

The dataset is split so that information available at issuance cannot be confused with ex-post supervision.

| Table | Role | May be used as reranker input? |
|---|---|---|
| `query_features.parquet` | Query issuance/calendar/context descriptors and frozen direct Kumo forecast | Yes |
| `candidate_library.parquet` | Unique historical contexts, completed 72-step continuations, and historical dispatch schedules | Yes, after causal eligibility join |
| `candidate_features.parquet` | Query-candidate similarity, causal ranks, and pair descriptors | Yes |
| `query_labels.parquet` | Realized query trajectory, oracle/direct outcomes, and ex-post regime labels | No |
| `candidate_labels.parquet` | Continuation error, ranking/spread quality, realized battery value, throughput, and regret | No |

All feature names use `query__`, `candidate__`, or `pair__` prefixes. All ex-post columns use `label__`. Candidate continuations are historical observations that completed before the associated query issuance; query outcomes are never feature columns.

The custom adapter uses NVIDIA's pinned `model.embed`, L2 normalization, dot-product cosine similarity, top-k ordering, and temperature-softmax equations. It additionally exposes identities, ranks, and continuations that the official `knn_forecast` return value does not expose. Numerical equivalence of the weighted forecast is audited against NVIDIA's function.
