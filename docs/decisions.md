# Architecture decision log

Each decision must be backed by an experiment id from `experiments/results/`. Until the numbers exist,
the entry is marked **PENDING** and the current default is only a starting hypothesis.

| # | Decision | Options compared | Metric that decides | Result / experiment id | Status |
|---|---|---|---|---|---|
| 1 | Chunking strategy | fixed-256/512/1024, recursive, section, parent/child | Recall@5, nDCG@10, context tokens | — | PENDING (default: section-512/64) |
| 2 | Embedding model | text-embedding-3-small, bge-small-en-v1.5, e5-base-v2 | Recall@5, MRR, latency, $/1k queries | — | PENDING |
| 3 | Retriever | dense, BM25, hybrid (RRF k=60) | Recall@5 on exact-reference vs terminology questions | — | PENDING (hypothesis: BM25 wins on article numbers, hybrid best overall) |
| 4 | Reranker | off, ms-marco-MiniLM-L-6-v2 | nDCG@10, added p95 latency | — | PENDING |
| 5 | Query rewriting | off, LLM rewrite | Recall@5 on ambiguous/noisy questions, cost | — | PENDING (disable if negative) |
| 6 | Multi-query | off, 3 variants | Recall@10, latency, cost | — | PENDING |
| 7 | Filter inference | off, boost, restrict | Recall@5 on authority-specific questions | — | PENDING |
| 8 | Context budget | 3k, 6k, 10k tokens | citation correctness, cost | — | PENDING |

## Fixed decisions (non-experimental)

- **Qdrant, self-hosted in Docker** — spec requirement; no dependency on a managed account.
- **Citations as chunk ids only** — the model never produces citation strings; eliminates fabricated
  sources by construction (verified by `invalid_citation_rate`).
- **Page-level ground truth** — makes one benchmark valid across chunking strategies.
- **RRF over score blending** — BM25 and cosine scores are on incomparable scales; RRF uses ranks only.
