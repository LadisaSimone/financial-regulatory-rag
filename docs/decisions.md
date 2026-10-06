# Architecture decision log

Each decision must be backed by an experiment id from `experiments/results/`. Until the numbers exist,
the entry is marked **PENDING** and the current default is only a starting hypothesis.

| # | Decision | Options compared | Metric that decides | Result / experiment id | Status |
|---|---|---|---|---|---|
| 1 | Chunking strategy | fixed-256/512/1024, recursive, section, parent/child | Recall@5, nDCG@10, context tokens | — | PENDING (default: section-512/64) |
| 2 | Embedding model | text-embedding-3-small, bge-small-en-v1.5, e5-base-v2 | Recall@5, MRR, latency, $/1k queries | — | PENDING |
| 3 | Retriever | dense, BM25, hybrid (RRF k=60, equal weights) | Recall@5, MRR overall and per question type | Dense best: Recall@5 0.70 vs hybrid 0.67 vs BM25 0.61; MRR 0.65 / 0.63 / 0.55 (`dense_bge-5b0076e4`, `hybrid_bge-2fb3ba09`, `bm25-6b6efba9`) | **PROVISIONAL: dense.** Hypothesis "BM25 wins on article numbers" refuted. Hybrid kept as candidate until reranker + RRF weights are tested |
| 4 | Reranker | off, `cross-encoder/ms-marco-MiniLM-L-6-v2` (30 candidates → top 10) on dense and hybrid | Recall@10, Hit@10, nDCG@10, p95 latency | Hybrid+rerank: Recall@10 0.82, Hit@10 0.92, nDCG 0.66 vs dense 0.77 / 0.86 / 0.62; p50 latency 6 → 214 ms (`hybrid_rerank-11bcee69`, `dense_rerank-4592e9ee`) | **PROVISIONAL: hybrid + reranker.** Gains are in the tail (top 10), not at rank 1–5; hurts exact references. +210 ms is small vs ~4 s LLM time |
| 5 | Query rewriting | off, LLM rewrite | Recall@5 on ambiguous/noisy questions, cost | — | PENDING (disable if negative) |
| 6 | Multi-query | off, 3 variants | Recall@10, latency, cost | — | PENDING |
| 7 | Filter inference | off, boost, restrict | Recall@5 on authority-specific questions | — | PENDING |
| 8 | Context budget | 3k, 6k, 10k tokens | citation correctness, cost | — | PENDING |

## Decision 3 — Retriever (2026-10-06)

Setup: section chunking (512/64), `BAAI/bge-small-en-v1.5`, top-k 10, 100 validated questions with
page-level ground truth, no reranker, no query rewriting. Retrieval only (no LLM).

| Retriever | Recall@1 | Recall@5 | Recall@10 | Hit@10 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|---|---|
| BM25 | 0.30 | 0.61 | 0.73 | 0.82 | 0.55 | 0.58 | 5.6 |
| Dense (BGE-small) | **0.39** | **0.70** | **0.77** | **0.86** | **0.65** | 0.62 | 5.8 |
| Hybrid (RRF, 1:1) | 0.38 | 0.67 | **0.77** | **0.86** | 0.63 | **0.63** | 7.1 |

Recall@5 / MRR by question type:

| Type (n) | BM25 | Dense | Hybrid |
|---|---|---|---|
| factual (60) | 0.66 / 0.60 | **0.74 / 0.70** | 0.72 / 0.68 |
| exact_reference (17) | 0.56 / 0.41 | **0.71 / 0.61** | 0.65 / 0.48 |
| terminology (11) | 0.64 / 0.59 | **0.77** / 0.69 | 0.73 / **0.71** |
| cross_document (8) | 0.61 / 0.72 | 0.56 / 0.53 | **0.63 / 0.73** |

Findings:
- Dense beats BM25 on every type except cross-document, including exact references ("Article 18",
  "Recommendation 10"). The initial hypothesis that BM25 would win on identifiers is **not supported**:
  BM25 matches the identifier but not the document ("Directive 2015/849" vs EBA texts that cite it).
- Equal-weight RRF pulls BM25 noise into the top ranks: hybrid loses 3 points of Recall@5 vs dense,
  gaining only on cross-document questions, where BM25's exact term overlap helps.
- 9 questions are missed in the top 10 by all three retrievers (q007, q031, q035, q039, q073, q081, q096,
  q098, q100) → candidates for reranking, metadata filtering or chunking fixes.

Follow-up: weighting dense 2:1 inside RRF (`hybrid_w2`) did not help (Recall@5 0.68, MRR 0.63, nDCG 0.62),
so the weights stay 1:1. Note: `hybrid_w2` shares the config hash of `hybrid_bge` because RRF weights were not
yet recorded in the experiment config (fixed in `runner.py` after this run).

## Decision 4 — Reranker (2026-10-06)

Same setup as Decision 3. Reranker re-scores the top 30 candidates and keeps 10.

| Configuration | Recall@1 | Recall@5 | Recall@10 | Hit@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| Dense | 0.39 | **0.70** | 0.77 | 0.86 | 0.65 | 0.62 | 6 | 7 |
| Dense + rerank | **0.41** | **0.71** | 0.81 | 0.90 | **0.65** | 0.65 | 212 | 226 |
| Hybrid | 0.38 | 0.67 | 0.77 | 0.86 | 0.63 | 0.63 | 7 | 9 |
| **Hybrid + rerank** | **0.41** | 0.70 | **0.82** | **0.92** | **0.66** | **0.66** | 214 | 232 |
| Hybrid 2:1 (no rerank) | 0.38 | 0.68 | 0.78 | 0.86 | 0.63 | 0.62 | 7 | 9 |

Recall@5 / MRR by type: reranking helps factual (0.74 → 0.76) and noisy/adversarial queries
(0.07 → 0.64 on 2 questions — too few to generalise), but **hurts exact references** (dense 0.71 → 0.65,
hybrid+rerank 0.59): the general-purpose MS MARCO model does not value identifiers like "Article 18".

Findings:
- The reranker's main effect is recovering evidence into the top 10 (Hit@10 0.86 → 0.92; questions
  missed by every retriever drop from 9 to 6–8). Part of this comes from the larger candidate pool (30 vs 10).
- Rank-1 quality barely moves (MRR 0.65 → 0.66), so the gain matters only because the LLM receives 6–8 chunks.
- Cost: ~210 ms per query on a MacBook CPU, negligible against ~4–5 s of LLM generation.
- Still missed by the best configuration: q028, q031, q034, q035, q044, q073, q096, q098.

Next: inspect the 8 remaining misses (chunking vs ground truth vs retrieval); try a domain-stronger
reranker (e.g. `BAAI/bge-reranker-base`); then run end-to-end generation metrics on the chosen configuration.

## Error analysis — questions missed in the top 10 by hybrid + rerank (2026-10-06)

8 of 100 answerable questions had no ground-truth page in the top 10 (`hybrid_rerank-11bcee69`).
Each was inspected by comparing the retrieved chunks with the ground-truth chunks.

| Cause | Questions | What happened | Fix |
|---|---|---|---|
| **Document confusion** (retrieval) | q028, q031, q034, q035 | The question names *Directive 2015/849*; the top results are EBA guideline paragraphs that **cite** the Directive ("To comply with Article 19 of Directive (EU) 2015/849…"), or AMLR articles on the same topic. The Directive's own Articles 3, 14, 18, 19 are not in the top 10. | Document-aware retrieval: infer an explicit document reference from the query (boost/restrict) and a contextual header (authority — title — section) in the indexed text |
| **Authority ignored** (retrieval) | q073 | "What … does **the EBA** expect for PEPs?" → AMLR Art. 42, AMLD Art. 20, FATF R.12 ranked above the EBA PEP section (pp. 36–37). Filter inference was disabled. | Same as above (authority boost) |
| **Ground truth too narrow** (dataset) | q044 | Recommendation 19's own text (p. 19–20) was retrieved at rank 2, but the ground truth only listed the Interpretive Note (pp. 92–93). The R.19 text sits in a chunk labelled "18. Internal controls…" (heading not detected). | Added pp. 19–20 to the ground truth; parser note below |
| **Ground truth too narrow** (dataset) | q096 | "What does the EBA say about EDD?" — retrieved the EBA's sectoral EDD sections (pp. 68–69, 119–120…), which are valid answers; the ground truth only listed Title I (pp. 35–36). | Ground truth extended to all 8 EBA "Enhanced customer due diligence" sections |
| **Ambiguous question** (dataset) | q098 | "What does it say about risky customers?" — retrieved EBA customer-risk-factor sections; the ground truth only had EDD. Both readings are legitimate. | Ground truth extended to Customer risk factors (pp. 13–17) + EDD (pp. 35–36) |

Summary: **5 of 8 misses are retrieval failures with a single root cause (the system ignores which
document or authority the user names); 3 of 8 were evaluation-set errors.** The changed ground truth
(q044, q096, q098) is marked `validated: false` until reviewed; all earlier runs must be re-run on the
corrected set before comparing with new experiments.

Caveat: ~60% of questions name a document or authority explicitly because they were written against
known sources. Gains from document-aware retrieval should therefore also be reported on the questions
that do **not** name one, to check they are not hurt.

Parser follow-up: FATF Recommendation headings that follow another Recommendation on the same page
(e.g. "19. Higher-risk countries" inside the R.18 chunk) are not detected as headings, so the chunk keeps
the previous section label.

## Fixed decisions (non-experimental)

- **Qdrant, self-hosted in Docker** — spec requirement; no dependency on a managed account.
- **Citations as chunk ids only** — the model never produces citation strings; eliminates fabricated
  sources by construction (verified by `invalid_citation_rate`).
- **Page-level ground truth** — makes one benchmark valid across chunking strategies.
- **RRF over score blending** — BM25 and cosine scores are on incomparable scales; RRF uses ranks only.
