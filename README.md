# Financial Regulatory Intelligence RAG

A production-oriented Retrieval-Augmented Generation system over public European financial-regulation
documents (AML/CFT, KYC, customer due diligence, PEPs, beneficial ownership, sanctions, banking risk).
It answers questions such as *"What enhanced due diligence measures apply to high-risk customers?"* with
a concise answer, supporting evidence, and **verifiable citations** (authority, document, section, page,
URL) — and says so explicitly when the corpus does not contain enough evidence.

The goal is not a big chatbot: it is a technically rigorous RAG system where **every architectural
choice is backed by an experiment** (see [Ablation study](#evaluation--ablation)).

> **Disclaimer.** This is a technical demonstration built on publicly available regulatory documents.
> It is **not legal advice**, not a compliance decision engine, not a replacement for compliance
> professionals, and must not be used to decide whether a real person or company is suspicious.
> Always verify answers against the original source.

---

## Architecture

```
                        ┌─────────────── ingestion (regrag ingest) ───────────────┐
 data/manifest.yaml ──► download ─► validate ─► parse (PyMuPDF) ─► clean ─► structure/sections
                                                                        │
                                   chunk (fixed | recursive | section | parent_child)
                                                                        │
                                  embed (OpenAI | BGE/E5 local) ─► Qdrant (versioned collection)
                                                                  └► chunks_<strategy>.jsonl (BM25, neighbors)

 query ─► normalize ─► [filter inference] ─► [LLM rewrite | multi-query]
       ─► BM25 ┐
       ─► dense┴► RRF fusion ─► dedup ─► [cross-encoder rerank] ─► ContextBuilder (budget, diversity,
                                                                    neighbors / parent) 
       ─► LLM (structured JSON, versioned prompt) ─► Pydantic parse ─► citation validation
       ─► answer + validated citations + usage/latency/cost trace
```

| Spec section | Where |
|---|---|
| 5–6 Manifest, reproducible ingestion | `data/manifest.yaml`, `src/regrag/ingestion/` |
| 7 PDF parsing (headings, sections, header/footer, hyphenation) | `ingestion/parsing.py`, `ingestion/cleaning.py` |
| 8 Data-quality validation + report | `ingestion/validation.py` → `data/processed/ingestion_report_*.json` |
| 9–10 Chunking strategies A–D + chunk metadata | `chunking/` |
| 11–12 Embedding abstraction + comparison | `embeddings/base.py`, `configs/experiments/local_*.yaml` |
| 13 Qdrant (Docker, payload filters, versioned collections) | `vectorstore.py`, `docker-compose.yml` |
| 14–17 BM25, dense, metadata filters, hybrid RRF | `retrieval/` |
| 18–19 Deduplication, reranking | `retrieval/fusion.py`, `retrieval/reranker.py` |
| 20–22 Query processing, rewriting, multi-query | `retrieval/query.py` |
| 23–24 Context builder, neighbor expansion | `generation/context.py` |
| 25–29 LLM abstraction, prompts, structured output, citations, validation | `generation/` |
| 30–31 Hallucination mitigation, answerability | see [below](#hallucination-mitigation) |
| 32–37 Eval dataset, retrieval/generation/E2E eval, ablation, tracking | `evaluation/`, `eval/questions.jsonl` |
| 38 Configuration | `configs/`, `config.py`, `.env.example` |
| 39–47 Pipeline, FastAPI, validation, errors, retry, caching, observability, logging, metrics | `pipeline.py`, `api/app.py`, `retry.py`, `observability.py`, `logging_utils.py` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local]"          # 'local' = sentence-transformers (BGE/E5, cross-encoder)
cp .env.example .env                    # add OPENAI_API_KEY
docker compose up -d qdrant             # local Qdrant, no managed account needed

regrag ingest                           # download → parse → chunk → embed → index
regrag ask "What guidance exists regarding politically exposed persons?"
regrag serve                            # http://localhost:8000/docs
```

No API key? Run the whole pipeline offline (plumbing check, not a quality baseline):

```bash
regrag --config configs/experiments/offline.yaml ingest
regrag --config configs/experiments/offline.yaml ask "What is enhanced due diligence?"
```

Full stack in Docker: `docker compose up -d --build` (then `docker compose exec api regrag ingest`).

### API

| Method | Path | Purpose |
|---|---|---|
| POST | `/query` | answer + validated citations + latency/usage (`debug: true` adds chunk text when `REGRAG_DEBUG=1`) |
| POST | `/retrieve` | retrieval only |
| POST | `/ingest` | rebuild index (disabled unless `REGRAG_ALLOW_INGEST=1`) |
| GET | `/documents`, `/health`, `/metrics` (`?format=prometheus`) | ops |

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query": "What does the EBA say about enhanced due diligence?", "filters": {"authority": ["EBA"]}}'
```

## Evaluation & ablation

The benchmark (`eval/questions.jsonl`) stores ground truth at **(document_id, page)** level, so one
benchmark scores every chunking strategy (chunk ids change between strategies, pages don't).

```bash
regrag eval --retrieval-only --name dense_baseline     # Recall@1/3/5/10, Precision@K, Hit, MRR, nDCG@10
regrag eval --name e2e                                 # + citation correctness/completeness, abstention
regrag eval --name e2e_judge --judge                   # + LLM-as-a-judge (labelled as such)
regrag ablation --ingest                               # baseline → +section → BM25 → hybrid → +rerank → +rewrite → +multi-query → parent/child
```

Every run writes `experiments/results/<name>-<config-hash>.json` with the full config, git SHA,
metrics (overall and per question type), latency p50/p95, tokens and cost. The ablation writes a
markdown table. **If a technique makes things worse (e.g. query rewriting), the table shows it and it is
disabled in `configs/default.yaml`** — negative results are kept.

Results table: *to be filled after the first real run* → [docs/decisions.md](docs/decisions.md).

## Hallucination mitigation

Hallucination cannot be "solved" by prompting. It is reduced — and made measurable — in layers:

1. **Retrieval** — hybrid BM25 + dense with RRF, reranking; measured with Recall/MRR/nDCG.
2. **Context** — deduplicated, budgeted, source-diverse evidence with authority/section/page headers.
3. **Prompt** — versioned system prompt: use only supplied evidence, preserve "should" vs "must",
   attribute statements to authorities, abstain when evidence is insufficient.
4. **Generation** — structured JSON; the model may only return `chunk_id`s, never free-text citations.
5. **Validation** — every cited id must exist and must have been in the context shown to the model;
   titles/pages/URLs are filled from our metadata. An answer with zero valid citations is withheld.
6. **Evaluation** — deterministic citation correctness, abstention accuracy on intentionally
   unanswerable questions, and (clearly labelled) LLM-judge faithfulness.

## Growing the corpus and the benchmark (next steps)

- [ ] Extend `data/manifest.yaml` from 9 to 30–100 documents (EBA, ECB, EC/EUR-Lex, FATF, AMLA, DNB).
      Prefer overlapping documents (EBA vs FATF on PEPs) so retrieval is non-trivial.
- [ ] Extend `eval/questions.jsonl` to 75–150 questions across all types; add `relevant_evidence`
      page numbers and set `validated: true` after manual checking.
- [ ] Run embedding comparison (OpenAI small vs BGE vs E5) and the ablation; record the numbers and the
      resulting decisions in `docs/decisions.md`.
- [ ] Optional: MLflow tracking (`pip install -e ".[tracking]"`).

## Security & privacy notes

API keys come only from environment variables; `.env` is git-ignored. Ingestion via API is off by
default. Error responses never expose stack traces. Traces store query text for evaluation — in a real
deployment queries may contain customer data, so they must be masked or dropped, access-controlled, and
subject to a retention policy. Input size and filters are validated with Pydantic.

## Development

```bash
make test     # fully offline: synthetic PDFs, hashing embeddings, in-memory Qdrant, fake LLM
make lint
```

## License

MIT — see [LICENSE](LICENSE). Regulatory documents remain the property of their publishers; they are
downloaded from the source URLs in the manifest and not redistributed in this repository.
