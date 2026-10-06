"""RAGPipeline — orchestration layer. Ingestion, retrieval and generation stay separable:
this class only wires components chosen by configuration."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from regrag import DISCLAIMER
from regrag.chunk_store import ChunkStore, chunks_path
from regrag.config import Settings
from regrag.embeddings.base import get_embedding_provider
from regrag.generation.citations import INSUFFICIENT_MSG, parse_llm_answer, validate_citations
from regrag.generation.context import BuiltContext, ContextBuilder
from regrag.generation.llm import LLMProvider, get_llm
from regrag.generation.prompts import USER_TEMPLATE, get_system_prompt
from regrag.logging_utils import get_logger, log, new_request_id, request_id_var
from regrag.observability import Metrics, TraceWriter, estimate_cost, timer
from regrag.retrieval.base import Retriever
from regrag.retrieval.bm25 import BM25Retriever
from regrag.retrieval.dense import DenseRetriever
from regrag.retrieval.fusion import HybridRetriever, deduplicate, reciprocal_rank_fusion
from regrag.retrieval.query import ProcessedQuery, QueryProcessor, apply_boost
from regrag.retrieval.reranker import CrossEncoderReranker, Reranker
from regrag.schemas import LLMAnswer, MetadataFilter, RAGResponse, RetrievedChunk, Usage
from regrag.vectorstore import QdrantStore

logger = get_logger("pipeline")


class RAGPipeline:
    def __init__(self, settings: Settings, *, store: ChunkStore | None = None,
                 vector_store: QdrantStore | None = None, llm: LLMProvider | None = None,
                 reranker: Reranker | None = None, embedder=None, metrics: Metrics | None = None):
        self.s = settings
        cache = settings.path("cache_dir")
        self.store = store or ChunkStore.load(chunks_path(settings.path("processed_dir"), settings.chunking.strategy))
        self.embedder = embedder or get_embedding_provider(settings.embeddings, cache)
        self.vs = vector_store or QdrantStore(settings.vector_store.url, settings.collection_name())
        self.llm = llm or get_llm(settings.llm, cache, settings.cache.llm_responses)
        self.retriever = self._build_retriever()
        self.reranker = reranker
        if self.reranker is None and settings.reranker.enabled:
            self.reranker = CrossEncoderReranker(settings.reranker.model)
        self.query_processor = QueryProcessor(settings.query, settings.retrieval, self.llm)
        self.context_builder = ContextBuilder(
            max_tokens=settings.context.max_tokens,
            max_chunks_per_document=settings.context.max_chunks_per_document,
            neighbor_expansion=settings.context.neighbor_expansion,
            neighbor_window=settings.context.neighbor_window,
            store=self.store, tokenizer=settings.chunking.tokenizer,
        )
        self.metrics = metrics or Metrics()
        self.traces = TraceWriter(settings.path("traces_file"))

    def _build_retriever(self) -> Retriever:
        r = self.s.retrieval
        if r.type == "bm25":
            return BM25Retriever(self.store, contextual_header=self.s.chunking.contextual_header)
        dense = DenseRetriever(self.vs, self.embedder)
        if r.type == "dense":
            return dense
        return HybridRetriever(dense, BM25Retriever(self.store, contextual_header=self.s.chunking.contextual_header), r.dense_top_k, r.sparse_top_k,
                               r.rrf_k, r.dense_weight, r.sparse_weight)

    # ---------------- retrieval ----------------
    def retrieve(self, query: str | ProcessedQuery, flt: MetadataFilter | None = None,
                 top_k: int | None = None, latency: dict | None = None) -> tuple[list[RetrievedChunk], ProcessedQuery]:
        latency = latency if latency is not None else {}
        pq = query if isinstance(query, ProcessedQuery) else self.query_processor.process(query)
        r = self.s.retrieval
        final_k = top_k or r.final_top_k
        candidates_k = max(self.s.reranker.candidates if self.reranker else final_k, final_k)

        eff_filter, boost_filter = flt, None
        if pq.inferred_filter and (flt is None or flt.is_empty()):
            if r.filter_mode == "restrict":
                eff_filter = pq.inferred_filter
            else:
                boost_filter = pq.inferred_filter

        with timer(latency, "retrieval_ms"):
            lists = [self.retriever.retrieve(q, candidates_k, eff_filter) for q in pq.retrieval_queries]
            results = lists[0] if len(lists) == 1 else reciprocal_rank_fusion(lists, k=r.rrf_k)
            results = deduplicate(results, r.dedup_text_similarity)
        if self.reranker:
            with timer(latency, "reranking_ms"):
                results = self.reranker.rerank(pq.rewritten or pq.normalized, results, self.s.reranker.top_k)
        if boost_filter:  # applied last so a reranker cannot undo it
            results = apply_boost(results, boost_filter, r.boost_ranks)
        return results[:final_k], pq

    # ---------------- generation ----------------
    def generate(self, query: str, context: BuiltContext):
        if not context.chunks:
            return None, LLMAnswer(answer=INSUFFICIENT_MSG, citations=[], insufficient_evidence=True)
        system = get_system_prompt(self.s.llm.prompt_version)
        user = USER_TEMPLATE.format(context=context.text, question=query)
        res = self.llm.generate(system, user, json_mode=True)
        return res, parse_llm_answer(res.text)

    # ---------------- end-to-end ----------------
    def answer(self, query: str, flt: MetadataFilter | None = None, top_k: int | None = None,
               debug: bool = False) -> RAGResponse:
        rid = request_id_var.get() or new_request_id()
        latency: dict[str, float] = {}
        t0 = time.perf_counter()
        try:
            with timer(latency, "query_processing_ms"):
                pq = self.query_processor.process(query)
            chunks, pq = self.retrieve(pq, flt, top_k, latency)
            context = self.context_builder.build(chunks)
            with timer(latency, "generation_ms"):
                res, parsed = self.generate(query, context)
            valid, invalid = validate_citations(parsed, context.chunks, self.store)
            insufficient = parsed.insufficient_evidence or (not valid and not parsed.insufficient_evidence and bool(parsed.answer))
            answer_text = parsed.answer
            if not valid and not parsed.insufficient_evidence:
                # An answer with no verifiable evidence is not shown as a regulatory statement.
                answer_text = INSUFFICIENT_MSG + " (The model's draft answer had no valid citations and was withheld.)"
            latency["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            pt = res.prompt_tokens if res else 0
            ct = res.completion_tokens if res else 0
            cost = estimate_cost(self.s.pricing_usd_per_1m_tokens, self.s.llm.model, pt, ct)
            resp = RAGResponse(
                request_id=rid, query=query, rewritten_query=pq.rewritten,
                answer=answer_text, insufficient_evidence=insufficient,
                citations=valid, invalid_citations=invalid,
                retrieval={
                    "strategy": self.s.retrieval.type,
                    "reranker": bool(self.reranker),
                    "context_tokens": context.token_count,
                    "chunks": [
                        {"chunk_id": c.chunk_id, "score": round(c.score, 5), "retriever": c.retriever,
                         **({"text": c.text[:500]} if debug else {})}
                        for c in chunks
                    ],
                },
                latency_ms=latency,
                usage=Usage(prompt_tokens=pt, completion_tokens=ct, estimated_cost_usd=round(cost, 6)),
                disclaimer=DISCLAIMER,
            )
            self.metrics.record(latency, context.token_count, cost)
            self.traces.write({
                "request_id": rid, "timestamp": datetime.now(UTC).isoformat(), "query": query,
                "rewritten_query": pq.rewritten, "variants": pq.variants,
                "retrieval_strategy": self.s.retrieval.type, **latency,
                "retrieved_chunk_ids": [c.chunk_id for c in chunks],
                "retrieval_scores": [c.score for c in chunks],
                "context_chunk_ids": [c.chunk_id for c in context.chunks],
                "prompt_tokens": pt, "completion_tokens": ct, "estimated_cost": cost,
                "model": self.s.llm.model, "prompt_version": self.s.llm.prompt_version,
                "invalid_citations": invalid, "insufficient_evidence": insufficient, "errors": [],
            })
            log(logger, "query_answered", total_ms=latency["total_ms"], n_citations=len(valid),
                invalid_citations=len(invalid), insufficient=insufficient)
            return resp
        except Exception as e:
            self.metrics.record_error()
            log(logger, "query_failed", error=f"{type(e).__name__}: {e}")
            raise
