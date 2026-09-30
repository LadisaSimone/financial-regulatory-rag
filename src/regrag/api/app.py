"""FastAPI service.

POST /query     full RAG answer with validated citations
POST /retrieve  retrieval only (debugging / evaluation)
POST /ingest    (re)build the index from the manifest — disabled unless REGRAG_ALLOW_INGEST=1
GET  /documents corpus listing
GET  /health    liveness + dependency checks
GET  /metrics   JSON, or Prometheus text with ?format=prometheus
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from regrag import DISCLAIMER, __version__
from regrag.config import Settings, load_settings
from regrag.embeddings.base import EmbeddingError
from regrag.generation.llm import LLMError
from regrag.logging_utils import get_logger, log, new_request_id
from regrag.pipeline import RAGPipeline
from regrag.schemas import MetadataFilter, RAGResponse
from regrag.vectorstore import VectorStoreUnavailable

logger = get_logger("api")

ALLOWED_AUTHORITIES = {"EBA", "ECB", "EC", "FATF", "ESMA", "EIOPA", "AMLA", "DNB", "AFM", "BCBS", "OTHER"}


class FilterIn(BaseModel):
    model_config = {"extra": "forbid"}
    authority: list[str] | None = None
    document_type: list[str] | None = None
    publication_date_gte: date | None = None
    publication_date_lte: date | None = None

    @field_validator("authority")
    @classmethod
    def _auth(cls, v):
        if v:
            bad = set(v) - ALLOWED_AUTHORITIES
            if bad:
                raise ValueError(f"unsupported authority: {sorted(bad)}")
        return v


class QueryIn(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = Field(min_length=3, max_length=1000)
    filters: FilterIn | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    debug: bool = False

    @field_validator("query")
    @classmethod
    def _q(cls, v: str):
        if not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()


class IngestIn(BaseModel):
    model_config = {"extra": "forbid"}
    download: bool = False
    chunking_strategy: Literal["fixed", "recursive", "section", "parent_child"] | None = None


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    state: dict = {"settings": settings, "pipeline": pipeline, "lock": threading.Lock()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["settings"] = state["settings"] or load_settings()
        if state["pipeline"] is None:
            try:
                state["pipeline"] = RAGPipeline(state["settings"])
            except Exception as e:  # start anyway; /health reports the problem
                log(logger, "pipeline_init_failed", error=str(e))
        yield

    app = FastAPI(title="Financial Regulatory Intelligence RAG", version=__version__,
                  description=DISCLAIMER, lifespan=lifespan)

    def pipe() -> RAGPipeline:
        p = state["pipeline"]
        if p is None:
            raise HTTPException(503, "Index not available. Run ingestion first (see README).")
        return p

    @app.middleware("http")
    async def request_id_mw(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_request_id()
        from regrag.logging_utils import request_id_var
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    # ---- error mapping: no stack traces to API users ----
    @app.exception_handler(VectorStoreUnavailable)
    async def _vs(_, exc):
        return JSONResponse(status_code=503, content={"error": "vector_store_unavailable", "detail": "Vector database unavailable."})

    @app.exception_handler(LLMError)
    async def _llm(_, exc):
        return JSONResponse(status_code=502, content={"error": "llm_unavailable", "detail": "Language model call failed. Try again later."})

    @app.exception_handler(EmbeddingError)
    async def _emb(_, exc):
        return JSONResponse(status_code=502, content={"error": "embedding_failed", "detail": "Embedding service failed."})

    @app.exception_handler(Exception)
    async def _any(_, exc):
        log(logger, "unhandled_error", error=f"{type(exc).__name__}: {exc}")
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "Unexpected error."})

    def _filter(f: FilterIn | None) -> MetadataFilter | None:
        return MetadataFilter(**f.model_dump()) if f else None

    @app.post("/query", response_model=RAGResponse, response_model_exclude_none=True)
    def query(body: QueryIn):
        p = pipe()
        debug = body.debug and (state["settings"].api.debug or os.environ.get("REGRAG_DEBUG") == "1")
        resp = p.answer(body.query, _filter(body.filters), body.top_k, debug=debug)
        if not debug:
            resp.retrieval = {k: v for k, v in resp.retrieval.items() if k != "chunks"} | {
                "chunk_ids": [c["chunk_id"] for c in resp.retrieval.get("chunks", [])]}
        return resp

    @app.post("/retrieve")
    def retrieve(body: QueryIn):
        p = pipe()
        latency: dict = {}
        chunks, pq = p.retrieve(body.query, _filter(body.filters), body.top_k, latency)
        return {"query": body.query, "processed_query": pq.rewritten or pq.normalized,
                "latency_ms": latency,
                "chunks": [{"chunk_id": c.chunk_id, "score": c.score, "retriever": c.retriever,
                            "document_id": c.document_id, "title": c.metadata["title"],
                            "authority": c.metadata["authority"], "section": c.metadata.get("section"),
                            "page_start": c.page_start, "page_end": c.page_end, "text": c.text}
                           for c in chunks]}

    @app.post("/ingest", status_code=202)
    def ingest(body: IngestIn):
        if os.environ.get("REGRAG_ALLOW_INGEST") != "1":
            raise HTTPException(403, "Ingestion via API is disabled. Set REGRAG_ALLOW_INGEST=1 or use the CLI.")
        if not state["lock"].acquire(blocking=False):
            raise HTTPException(409, "Ingestion already running.")
        try:
            from regrag.ingestion.pipeline import run_ingestion

            s = state["settings"]
            if body.chunking_strategy:
                s = s.model_copy(update={"chunking": s.chunking.model_copy(update={"strategy": body.chunking_strategy})})
            report = run_ingestion(s, download=body.download)
            state["pipeline"] = RAGPipeline(s)
            return report.as_dict()
        finally:
            state["lock"].release()

    @app.get("/documents")
    def documents():
        return {"documents": list(pipe().store.documents().values())}

    @app.get("/health")
    def health():
        p = state["pipeline"]
        vs_ok = bool(p and p.vs.health())
        status = "ok" if p and vs_ok else "degraded"
        return JSONResponse(status_code=200 if status == "ok" else 503, content={
            "status": status, "version": __version__, "index_loaded": p is not None,
            "vector_store": vs_ok, "chunks": len(p.store.chunks) if p else 0,
            "collection": state["settings"].collection_name() if state["settings"] else None,
        })

    @app.get("/metrics")
    def metrics(format: Literal["json", "prometheus"] = Query("json")):
        m = state["pipeline"].metrics if state["pipeline"] else None
        if m is None:
            return {}
        return PlainTextResponse(m.prometheus()) if format == "prometheus" else m.snapshot()

    return app


app = create_app()
