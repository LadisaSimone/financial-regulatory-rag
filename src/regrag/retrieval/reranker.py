"""Cross-encoder reranking (optional, config-driven). Broad candidate set in, top-k out."""

from __future__ import annotations

from abc import ABC, abstractmethod

from regrag.schemas import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError("pip install '.[local]' to enable the cross-encoder reranker") from e
        self.model = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self.model.predict([(query, c.text) for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
        return [
            c.model_copy(update={"score": float(s), "metadata": {**c.metadata, "pre_rerank_score": c.score}})
            for c, s in ranked[:top_k]
        ]


class LexicalOverlapReranker(Reranker):
    """Cheap deterministic stand-in used in tests / offline runs."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        from regrag.retrieval.bm25 import bm25_tokenize

        q = set(bm25_tokenize(query))
        scored = [(c, len(q & set(bm25_tokenize(c.text))) / (len(q) or 1)) for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c.model_copy(update={"score": s}) for c, s in scored[:top_k]]
