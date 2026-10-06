"""Sparse retrieval (BM25). Regulatory text is full of exact identifiers — "Article 18a",
"2015/849", "PEP", "Recommendation 10" — so the tokenizer keeps those intact."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from regrag.chunk_store import ChunkStore
from regrag.retrieval.base import Retriever
from regrag.schemas import MetadataFilter, RetrievedChunk

_TOKEN = re.compile(r"\d+(?:[./]\d+)*(?:\(\d+\))*[a-z]?|[a-z]+(?:[-'][a-z]+)*", re.IGNORECASE)
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "which", "with", "what", "does", "do", "should", "shall", "may", "can", "how", "when", "where", "who", "whom", "whose", "why"]
)


def bm25_tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text) if t.lower() not in STOPWORDS]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, store: ChunkStore, k1: float = 1.5, b: float = 0.75, contextual_header: bool = False):
        self.store = store
        self.chunks = store.chunks
        docs = [c.index_text(True) if contextual_header else c.text + " " + (c.section or "") for c in self.chunks]
        self.index = BM25Okapi([bm25_tokenize(d) for d in docs], k1=k1, b=b)

    def retrieve(self, query: str, top_k: int, flt: MetadataFilter | None = None) -> list[RetrievedChunk]:
        scores = self.index.get_scores(bm25_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            meta = c.payload()
            if flt and not flt.matches(meta):
                continue
            out.append(RetrievedChunk(chunk_id=c.chunk_id, text=c.text, score=float(scores[i]),
                                      metadata=meta, retriever=self.name))
            if len(out) >= top_k:
                break
        return out
