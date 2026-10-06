"""Rank fusion, hybrid retrieval and deduplication."""

from __future__ import annotations

import re

from regrag.retrieval.base import Retriever
from regrag.schemas import MetadataFilter, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """score(d) = sum_i w_i / (k + rank_i(d)). Scale-free: BM25 and cosine scores are never
    compared directly, only ranks are."""
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    sources: dict[str, list[str]] = {}
    for w, lst in zip(weights, ranked_lists, strict=True):
        for rank, item in enumerate(lst, start=1):
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + w / (k + rank)
            first_seen.setdefault(item.chunk_id, item)
            sources.setdefault(item.chunk_id, []).append(item.retriever)
    out = []
    for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        base = first_seen[cid]
        out.append(base.model_copy(update={"score": score, "retriever": "+".join(sorted(set(sources[cid])))}))
    return out


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, dense: Retriever, sparse: Retriever, dense_top_k: int = 20, sparse_top_k: int = 20,
                 rrf_k: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0):
        self.dense, self.sparse = dense, sparse
        self.dense_top_k, self.sparse_top_k = dense_top_k, sparse_top_k
        self.rrf_k, self.weights = rrf_k, [dense_weight, sparse_weight]

    def retrieve(self, query: str, top_k: int, flt: MetadataFilter | None = None) -> list[RetrievedChunk]:
        d = self.dense.retrieve(query, self.dense_top_k, flt)
        s = self.sparse.retrieve(query, self.sparse_top_k, flt)
        return reciprocal_rank_fusion([d, s], k=self.rrf_k, weights=self.weights)[:top_k]


def interleave(primary: list[RetrievedChunk], secondary: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    """Alternate 1:1 between two ranked lists (primary first), skipping chunk ids already taken.
    Parameter-free: no scores are compared, so lists with different score scales can be merged."""
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for i in range(max(len(primary), len(secondary))):
        for lst in (primary, secondary):
            if i < len(lst) and lst[i].chunk_id not in seen:
                out.append(lst[i])
                seen.add(lst[i].chunk_id)
                if len(out) == k:
                    return out
    return out


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i : i + n]) for i in range(max(1, len(toks) - n + 1))}


def jaccard(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate(chunks: list[RetrievedChunk], text_similarity: float = 0.9) -> list[RetrievedChunk]:
    """Remove (1) repeated chunk ids, (2) chunks fully contained in the page range of a
    higher-ranked chunk from the same document with heavy text overlap (overlap windows),
    (3) near-identical text across documents (e.g. the same paragraph re-published)."""
    kept: list[RetrievedChunk] = []
    seen: set[str] = set()
    for c in chunks:
        if c.chunk_id in seen:
            continue
        dup = False
        for k in kept:
            same_doc_overlap = (
                k.document_id == c.document_id
                and c.page_start >= k.page_start and c.page_end <= k.page_end
                and jaccard(k.text, c.text) >= 0.5
            )
            if same_doc_overlap or (text_similarity < 1.0 and jaccard(k.text, c.text) >= text_similarity):
                dup = True
                break
        if not dup:
            kept.append(c)
            seen.add(c.chunk_id)
    return kept
