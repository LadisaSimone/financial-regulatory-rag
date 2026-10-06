"""Deterministic retrieval and generation metrics."""

from __future__ import annotations

import math

from regrag.evaluation.dataset import EvalItem
from regrag.schemas import Citation, RetrievedChunk


def evidence_units(item: EvalItem) -> list[tuple[str, int | None]]:
    """Ground-truth units: (doc, page) pairs, or (doc, None) if only document-level."""
    if item.relevant_evidence:
        return [(e.document_id, p) for e in item.relevant_evidence for p in e.pages]
    return [(d, None) for d in item.relevant_document_ids]


def is_relevant(meta: dict, chunk_id: str, item: EvalItem) -> set[tuple[str, int | None]]:
    """Return the ground-truth units this chunk covers (empty set = not relevant)."""
    if chunk_id in item.relevant_chunk_ids:
        return {("__chunk__", hash(chunk_id))}  # type: ignore[arg-type]
    covered = set()
    for doc, page in evidence_units(item):
        if meta["document_id"] != doc:
            continue
        if page is None or int(meta["page_start"]) <= page <= int(meta["page_end"]):
            covered.add((doc, page))
    return covered


def retrieval_metrics(retrieved: list[RetrievedChunk], item: EvalItem, ks=(1, 3, 5, 10)) -> dict[str, float]:
    units = set(evidence_units(item)) or {("__chunk__", hash(c)) for c in item.relevant_chunk_ids}
    rel = [is_relevant(c.metadata, c.chunk_id, item) for c in retrieved]
    out: dict[str, float] = {}
    for k in ks:
        top = rel[:k]
        covered = set().union(*top) if top else set()
        out[f"recall@{k}"] = len(covered & units) / len(units) if units else 0.0
        out[f"precision@{k}"] = sum(1 for r in top if r) / k
        out[f"hit@{k}"] = float(any(top))
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    # nDCG over ground-truth *units* (document pages), consistent with recall: a chunk earns gain only
    # if it covers at least one unit not covered by a higher-ranked chunk. (Fix: previously every
    # relevant chunk earned gain, so several chunks on the same page could push nDCG above 1.)
    k = 10
    seen: set = set()
    dcg = 0.0
    for i, r in enumerate(rel[:k]):
        if r - seen:
            dcg += 1.0 / math.log2(i + 2)
            seen |= r
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(units))))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    return out


def citation_metrics(citations: list[Citation], invalid: list[str], item: EvalItem) -> dict[str, float]:
    """citation_correctness: share of valid citations that point at ground-truth evidence.
    citation_completeness: share of ground-truth documents cited at least once.
    invalid_citation_rate: citations rejected by validation / all emitted citations."""
    n_emitted = len(citations) + len(invalid)
    if not item.answerable:
        return {"invalid_citation_rate": len(invalid) / n_emitted if n_emitted else 0.0}
    correct = sum(1 for c in citations if is_relevant(c.model_dump(), c.chunk_id, item))
    docs_cited = {c.document_id for c in citations}
    gt_docs = set(item.relevant_document_ids)
    return {
        "citation_correctness": correct / len(citations) if citations else 0.0,
        "citation_completeness": len(docs_cited & gt_docs) / len(gt_docs) if gt_docs else 0.0,
        "invalid_citation_rate": len(invalid) / n_emitted if n_emitted else 0.0,
    }


def answerability_metrics(pred_insufficient: bool, item: EvalItem) -> dict[str, float]:
    if item.answerable:
        return {"answerable_success": float(not pred_insufficient)}
    return {"abstention_accuracy": float(pred_insufficient)}


def mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return out
