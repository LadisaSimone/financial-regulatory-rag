"""Structured-output parsing and post-generation citation validation.

The LLM may only reference chunk_ids. Titles, pages, sections and URLs are ALWAYS filled
from our own metadata — model-generated citation strings are never trusted.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from regrag.chunk_store import ChunkStore
from regrag.schemas import Citation, LLMAnswer, RetrievedChunk

INSUFFICIENT_MSG = "The provided documents do not contain sufficient information to answer this question reliably."


def parse_llm_answer(raw: str) -> LLMAnswer:
    """Parse JSON output; tolerate code fences; fall back to an abstention on garbage."""
    txt = raw.strip()
    txt = re.sub(r"^```(json)?|```$", "", txt, flags=re.M).strip()
    try:
        return LLMAnswer.model_validate(json.loads(txt))
    except (json.JSONDecodeError, ValidationError):
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return LLMAnswer.model_validate(json.loads(m.group(0)))
            except (json.JSONDecodeError, ValidationError):
                pass
    return LLMAnswer(answer=INSUFFICIENT_MSG, citations=[], insufficient_evidence=True)


def format_label(c: Citation) -> str:
    pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
    sec = f", §{c.section}" if c.section else ""
    return f"[{c.authority} — {c.title}{sec}, {pages}]"


def validate_citations(
    answer: LLMAnswer,
    context_chunks: list[RetrievedChunk],
    store: ChunkStore | None = None,
) -> tuple[list[Citation], list[str]]:
    """Checks: cited chunk exists, document exists, page metadata exists, and the chunk was
    part of the context actually shown to the model. Returns (valid, invalid_ids)."""
    in_context = {c.chunk_id: c for c in context_chunks}
    valid: list[Citation] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for rc in answer.citations:
        cid = rc.chunk_id.strip()
        if cid in seen:
            continue
        seen.add(cid)
        ctx = in_context.get(cid)
        exists = store is None or cid in store.by_id
        m = ctx.metadata if ctx else {}
        ok = (
            ctx is not None and exists
            and bool(m.get("document_id")) and m.get("page_start") is not None
        )
        if not ok:
            invalid.append(cid)
            continue
        cit = Citation(
            chunk_id=cid, document_id=m["document_id"], title=m["title"], authority=m["authority"],
            page_start=int(m["page_start"]), page_end=int(m["page_end"]), section=m.get("section"),
            source_url=m.get("source_url"),
        )
        cit.label = format_label(cit)
        valid.append(cit)
    return valid, invalid


def citation_coverage(answer_text: str, n_valid_citations: int) -> float:
    """Rough deterministic proxy: share of answer sentences carrying a [n] marker, or 1.0 if
    the answer has a single sentence and at least one valid citation."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer_text.strip()) if len(s) > 20]
    if not sents:
        return 0.0
    if n_valid_citations and len(sents) == 1:
        return 1.0
    marked = sum(1 for s in sents if re.search(r"\[\d+\]", s))
    return marked / len(sents)
