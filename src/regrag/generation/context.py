"""ContextBuilder: turns ranked chunks into a bounded, ordered, attributable evidence block.

Responsibilities: token budget, ordering, dedup, source diversity, metadata preservation,
page references, optional neighbor expansion / parent substitution. Never blind concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from regrag.chunk_store import ChunkStore
from regrag.retrieval.fusion import deduplicate
from regrag.schemas import RetrievedChunk
from regrag.tokenization import get_tokenizer


@dataclass
class BuiltContext:
    text: str
    chunks: list[RetrievedChunk]  # exactly the chunks the LLM saw (citation whitelist)
    token_count: int
    dropped_for_budget: int = 0
    expanded_from: dict[str, list[str]] = field(default_factory=dict)

    @property
    def allowed_ids(self) -> set[str]:
        return {c.chunk_id for c in self.chunks}


def format_header(c: RetrievedChunk) -> str:
    m = c.metadata
    pages = f"p.{m['page_start']}" if m["page_start"] == m["page_end"] else f"pp.{m['page_start']}-{m['page_end']}"
    sec = f" | §{m['section']}" if m.get("section") else ""
    return f"[chunk_id: {c.chunk_id}] {m['authority']} — {m['title']}{sec} | {pages} | {m.get('publication_date', '')}"


class ContextBuilder:
    def __init__(self, max_tokens: int = 6000, max_chunks_per_document: int = 4,
                 neighbor_expansion: bool = False, neighbor_window: int = 1,
                 store: ChunkStore | None = None, tokenizer: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.max_per_doc = max_chunks_per_document
        self.neighbor_expansion, self.window, self.store = neighbor_expansion, neighbor_window, store
        self.tok = get_tokenizer(tokenizer)

    def _expand(self, c: RetrievedChunk) -> RetrievedChunk:
        # Parent/child: hand the parent section to the LLM, keep the child's id for citation.
        if c.metadata.get("parent_text"):
            return c.model_copy(update={"text": c.metadata["parent_text"]})
        if not (self.neighbor_expansion and self.store):
            return c
        neigh = self.store.neighbors(c.chunk_id, self.window)
        if len(neigh) <= 1:
            return c
        text = "\n".join(n.text for n in neigh)
        meta = {**c.metadata, "page_start": min(n.page_start for n in neigh),
                "page_end": max(n.page_end for n in neigh), "expanded_with": [n.chunk_id for n in neigh]}
        return c.model_copy(update={"text": text, "metadata": meta})

    def build(self, retrieved: list[RetrievedChunk], max_tokens: int | None = None) -> BuiltContext:
        budget = max_tokens or self.max_tokens
        chunks = deduplicate(retrieved)
        per_doc: dict[str, int] = {}
        selected: list[RetrievedChunk] = []
        blocks: list[str] = []
        used, dropped = 0, 0
        seen_parents: set[str] = set()
        for c in chunks:  # already in relevance order
            if per_doc.get(c.document_id, 0) >= self.max_per_doc:
                continue
            pid = c.metadata.get("parent_id")
            if pid and pid in seen_parents:
                continue  # sibling child of an already-included parent
            c2 = self._expand(c)
            block = f"{format_header(c2)}\n{c2.text.strip()}"
            n = self.tok.count(block)
            if used + n > budget:
                dropped += 1
                continue
            if pid:
                seen_parents.add(pid)
            per_doc[c.document_id] = per_doc.get(c.document_id, 0) + 1
            selected.append(c2)
            blocks.append(block)
            used += n
        return BuiltContext(text="\n\n".join(blocks), chunks=selected, token_count=used, dropped_for_budget=dropped)
