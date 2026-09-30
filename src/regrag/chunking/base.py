"""Chunking strategies — treated as an experimental variable.

A: fixed        fixed token windows with overlap (ignores structure)
B: recursive    paragraph -> sentence -> token boundaries, greedy packing
C: section      like B, but never crosses a section/heading boundary
D: parent_child small child chunks for retrieval, parent section text passed to the LLM

All strategies keep page traceability (page_start / page_end) and produce deterministic
chunk ids: "<document_id>:<strategy>:<position>".
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from regrag.config import Chunking as ChunkingConfig
from regrag.schemas import Chunk, DocumentMeta, DocumentPage
from regrag.tokenization import Tokenizer, get_tokenizer

_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"'])")


@dataclass
class Unit:
    """A paragraph-level piece of text with its provenance."""

    text: str
    page: int
    section: str | None
    tokens: int


def iter_units(pages: list[DocumentPage], tok: Tokenizer) -> list[Unit]:
    units: list[Unit] = []
    current = None
    for p in pages:
        heads = [h.strip() for h in p.headings]
        if current is None:
            current = p.section
        for para in p.text.split("\n"):
            para = para.strip()
            if not para:
                continue
            for h in heads:
                if para.startswith(h[:60]):
                    current = h
                    break
            units.append(Unit(para, p.page_number, current, tok.count(para)))
    return units


def split_oversized(unit: Unit, max_tokens: int, tok: Tokenizer) -> list[Unit]:
    """Sentence split, then hard token split as last resort."""
    if unit.tokens <= max_tokens:
        return [unit]
    out: list[Unit] = []
    for sent in _SENTENCE.split(unit.text):
        n = tok.count(sent)
        if n <= max_tokens:
            out.append(Unit(sent, unit.page, unit.section, n))
            continue
        ids = tok.encode(sent)
        for i in range(0, len(ids), max_tokens):
            piece = tok.decode(ids[i : i + max_tokens])
            out.append(Unit(piece, unit.page, unit.section, tok.count(piece)))
    return out


def pack_units(units: list[Unit], max_tokens: int, overlap: int) -> list[list[Unit]]:
    """Greedy packing of units into groups <= max_tokens, carrying ~`overlap` tokens of
    trailing units into the next group."""
    groups: list[list[Unit]] = []
    cur: list[Unit] = []
    size = 0
    for u in units:
        if cur and size + u.tokens > max_tokens:
            groups.append(cur)
            carry, carried = [], 0
            for prev in reversed(cur):
                if carried + prev.tokens > overlap:
                    break
                carry.insert(0, prev)
                carried += prev.tokens
            cur, size = list(carry), carried
        cur.append(u)
        size += u.tokens
    if cur:
        groups.append(cur)
    return groups


class Chunker(ABC):
    name: str

    def __init__(self, cfg: ChunkingConfig):
        self.cfg = cfg
        self.tok = get_tokenizer(cfg.tokenizer)

    @abstractmethod
    def chunk(self, pages: list[DocumentPage], meta: DocumentMeta) -> list[Chunk]: ...

    def _make(self, meta: DocumentMeta, units: list[Unit], position: int, **extra) -> Chunk:
        text = "\n".join(u.text for u in units)
        sections = [u.section for u in units if u.section]
        return Chunk(
            chunk_id=f"{meta.document_id}:{self.name}:{position:05d}",
            document_id=meta.document_id,
            text=text,
            title=meta.title,
            authority=meta.authority,
            document_type=meta.document_type,
            section=sections[0] if sections else None,
            page_start=min(u.page for u in units),
            page_end=max(u.page for u in units),
            publication_date=meta.publication_date.isoformat(),
            source_url=str(meta.source_url),
            chunking_strategy=self.name,
            token_count=self.tok.count(text),
            position=position,
            **extra,
        )


def get_chunker(cfg: ChunkingConfig) -> Chunker:
    from regrag.chunking.strategies import (
        FixedTokenChunker,
        ParentChildChunker,
        RecursiveChunker,
        SectionChunker,
    )

    registry = {
        "fixed": FixedTokenChunker,
        "recursive": RecursiveChunker,
        "section": SectionChunker,
        "parent_child": ParentChildChunker,
    }
    return registry[cfg.strategy](cfg)
