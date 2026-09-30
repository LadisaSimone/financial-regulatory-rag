from __future__ import annotations

from itertools import groupby

from regrag.chunking.base import Chunker, Unit, iter_units, pack_units, split_oversized
from regrag.schemas import Chunk, DocumentMeta, DocumentPage


class FixedTokenChunker(Chunker):
    """Strategy A — fixed token windows. Page of each token is tracked for traceability."""

    name = "fixed"

    def chunk(self, pages: list[DocumentPage], meta: DocumentMeta) -> list[Chunk]:
        units = iter_units(pages, self.tok)
        ids: list = []
        page_of: list[int] = []
        section_of: list[str | None] = []
        for u in units:
            t = self.tok.encode(u.text + "\n")
            ids.extend(t)
            page_of.extend([u.page] * len(t))
            section_of.extend([u.section] * len(t))
        size, step = self.cfg.max_tokens, max(1, self.cfg.max_tokens - self.cfg.overlap)
        chunks = []
        for pos, start in enumerate(range(0, max(len(ids), 1), step)):
            window = ids[start : start + size]
            if not window:
                break
            text = self.tok.decode(window).strip()
            pages_w = page_of[start : start + size]
            u = [Unit(text, pages_w[0], section_of[start], len(window))]
            c = self._make(meta, u, pos)
            c.page_end = max(pages_w)
            chunks.append(c)
            if start + size >= len(ids):
                break
        return chunks


class RecursiveChunker(Chunker):
    """Strategy B — paragraph/sentence boundaries before token boundaries."""

    name = "recursive"

    def _units(self, pages, max_tokens):
        units = iter_units(pages, self.tok)
        return [s for u in units for s in split_oversized(u, max_tokens, self.tok)]

    def chunk(self, pages: list[DocumentPage], meta: DocumentMeta) -> list[Chunk]:
        units = self._units(pages, self.cfg.max_tokens)
        groups = pack_units(units, self.cfg.max_tokens, self.cfg.overlap)
        return [self._make(meta, g, i) for i, g in enumerate(groups)]


class SectionChunker(RecursiveChunker):
    """Strategy C — never mixes two sections in one chunk; keeps related paragraphs together."""

    name = "section"

    def chunk(self, pages: list[DocumentPage], meta: DocumentMeta) -> list[Chunk]:
        units = self._units(pages, self.cfg.max_tokens)
        chunks, pos = [], 0
        for _section, grp in groupby(units, key=lambda u: u.section):
            for g in pack_units(list(grp), self.cfg.max_tokens, self.cfg.overlap):
                chunks.append(self._make(meta, g, pos))
                pos += 1
        return chunks


class ParentChildChunker(RecursiveChunker):
    """Strategy D — retrieve small children, hand the parent section to the LLM."""

    name = "parent_child"

    def chunk(self, pages: list[DocumentPage], meta: DocumentMeta) -> list[Chunk]:
        parent_max = self.cfg.max_tokens * 2
        units = self._units(pages, self.cfg.child_tokens)
        chunks, pos, parent_idx = [], 0, 0
        for _section, grp in groupby(units, key=lambda u: u.section):
            for parent in pack_units(list(grp), parent_max, 0):
                parent_id = f"{meta.document_id}:parent:{parent_idx:05d}"
                parent_text = "\n".join(u.text for u in parent)
                parent_idx += 1
                for child in pack_units(parent, self.cfg.child_tokens, self.cfg.overlap // 2):
                    chunks.append(
                        self._make(meta, child, pos, parent_id=parent_id, parent_text=parent_text)
                    )
                    pos += 1
        return chunks
