"""On-disk chunk store (JSONL per chunking strategy). Source for BM25 and neighbor expansion,
and a reproducible artifact of each ingestion run."""

from __future__ import annotations

import json
from pathlib import Path

from regrag.schemas import Chunk


class ChunkStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.by_pos = {(c.document_id, c.position): c for c in chunks}

    @classmethod
    def load(cls, path: Path) -> ChunkStore:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `regrag ingest` first")
        with path.open() as f:
            return cls([Chunk.model_validate_json(line) for line in f if line.strip()])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")

    def neighbors(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        c = self.by_id.get(chunk_id)
        if c is None:
            return []
        out = []
        for d in range(-window, window + 1):
            n = self.by_pos.get((c.document_id, c.position + d))
            if n is not None:
                out.append(n)
        return out

    def documents(self) -> dict[str, dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c.document_id, {
                "document_id": c.document_id, "title": c.title, "authority": c.authority,
                "document_type": c.document_type, "publication_date": c.publication_date,
                "source_url": c.source_url, "chunks": 0, "pages": 0,
            })
            d["chunks"] += 1
            d["pages"] = max(d["pages"], c.page_end)
        return docs


def chunks_path(processed_dir: Path, strategy: str) -> Path:
    return processed_dir / f"chunks_{strategy}.jsonl"


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
