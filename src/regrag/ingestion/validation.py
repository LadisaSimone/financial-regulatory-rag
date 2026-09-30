"""Data-quality checks and the ingestion report."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

from regrag.schemas import DocumentPage


@dataclass
class IngestionReport:
    documents_in_manifest: int = 0
    documents_processed: int = 0
    documents_failed: list[str] = field(default_factory=list)
    duplicate_documents: list[str] = field(default_factory=list)
    empty_documents: list[str] = field(default_factory=list)
    pages_processed: int = 0
    near_empty_pages: int = 0
    failed_pages: int = 0
    duplicate_pages_removed: int = 0
    chunks_generated: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Documents processed: {self.documents_processed}/{self.documents_in_manifest}\n"
            f"Documents failed: {len(self.documents_failed)} {self.documents_failed or ''}\n"
            f"Duplicate documents: {len(self.duplicate_documents)}\n"
            f"Pages processed: {self.pages_processed:,}\n"
            f"Near-empty pages: {self.near_empty_pages}\n"
            f"Failed pages: {self.failed_pages}\n"
            f"Duplicate pages removed: {self.duplicate_pages_removed}\n"
            f"Chunks generated: {self.chunks_generated:,}"
        )


def text_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


def validate_pages(
    pages: list[DocumentPage],
    report: IngestionReport,
    seen_page_hashes: set[str],
    min_page_chars: int = 80,
) -> list[DocumentPage]:
    """Drop duplicate pages (across the corpus), count near-empty ones. Near-empty pages are
    kept out of chunking but still counted, so page numbering stays correct."""
    kept = []
    for p in pages:
        report.pages_processed += 1
        if len(p.text) < min_page_chars:
            report.near_empty_pages += 1
            continue
        h = text_hash(p.text)
        if h in seen_page_hashes:
            report.duplicate_pages_removed += 1
            continue
        seen_page_hashes.add(h)
        kept.append(p)
    return kept
