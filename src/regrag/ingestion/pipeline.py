"""Reproducible ingestion pipeline:

manifest -> download -> validate -> parse -> clean -> structure -> metadata -> chunk -> embed -> index

Rerunnable: downloads are cached by hash, embeddings are cached by text hash, and the Qdrant
collection for (strategy, model, version) is recreated from scratch each run.
"""

from __future__ import annotations

from regrag.chunk_store import ChunkStore, chunks_path, dump_json
from regrag.chunking import get_chunker
from regrag.config import Settings
from regrag.embeddings.base import get_embedding_provider
from regrag.ingestion.download import download_document
from regrag.ingestion.manifest import load_manifest, save_manifest
from regrag.ingestion.parsing import ParseError, parse_pdf
from regrag.ingestion.validation import IngestionReport, text_hash, validate_pages
from regrag.logging_utils import get_logger, log
from regrag.schemas import Chunk
from regrag.vectorstore import QdrantStore

logger = get_logger("ingestion")


def build_chunks(settings: Settings, report: IngestionReport, download: bool = True) -> list[Chunk]:
    manifest_path = settings.path("manifest")
    docs = load_manifest(manifest_path)
    report.documents_in_manifest = len(docs)
    raw_dir = settings.path("raw_dir")
    chunker = get_chunker(settings.chunking)
    seen_pages: set[str] = set()
    seen_docs: dict[str, str] = {}
    chunks: list[Chunk] = []

    for meta in docs:
        pdf = raw_dir / f"{meta.document_id}.pdf"
        try:
            if download:
                pdf = download_document(meta, raw_dir, settings.ingestion.timeout_s)
            elif not pdf.exists():
                raise FileNotFoundError(f"{pdf} missing (run without --no-download)")
            pages = parse_pdf(pdf, meta, settings.ingestion.header_footer_min_ratio)
        except (ParseError, FileNotFoundError, ValueError, OSError) as e:
            report.documents_failed.append(meta.document_id)
            report.errors.append(str(e))
            log(logger, "document_failed", document_id=meta.document_id, error=str(e))
            continue
        except Exception as e:  # network etc. — never abort the whole run for one document
            report.documents_failed.append(meta.document_id)
            report.errors.append(f"{meta.document_id}: {type(e).__name__}: {e}")
            log(logger, "document_failed", document_id=meta.document_id, error=str(e))
            continue

        full_hash = text_hash(" ".join(p.text for p in pages))
        if full_hash in seen_docs:
            report.duplicate_documents.append(f"{meta.document_id} == {seen_docs[full_hash]}")
            continue
        seen_docs[full_hash] = meta.document_id

        failed = sum(1 for p in pages if not p.text)
        report.failed_pages += failed
        kept = validate_pages(pages, report, seen_pages, settings.ingestion.min_page_chars)
        if not kept:
            report.empty_documents.append(meta.document_id)
            continue

        doc_chunks = chunker.chunk(kept, meta)
        chunks.extend(doc_chunks)
        report.documents_processed += 1
        log(logger, "document_ingested", document_id=meta.document_id, pages=len(pages), chunks=len(doc_chunks))

    if download:
        save_manifest(manifest_path, docs)  # persist download_date / sha256
    report.chunks_generated = len(chunks)
    return chunks


def run_ingestion(settings: Settings, download: bool = True, index: bool = True,
                  vector_store: QdrantStore | None = None, embedder=None) -> IngestionReport:
    report = IngestionReport()
    chunks = build_chunks(settings, report, download)
    processed = settings.path("processed_dir")
    ChunkStore(chunks).save(chunks_path(processed, settings.chunking.strategy))

    if index and chunks:
        embedder = embedder or get_embedding_provider(settings.embeddings, settings.path("cache_dir"))
        vs = vector_store or QdrantStore(settings.vector_store.url, settings.collection_name())
        vectors = embedder.embed_documents([c.text for c in chunks])
        vs.recreate(embedder.dim)
        vs.upsert(chunks, vectors)
        log(logger, "indexed", collection=settings.collection_name(), points=len(chunks))

    dump_json(report.as_dict(), processed / f"ingestion_report_{settings.chunking.strategy}.json")
    return report
