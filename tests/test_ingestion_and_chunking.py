import pytest

from regrag.chunk_store import ChunkStore, chunks_path
from regrag.chunking import get_chunker
from regrag.config import Chunking
from regrag.ingestion.manifest import ManifestError, load_manifest
from regrag.ingestion.parsing import ParseError, parse_pdf
from regrag.ingestion.pipeline import run_ingestion


def test_parse_keeps_pages_sections_and_strips_headers(settings):
    meta = load_manifest(settings.path("manifest"))[0]
    pages = parse_pdf(settings.path("raw_dir") / "doc_a.pdf", meta)
    assert [p.page_number for p in pages] == [1, 2, 3, 4]
    assert pages[1].section and pages[1].section.startswith("2. Enhanced due diligence")
    assert all("EBA PUBLIC" not in p.text for p in pages)
    assert "relationship." in pages[1].text  # hyphenation across line fixed


def test_corrupted_pdf_raises(settings, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    meta = load_manifest(settings.path("manifest"))[0]
    with pytest.raises(ParseError):
        parse_pdf(bad, meta)


def test_invalid_manifest(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("documents:\n  - document_id: BAD ID\n    title: x\n")
    with pytest.raises(ManifestError):
        load_manifest(p)


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "section", "parent_child"])
def test_all_strategies_traceable_and_deterministic(settings, strategy):
    meta = load_manifest(settings.path("manifest"))[0]
    pages = parse_pdf(settings.path("raw_dir") / "doc_a.pdf", meta)
    ch = get_chunker(Chunking(strategy=strategy, max_tokens=64, overlap=8, child_tokens=32))
    a, b = ch.chunk(pages, meta), ch.chunk(pages, meta)
    assert a and [c.chunk_id for c in a] == [c.chunk_id for c in b]
    for c in a:
        assert 1 <= c.page_start <= c.page_end <= 4
        assert c.chunking_strategy == strategy
        assert c.token_count <= 64 + 16  # small slack for joins
    if strategy == "parent_child":
        assert all(c.parent_id and c.parent_text for c in a)


def test_section_chunks_do_not_cross_sections(settings):
    meta = load_manifest(settings.path("manifest"))[0]
    pages = parse_pdf(settings.path("raw_dir") / "doc_a.pdf", meta)
    chunks = get_chunker(Chunking(strategy="section", max_tokens=512, overlap=0)).chunk(pages, meta)
    for c in chunks:
        assert c.page_start == c.page_end  # each synthetic section is one page


def test_ingestion_report_and_duplicates(settings):
    import shutil

    shutil.copy(settings.path("raw_dir") / "doc_a.pdf", settings.path("raw_dir") / "doc_c.pdf")
    import yaml

    m = yaml.safe_load(settings.path("manifest").read_text())
    m["documents"].append({**m["documents"][0], "document_id": "doc_c"})
    settings.path("manifest").write_text(yaml.safe_dump(m))
    report = run_ingestion(settings, download=False, index=False)
    assert report.documents_processed == 2
    assert report.duplicate_documents and "doc_c" in report.duplicate_documents[0]
    store = ChunkStore.load(chunks_path(settings.path("processed_dir"), "section"))
    assert len(store.chunks) == report.chunks_generated
