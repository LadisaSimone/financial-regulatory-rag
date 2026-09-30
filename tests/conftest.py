"""Fully offline fixtures: synthetic regulatory PDFs, hashing embeddings, in-memory Qdrant,
extractive FakeLLM. No network, no API keys, no model downloads."""

from __future__ import annotations

import pytest
import yaml

from regrag.config import load_settings
from regrag.embeddings.base import HashingEmbeddingProvider

DOC_A = {
    "title": "Guidelines on Customer Risk",
    "sections": [
        ("1. Subject matter", "These guidelines set out factors firms should consider when assessing the "
         "money laundering and terrorist financing risk associated with a business relationship."),
        ("2. Enhanced due diligence", "Firms should apply enhanced due diligence measures to high-risk "
         "customers. Enhanced due diligence measures include obtaining additional information on the "
         "customer and the beneficial owner, obtaining information on the source of funds and source of "
         "wealth, and obtaining senior management approval for the business relationship. Firms should "
         "increase the frequency and intensity of ongoing monitoring of the relation-\nship."),
        ("3. Politically exposed persons", "Where the customer or the beneficial owner is a politically "
         "exposed person, firms should take adequate measures to establish the source of wealth and "
         "source of funds and conduct enhanced ongoing monitoring."),
        ("4. High-risk third countries", "Business relationships involving high-risk third countries "
         "require additional controls, including enhanced monitoring of transactions and restrictions "
         "on correspondent relationships."),
    ],
}
DOC_B = {
    "title": "Recommendations on Transparency",
    "sections": [
        ("Recommendation 10. Customer due diligence", "Financial institutions should be required to "
         "identify the customer and verify that customer's identity using reliable, independent source "
         "documents. Article 13(1) applies."),
        ("Recommendation 24. Beneficial ownership", "Countries should ensure that there is adequate, "
         "accurate and timely information on the beneficial ownership and control of legal persons."),
    ],
}


def make_pdf(path, doc, header="EBA PUBLIC — Confidentiality: none", filler=6):
    import fitz

    pdf = fitz.open()
    for i, (head, body) in enumerate(doc["sections"]):
        page = pdf.new_page()
        page.insert_text((50, 30), header, fontsize=8)
        page.insert_text((50, 80), head, fontsize=15, fontname="helvetica-bold")
        y = 110
        text = (body + " ") * 1 + " ".join(["Firms should document their risk assessment."] * filler)
        box = fitz.Rect(50, y, 550, 780)
        page.insert_textbox(box, text, fontsize=10)
        page.insert_text((290, 820), f"{i + 1}", fontsize=8)
    pdf.save(path)


@pytest.fixture()
def corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "doc_a.pdf", DOC_A)
    make_pdf(raw / "doc_b.pdf", DOC_B, header="FATF — International Standards")
    manifest = {"documents": [
        {"document_id": "doc_a", "title": DOC_A["title"], "authority": "EBA", "document_type": "guideline",
         "publication_date": "2021-03-01", "source_url": "https://example.org/a.pdf"},
        {"document_id": "doc_b", "title": DOC_B["title"], "authority": "FATF",
         "document_type": "recommendation", "publication_date": "2012-02-16",
         "source_url": "https://example.org/b.pdf"},
    ]}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return tmp_path


@pytest.fixture()
def settings(corpus):
    return load_settings(overrides={
        "paths": {
            "manifest": str(corpus / "manifest.yaml"), "raw_dir": str(corpus / "raw"),
            "processed_dir": str(corpus / "processed"), "cache_dir": str(corpus / "cache"),
            "results_dir": str(corpus / "results"), "traces_file": str(corpus / "traces.jsonl"),
        },
        "chunking": {"strategy": "section", "max_tokens": 128, "overlap": 16},
        "embeddings": {"provider": "hashing", "model": "hashing", "cache": False},
        "vector_store": {"url": ":memory:"},
        "llm": {"provider": "fake", "model": "fake-extractive"},
        "retrieval": {"type": "hybrid", "final_top_k": 5},
    })


@pytest.fixture()
def pipeline(settings):
    from regrag.ingestion.pipeline import run_ingestion
    from regrag.pipeline import RAGPipeline
    from regrag.vectorstore import QdrantStore

    vs = QdrantStore(":memory:", settings.collection_name())
    emb = HashingEmbeddingProvider()
    report = run_ingestion(settings, download=False, vector_store=vs, embedder=emb)
    assert report.documents_processed == 2
    return RAGPipeline(settings, vector_store=vs, embedder=emb)
