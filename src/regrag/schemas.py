"""Core data model shared by ingestion, retrieval, generation and the API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

Authority = Literal["EBA", "ECB", "EC", "FATF", "ESMA", "EIOPA", "AMLA", "DNB", "AFM", "BCBS", "OTHER"]
DocumentType = Literal[
    "guideline", "regulation", "directive", "recommendation", "report",
    "opinion", "standard", "guidance", "typology", "other",
]


class DocumentMeta(BaseModel):
    """Provenance metadata, one per source document (lives in data/manifest.yaml)."""

    document_id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    authority: Authority
    document_type: DocumentType
    publication_date: date
    source_url: HttpUrl
    download_date: date | None = None
    version: str | None = None
    language: str = "en"
    sha256: str | None = None
    tags: list[str] = []


class DocumentPage(BaseModel):
    document_id: str
    page_number: int  # 1-based, as printed by PDF viewers
    text: str
    section: str | None = None
    headings: list[str] = []
    metadata: dict[str, Any] = {}


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    title: str
    authority: str
    document_type: str
    section: str | None = None
    page_start: int
    page_end: int
    publication_date: str | None = None
    source_url: str | None = None
    chunking_strategy: str
    token_count: int
    position: int  # order of the chunk inside its document (for neighbor expansion)
    parent_id: str | None = None
    parent_text: str | None = None

    def payload(self) -> dict[str, Any]:
        return self.model_dump()

    def index_text(self, contextual_header: bool = False) -> str:
        """Text used for embedding / BM25. With a contextual header, chunks that only *cite*
        "Directive (EU) 2015/849" become distinguishable from the Directive's own chunks."""
        if not contextual_header:
            return self.text
        sec = f" — {self.section}" if self.section else ""
        return f"{self.authority} — {self.title}{sec}\n{self.text}"


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    retriever: str = ""

    @property
    def document_id(self) -> str:
        return self.metadata["document_id"]

    @property
    def page_start(self) -> int:
        return int(self.metadata["page_start"])

    @property
    def page_end(self) -> int:
        return int(self.metadata["page_end"])


class MetadataFilter(BaseModel):
    authority: list[str] | None = None
    document_type: list[str] | None = None
    publication_date_gte: date | None = None
    publication_date_lte: date | None = None
    document_id: list[str] | None = None

    def is_empty(self) -> bool:
        return not any(v for v in self.model_dump().values())

    def matches(self, meta: dict[str, Any]) -> bool:
        if self.authority and meta.get("authority") not in self.authority:
            return False
        if self.document_type and meta.get("document_type") not in self.document_type:
            return False
        if self.document_id and meta.get("document_id") not in self.document_id:
            return False
        pub = meta.get("publication_date")
        if pub and (self.publication_date_gte or self.publication_date_lte):
            d = date.fromisoformat(str(pub)[:10])
            if self.publication_date_gte and d < self.publication_date_gte:
                return False
            if self.publication_date_lte and d > self.publication_date_lte:
                return False
        return True


# ---------- Generation ----------

class RawCitation(BaseModel):
    """What the LLM is allowed to emit: a reference to a chunk it was shown. Nothing else."""

    chunk_id: str


class LLMAnswer(BaseModel):
    """Structured output requested from the LLM (validated; never trusted blindly)."""

    answer: str
    citations: list[RawCitation] = []
    insufficient_evidence: bool = False


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    authority: str
    page_start: int
    page_end: int
    section: str | None = None
    source_url: str | None = None
    valid: bool = True
    label: str = ""


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RAGResponse(BaseModel):
    request_id: str
    query: str
    rewritten_query: str | None = None
    answer: str
    insufficient_evidence: bool
    citations: list[Citation]
    invalid_citations: list[str] = []
    retrieval: dict[str, Any] = {}
    latency_ms: dict[str, float] = {}
    usage: Usage = Usage()
    disclaimer: str = ""
