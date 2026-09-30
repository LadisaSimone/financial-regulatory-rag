from __future__ import annotations

from abc import ABC, abstractmethod

from regrag.schemas import MetadataFilter, RetrievedChunk


class Retriever(ABC):
    """Common interface for BM25, dense and hybrid retrieval."""

    name: str

    @abstractmethod
    def retrieve(self, query: str, top_k: int, flt: MetadataFilter | None = None) -> list[RetrievedChunk]: ...
