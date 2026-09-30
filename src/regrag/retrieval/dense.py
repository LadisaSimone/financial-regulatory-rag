from __future__ import annotations

from regrag.embeddings.base import EmbeddingProvider
from regrag.retrieval.base import Retriever
from regrag.schemas import MetadataFilter, RetrievedChunk
from regrag.vectorstore import QdrantStore


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, store: QdrantStore, embedder: EmbeddingProvider):
        self.store, self.embedder = store, embedder

    def retrieve(self, query: str, top_k: int, flt: MetadataFilter | None = None) -> list[RetrievedChunk]:
        return self.store.search(self.embedder.embed_query(query), top_k, flt)
