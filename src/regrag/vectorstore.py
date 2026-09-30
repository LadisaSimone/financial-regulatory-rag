"""Qdrant vector store (self-hosted via Docker, or in-memory for tests).

Collection versioning: the collection name encodes chunking strategy + embedding model +
version (see Settings.collection_name), so experiments never overwrite each other and an
index can be rebuilt side by side before switching.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from regrag.schemas import Chunk, MetadataFilter, RetrievedChunk


class VectorStoreUnavailable(RuntimeError):
    pass


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.collection = collection
        try:
            self.client = QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(url=url, timeout=10)
        except Exception as e:
            raise VectorStoreUnavailable(str(e)) from e

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def recreate(self, dim: int) -> None:
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
            for field, schema in [
                ("authority", qm.PayloadSchemaType.KEYWORD),
                ("document_type", qm.PayloadSchemaType.KEYWORD),
                ("document_id", qm.PayloadSchemaType.KEYWORD),
                ("publication_date", qm.PayloadSchemaType.DATETIME),
            ]:
                self.client.create_payload_index(self.collection, field, schema)
        except Exception as e:
            raise VectorStoreUnavailable(f"cannot (re)create {self.collection}: {e}") from e

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch: int = 256) -> None:
        for i in range(0, len(chunks), batch):
            pts = [
                qm.PointStruct(id=point_id(c.chunk_id), vector=v, payload=c.payload())
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch], strict=True)
            ]
            self.client.upsert(self.collection, points=pts, wait=True)

    @staticmethod
    def to_qdrant_filter(f: MetadataFilter | None) -> qm.Filter | None:
        if f is None or f.is_empty():
            return None
        must: list[Any] = []
        if f.authority:
            must.append(qm.FieldCondition(key="authority", match=qm.MatchAny(any=f.authority)))
        if f.document_type:
            must.append(qm.FieldCondition(key="document_type", match=qm.MatchAny(any=f.document_type)))
        if f.document_id:
            must.append(qm.FieldCondition(key="document_id", match=qm.MatchAny(any=f.document_id)))
        if f.publication_date_gte or f.publication_date_lte:
            must.append(
                qm.FieldCondition(
                    key="publication_date",
                    range=qm.DatetimeRange(
                        gte=f.publication_date_gte.isoformat() if f.publication_date_gte else None,
                        lte=f.publication_date_lte.isoformat() if f.publication_date_lte else None,
                    ),
                )
            )
        return qm.Filter(must=must)

    def search(self, vector: list[float], top_k: int, flt: MetadataFilter | None = None) -> list[RetrievedChunk]:
        try:
            res = self.client.query_points(
                self.collection, query=vector, limit=top_k,
                query_filter=self.to_qdrant_filter(flt), with_payload=True,
            ).points
        except Exception as e:
            raise VectorStoreUnavailable(f"search failed: {e}") from e
        return [
            RetrievedChunk(chunk_id=p.payload["chunk_id"], text=p.payload["text"], score=float(p.score),
                           metadata=p.payload, retriever="dense")
            for p in res
        ]

    def all_chunks(self) -> list[dict]:
        """Scroll the whole collection (used to build BM25 and neighbor lookup)."""
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=True)
            out.extend(p.payload for p in pts)
            if offset is None:
                break
        return out

    def count(self) -> int:
        return self.client.count(self.collection).count
