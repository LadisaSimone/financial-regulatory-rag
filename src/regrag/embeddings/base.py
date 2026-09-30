"""Embedding provider abstraction + a disk cache keyed by (model, text hash).

Providers:
  openai                 API-based (text-embedding-3-small / -large)
  sentence_transformers  local open-source (BGE / E5 / MiniLM ...)
  hashing                deterministic, dependency-free; for tests and offline CI ONLY
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import struct
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from regrag.config import Embeddings as EmbeddingsConfig


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(ABC):
    model: str
    dim: int
    # Some models (E5, BGE) expect instruction prefixes for queries vs passages.
    query_prefix: str = ""
    document_prefix: str = ""

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    # token accounting (API providers fill this)
    tokens_used: int = 0


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wraps any provider with a SQLite cache. Avoids recomputing identical embeddings
    across experiments (e.g. same chunk text under two retriever configs)."""

    def __init__(self, inner: EmbeddingProvider, cache_dir: Path):
        self.inner = inner
        self.model, self.dim = inner.model, inner.dim
        self.query_prefix, self.document_prefix = inner.query_prefix, inner.document_prefix
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_dir / "embeddings.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def tokens_used(self) -> int:  # type: ignore[override]
        return self.inner.tokens_used

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        found: dict[str, list[float]] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                batch = keys[i : i + 500]
                q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(batch))})"
                for k, v in self._db.execute(q, batch):
                    found[k] = list(struct.unpack(f"{len(v) // 4}f", v))
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in found]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vecs = self.inner._embed([t for _, t in missing])
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                    [(k, struct.pack(f"{len(v)}f", *v)) for (k, _), v in zip(missing, vecs, strict=True)],
                )
                self._db.commit()
            found.update({k: v for (k, _), v in zip(missing, vecs, strict=True)})
        return [found[k] for k in keys]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Feature-hashing bag of words (+bigrams). NOT a semantic model: used so the full
    pipeline and CI run with zero network and zero model downloads."""

    def __init__(self, dim: int = 384):
        self.model, self.dim = "hashing", dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            toks = re.findall(r"\w+", t.lower())
            for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:], strict=False)]:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class OpenAIEmbeddingProvider(EmbeddingProvider):
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}

    def __init__(self, model: str, batch_size: int = 64):
        from openai import OpenAI

        from regrag.retry import api_retry

        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.model, self.dim, self.batch_size = model, self.DIMS.get(model, 1536), batch_size
        self._call = api_retry(self._call_once)
        self.tokens_used = 0

    def _call_once(self, batch: list[str]):
        return self.client.embeddings.create(model=self.model, input=batch)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            try:
                resp = self._call(texts[i : i + self.batch_size])
            except Exception as e:
                raise EmbeddingError(f"OpenAI embedding failed: {e}") from e
            self.tokens_used += resp.usage.total_tokens
            out.extend(d.embedding for d in resp.data)
        return out


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model: str, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError("pip install '.[local]' to use sentence-transformers") from e
        self.st = SentenceTransformer(model)
        self.model, self.batch_size = model, batch_size
        self.dim = self.st.get_sentence_embedding_dimension()
        name = model.lower()
        if "e5" in name:
            self.query_prefix, self.document_prefix = "query: ", "passage: "
        elif "bge" in name and "en" in name:
            self.query_prefix = "Represent this sentence for searching relevant passages: "

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.st.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def get_embedding_provider(cfg: EmbeddingsConfig, cache_dir: Path | None = None) -> EmbeddingProvider:
    if cfg.provider == "openai":
        p: EmbeddingProvider = OpenAIEmbeddingProvider(cfg.model, cfg.batch_size)
    elif cfg.provider == "sentence_transformers":
        p = SentenceTransformerProvider(cfg.model, cfg.batch_size)
    elif cfg.provider == "hashing":
        p = HashingEmbeddingProvider()
    else:
        raise ValueError(f"unknown embedding provider {cfg.provider}")
    if cfg.cache and cache_dir is not None:
        return CachedEmbeddingProvider(p, cache_dir)
    return p
