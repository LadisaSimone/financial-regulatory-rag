"""Configuration: YAML file + environment overrides, validated with Pydantic.

Resolution order (later wins):
  1. configs/default.yaml
  2. file in REGRAG_CONFIG (deep-merged)
  3. env vars REGRAG__SECTION__KEY=value
Secrets (OPENAI_API_KEY, ...) are read from env only and never stored in config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Paths(BaseModel):
    manifest: str = "data/manifest.yaml"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    eval_dataset: str = "eval/questions.jsonl"
    results_dir: str = "experiments/results"
    traces_file: str = "experiments/traces.jsonl"


class Ingestion(BaseModel):
    min_page_chars: int = 80
    header_footer_min_ratio: float = 0.5
    timeout_s: int = 60


class Chunking(BaseModel):
    strategy: Literal["fixed", "recursive", "section", "parent_child"] = "section"
    max_tokens: int = Field(512, gt=16)
    overlap: int = Field(64, ge=0)
    child_tokens: int = 128
    tokenizer: str = "cl100k_base"


class Embeddings(BaseModel):
    provider: Literal["openai", "sentence_transformers", "hashing"] = "openai"
    model: str = "text-embedding-3-small"
    batch_size: int = 64
    cache: bool = True


class VectorStore(BaseModel):
    url: str = "http://localhost:6333"
    collection_prefix: str = "regrag"
    collection_version: int = 1


class Retrieval(BaseModel):
    type: Literal["dense", "bm25", "hybrid"] = "hybrid"
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 8
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    dedup_text_similarity: float = 0.9
    infer_filters: bool = False
    filter_mode: Literal["boost", "restrict"] = "boost"


class Reranker(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 30
    top_k: int = 6


class Query(BaseModel):
    normalize: bool = True
    rewriting: bool = False
    multi_query: bool = False
    multi_query_n: int = 3


class Context(BaseModel):
    max_tokens: int = 6000
    neighbor_expansion: bool = False
    neighbor_window: int = 1
    max_chunks_per_document: int = 4


class LLM(BaseModel):
    provider: Literal["openai", "fake"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_output_tokens: int = 800
    timeout_s: int = 60
    max_retries: int = 3
    prompt_version: str = "v1"


class Cache(BaseModel):
    llm_responses: bool = True


class Api(BaseModel):
    max_query_chars: int = 1000
    debug: bool = False


class Settings(BaseModel):
    paths: Paths = Paths()
    ingestion: Ingestion = Ingestion()
    chunking: Chunking = Chunking()
    embeddings: Embeddings = Embeddings()
    vector_store: VectorStore = VectorStore()
    retrieval: Retrieval = Retrieval()
    reranker: Reranker = Reranker()
    query: Query = Query()
    context: Context = Context()
    llm: LLM = LLM()
    cache: Cache = Cache()
    pricing_usd_per_1m_tokens: dict[str, dict[str, float]] = {}
    api: Api = Api()

    def path(self, name: str) -> Path:
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else ROOT / p

    def collection_name(self) -> str:
        model = self.embeddings.model.replace("/", "_").replace("-", "_")
        return (
            f"{self.vector_store.collection_prefix}_{self.chunking.strategy}"
            f"_{model}_v{self.vector_store.collection_version}"
        ).lower()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides(prefix: str = "REGRAG__") -> dict:
    out: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)  # "5" -> 5, "true" -> True
    return out


def load_settings(path: str | Path | None = None, overrides: dict | None = None) -> Settings:
    data: dict = {}
    if DEFAULT_CONFIG.exists():
        data = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    extra = path or os.environ.get("REGRAG_CONFIG")
    if extra:
        data = _deep_merge(data, yaml.safe_load(Path(extra).read_text()) or {})
    data = _deep_merge(data, _env_overrides())
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings.model_validate(data)
