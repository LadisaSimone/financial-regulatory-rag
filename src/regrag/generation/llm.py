"""LLM provider abstraction. The RAG core only sees `LLMProvider.generate`."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from regrag.config import LLM as LLMConfig


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    cached: bool = False


class LLMProvider(ABC):
    model: str

    @abstractmethod
    def generate(self, system: str, user: str, json_mode: bool = False) -> LLMResult: ...


class OpenAILLM(LLMProvider):
    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI

        from regrag.retry import api_retry

        self.cfg, self.model = cfg, cfg.model
        self.client = OpenAI(timeout=cfg.timeout_s, max_retries=0)  # retries handled by us
        self._call = api_retry(self._call_once, attempts=cfg.max_retries)

    def _call_once(self, **kw):
        return self.client.chat.completions.create(**kw)

    def generate(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        kw = dict(
            model=self.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_output_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        try:
            r = self._call(**kw)
        except Exception as e:
            raise LLMError(f"LLM call failed: {type(e).__name__}: {e}") from e
        return LLMResult(
            text=r.choices[0].message.content or "",
            prompt_tokens=r.usage.prompt_tokens if r.usage else 0,
            completion_tokens=r.usage.completion_tokens if r.usage else 0,
            model=self.model,
        )


class FakeLLM(LLMProvider):
    """Deterministic extractive 'LLM' for tests and offline pipeline runs.
    Picks the context chunk with the highest word overlap with the question; abstains when
    overlap is too low. Honors the JSON contract of the real prompt."""

    model = "fake-extractive"

    def generate(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        if not json_mode:
            q = user.split("Question:")[-1].strip()
            return LLMResult(text=q, model=self.model)
        q = user.split("QUESTION:")[-1].strip().lower()
        qw = set(re.findall(r"[a-z]{4,}", q))
        best, best_score = None, 0.0
        for m in re.finditer(r"\[chunk_id: ([^\]]+)\][^\n]*\n(.*?)(?=\n\[chunk_id:|\n=== END)", user, re.S):
            words = set(re.findall(r"[a-z]{4,}", m.group(2).lower()))
            score = len(qw & words) / (len(qw) or 1)
            if score > best_score:
                best, best_score = m, score
        if best is None or best_score < 0.3:
            out = {"answer": "The provided documents do not contain sufficient information to answer this question reliably.",
                   "citations": [], "insufficient_evidence": True}
        else:
            sent = re.split(r"(?<=[.;])\s", best.group(2).strip())[0][:400]
            out = {"answer": sent, "citations": [{"chunk_id": best.group(1)}], "insufficient_evidence": False}
        return LLMResult(text=json.dumps(out), prompt_tokens=len(user) // 4, completion_tokens=60, model=self.model)


class CachedLLM(LLMProvider):
    def __init__(self, inner: LLMProvider, cache_dir: Path):
        self.inner, self.model = inner, inner.model
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_dir / "llm.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def generate(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        k = hashlib.sha256(f"{self.model}\x00{json_mode}\x00{system}\x00{user}".encode()).hexdigest()
        with self._lock:
            row = self._db.execute("SELECT v FROM llm WHERE k=?", (k,)).fetchone()
        if row:
            d = json.loads(row[0])
            return LLMResult(**{**d, "cached": True})
        res = self.inner.generate(system, user, json_mode)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (k, json.dumps(res.__dict__)))
            self._db.commit()
        return res


def get_llm(cfg: LLMConfig, cache_dir: Path | None = None, use_cache: bool = True) -> LLMProvider:
    llm: LLMProvider = OpenAILLM(cfg) if cfg.provider == "openai" else FakeLLM()
    if use_cache and cache_dir is not None and cfg.provider != "fake":
        llm = CachedLLM(llm, cache_dir)
    return llm
