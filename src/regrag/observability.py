"""Lightweight metrics, traces and cost estimation.

Privacy note: traces store the query text for debugging/evaluation. In a real deployment
queries may contain customer data — mask or drop `query` before persisting, restrict access,
and apply a retention policy. This demo uses only public documents and synthetic questions.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path


def estimate_cost(pricing: dict, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = pricing.get(model)
    if not p:
        return 0.0
    return (prompt_tokens * p.get("input", 0) + completion_tokens * p.get("output", 0)) / 1_000_000


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round((time.perf_counter() - t0) * 1000, 2)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


class Metrics:
    """In-process metrics (sliding window for latency percentiles)."""

    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.total = deque(maxlen=window)
        self.retrieval = deque(maxlen=window)
        self.generation = deque(maxlen=window)
        self.context_tokens = deque(maxlen=window)
        self.cost = deque(maxlen=window)

    def record(self, latency: dict, context_tokens: int, cost: float) -> None:
        with self._lock:
            self.request_count += 1
            self.total.append(latency.get("total_ms", 0))
            self.retrieval.append(latency.get("retrieval_ms", 0))
            self.generation.append(latency.get("generation_ms", 0))
            self.context_tokens.append(context_tokens)
            self.cost.append(cost)

    def record_error(self) -> None:
        with self._lock:
            self.request_count += 1
            self.error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg = lambda d: round(sum(d) / len(d), 4) if d else 0.0  # noqa: E731
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "p50_latency_ms": _pct(list(self.total), 0.5),
                "p95_latency_ms": _pct(list(self.total), 0.95),
                "avg_retrieval_latency_ms": avg(self.retrieval),
                "avg_generation_latency_ms": avg(self.generation),
                "average_context_tokens": avg(self.context_tokens),
                "average_cost_per_query_usd": avg(self.cost),
            }

    def prometheus(self) -> str:
        s = self.snapshot()
        return "".join(f"regrag_{k} {v}\n" for k, v in s.items())


class TraceWriter:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: dict) -> None:
        if not self.path:
            return
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(trace, default=str) + "\n")
