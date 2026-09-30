"""Ablation study: start from a baseline and add one technique at a time.

Each step is an override on top of the base settings. Steps that change chunking or the
embedding model need their own index (built automatically with --ingest). The summary table
is written to experiments/results/ablation_<ts>.md. Negative results are kept on purpose.
"""

from __future__ import annotations

from datetime import UTC, datetime

from regrag.config import Settings, load_settings
from regrag.evaluation.runner import run_evaluation
from regrag.ingestion.pipeline import run_ingestion

ABLATION_STEPS: list[tuple[str, dict]] = [
    ("01_baseline_dense_fixed", {"chunking": {"strategy": "fixed"}, "retrieval": {"type": "dense"}}),
    ("02_+section_chunking", {"chunking": {"strategy": "section"}, "retrieval": {"type": "dense"}}),
    ("03_bm25_only", {"chunking": {"strategy": "section"}, "retrieval": {"type": "bm25"}}),
    ("04_+hybrid_rrf", {"chunking": {"strategy": "section"}, "retrieval": {"type": "hybrid"}}),
    ("05_+reranker", {"chunking": {"strategy": "section"}, "retrieval": {"type": "hybrid"}, "reranker": {"enabled": True}}),
    ("06_+query_rewriting", {"chunking": {"strategy": "section"}, "retrieval": {"type": "hybrid"},
                             "reranker": {"enabled": True}, "query": {"rewriting": True}}),
    ("07_+multi_query", {"chunking": {"strategy": "section"}, "retrieval": {"type": "hybrid"},
                         "reranker": {"enabled": True}, "query": {"multi_query": True}}),
    ("08_parent_child", {"chunking": {"strategy": "parent_child"}, "retrieval": {"type": "hybrid"},
                         "reranker": {"enabled": True}}),
]

COLUMNS = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]
GEN_COLUMNS = ["citation_correctness", "answerable_success", "abstention_accuracy"]


def run_ablation(base: Settings | None = None, ingest: bool = False, generation: bool = True,
                 steps: list[str] | None = None) -> str:
    base = base or load_settings()
    built: set[str] = set()
    rows = []
    for name, override in ABLATION_STEPS:
        if steps and name not in steps:
            continue
        s = load_settings(overrides=_merge(base.model_dump(), override))
        key = s.collection_name()
        if ingest and key not in built:
            run_ingestion(s, download=False)
            built.add(key)
        res = run_evaluation(s, name=name, generation=generation)
        rows.append((name, res))

    lines = ["| step | " + " | ".join(COLUMNS + GEN_COLUMNS) + " | p50 ms | cost $ |",
             "|" + "---|" * (len(COLUMNS) + len(GEN_COLUMNS) + 3)]
    for name, r in rows:
        rm, gm = r["retrieval_metrics"], r["generation_metrics"]
        vals = [f"{rm.get(c, 0):.3f}" for c in COLUMNS] + [f"{gm.get(c, 0):.3f}" for c in GEN_COLUMNS]
        lines.append(f"| {name} | " + " | ".join(vals) + f" | {r['latency_ms']['total_p50']} | {r['cost_usd']} |")
    table = "\n".join(lines)
    out = base.path("results_dir") / f"ablation_{datetime.now(UTC):%Y%m%dT%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Ablation study\n\n" + table + "\n")
    return table


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
