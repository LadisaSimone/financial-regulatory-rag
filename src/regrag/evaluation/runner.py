"""Experiment runner: retrieval-only and end-to-end evaluation, with results stored as JSON
(one file per experiment) so every architectural decision can be backed by numbers."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from regrag.chunk_store import dump_json
from regrag.config import Settings
from regrag.evaluation.dataset import EvalItem, load_dataset
from regrag.evaluation.judge import judge
from regrag.evaluation.metrics import (
    answerability_metrics,
    citation_metrics,
    mean_dicts,
    retrieval_metrics,
)
from regrag.logging_utils import get_logger, log
from regrag.pipeline import RAGPipeline
from regrag.retrieval.query import infer_filter

logger = get_logger("evaluation")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def experiment_config(s: Settings) -> dict:
    return {
        "chunking_strategy": s.chunking.strategy, "chunk_size": s.chunking.max_tokens,
        "chunk_overlap": s.chunking.overlap, "embedding_model": f"{s.embeddings.provider}:{s.embeddings.model}",
        "retriever": s.retrieval.type, "top_k": s.retrieval.final_top_k, "rrf_k": s.retrieval.rrf_k,
        "dense_top_k": s.retrieval.dense_top_k, "sparse_top_k": s.retrieval.sparse_top_k,
        "dense_weight": s.retrieval.dense_weight, "sparse_weight": s.retrieval.sparse_weight,
        "reranker_candidates": s.reranker.candidates if s.reranker.enabled else None,
        "infer_filters": s.retrieval.infer_filters,
        "filter_mode": s.retrieval.filter_mode if s.retrieval.infer_filters else None,
        "boost_ranks": s.retrieval.boost_ranks if s.retrieval.infer_filters else None,
        "contextual_header": s.chunking.contextual_header,
        "reranker": s.reranker.model if s.reranker.enabled else None, "reranker_top_k": s.reranker.top_k,
        "query_rewriting": s.query.rewriting, "multi_query": s.query.multi_query,
        "neighbor_expansion": s.context.neighbor_expansion, "llm": f"{s.llm.provider}:{s.llm.model}",
        "prompt_version": s.llm.prompt_version,
        "eval_dataset": s.paths.eval_dataset,
    }


def experiment_id(cfg: dict, name: str | None) -> str:
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{name}-{h}" if name else h


def run_evaluation(settings: Settings, name: str | None = None, generation: bool = True,
                   use_judge: bool = False, pipeline: RAGPipeline | None = None,
                   items: list[EvalItem] | None = None, out_dir: Path | None = None) -> dict:
    pipe = pipeline or RAGPipeline(settings)
    items = items if items is not None else load_dataset(settings.path("eval_dataset"))
    cfg = experiment_config(settings)
    exp_id = experiment_id(cfg, name)
    per_q, retr_rows, gen_rows, lat_retr, lat_total = [], [], [], [], []
    tokens_in = tokens_out = 0
    cost = 0.0

    for it in items:
        row: dict = {"id": it.id, "question_type": it.question_type, "answerable": it.answerable}
        if it.answerable:
            t0 = time.perf_counter()
            retrieved, _ = pipe.retrieve(it.question, top_k=max(10, settings.retrieval.final_top_k))
            lat_retr.append((time.perf_counter() - t0) * 1000)
            rm = retrieval_metrics(retrieved, it)
            retr_rows.append(rm)
            row["retrieval"] = rm
            row["retrieved"] = [c.chunk_id for c in retrieved[:10]]
        if generation:
            resp = pipe.answer(it.question, debug=use_judge)
            lat_total.append(resp.latency_ms.get("total_ms", 0))
            gm = {**citation_metrics(resp.citations, resp.invalid_citations, it),
                  **answerability_metrics(resp.insufficient_evidence, it)}
            if use_judge and it.answerable and not resp.insufficient_evidence:
                ctx = "\n\n".join(c.get("text", "") for c in resp.retrieval.get("chunks", []))
                gm.update(judge(pipe.llm, it.question, resp.answer, ctx))
            gen_rows.append(gm)
            tokens_in += resp.usage.prompt_tokens
            tokens_out += resp.usage.completion_tokens
            cost += resp.usage.estimated_cost_usd
            row.update(generation=gm, answer=resp.answer, insufficient=resp.insufficient_evidence,
                       citations=[c.chunk_id for c in resp.citations])
        per_q.append(row)

    def pct(v, q):
        return round(statistics.quantiles(v, n=100)[q - 1], 2) if len(v) >= 2 else (round(v[0], 2) if v else 0.0)

    by_type: dict[str, list] = {}
    by_ref: dict[str, list] = {}
    for r, it in zip(per_q, items, strict=True):
        if "retrieval" in r:
            by_type.setdefault(r["question_type"], []).append(r["retrieval"])
            named = "names_document_or_authority" if infer_filter(it.question) else "no_explicit_reference"
            by_ref.setdefault(named, []).append(r["retrieval"])

    result = {
        "experiment_id": exp_id, "name": name, "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(), "config": cfg, "n_questions": len(items),
        "n_answerable": sum(i.answerable for i in items),
        "retrieval_metrics": mean_dicts(retr_rows),
        "retrieval_metrics_by_type": {k: mean_dicts(v) for k, v in by_type.items()},
        "retrieval_metrics_by_reference": {k: {**mean_dicts(v), "n": len(v)} for k, v in by_ref.items()},
        "generation_metrics": mean_dicts(gen_rows),
        "generation_metrics_note": "keys prefixed llm_judge_ are LLM-as-a-judge estimates, not ground truth",
        "latency_ms": {"retrieval_p50": pct(lat_retr, 50), "retrieval_p95": pct(lat_retr, 95),
                       "total_p50": pct(lat_total, 50), "total_p95": pct(lat_total, 95)},
        "tokens": {"prompt": tokens_in, "completion": tokens_out},
        "cost_usd": round(cost, 4),
        "per_question": per_q,
    }
    out = (out_dir or settings.path("results_dir")) / f"{exp_id}.json"
    dump_json(result, out)
    log(logger, "experiment_done", experiment_id=exp_id, path=str(out), **result["retrieval_metrics"])
    return result
