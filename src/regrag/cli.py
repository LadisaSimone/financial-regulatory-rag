"""Command line entry point.

  regrag ingest [--no-download] [--no-index] [--config X]
  regrag ask "What EDD measures apply to high-risk customers?"
  regrag eval [--name N] [--retrieval-only] [--judge]
  regrag ablation [--ingest] [--retrieval-only]
  regrag serve
"""

from __future__ import annotations

import argparse
import json
import sys

from regrag.config import load_settings


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set."""
    import os
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="regrag")
    ap.add_argument("--config", help="extra YAML merged over configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--no-index", action="store_true")

    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--authority", nargs="*")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--name")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--judge", action="store_true")

    p = sub.add_parser("ablation")
    p.add_argument("--ingest", action="store_true")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--steps", nargs="*")

    p = sub.add_parser("serve")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)
    s = load_settings(args.config)

    if args.cmd == "ingest":
        from regrag.ingestion.pipeline import run_ingestion

        report = run_ingestion(s, download=not args.no_download, index=not args.no_index)
        print(report.summary())
        return 1 if report.documents_processed == 0 else 0

    if args.cmd == "ask":
        from regrag.pipeline import RAGPipeline
        from regrag.schemas import MetadataFilter

        flt = MetadataFilter(authority=args.authority) if args.authority else None
        r = RAGPipeline(s).answer(args.question, flt)
        if args.json:
            print(r.model_dump_json(indent=2))
            return 0
        print(f"\n{r.answer}\n")
        for i, c in enumerate(r.citations, 1):
            print(f"  [{i}] {c.label}  {c.source_url or ''}")
        if r.invalid_citations:
            print(f"  ! rejected citations: {r.invalid_citations}")
        print(f"\n  latency {r.latency_ms.get('total_ms')} ms · tokens {r.usage.prompt_tokens}+{r.usage.completion_tokens}"
              f" · ~${r.usage.estimated_cost_usd}\n  {r.disclaimer}")
        return 0

    if args.cmd == "eval":
        from regrag.evaluation.runner import run_evaluation

        res = run_evaluation(s, name=args.name, generation=not args.retrieval_only, use_judge=args.judge)
        print(json.dumps({k: res[k] for k in ("experiment_id", "retrieval_metrics", "generation_metrics",
                                              "latency_ms", "cost_usd")}, indent=2))
        return 0

    if args.cmd == "ablation":
        from regrag.evaluation.ablation import run_ablation

        print(run_ablation(s, ingest=args.ingest, generation=not args.retrieval_only, steps=args.steps))
        return 0

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("regrag.api.app:app", host=args.host, port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
