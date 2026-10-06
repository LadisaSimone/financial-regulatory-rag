"""Build eval/review.md: for every answerable question, show the best-matching passage on the
ground-truth pages so a human can confirm (or correct) the evidence, then set
"validated": true in eval/questions.jsonl.

    python scripts/review_eval.py [--strategy section]
"""

from __future__ import annotations

import argparse

from rank_bm25 import BM25Okapi

from regrag.chunk_store import ChunkStore, chunks_path
from regrag.config import load_settings
from regrag.evaluation.dataset import load_dataset
from regrag.retrieval.bm25 import bm25_tokenize


def best_window(text: str, query_tokens: list[str], size: int = 600, step: int = 100) -> str:
    """Show the part of a long chunk that overlaps the question most, not just its beginning."""
    if len(text) <= size:
        return text
    q = set(query_tokens)
    starts = range(0, len(text) - size + step, step)
    best = max(starts, key=lambda i: len(q & set(bm25_tokenize(text[i:i + size]))))
    return ("…" if best else "") + text[best:best + size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="section")
    args = ap.parse_args()
    s = load_settings()
    store = ChunkStore.load(chunks_path(s.path("processed_dir"), args.strategy))
    items = load_dataset(s.path("eval_dataset"))
    out = ["# Evaluation set — evidence review\n",
           "For each question: read the passage, check the pages answer the question, then set "
           '`"validated": true` in `eval/questions.jsonl` (or fix the pages).\n']
    for it in items:
        if not it.answerable:
            continue
        mark = "✅" if it.validated else "⬜"
        out.append(f"\n## {mark} {it.id} · {it.question_type} · {it.difficulty}\n\n**{it.question}**\n")
        q = bm25_tokenize(it.question)
        for ev in it.relevant_evidence:
            cands = [c for c in store.chunks if c.document_id == ev.document_id
                     and c.page_start <= max(ev.pages) and c.page_end >= min(ev.pages)]
            if not cands:
                out.append(f"- ⚠️ `{ev.document_id}` pp.{ev.pages}: no chunk found on these pages\n")
                continue
            bm = BM25Okapi([bm25_tokenize(f"{c.section or ''} {c.text}") for c in cands])
            scores = bm.get_scores(q)
            top = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)[:2]
            out.append(f"- `{ev.document_id}` pp.{min(ev.pages)}–{max(ev.pages)}\n")
            for i in top:
                c = cands[i]
                excerpt = best_window(" ".join(c.text.split()), q)
                out.append(f"  - p.{c.page_start}–{c.page_end} · §{c.section or '—'}\n\n    > {excerpt}…\n")
    path = s.path("eval_dataset").with_name("review.md")
    path.write_text("\n".join(out))
    print(f"wrote {path} ({sum(i.answerable for i in items)} questions)")


if __name__ == "__main__":
    main()
