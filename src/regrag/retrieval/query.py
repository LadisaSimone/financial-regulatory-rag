"""Query processing: normalization -> analysis (conservative filter inference) -> optional
LLM rewriting / multi-query expansion. The original query is always kept in the trace."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from regrag.schemas import MetadataFilter

# Only explicit mentions trigger inference ("What does the EBA say ..."). No guessing from topic.
AUTHORITY_PATTERNS = {
    "EBA": r"\b(EBA|European Banking Authority)\b",
    "ECB": r"\b(ECB|European Central Bank)\b",
    "EC": r"\b(European Commission)\b",
    "FATF": r"\b(FATF|Financial Action Task Force)\b",
    "AMLA": r"\b(AMLA|Anti-Money Laundering Authority)\b",
    "ESMA": r"\b(ESMA)\b",
}

ACRONYMS = {  # light expansion helps BM25 when users write the acronym only
    "edd": "enhanced due diligence",
    "cdd": "customer due diligence",
    "pep": "politically exposed person",
    "peps": "politically exposed persons",
    "ubo": "ultimate beneficial owner",
    "ml/tf": "money laundering terrorist financing",
    "kyc": "know your customer",
}


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    inferred_filter: MetadataFilter | None = None
    rewritten: str | None = None
    variants: list[str] = field(default_factory=list)

    @property
    def retrieval_queries(self) -> list[str]:
        base = self.rewritten or self.normalized
        return [base] + [v for v in self.variants if v != base]


def normalize(q: str) -> str:
    q = unicodedata.normalize("NFKC", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def expand_acronyms(q: str) -> str:
    extra = [v for k, v in ACRONYMS.items() if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", q, re.I)]
    return q if not extra else f"{q} ({'; '.join(extra)})"


def infer_filter(q: str) -> MetadataFilter | None:
    found = [a for a, pat in AUTHORITY_PATTERNS.items() if re.search(pat, q, re.IGNORECASE)]
    return MetadataFilter(authority=found) if found else None


def apply_boost(chunks, flt: MetadataFilter, factor: float = 1.25):
    """'boost' mode: prefer matching authority without discarding everything else."""
    boosted = [c.model_copy(update={"score": c.score * (factor if flt.matches(c.metadata) else 1.0)}) for c in chunks]
    return sorted(boosted, key=lambda c: c.score, reverse=True)


REWRITE_PROMPT = """Rewrite the user's question into a precise, self-contained search query for
European financial-regulation documents (AML/CFT, KYC, CDD, sanctions). Expand vague terms into
the regulatory terminology, keep any article/regulation numbers exactly, do not add facts.
Return only the rewritten query.

Question: {q}"""

MULTI_QUERY_PROMPT = """Generate {n} different search queries that together cover the user's
question for retrieval over European AML/CFT regulatory documents. Use regulatory terminology,
keep identifiers exactly. One query per line, no numbering.

Question: {q}"""


class QueryProcessor:
    def __init__(self, cfg_query, cfg_retrieval, llm=None):
        self.cq, self.cr, self.llm = cfg_query, cfg_retrieval, llm

    def process(self, query: str) -> ProcessedQuery:
        norm = normalize(query) if self.cq.normalize else query
        pq = ProcessedQuery(original=query, normalized=expand_acronyms(norm))
        if self.cr.infer_filters:
            pq.inferred_filter = infer_filter(norm)
        if self.cq.rewriting and self.llm is not None:
            out = self.llm.generate(system="You rewrite search queries.", user=REWRITE_PROMPT.format(q=norm))
            pq.rewritten = out.text.strip().strip('"') or None
        if self.cq.multi_query and self.llm is not None:
            out = self.llm.generate(
                system="You generate search queries.",
                user=MULTI_QUERY_PROMPT.format(q=norm, n=self.cq.multi_query_n),
            )
            pq.variants = [ln.strip("-• ").strip() for ln in out.text.splitlines() if ln.strip()][: self.cq.multi_query_n]
        return pq
