"""LLM-as-a-judge metrics (faithfulness, answer relevance, context relevance).

LABELED AS LLM-JUDGED in every report: these are noisy estimates, not ground truth. Use them
to compare configurations, and spot-check a sample manually. Deterministic metrics in
metrics.py are preferred wherever possible.
"""

from __future__ import annotations

import json

from regrag.generation.llm import LLMProvider

JUDGE_PROMPT = """You are grading a retrieval-augmented answer about financial regulation.

EVIDENCE:
{context}

QUESTION: {question}
ANSWER: {answer}

Score each from 0 to 1:
- faithfulness: fraction of the answer's factual claims that are directly supported by the EVIDENCE.
- answer_relevance: does the answer address the question?
- context_relevance: fraction of the EVIDENCE that is relevant to the question.
Return JSON: {{"faithfulness": x, "answer_relevance": y, "context_relevance": z}}"""


def judge(llm: LLMProvider, question: str, answer: str, context: str) -> dict[str, float]:
    res = llm.generate("You are a strict, calibrated evaluator.",
                       JUDGE_PROMPT.format(context=context[:12000], question=question, answer=answer),
                       json_mode=True)
    try:
        d = json.loads(res.text)
        return {f"llm_judge_{k}": float(d[k]) for k in ("faithfulness", "answer_relevance", "context_relevance")}
    except Exception:
        return {}
