"""Evaluation dataset schema.

Ground truth is expressed at (document_id, page) level — `relevant_evidence` — so the SAME
benchmark can score every chunking strategy (chunk ids differ between strategies; pages don't).
`relevant_chunk_ids` is optional and strategy-specific.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

QuestionType = Literal[
    "factual", "multi_section", "cross_document", "terminology", "exact_reference",
    "ambiguous", "unanswerable", "adversarial",
]


class Evidence(BaseModel):
    document_id: str
    pages: list[int]


class EvalItem(BaseModel):
    id: str
    question: str
    answerable: bool = True
    question_type: QuestionType = "factual"
    relevant_document_ids: list[str] = []
    relevant_evidence: list[Evidence] = []
    relevant_chunk_ids: list[str] = []
    reference_answer: str | None = None
    tags: list[str] = []
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    validated: bool = False  # set true once a human checked the ground truth

    @model_validator(mode="after")
    def _check(self):
        if self.answerable and not (self.relevant_evidence or self.relevant_document_ids or self.relevant_chunk_ids):
            raise ValueError(f"{self.id}: answerable question without any ground truth")
        if not self.relevant_document_ids and self.relevant_evidence:
            self.relevant_document_ids = sorted({e.document_id for e in self.relevant_evidence})
        return self


def load_dataset(path: Path) -> list[EvalItem]:
    items = [EvalItem.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in eval dataset")
    return items
