import pytest

from regrag.evaluation.dataset import EvalItem, Evidence, load_dataset
from regrag.evaluation.metrics import retrieval_metrics
from regrag.evaluation.runner import run_evaluation
from regrag.schemas import RetrievedChunk


def rc(cid, doc, p):
    return RetrievedChunk(chunk_id=cid, text="t", score=1, metadata={"document_id": doc, "page_start": p, "page_end": p})


def test_retrieval_metrics_page_level():
    item = EvalItem(id="q", question="x", relevant_evidence=[Evidence(document_id="A", pages=[3, 4])])
    ret = [rc("1", "B", 1), rc("2", "A", 3), rc("3", "A", 4)]
    m = retrieval_metrics(ret, item)
    assert m["recall@1"] == 0 and m["recall@3"] == 1.0
    assert m["mrr"] == 0.5 and m["hit@3"] == 1.0
    assert abs(m["precision@3"] - 2 / 3) < 1e-9
    assert 0 < m["ndcg@10"] < 1


def test_answerable_requires_ground_truth():
    with pytest.raises(ValueError):
        EvalItem(id="q", question="x", answerable=True)


def test_repo_eval_dataset_is_valid():
    from regrag.config import ROOT

    items = load_dataset(ROOT / "eval" / "questions.jsonl")
    assert any(not i.answerable for i in items)


def test_run_evaluation_offline(pipeline, settings):
    items = [
        EvalItem(id="q1", question="enhanced due diligence high-risk customers",
                 relevant_evidence=[Evidence(document_id="doc_a", pages=[2])]),
        EvalItem(id="q2", question="What is Apple's current dividend?", answerable=False),
    ]
    res = run_evaluation(settings, name="test", pipeline=pipeline, items=items)
    assert res["retrieval_metrics"]["hit@5"] == 1.0
    assert res["generation_metrics"]["abstention_accuracy"] == 1.0
    assert (settings.path("results_dir") / f"{res['experiment_id']}.json").exists()
