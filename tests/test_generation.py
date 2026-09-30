from regrag.generation.citations import parse_llm_answer, validate_citations
from regrag.generation.context import ContextBuilder
from regrag.schemas import LLMAnswer, RawCitation, RetrievedChunk


def rc(cid, doc="d1", text="Some regulatory text about due diligence.", p=1):
    return RetrievedChunk(chunk_id=cid, text=text, score=1.0, metadata={
        "document_id": doc, "title": "T", "authority": "EBA", "page_start": p, "page_end": p,
        "section": "4.2 EDD", "publication_date": "2021-03-01", "source_url": "https://x"})


def test_parse_handles_fences_and_garbage():
    a = parse_llm_answer('```json\n{"answer": "x", "citations": [{"chunk_id": "c1"}], "insufficient_evidence": false}\n```')
    assert a.citations[0].chunk_id == "c1"
    bad = parse_llm_answer("I think the answer is yes")
    assert bad.insufficient_evidence and not bad.citations


def test_citations_must_come_from_context():
    ctx = [rc("c1"), rc("c2")]
    ans = LLMAnswer(answer="x", citations=[RawCitation(chunk_id="c1"), RawCitation(chunk_id="invented:99")])
    valid, invalid = validate_citations(ans, ctx)
    assert [c.chunk_id for c in valid] == ["c1"] and invalid == ["invented:99"]
    assert valid[0].label == "[EBA — T, §4.2 EDD, p.1]"


def test_context_builder_budget_and_diversity():
    long = " ".join(["word"] * 400)
    chunks = [rc(f"a{i}", doc="A", text=long + str(i), p=i) for i in range(6)] + [rc("b0", doc="B", text="other text", p=1)]
    cb = ContextBuilder(max_tokens=1300, max_chunks_per_document=2)
    ctx = cb.build(chunks)
    docs = [c.document_id for c in ctx.chunks]
    assert docs.count("A") <= 2 and "B" in docs
    assert ctx.token_count <= 1300
    assert "[chunk_id: a0]" in ctx.text and "p.0" in ctx.text


def test_end_to_end_answer_is_grounded(pipeline):
    r = pipeline.answer("What enhanced due diligence measures apply to high-risk customers?")
    assert not r.insufficient_evidence
    assert r.citations and all(c.document_id in {"doc_a", "doc_b"} for c in r.citations)
    assert r.disclaimer


def test_unanswerable_abstains(pipeline):
    r = pipeline.answer("What is Apple's current dividend yield in 2024?")
    assert r.insufficient_evidence and not r.citations
