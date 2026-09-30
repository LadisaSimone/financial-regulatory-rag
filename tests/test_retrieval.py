from regrag.retrieval.bm25 import bm25_tokenize
from regrag.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from regrag.retrieval.query import expand_acronyms, infer_filter
from regrag.schemas import MetadataFilter, RetrievedChunk


def rc(cid, doc="d", p=(1, 1), text=None, score=1.0, r="x"):
    return RetrievedChunk(chunk_id=cid, text=text or f"text {cid} unique {cid}", score=score, retriever=r,
                          metadata={"document_id": doc, "page_start": p[0], "page_end": p[1]})


def test_bm25_tokenizer_keeps_identifiers():
    toks = bm25_tokenize("What does Article 13(1) of Directive 2015/849 say about PEPs?")
    assert "13(1)" in toks and "2015/849" in toks and "peps" in toks and "what" not in toks


def test_rrf_prefers_items_ranked_by_both():
    dense = [rc("a", r="dense"), rc("b", r="dense"), rc("c", r="dense")]
    sparse = [rc("c", r="bm25"), rc("a", r="bm25"), rc("d", r="bm25")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert [f.chunk_id for f in fused][:2] == ["a", "c"]
    assert fused[0].retriever == "bm25+dense"


def test_dedup_by_id_and_text():
    same = "enhanced due diligence measures apply to high risk customers in all cases"
    items = [rc("a", text=same), rc("a", text=same), rc("b", doc="other", text=same), rc("c")]
    out = deduplicate(items, text_similarity=0.9)
    assert [c.chunk_id for c in out] == ["a", "c"]


def test_filter_inference_is_conservative():
    assert infer_filter("What does the EBA say about EDD?").authority == ["EBA"]
    assert infer_filter("What is enhanced due diligence?") is None


def test_acronym_expansion():
    assert "enhanced due diligence" in expand_acronyms("EDD for PEPs")


def test_metadata_filter_matches():
    f = MetadataFilter(authority=["EBA"], publication_date_gte="2020-01-01")
    assert f.matches({"authority": "EBA", "publication_date": "2021-03-01"})
    assert not f.matches({"authority": "FATF", "publication_date": "2021-03-01"})
    assert not f.matches({"authority": "EBA", "publication_date": "2012-01-01"})


def test_hybrid_end_to_end_retrieval(pipeline):
    chunks, _ = pipeline.retrieve("enhanced due diligence high-risk customers source of wealth")
    assert chunks and chunks[0].document_id == "doc_a"
    chunks, _ = pipeline.retrieve("Recommendation 24 beneficial ownership legal persons")
    assert chunks[0].document_id == "doc_b"


def test_filter_restricts_results(pipeline):
    chunks, _ = pipeline.retrieve("beneficial owner", flt=MetadataFilter(authority=["EBA"]))
    assert chunks and all(c.metadata["authority"] == "EBA" for c in chunks)
