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


def test_document_reference_inference():
    f = infer_filter("What does Article 18 of Directive 2015/849 require?")
    assert f.document_id == ["eu_directive_2015_849_amld4"] and not f.authority
    assert infer_filter("When must CDD be applied under the AMLR?").document_id == ["eu_regulation_2024_1624_amlr"]
    assert infer_filter("What does FATF Recommendation 10 require?").document_id == ["fatf_recommendations"]
    assert infer_filter("What does the EBA say about PEPs?").authority == ["EBA"]


def test_rank_boost_moves_matches_up_without_dropping():
    from regrag.retrieval.query import apply_boost

    items = [rc(f"c{i}", doc="target" if i == 6 else "other") for i in range(8)]
    out = apply_boost(items, MetadataFilter(document_id=["target"]), ranks=5)
    assert [c.chunk_id for c in out][:3] == ["c0", "c1", "c6"] and len(out) == 8  # rank 7 -> 2


def test_contextual_index_text():
    from regrag.schemas import Chunk

    c = Chunk(chunk_id="x", document_id="d", text="body", title="Directive (EU) 2015/849", authority="EC",
              document_type="directive", section="Article 18", page_start=1, page_end=1,
              chunking_strategy="section", token_count=1, position=0)
    assert c.index_text() == "body"
    assert c.index_text(True).startswith("EC — Directive (EU) 2015/849 — Article 18\n")
