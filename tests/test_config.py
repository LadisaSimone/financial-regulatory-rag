from regrag.config import load_settings


def test_env_override(monkeypatch):
    monkeypatch.setenv("REGRAG__RETRIEVAL__FINAL_TOP_K", "3")
    monkeypatch.setenv("REGRAG__RERANKER__ENABLED", "true")
    s = load_settings()
    assert s.retrieval.final_top_k == 3 and s.reranker.enabled is True


def test_collection_name_versioned():
    s = load_settings(overrides={"chunking": {"strategy": "fixed"}, "vector_store": {"collection_version": 2}})
    assert s.collection_name() == "regrag_fixed_text_embedding_3_small_v2"
