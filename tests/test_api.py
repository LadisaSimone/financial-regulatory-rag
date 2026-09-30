from fastapi.testclient import TestClient

from regrag.api.app import create_app


def client(settings, pipeline):
    return TestClient(create_app(settings, pipeline))


def test_query_returns_citations(settings, pipeline):
    with client(settings, pipeline) as c:
        r = c.post("/query", json={"query": "What enhanced due diligence measures apply to high-risk customers?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["citations"] and body["disclaimer"]
        assert "x-request-id" in r.headers
        assert "chunks" not in body["retrieval"]  # debug info hidden by default


def test_validation_errors(settings, pipeline):
    with client(settings, pipeline) as c:
        assert c.post("/query", json={"query": "  "}).status_code == 422
        assert c.post("/query", json={"query": "x" * 2000}).status_code == 422
        assert c.post("/query", json={"query": "edd", "filters": {"authority": ["NASA"]}}).status_code == 422
        assert c.post("/query", json={"query": "edd rules", "unknown": 1}).status_code == 422


def test_retrieve_documents_health_metrics(settings, pipeline):
    with client(settings, pipeline) as c:
        assert c.post("/retrieve", json={"query": "beneficial ownership"}).json()["chunks"]
        assert len(c.get("/documents").json()["documents"]) == 2
        assert c.get("/health").json()["status"] == "ok"
        c.post("/query", json={"query": "politically exposed persons source of wealth"})
        m = c.get("/metrics").json()
        assert m["request_count"] >= 1
        assert "regrag_request_count" in c.get("/metrics?format=prometheus").text


def test_ingest_disabled_by_default(settings, pipeline):
    with client(settings, pipeline) as c:
        assert c.post("/ingest", json={}).status_code == 403
