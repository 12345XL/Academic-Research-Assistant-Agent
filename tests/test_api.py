import hashlib
import json

from fastapi.testclient import TestClient

from research_agent.api import create_app


def write_corpus(directory):
    papers = [{"paper_id": pid, "title": title, "abstract": "", "split": "train", "source": "qasper", "version": "test"}
              for pid, title in [("p1", "Alpha study"), ("p2", "Beta study")]]
    paragraphs = [{**p, "chunk_id": p["paper_id"] + ":s0:p0", "text": "alpha evidence", "section_name": "Results",
                   "section_index": 0, "paragraph_index": 0,
                   "text_sha256": hashlib.sha256(b"alpha evidence").hexdigest()} for p in papers]
    for name, rows in [("papers", papers), ("paragraphs", paragraphs)]:
        (directory / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    # The online service must neither parse nor expose this offline file.
    (directory / "questions.jsonl").write_text("INVALID JSON GOLD_SECRET_ONLY", encoding="utf-8")


def test_real_routes_scope_citations_and_gold_isolation(tmp_path):
    write_corpus(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["status"] == "ready"
        papers = client.get("/api/v1/papers?q=Alpha").json()
        assert papers["total"] == 1
        assert client.get("/api/v1/papers/p1").json()["title"] == "Alpha study"
        result = client.post("/api/v1/retrieve", json={"paper_id": "p1", "query": "alpha"})
        assert result.status_code == 200
        body = result.json()
        assert body["mode"] == "evidence_only"
        assert body["trace"]["model_calls"] == 0
        assert [c["chunk_id"] for c in body["citations"]] == ["p1:s0:p0"]
        assert body["citations"][0]["text"] == "alpha evidence"
        assert body["citations"][0]["text_sha256"] == hashlib.sha256(b"alpha evidence").hexdigest()
        assert "GOLD_SECRET" not in result.text
        no_hit = client.post("/api/v1/retrieve", json={"paper_id": "p1", "query": "GOLD_SECRET_ONLY"}).json()
        assert no_hit["status"] == "no_lexical_match"
        assert no_hit["citations"] == []
        assert "unanswerable" not in no_hit  # No lexical match is not a refusal verdict.


def test_input_and_missing_data_errors(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["status"] == "data_missing"
        assert client.get("/api/v1/papers").status_code == 503
    write_corpus(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/v1/papers/missing").status_code == 404
        assert client.post("/api/v1/retrieve", json={"paper_id": "missing", "query": "x"}).status_code == 404
        for payload in [{"paper_id": "p1", "query": "  "}, {"paper_id": "  ", "query": "x"},
                        {"paper_id": "p1", "query": "x", "top_k": 51}]:
            assert client.post("/api/v1/retrieve", json=payload).status_code == 422
