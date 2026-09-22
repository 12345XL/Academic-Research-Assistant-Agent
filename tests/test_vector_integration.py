"""Synthetic encoder here verifies DB governance; full QASPER report uses real BGE."""
import hashlib
import json
import os

from fastapi.testclient import TestClient
import pytest

from test_storage_integration import infrastructure, corpus
from research_agent.api import create_app
from research_agent.embeddings import VectorUnavailableError, collection_id
from research_agent.ingestion import ingest_qasper
from research_agent.service import CorpusChangedError, CorpusIntegrityError, PersistentEvidenceService
from research_agent.vector_store import VectorStore

pytestmark = pytest.mark.skipif(os.getenv("RUN_STORAGE_INTEGRATION") != "1", reason="requires local storage")


def unit(axis=0):
    values = [0.0] * 384
    values[axis] = 1.0
    return values


class Encoder:
    def encode_paragraphs(self, texts):
        return [[unit(0), unit(1)] for text in texts]

    def encode_queries(self, queries):
        return [unit(1) for query in queries]


def test_exact_vector_retrieval_readiness_version_scope_and_fact_check(infrastructure, tmp_path):
    settings, repo, store = infrastructure
    corpus(tmp_path)
    # A second paper has an equally close vector; it must never enter p1 retrieval.
    for name in ("papers", "paragraphs"):
        path = tmp_path / f"{name}.jsonl"
        record = json.loads(path.read_text())
        other = {**record, "paper_id": "p2"}
        if name == "paragraphs":
            other["chunk_id"] = "p2:s0:p0"
        path.write_text(json.dumps(record) + "\n" + json.dumps(other) + "\n")
    ingest_qasper(repo, store, tmp_path)
    vectors = VectorStore(repo)
    with pytest.raises(VectorUnavailableError):
        vectors.search(unit(), "p1", 1, 5)
    with TestClient(create_app(settings=settings)) as client:
        assert client.post("/api/v1/retrieve", json={"paper_id": "p1", "query": "x", "mode": "dense"}).status_code == 503
    built = vectors.build(Encoder())
    assert (built["paragraphs"], built["windows"]) == (2, 4)
    assert vectors.build(Encoder())["outcome"] == "reused"
    found = vectors.search(unit(1), "p1", 1, 5)
    assert len(found) == 1 and found[0]["score"] == pytest.approx(1)
    assert found[0]["chunk_id"] == "p1:s0:p0"
    service = PersistentEvidenceService(repo, Encoder(), vectors)
    for mode in ("dense", "hybrid"):
        result = service.retrieve("no_lexical_match", "p1", mode=mode)
        assert result["citations"][0]["text"] == "alpha evidence"
        assert result["trace"]["fact_check"] == "database"
    with repo.connect() as conn:
        conn.execute("UPDATE paragraphs SET paragraph_payload=jsonb_set(paragraph_payload,'{text}','\"tampered\"') "
                     "WHERE chunk_id='p1:s0:p0'")
    with pytest.raises(CorpusIntegrityError):
        service.retrieve("alpha", "p1", mode="hybrid")
    corpus(tmp_path, "changed evidence")
    ingest_qasper(repo, store, tmp_path)
    assert repo.revision() == 2
    with pytest.raises(CorpusChangedError):
        vectors.search(unit(), "p1", 1, 5)
    with pytest.raises(VectorUnavailableError):
        service.retrieve("changed", "p1", mode="hybrid")
    assert service.retrieve("changed", "p1")["citations"][0]["text"] == "changed evidence"


def test_interrupted_build_resumes_atomic_paragraphs(infrastructure, tmp_path):
    _, repo, store = infrastructure
    corpus(tmp_path)
    path = tmp_path / "paragraphs.jsonl"
    paragraph = json.loads(path.read_text())
    second = {**paragraph, "chunk_id": "p1:s0:p1", "paragraph_index": 1}
    path.write_text(json.dumps(paragraph) + "\n" + json.dumps(second) + "\n")
    ingest_qasper(repo, store, tmp_path)

    class Fails(Encoder):
        calls = 0
        def encode_paragraphs(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("inference interrupted")
            return super().encode_paragraphs(texts)

    vectors = VectorStore(repo)
    with pytest.raises(RuntimeError):
        vectors.build(Fails(), batch_size=1)
    assert vectors.status(1)["state"] == "building"
    with pytest.raises(VectorUnavailableError):
        vectors.search(unit(), "p1", 1, 5)
    resumed = vectors.build(Encoder(), batch_size=1)
    assert resumed["resumed_paragraphs"] == 1
    assert resumed["paragraphs"] == 2 and resumed["windows"] == 4
