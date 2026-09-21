"""Read-only full-corpus storage/API check against the frozen P1 retrieved IDs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.dataset import read_jsonl
from research_agent.ingestion import build_documents
from research_agent.retrieval import BM25Index
from research_agent.settings import Settings
from research_agent.storage import Repository, S3ObjectStore


def fetch(path: str, payload: dict | None = None):
    request = urllib.request.Request("http://127.0.0.1:8011" + path,
                                    data=json.dumps(payload).encode() if payload else None,
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    load_dotenv(ROOT / ".env")
    settings = Settings.from_env()
    repo, store = Repository(settings), S3ObjectStore(settings)
    revision, papers, paragraphs = repo.load_snapshot()
    counts = repo.summary()
    manifest, documents = build_documents(papers, paragraphs)
    assert len(papers) == counts["papers"] == 1169
    assert len(paragraphs) == counts["paragraphs"] == 61317
    assert manifest == counts["manifest_sha256"]
    assert store.health()["status"] == "ready"
    frozen = json.loads((ROOT / "reports/p1_retrieval_dev.json").read_text())
    questions = {q["question_id"]: q for q in read_jsonl(ROOT / "data/processed/questions.jsonl")}
    index = BM25Index(paragraphs)
    mismatches = []
    for case in frozen["cases"]:
        question = questions[case["question_id"]]
        actual = [hit.paragraph["chunk_id"] for hit in index.search(question["question"], question["paper_id"], 5)]
        if actual != case["retrieved_chunk_ids"]:
            mismatches.append(case["question_id"])
    assert not mismatches, f"P1 ranking mismatch for {len(mismatches)} cases"
    ready = fetch("/ready")
    assert ready["database"]["status"] == ready["object_store"]["status"] == "ready"
    first = papers[0]
    result = fetch("/api/v1/retrieve", {"paper_id": first["paper_id"], "query": "What method and dataset are used?", "top_k": 5})
    assert result["citations"] and result["trace"]["storage"] == "postgres"
    assert all(item["paper_id"] == first["paper_id"] for item in result["citations"])
    source = fetch(f"/api/v1/papers/{first['paper_id']}/source")
    assert source["paper"] == first and "annotations" not in source and "qas" not in source
    source_metadata = repo.get_paper_object(first["paper_id"])
    assert hashlib.sha256(store.read_bytes(source_metadata["object_key"])).hexdigest() == source_metadata["sha256"]
    with repo.connect() as conn:
        postgres_version = conn.execute("SHOW server_version").fetchone()["server_version"]
        migrated = [r["name"] for r in conn.execute("SELECT name FROM schema_migrations ORDER BY name")]
    report = {
        "stage": "P2A", "verified_at": datetime.now(timezone.utc).isoformat(),
        "postgres_version": postgres_version, "migrations": migrated,
        "corpus_revision": revision, "manifest_sha256": manifest,
        "counts": {"papers": len(papers), "paragraphs": len(paragraphs), "paper_objects": len(documents)},
        "p1_regression": {"compared_questions": len(frozen["cases"]), "ranking_mismatches": mismatches,
                          "metrics_unchanged": frozen["metrics"], "protocol": frozen["protocol"]},
        "http": {"ready": True, "retrieval_fact_check": result["trace"]["fact_check"],
                 "source_download": True, "source_checksum": True},
        "boundaries": ["Known-paper BM25 evidence retrieval only; no new answer-quality claim",
                       "Questions read by this offline verifier, never by the online service",
                       "Local single-node PostgreSQL and S3, no production HA/load validation"],
    }
    target = ROOT / "reports/p2a_storage_regression.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
