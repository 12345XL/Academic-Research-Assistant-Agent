"""Opt-in real PostgreSQL/S3 tests, isolated from the application database/bucket.

RUN_STORAGE_INTEGRATION=1 python -m pytest tests/test_storage_integration.py -v
Only local services configured by scripts/local_services.py are accepted.
"""
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse
import uuid

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from research_agent.api import create_app
from research_agent.ingestion import ingest_qasper
from research_agent.service import PersistentEvidenceService
from research_agent.settings import Settings
from research_agent.storage import ImportBusyError, Repository, S3ObjectStore, StorageError

pytestmark = pytest.mark.skipif(os.getenv("RUN_STORAGE_INTEGRATION") != "1", reason="requires local PostgreSQL and S3")


@pytest.fixture
def infrastructure():
    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    base = Settings.from_env(values)
    assert urlparse(base.database_url).hostname == "127.0.0.1"
    assert urlparse(base.s3_endpoint_url).hostname == "127.0.0.1"
    suffix = uuid.uuid4().hex
    database, bucket = "research_test_" + suffix, "research-test-" + suffix
    admin_url = base.database_url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    settings = replace(base, database_url=base.database_url.rsplit("/", 1)[0] + "/" + database, s3_bucket=bucket)
    repo, store = Repository(settings), S3ObjectStore(settings)
    try:
        repo.migrate()
        assert repo.migrate() == []
        store.ensure_bucket()
        yield settings, repo, store
    finally:
        # Exact random resources created by this fixture only; retain main corpus.
        for page in store.client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            for item in page.get("Contents", []):
                store.client.delete_object(Bucket=bucket, Key=item["Key"])
        store.client.delete_bucket(Bucket=bucket)
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


def corpus(directory, text="alpha evidence"):
    paper = {"paper_id": "p1", "title": "Test paper", "abstract": "Abstract", "split": "train", "source": "qasper", "version": "qasper-v0.3"}
    paragraph = {key: value for key, value in paper.items() if key != "abstract"}
    paragraph.update(chunk_id="p1:s0:p0", section_name="Results", section_index=0, paragraph_index=0,
                     text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())
    for name, row in (("papers", paper), ("paragraphs", paragraph)):
        (directory / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
    (directory / "questions.jsonl").write_text("INVALID GOLD_SECRET")


def test_real_import_idempotence_object_repair_and_api(infrastructure, tmp_path):
    settings, repo, store = infrastructure
    corpus(tmp_path)
    first = ingest_qasper(repo, store, tmp_path)
    second = ingest_qasper(repo, store, tmp_path)
    assert first["outcome"] == "imported" and second["outcome"] == "skipped"
    assert second["revision"] == 1 and second["objects_written"] == 0
    assert repo.summary()["papers"] == repo.summary()["paragraphs"] == repo.summary()["objects"] == 1
    metadata = repo.get_paper_object("p1")
    content = store.read_bytes(metadata["object_key"])
    assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
    assert "GOLD_SECRET" not in content.decode()
    store.client.delete_object(Bucket=store.bucket, Key=metadata["object_key"])
    repaired = ingest_qasper(repo, store, tmp_path)
    assert repaired["outcome"] == "skipped" and repaired["objects_written"] == 1
    assert repaired["revision"] == 1
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/api/v1/papers").json()["total"] == 1
        assert client.get("/api/v1/papers?q=%").json()["total"] == 0
        repo.upsert_paper_metadata([{
            "paper_id": "p1", "arxiv_submitted_at": "2020-01-02T00:00:00+00:00",
            "journal_ref": "EMNLP 2020", "doi": "", "ccf_venue": "EMNLP",
            "ccf_level": "B", "ccf_catalog_url": "https://www.ccf.org.cn/Academic_Evaluation/AI/",
            "primary_category": "cs.CL", "categories": ["cs.CL", "cs.LG"],
            "pdf_url": "https://arxiv.org/pdf/2001.00001v1", "research_direction": "nlp",
            "metadata_source": "test", "metadata_checked_at": "2020-01-03T00:00:00+00:00",
        }])
        listed = client.get("/api/v1/papers?direction=nlp&sort=ccf_best").json()["items"][0]
        assert listed["arxiv_submitted_at"].startswith("2020-01-02")
        assert (listed["ccf_venue"], listed["ccf_level"]) == ("EMNLP", "B")
        assert listed["research_direction_label"] == "自然语言处理"
        assert listed["arxiv_pdf_url"].startswith("https://arxiv.org/pdf/")
        detail = client.get("/api/v1/papers/p1").json()
        assert (detail["ccf_venue"], detail["ccf_level"]) == ("EMNLP", "B")
        assert client.get("/api/v1/papers/p1/paragraphs").json()["total"] == 1
        answer = client.post("/api/v1/retrieve", json={"paper_id": "p1", "query": "alpha"})
        assert answer.status_code == 200
        assert answer.json()["citations"][0]["text"] == "alpha evidence"
        assert client.get("/api/v1/papers/p1/source").content == content
        jobs = client.get("/api/v1/ingestions").json()["items"]
        assert jobs[0]["reused"] and jobs[0]["status"] == "completed"


def test_publication_rollback_orphan_records_and_retry(infrastructure, tmp_path, monkeypatch):
    _, repo, store = infrastructure
    corpus(tmp_path)
    ingest_qasper(repo, store, tmp_path)
    corpus(tmp_path, "beta replacement")
    import research_agent.ingestion as module
    publish = module._publish

    def fail_after_publication_sql(*args):
        publish(*args)
        raise RuntimeError("Injected pre-commit failure")

    monkeypatch.setattr(module, "_publish", fail_after_publication_sql)
    with pytest.raises(StorageError):
        ingest_qasper(repo, store, tmp_path)
    assert repo.revision() == 1
    assert repo.get_paragraphs("p1")["items"][0]["text"] == "alpha evidence"
    assert repo.list_jobs()[0]["status"] == "failed"
    assert repo.summary()["object_states"]["orphaned"] == 1
    monkeypatch.setattr(module, "_publish", publish)
    assert ingest_qasper(repo, store, tmp_path)["revision"] == 2
    assert repo.get_paragraphs("p1")["items"][0]["text"] == "beta replacement"


def test_lock_exclusion_and_interrupted_job_recovery(infrastructure):
    _, repo, _ = infrastructure
    with repo.import_lock() as conn:
        with pytest.raises(ImportBusyError):
            with repo.import_lock():
                pass
        job = uuid.uuid4()
        conn.execute("INSERT INTO ingestion_jobs(job_id,source,status) VALUES (%s,'qasper','running')", (job,))
    with repo.import_lock():
        assert repo.list_jobs()[0]["status"] == "failed"
        assert repo.list_jobs()[0]["error"] == "interrupted_process"


def test_republish_checks_archived_content_and_keeps_current_snapshot(infrastructure, tmp_path):
    _, repo, store = infrastructure
    corpus(tmp_path)
    ingest_qasper(repo, store, tmp_path)
    original_version = repo.get_paper_object("p1")["version_id"]
    corpus(tmp_path, "beta replacement")
    ingest_qasper(repo, store, tmp_path)
    with repo.connect() as conn:
        row = conn.execute("SELECT paragraph_payload FROM paragraphs WHERE version_id=%s", (original_version,)).fetchone()
        damaged = row["paragraph_payload"] | {"text": "changed without matching checksum"}
        conn.execute("UPDATE paragraphs SET paragraph_payload=%s WHERE version_id=%s", (Jsonb(damaged), original_version))
    corpus(tmp_path)
    with pytest.raises(StorageError):
        ingest_qasper(repo, store, tmp_path)
    assert repo.revision() == 2
    assert repo.get_paragraphs("p1")["items"][0]["text"] == "beta replacement"


def test_live_service_revision_refresh_and_api_dependency_errors(infrastructure, tmp_path, monkeypatch):
    settings, repo, store = infrastructure
    corpus(tmp_path)
    ingest_qasper(repo, store, tmp_path)
    service = PersistentEvidenceService(repo)
    assert service.retrieve("alpha", "p1")["citations"]
    corpus(tmp_path, "beta replacement")
    ingest_qasper(repo, store, tmp_path)
    assert service.retrieve("alpha", "p1")["citations"] == []
    assert service.retrieve("beta", "p1")["citations"][0]["text"] == "beta replacement"
    with TestClient(create_app(settings=settings)) as client:
        def unavailable(*args, **kwargs):
            raise StorageError("Sensitive details that must not reach clients")
        monkeypatch.setattr(S3ObjectStore, "read_bytes", unavailable)
        source = client.get("/api/v1/papers/p1/source")
        assert source.status_code == 503 and "Sensitive" not in source.text
        assert client.post("/api/v1/retrieve", json={"paper_id": "p1", "query": "beta"}).status_code == 200
        monkeypatch.setattr(Repository, "connect", unavailable)
        failed = client.post("/api/v1/retrieve", json={"paper_id": "p1", "query": "beta"})
        assert failed.status_code == 503 and "Sensitive" not in failed.text
        assert client.get("/ready").status_code == 503
    monkeypatch.undo()
