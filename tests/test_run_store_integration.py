"""Real PostgreSQL run recovery in a random schema, no corpus/model/S3 mutation.

RUN_STORAGE_INTEGRATION=1 python -m pytest tests/test_run_store_integration.py
"""
from pathlib import Path
import os
from urllib.parse import urlparse
import uuid

from dotenv import dotenv_values
import psycopg
from psycopg import sql
import pytest

from research_agent.run_store import PostgresRunStore, RunStoreBusyError, RunStoreCancelledError, RunStoreConflictError, RunStoreError
from research_agent.settings import Settings
from research_agent.storage import Repository
from test_run_store import running_snapshot, terminal_snapshot

pytestmark = pytest.mark.skipif(os.getenv("RUN_STORAGE_INTEGRATION") != "1", reason="requires local PostgreSQL")


@pytest.fixture
def run_repository(monkeypatch, tmp_path):
    # Same local-only configuration guard as storage integration tests, but only
    # a randomly named schema is needed; existing corpus tables stay untouched.
    settings = Settings.from_env(dotenv_values(Path(__file__).resolve().parents[1] / ".env"))
    assert urlparse(settings.database_url).hostname == "127.0.0.1"
    schema = "research_run_test_" + uuid.uuid4().hex
    monkeypatch.setattr("research_agent.run_store.RUN_WORKER_LOCK_ID", int(uuid.uuid4().hex[:14], 16))
    base = Repository(settings)
    with base.connect() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    class SchemaRepository(Repository):
        def connect(self, *, autocommit=False):
            conn = super().connect(autocommit=autocommit)
            conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            if not autocommit:
                conn.commit()
            return conn

    repo = SchemaRepository(settings)
    migration_dir = Path(__file__).resolve().parents[1] / "src/research_agent/migrations"
    names = ["005_answer_runs.sql", "006_answer_feedback.sql"]
    for name in names:
        (tmp_path / name).write_bytes((migration_dir / name).read_bytes())
    try:
        assert repo.migrate(tmp_path) == names
        assert repo.migrate(tmp_path) == []
        yield repo
    finally:
        with base.connect() as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_single_worker_restart_marks_only_running_and_never_replays(run_repository):
    store = PostgresRunStore(run_repository)
    store.start()
    blocked = PostgresRunStore(run_repository)
    try:
        running, finished = uuid.uuid4().hex, uuid.uuid4().hex
        store.create(running, "alice", "p1", {"mode": "bm25", "question": "PRIVATE"})
        store.save(running, "alice", running_snapshot())
        store.create(finished, "alice", "p1", {})
        completed = store.save(finished, "alice", terminal_snapshot())
        with pytest.raises(RunStoreBusyError):
            blocked.start()
        assert store.get(running, "alice")["state"] == "running"
    finally:
        blocked.close()
        store.close()
    restarted = PostgresRunStore(run_repository)
    try:
        restarted.start()
        record = restarted.get(running, "alice")
        assert (record["state"], record["reason"], record["revision"]) == ("interrupted", "server_restarted", 3)
        assert record["snapshot"]["run"]["state"] == "interrupted"
        assert record["snapshot"]["run"]["stages"][0]["status"] == "stopped"
        assert record["snapshot"]["run"]["attempts"][0]["status"] == "stopped"
        assert record["snapshot"]["run"]["budget"]["model_calls"] == 1
        assert restarted.get(finished, "alice") == completed
        with pytest.raises(RunStoreConflictError):
            restarted.save(running, "alice", terminal_snapshot())
    finally:
        restarted.close()


def test_db_owner_cancel_and_terminal_guards(run_repository):
    store = PostgresRunStore(run_repository)
    store.start()
    try:
        own, other = uuid.uuid4().hex, uuid.uuid4().hex
        store.create(own, "alice", "p1", {})
        store.create(other, "bob", "p2", {})
        assert store.get(own, "bob") is None and store.request_cancel(own, "bob") is None
        assert [r["run_id"] for r in store.list("alice")] == [own]
        assert store.list("alice", allowed_paper_ids={"p2"}) == []
        assert store.list("alice", allowed_paper_ids=set()) == []
        with pytest.raises(RunStoreConflictError):
            store.save(own, "bob", terminal_snapshot())
        assert store.request_cancel(own, "alice")["cancel_requested"]
        current = store.save(own, "alice", running_snapshot())
        assert current["cancel_requested"] and current["revision"] == 3
        with pytest.raises(RunStoreCancelledError):
            store.save(own, "alice", terminal_snapshot())
        assert store.get(own, "alice")["state"] == "running"
        final = store.save(own, "alice", terminal_snapshot("cancelled", "interrupted"))
        assert store.request_cancel(own, "alice") == final
        with pytest.raises(RunStoreConflictError):
            store.save(own, "alice", running_snapshot())
        with pytest.raises(RunStoreConflictError):
            store.create(own, "alice", "p1", {})
        assert store.get(own, "alice") == final  # duplicate did not poison autocommit session
    finally:
        store.close()


def test_session_loss_fails_closed_without_reconnecting(run_repository, monkeypatch):
    store = PostgresRunStore(run_repository)
    store.start()
    try:
        run_id = uuid.uuid4().hex
        store.create(run_id, "alice", "p1", {})
        pid = store._conn.info.backend_pid
        with run_repository.connect(autocommit=True) as killer:
            assert killer.execute("SELECT pg_terminate_backend(%s) AS killed", (pid,)).fetchone()["killed"]
        def unexpected_reconnect(**kwargs):
            pytest.fail("Run store must not reconnect a lost advisory-lock session")
        monkeypatch.setattr(run_repository, "connect", unexpected_reconnect)
        for action in (lambda: store.get(run_id, "alice"), lambda: store.save(run_id, "alice", terminal_snapshot()), store.start):
            with pytest.raises(RunStoreError):
                action()
    finally:
        store.close()
