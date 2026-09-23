"""Run ownership, immutability and content minimization without model calls."""
from concurrent.futures import ThreadPoolExecutor
import json
import uuid

import pytest

from research_agent.run_store import (
    MemoryRunStore, PostgresRunStore, RunStoreCancelledError, RunStoreConflictError, RunStoreError,
    sanitize_snapshot,
)
from research_agent.storage import StorageError


def running_snapshot():
    return {
        "run": {
            "trace_id": "abc123", "state": "running", "reason": None, "terminal_stage": None,
            "latency_ms": 12.5,
            "stages": [{"name": "generate", "status": "running", "latency_ms": None}],
            "attempts": [{"name": "generate", "status": "running", "latency_ms": None, "attempt": 0}],
            "limits": {"deadline_seconds": 90.0, "max_model_calls": 4, "max_prompt_chars": 60000,
                       "max_completion_tokens": 2048, "max_repairs": 1},
            "budget": {"model_calls": 1, "completion_tokens_reserved": 2048,
                       "reported_prompt_tokens": 3, "reported_completion_tokens": 4, "usage_unknown_calls": 1},
        },
        "generation": {
            "prompt_version": "paper-claims-v2", "answer_language": "zh", "model_calls": 1,
            "context": {"budget_unit": "unicode_characters", "limit": 18000, "used": 100,
                        "included_ids": ["p1:s0:p0"], "skipped_ids": ["p1:s0:p1"]},
            "calls": [{"stage": "generate", "status": "started", "attempt": 0,
                       "requested_model": "deepseek-flash", "usage": {"prompt_tokens": 3},
                       "latency_ms": 1.2, "output_sha256": "a" * 64}],
            "checks": {"citation_integrity": "not_run", "semantic_support": "not_run"},
            "evidence_sha256": "b" * 64,
        },
        "access": {"principal_id": "alice", "mode": "bearer_policy"},
    }


def terminal_snapshot(reason="published", state="completed"):
    snapshot = running_snapshot()
    snapshot["run"].update(reason=reason, state=state, terminal_stage="publish")
    snapshot["run"]["stages"][0]["status"] = "completed"
    return snapshot


@pytest.fixture
def store():
    store = MemoryRunStore()
    store.start()
    yield store
    store.close()


def test_owner_scope_filter_cancel_and_terminal_immutability(store):
    own, other, hidden = [uuid.uuid4().hex for _ in range(3)]
    store.create(own, "alice", "p1", {"mode": "bm25"})
    store.create(other, "bob", "p1", {})
    store.create(hidden, "alice", "p2", {})
    assert store.get(own, "bob") is None
    assert store.request_cancel(own, "bob") is None
    assert len(store.list("alice")) == 2
    assert [r["run_id"] for r in store.list("alice", allowed_paper_ids={"p1"})] == [own]
    assert store.list("alice", allowed_paper_ids=set()) == []
    with pytest.raises(RunStoreConflictError):
        store.save(own, "bob", terminal_snapshot())
    cancelled = store.request_cancel(own, "alice")
    assert cancelled["cancel_requested"] and cancelled["state"] == "running"
    assert cancelled["revision"] == 2
    assert store.request_cancel(own, "alice")["revision"] == 2
    with pytest.raises(RunStoreCancelledError):
        store.save(own, "alice", terminal_snapshot())
    assert store.get(own, "alice")["state"] == "running"
    terminal = store.save(own, "alice", terminal_snapshot("cancelled", "interrupted"))
    assert terminal["cancel_requested"] and terminal["revision"] == 3
    assert store.request_cancel(own, "alice") == terminal
    for snapshot in (running_snapshot(), terminal_snapshot(), terminal_snapshot("cancelled", "interrupted")):
        with pytest.raises(RunStoreConflictError):
            store.save(own, "alice", snapshot)
    assert store.get(own, "alice") == terminal


def test_nested_allowlists_preserve_controls_but_no_content_or_credentials(store):
    run_id = uuid.uuid4().hex
    snapshot = running_snapshot()
    secret = "RAW PRIVATE QUESTION EVIDENCE DRAFT ANSWER TOKEN"
    snapshot.update(question=secret, citations=[{"text": secret}], answer=secret)
    snapshot["run"].update(question=secret, provider_error=secret)
    snapshot["run"]["limits"]["api_key"] = secret
    snapshot["run"]["budget"]["usage_unknown_calls"] = {"token": secret}
    snapshot["run"]["stages"][0]["question"] = secret
    snapshot["generation"].update(draft=secret, answer=secret)
    snapshot["generation"]["context"].update(evidence=secret, included_ids=["p1:s0:p0", {"text": secret}])
    snapshot["generation"]["calls"][0].update(provider_body=secret, token=secret, model={"key": secret})
    snapshot["generation"]["calls"][0]["usage"]["secret"] = secret
    snapshot["generation"]["checks"]["question"] = secret
    snapshot["access"]["token"] = secret
    record = store.create(run_id, "alice", "p1", {"mode": "bm25", "model": "deepseek-flash", "question": secret})
    assert record["metadata"] == {"mode": "bm25", "model": "deepseek-flash"}
    record = store.save(run_id, "alice", snapshot)
    assert secret not in json.dumps(record)
    clean = record["snapshot"]
    assert clean["access"] == {"principal_id": "alice", "mode": "bearer_policy"}
    assert clean["run"]["limits"]["max_model_calls"] == 4
    assert clean["run"]["attempts"][0]["attempt"] == 0
    assert clean["generation"]["calls"][0]["status"] == "started"
    assert clean["generation"]["context"]["included_ids"] == ["p1:s0:p0"]
    # Callers cannot mutate saved history after a return or save.
    record["metadata"]["mode"] = "dense"
    snapshot["run"]["limits"]["max_model_calls"] = 100
    assert store.get(run_id, "alice")["metadata"]["mode"] == "bm25"
    assert store.get(run_id, "alice")["snapshot"]["run"]["limits"]["max_model_calls"] == 4


def test_concurrent_cancel_and_save_never_lose_cancel_request(store):
    run_id = uuid.uuid4().hex
    store.create(run_id, "alice", "p1", {})
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(store.save, run_id, "alice", running_snapshot()) for _ in range(20)]
        futures.append(executor.submit(store.request_cancel, run_id, "alice"))
        for future in futures:
            future.result()
    record = store.get(run_id, "alice")
    assert record["cancel_requested"] and record["revision"] == 22


@pytest.mark.parametrize("snapshot", [None, {}, {"run": []}, {"run": {"state": {}}},
    {"run": {"state": "failed", "reason": "SECRET"}},
    {"run": {"state": "running", "reason": "published"}}])
def test_invalid_snapshots_have_only_fixed_public_error(snapshot):
    with pytest.raises(RunStoreConflictError) as error:
        sanitize_snapshot(snapshot)
    assert str(error.value) == "Answer run cannot be changed"


def test_malformed_allowlisted_nested_fields_are_discarded():
    snapshot = running_snapshot()
    snapshot["run"]["stages"] = [{"name": {"secret": "key"}}]
    snapshot["run"]["limits"]["deadline_seconds"] = float("nan")
    snapshot["generation"]["calls"][0]["usage"]["prompt_tokens"] = True
    clean = sanitize_snapshot(snapshot)
    assert clean["run"]["stages"] == []
    assert "deadline_seconds" not in clean["run"]["limits"]
    assert clean["generation"]["calls"][0]["usage"] == {}


def test_lifecycle_duplicate_and_limit_validation(store):
    run_id = uuid.uuid4().hex
    store.create(run_id, "alice", "p1", {})
    with pytest.raises(RunStoreConflictError):
        store.create(run_id, "alice", "p1", {})
    for value in (0, 101, True, "20"):
        with pytest.raises(RunStoreConflictError):
            store.list("alice", limit=value)
    store.close()
    with pytest.raises(RunStoreError):
        store.get(run_id, "alice")
    with pytest.raises(RunStoreError):
        store.start()


def test_connect_failure_is_sanitized_and_never_retried():
    class Repository:
        calls = 0
        def connect(self, **kwargs):
            self.calls += 1
            raise StorageError("postgres://admin:SUPERSECRET@private/db")
    repo = Repository()
    store = PostgresRunStore(repo)
    for _ in range(2):
        with pytest.raises(RunStoreError) as error:
            store.start()
        assert "SUPERSECRET" not in str(error.value)
    assert repo.calls == 1
