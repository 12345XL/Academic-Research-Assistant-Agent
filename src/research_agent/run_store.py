"""Durable, owner-scoped answer run metadata; no question or answer text.

A PostgreSQL store owns one session advisory lock for its lifetime. Losing that
session fails closed: it must never reconnect and accidentally revive work that
another process already marked interrupted. This is a single-worker design.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import re
import threading
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .storage import StorageError
from .feedback import FeedbackConflictError, FeedbackInput, next_feedback

RUN_WORKER_LOCK_ID = 71420520923
STATES = {"running", "completed", "abstained", "blocked", "failed", "interrupted"}
STAGES = {"retrieve", "context", "generate", "citation_check", "verify", "publication_check", "publish"}
STATUSES = {"not_run", "running", "completed", "stopped", "failed"}
REASONS = {
    "published", "no_retrieved_evidence", "context_budget_excluded_all", "generator_abstained",
    "citation_invalid", "semantic_verification_failed", "corpus_changed", "evidence_integrity_failed",
    "model_refused", "generation_not_configured", "provider_timeout", "provider_http_error",
    "provider_connection_error", "invalid_output", "incomplete_output", "vector_unavailable",
    "reranker_unavailable", "paper_not_found", "invalid_request", "dependency_unavailable",
    "internal_error", "cancelled", "deadline_exceeded", "call_budget_exceeded",
    "input_budget_exceeded", "repair_exhausted", "access_denied", "run_store_unavailable",
    "server_restarted", "tool_not_allowed",
}
_ID = re.compile(r"[A-Za-z0-9_.:/@+\-]{1,256}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class RunStoreError(RuntimeError):
    """Fixed public text: never expose a database endpoint or its error body."""

    code = "run_store_unavailable"

    def __init__(self):
        super().__init__("Answer run storage is unavailable")


class RunStoreConflictError(RunStoreError):
    code = "run_conflict"

    def __init__(self):
        RuntimeError.__init__(self, "Answer run cannot be changed")


class RunStoreCancelledError(RunStoreError):
    code = "cancelled"

    def __init__(self):
        RuntimeError.__init__(self, "Answer run was cancelled before publication")


class RunStoreBusyError(RunStoreError):
    code = "run_store_busy"

    def __init__(self):
        RuntimeError.__init__(self, "Another answer worker is active")


def _identifier(value):
    return isinstance(value, str) and bool(_ID.fullmatch(value))


def _number(value):
    return type(value) in (int, float) and 0 <= value <= 10**15 and math.isfinite(value)


def _integer(value):
    return type(value) is int and 0 <= value <= 10**15


def _copy_fields(source, rules):
    if not isinstance(source, dict):
        return {}
    return {key: deepcopy(source[key]) for key, check in rules.items()
            if key in source and check(source[key])}


def _enum(*values):
    allowed = set(values)
    return lambda value: isinstance(value, str) and value in allowed


def sanitize_metadata(value):
    """Allow only configuration, never user inputs or arbitrary nested objects."""
    return _copy_fields(value, {
        "mode": _enum("bm25", "dense", "hybrid"), "rerank": lambda v: type(v) is bool,
        "top_k": _integer, "profile": _enum("v1", "v2"), "language": _enum("zh", "en"),
        "allow_repair": lambda v: type(v) is bool, "rrf_constant": _number,
        "dense_weight": _number, "model": _identifier,
        "conversation_id": lambda v: isinstance(v, str) and bool(_RUN_ID.fullmatch(v)),
        "conversation_revision": _integer,
    })


def _stage_rows(value, *, attempts=False):
    if not isinstance(value, list):
        return []
    result = []
    for entry in value[:100]:
        if not isinstance(entry, dict) or not _enum(*STAGES)(entry.get("name")):
            continue
        clean = _copy_fields(entry, {
            "name": _enum(*STAGES), "status": _enum(*STATUSES),
            "latency_ms": lambda v: v is None or _number(v),
            **({"attempt": _integer} if attempts else {}),
        })
        result.append(clean)
    return result


def sanitize_snapshot(snapshot):
    """Nested allowlists exclude prompts, evidence, model bodies, answers and keys.

    String fields are restricted to known codes, hashes or bounded identifiers;
    copying an allowlisted key never copies an arbitrary object beneath it.
    """
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("run"), dict):
        raise RunStoreConflictError()
    original = snapshot["run"]
    if not _enum(*STATES)(original.get("state")):
        raise RunStoreConflictError()
    state = original["state"]
    reason = original.get("reason")
    if (state == "running" and reason is not None) or (state != "running" and not _enum(*REASONS)(reason)):
        raise RunStoreConflictError()
    run = _copy_fields(original, {
        "trace_id": _identifier, "state": _enum(*STATES),
        "reason": lambda v: v is None or (isinstance(v, str) and v in REASONS),
        "terminal_stage": lambda v: v is None or (isinstance(v, str) and v in STAGES),
        "latency_ms": _number,
    })
    run.setdefault("reason", None)
    if "stages" in original:
        run["stages"] = _stage_rows(original["stages"])
    if "attempts" in original:
        run["attempts"] = _stage_rows(original["attempts"], attempts=True)
    if "limits" in original:
        run["limits"] = _copy_fields(original["limits"], {
            "deadline_seconds": _number, "max_model_calls": _integer,
            "max_prompt_chars": _integer, "max_completion_tokens": _integer, "max_repairs": _integer,
        })
    if "budget" in original:
        run["budget"] = _copy_fields(original["budget"], {key: _integer for key in (
            "model_calls", "completion_tokens_reserved", "reported_prompt_tokens",
            "reported_completion_tokens", "usage_unknown_calls",
        )})
    result = {"run": run}
    trace = snapshot.get("generation")
    if isinstance(trace, dict):
        generation = _copy_fields(trace, {
            "prompt_version": _enum("paper-claims-v1", "paper-claims-v2"),
            "answer_language": _enum("zh", "en"), "model_calls": _integer,
            "latency_ms": _number,
            "memory_used_turns": _integer,
            "memory_sha256": lambda v: isinstance(v, str) and bool(_SHA.fullmatch(v)),
            "evidence_sha256": lambda v: isinstance(v, str) and bool(_SHA.fullmatch(v)),
            "draft_sha256": lambda v: isinstance(v, str) and bool(_SHA.fullmatch(v)),
            "verified_draft_sha256": lambda v: isinstance(v, str) and bool(_SHA.fullmatch(v)),
            "feedback_sha256": lambda v: isinstance(v, str) and bool(_SHA.fullmatch(v)),
            "publication_check": _enum("immutable_file_snapshot", "database_revision_and_hash"),
        })
        if "context" in trace:
            context = trace["context"]
            generation["context"] = _copy_fields(context, {
                "budget_unit": _enum("unicode_characters"), "limit": _integer, "used": _integer,
            })
            if isinstance(context, dict):
                for key in ("included_ids", "skipped_ids"):
                    if isinstance(context.get(key), list):
                        generation["context"][key] = [v for v in context[key][:100] if _identifier(v)]
        if "checks" in trace:
            generation["checks"] = _copy_fields(trace["checks"], {
                key: _enum("not_run", "passed", "failed")
                for key in ("citation_integrity", "semantic_support")
            })
        if isinstance(trace.get("calls"), list):
            generation["calls"] = []
            for call in trace["calls"][:100]:
                if not isinstance(call, dict):
                    continue
                clean = _copy_fields(call, {
                    "stage": _enum(*STAGES), "status": _enum("started", "completed", *REASONS),
                    "requested_model": _identifier, "model": _identifier,
                    "system_fingerprint": _identifier, "latency_ms": _number,
                    "output_sha256": lambda v: isinstance(v, str) and bool(_SHA.fullmatch(v)),
                    "http_status": lambda v: type(v) is int and 100 <= v <= 599,
                    "attempt": _integer, "prompt_chars": _integer,
                    "max_tokens": _integer, "max_completion_tokens": _integer,
                    "repair_reason": _enum(*REASONS),
                })
                if "usage" in call:
                    clean["usage"] = _copy_fields(call["usage"], {key: _integer for key in (
                        "prompt_tokens", "completion_tokens", "total_tokens",
                        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                    )})
                generation["calls"].append(clean)
        result["generation"] = generation
    if "access" in snapshot:
        result["access"] = _copy_fields(snapshot["access"], {
            "principal_id": _identifier, "mode": _enum("local_public", "bearer_policy"),
        })
    return result


def _new_record(run_id, owner_id, paper_id, metadata):
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or not _identifier(owner_id) or not _identifier(paper_id):
        raise RunStoreConflictError()
    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id, "owner_id": owner_id, "paper_id": paper_id,
        "revision": 1, "state": "running", "reason": None,
        "created_at": now, "updated_at": now, "metadata": sanitize_metadata(metadata),
        "snapshot": {"run": {"trace_id": run_id, "state": "running", "reason": None,
                               "terminal_stage": None, "stages": [], "attempts": []}},
        "cancel_requested": False,
    }


def _page_limit(limit):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise RunStoreConflictError()


def _interrupt_snapshot(snapshot):
    clean = sanitize_snapshot(snapshot)
    clean["run"].update(state="interrupted", reason="server_restarted")
    for key in ("stages", "attempts"):
        for stage in clean["run"].get(key, []):
            if stage.get("status") == "running":
                stage["status"] = "stopped"
                clean["run"]["terminal_stage"] = stage["name"]
    return clean


class MemoryRunStore:
    """Same ownership/immutability contract, deliberately lost at process exit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._feedback: dict[str, dict[str, Any]] = {}
        self._started = False
        self._closed = False

    def start(self):
        with self._lock:
            if self._closed:
                raise RunStoreError()
            self._started = True

    def close(self):
        with self._lock:
            self._closed = True
            self._started = False

    def _ready(self):
        if not self._started or self._closed:
            raise RunStoreError()

    def create(self, run_id, owner_id, paper_id, metadata):
        record = _new_record(run_id, owner_id, paper_id, metadata)
        with self._lock:
            self._ready()
            if run_id in self._records:
                raise RunStoreConflictError()
            self._records[run_id] = record
            return deepcopy(record)

    def save(self, run_id, owner_id, snapshot):
        clean = sanitize_snapshot(snapshot)
        with self._lock:
            self._ready()
            record = self._records.get(run_id)
            if record is None or record["owner_id"] != owner_id or record["state"] != "running":
                raise RunStoreConflictError()
            if record["cancel_requested"] and clean["run"]["reason"] == "published":
                raise RunStoreCancelledError()
            record.update(snapshot=clean, state=clean["run"]["state"], reason=clean["run"]["reason"],
                          updated_at=datetime.now(timezone.utc).isoformat(), revision=record["revision"] + 1)
            return deepcopy(record)

    def get(self, run_id, owner_id):
        with self._lock:
            self._ready()
            record = self._records.get(run_id)
            return deepcopy(record) if record and record["owner_id"] == owner_id else None

    def list(self, owner_id, limit=20, allowed_paper_ids=None):
        _page_limit(limit)
        with self._lock:
            self._ready()
            records = [r for r in self._records.values() if r["owner_id"] == owner_id
                       and (allowed_paper_ids is None or r["paper_id"] in allowed_paper_ids)]
            records.sort(key=lambda r: (r["created_at"], r["run_id"]), reverse=True)
            return deepcopy(records[:limit])

    def request_cancel(self, run_id, owner_id):
        with self._lock:
            self._ready()
            record = self._records.get(run_id)
            if record is None or record["owner_id"] != owner_id:
                return None
            if record["state"] == "running" and not record["cancel_requested"]:
                record.update(cancel_requested=True, revision=record["revision"] + 1,
                              updated_at=datetime.now(timezone.utc).isoformat())
            return deepcopy(record)


    def get_feedback(self, run_id, owner_id):
        with self._lock:
            self._ready()
            run = self._records.get(run_id)
            return deepcopy(self._feedback.get(run_id)) if run and run["owner_id"] == owner_id else None

    def save_feedback(self, run_id, owner_id, body: FeedbackInput):
        with self._lock:
            self._ready()
            run = self._records.get(run_id)
            if run is None or run["owner_id"] != owner_id:
                raise FeedbackConflictError("target_mismatch")
            result = next_feedback(run, self._feedback.get(run_id), body)
            self._feedback[run_id] = result
            return deepcopy(result)


class PostgresRunStore:
    """One dedicated session, one worker lock, no reconnection or paid replay."""

    def __init__(self, repository):
        self.repository = repository
        self._lock = threading.Lock()
        self._conn = None
        self._started = False
        self._closed = False
        self._failed = False

    def start(self):
        with self._lock:
            if self._closed or self._failed:
                raise RunStoreError()
            if self._started:
                self._ready()
                return
            conn = None
            try:
                conn = self.repository.connect(autocommit=True)
                conn.execute("SET statement_timeout = '2s'")
                locked = conn.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (RUN_WORKER_LOCK_ID,)).fetchone()
                if not locked["acquired"]:
                    raise RunStoreBusyError()
                with conn.transaction():
                    rows = conn.execute("SELECT run_id,snapshot FROM answer_runs WHERE state='running' FOR UPDATE").fetchall()
                    for row in rows:
                        snapshot = _interrupt_snapshot(row["snapshot"])
                        conn.execute("UPDATE answer_runs SET state='interrupted',reason='server_restarted',"
                                     "snapshot=%s,revision=revision+1,updated_at=clock_timestamp() WHERE run_id=%s",
                                     (Jsonb(snapshot), row["run_id"]))
                self._conn, self._started = conn, True
            except RunStoreError:
                if conn is not None:
                    conn.close()
                self._failed = True
                raise
            except (psycopg.Error, StorageError):
                if conn is not None:
                    conn.close()
                self._failed = True
                raise RunStoreError() from None

    def close(self):
        with self._lock:
            if self._conn is not None:
                # Closing the dedicated session releases its advisory lock even
                # if PostgreSQL is unavailable and an explicit unlock would fail.
                self._conn.close()
                self._conn = None
            self._started = False
            self._closed = True

    def _ready(self):
        if not self._started or self._closed or self._failed or self._conn is None or self._conn.closed:
            raise RunStoreError()
        return self._conn

    def _execute(self, statement, parameters=()):
        conn = self._ready()
        try:
            return conn.execute(statement, parameters)
        except psycopg.errors.UniqueViolation:
            raise RunStoreConflictError() from None
        except psycopg.Error:
            self._failed = True
            # Do not leave a lock-owning but unusable store indefinitely alive.
            conn.close()
            raise RunStoreError() from None

    @staticmethod
    def _record(row):
        if row is None:
            return None
        result = dict(row)
        for key in ("created_at", "updated_at"):
            if isinstance(result[key], datetime):
                result[key] = result[key].isoformat()
        return result

    def create(self, run_id, owner_id, paper_id, metadata):
        record = _new_record(run_id, owner_id, paper_id, metadata)
        with self._lock:
            row = self._execute(
                "INSERT INTO answer_runs(run_id,owner_id,paper_id,metadata,snapshot) VALUES(%s,%s,%s,%s,%s) RETURNING *",
                (run_id, owner_id, paper_id, Jsonb(record["metadata"]), Jsonb(record["snapshot"])),
            ).fetchone()
            return self._record(row)

    def save(self, run_id, owner_id, snapshot):
        clean = sanitize_snapshot(snapshot)
        with self._lock:
            row = self._execute(
                "UPDATE answer_runs SET snapshot=%s,state=%s,reason=%s,revision=revision+1,"
                "updated_at=clock_timestamp() WHERE run_id=%s AND owner_id=%s AND state='running' "
                "AND (NOT cancel_requested OR %s IS DISTINCT FROM 'published') RETURNING *",
                (Jsonb(clean), clean["run"]["state"], clean["run"]["reason"], run_id, owner_id, clean["run"]["reason"]),
            ).fetchone()
            if row is None:
                existing = self._execute("SELECT state,cancel_requested FROM answer_runs WHERE run_id=%s AND owner_id=%s",
                                         (run_id, owner_id)).fetchone()
                if (existing and existing["state"] == "running" and existing["cancel_requested"]
                        and clean["run"]["reason"] == "published"):
                    raise RunStoreCancelledError()
                raise RunStoreConflictError()
            return self._record(row)

    def get(self, run_id, owner_id):
        with self._lock:
            return self._record(self._execute("SELECT * FROM answer_runs WHERE run_id=%s AND owner_id=%s",
                                             (run_id, owner_id)).fetchone())

    def list(self, owner_id, limit=20, allowed_paper_ids=None):
        _page_limit(limit)
        with self._lock:
            if allowed_paper_ids is None:
                rows = self._execute("SELECT * FROM answer_runs WHERE owner_id=%s ORDER BY created_at DESC,run_id DESC LIMIT %s",
                                     (owner_id, limit)).fetchall()
            else:
                rows = self._execute("SELECT * FROM answer_runs WHERE owner_id=%s AND paper_id=ANY(%s) "
                                     "ORDER BY created_at DESC,run_id DESC LIMIT %s",
                                     (owner_id, list(allowed_paper_ids), limit)).fetchall()
            return [self._record(row) for row in rows]

    def request_cancel(self, run_id, owner_id):
        with self._lock:
            row = self._execute(
                "UPDATE answer_runs SET cancel_requested=true,revision=revision+1,updated_at=clock_timestamp() "
                "WHERE run_id=%s AND owner_id=%s AND state='running' AND NOT cancel_requested RETURNING *",
                (run_id, owner_id),
            ).fetchone()
            if row is None:
                row = self._execute("SELECT * FROM answer_runs WHERE run_id=%s AND owner_id=%s",
                                    (run_id, owner_id)).fetchone()
            return self._record(row)

    def get_feedback(self, run_id, owner_id):
        with self._lock:
            return self._record(self._execute(
                "SELECT f.* FROM answer_feedback f JOIN answer_runs r USING(run_id) "
                "WHERE r.run_id=%s AND r.owner_id=%s", (run_id, owner_id)).fetchone())

    def save_feedback(self, run_id, owner_id, body: FeedbackInput):
        with self._lock:
            run = self._execute("SELECT * FROM answer_runs WHERE run_id=%s AND owner_id=%s",
                                (run_id, owner_id)).fetchone()
            if run is None:
                raise FeedbackConflictError("target_mismatch")
            current = self._record(self._execute("SELECT * FROM answer_feedback WHERE run_id=%s", (run_id,)).fetchone())
            result = next_feedback(run, current, body)
            if current and result["revision"] == current["revision"]:
                return current
            if current is None:
                row = self._execute(
                    "INSERT INTO answer_feedback(run_id,revision,rating,note,target_sha256,target) "
                    "VALUES(%s,1,%s,%s,%s,%s) ON CONFLICT(run_id) DO NOTHING RETURNING *",
                    (run_id, body.rating, body.note, result["target_sha256"], Jsonb(body.target))).fetchone()
            else:
                row = self._execute(
                    "UPDATE answer_feedback SET rating=%s,note=%s,revision=revision+1,updated_at=clock_timestamp() "
                    "WHERE run_id=%s AND revision=%s RETURNING *",
                    (body.rating, body.note, run_id, body.expected_revision)).fetchone()
            if row is None:
                raise FeedbackConflictError()
            return self._record(row)
