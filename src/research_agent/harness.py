"""Request-local control flow. No persistence, scheduling or model-quality claims."""
from __future__ import annotations

import time
import threading
import uuid


STAGES = ("retrieve", "context", "generate", "citation_check", "verify", "publication_check", "publish")
NEXT = {
    None: {"retrieve"}, "retrieve": {"context"}, "context": {"generate"},
    "generate": {"citation_check", "publication_check"}, "citation_check": {"verify"},
    "verify": {"publication_check"}, "publication_check": {"publish"}, "publish": set(),
}
TERMINAL_STATES = {
    "published": "completed",
    "no_retrieved_evidence": "abstained", "context_budget_excluded_all": "abstained",
    "generator_abstained": "abstained",
    "citation_invalid": "blocked", "semantic_verification_failed": "blocked",
    "corpus_changed": "blocked", "evidence_integrity_failed": "blocked",
    "model_refused": "blocked",
    "generation_not_configured": "failed", "provider_timeout": "failed",
    "provider_http_error": "failed", "provider_connection_error": "failed",
    "invalid_output": "failed", "incomplete_output": "failed",
    "vector_unavailable": "failed", "reranker_unavailable": "failed",
    "paper_not_found": "failed", "invalid_request": "failed",
    "dependency_unavailable": "failed", "internal_error": "failed",
    "deadline_exceeded": "failed", "cancelled": "failed", "call_budget_exceeded": "failed",
    "input_budget_exceeded": "failed", "repair_exhausted": "blocked", "access_denied": "blocked",
    "run_store_unavailable": "failed", "tool_not_allowed": "blocked", "server_restarted": "interrupted",
}


class AnswerRun:
    """Only permitted edges can run; publication also requires every prior gate.

    The generator-abstention path rechecks the corpus but cannot publish claims.
    Reuse retrieval's trace ID once available; early failures still have an ID.
    All times are elapsed wall time, not CPU time or provider billing time.
    """

    def __init__(self, clock=time.perf_counter, *, control=None, trace_id=None, observer=None):
        self._clock = clock
        self._started = clock()
        self._stage_started = self._started
        self._current = None
        self._finished = False
        self._terminal = None
        self.control = control
        self._observer = observer
        self.lock = control.lock if control else threading.RLock()
        self.trace_id = trace_id or uuid.uuid4().hex
        self.fixed_id = trace_id is not None
        self.attempt = 0
        self.attempts = []
        self.stages = {name: {"name": name, "status": "not_run", "latency_ms": None} for name in STAGES}

    def check(self):
        if self.control:
            self.control.check()
        if self._finished:
            from .runtime import RunStopped
            raise RunStopped(self._terminal["reason"])

    def _close_stage(self, now, status):
        if self._current is not None:
            item = self.stages[self._current]
            item.update(status=status, latency_ms=round((now - self._stage_started) * 1000, 3))
            closed = {**item, "attempt": self.attempt}
            # A failed terminal write can close the same stage again as failed.
            # Preserve one final outcome for this attempt, not a false completion.
            if (self.attempts and self.attempts[-1]["name"] == self._current
                    and self.attempts[-1]["attempt"] == self.attempt):
                self.attempts[-1] = closed
            else:
                self.attempts.append(closed)

    def snapshot(self):
        with self.lock:
            if self._terminal is not None:
                return self._terminal
            stages = [dict(self.stages[name]) for name in STAGES]
            return {"trace_id": self.trace_id, "state": "running", "reason": None,
                    "terminal_stage": self._current, "latency_ms": round((self._clock() - self._started) * 1000, 3),
                    "stages": stages, "attempts": list(self.attempts),
                    **(self.control.snapshot() if self.control else {})}

    def emit(self):
        with self.lock:
            if self._observer:
                self._observer(self.snapshot())

    def enter(self, name):
        with self.lock:
            if self._finished or name not in NEXT[self._current]:
                raise RuntimeError("Invalid answer stage transition")
            self.check()
            if name == "publish" and any(self.stages[s]["status"] != "completed" for s in STAGES[:5]):
                raise RuntimeError("Cannot publish without completed verification gates")
            now = self._clock()
            self._close_stage(now, "completed")
            self._current, self._stage_started = name, now
            self.stages[name]["status"] = "running"
            self.emit()

    def repair(self):
        with self.lock:
            self.check()
            limit = self.control.limits.max_repairs if self.control else 0
            if self._current not in {"citation_check", "verify"} or self.attempt >= limit:
                raise RuntimeError("Invalid repair transition")
            self._close_stage(self._clock(), "stopped")
            self.attempt += 1
            # Old verification never authorizes a revised draft.
            for name in STAGES[2:]:
                self.stages[name].update(status="not_run", latency_ms=None)
            self._current = "generate"
            self._stage_started = self._clock()
            self.stages["generate"]["status"] = "running"
            self.emit()

    def finish(self, reason, *, force=False):
        with self.lock:
            if self._finished or reason not in TERMINAL_STATES:
                raise RuntimeError("Invalid answer termination")
            if reason == "published" and self._current != "publish":
                raise RuntimeError("Cannot finish before publication")
            if not force:
                self.check()
            now = self._clock()
            state = TERMINAL_STATES[reason]
            completed = state == "completed" or (reason == "generator_abstained" and self._current == "publication_check")
            self._close_stage(now, "completed" if completed else "failed" if state == "failed" else "stopped")
            record = {"trace_id": self.trace_id, "state": state, "reason": reason,
                      "terminal_stage": self._current, "latency_ms": round((now - self._started) * 1000, 3),
                      "stages": [dict(self.stages[name]) for name in STAGES], "attempts": list(self.attempts),
                      **(self.control.snapshot() if self.control else {})}
            # Persist before allowing the caller to return any published claims.
            if self._observer:
                self._observer(record)
            self._terminal, self._finished = record, True
            return record

    def abort(self, reason):
        with self.lock:
            if self._terminal:
                return self._terminal
            if self.control:
                self.control.stop(reason)
            try:
                return self.finish(reason, force=True)
            except Exception:
                # Storage unavailable: no publication, do not retry paid work.
                self._observer = None
                return self.finish("run_store_unavailable", force=True)


def exception_reason(exc):
    """Allowlisted codes only; never copy exception messages/credentials into runs."""
    from .runtime import RunStopped
    from .access import AccessError
    if isinstance(exc, RunStopped):
        return exc.code
    if isinstance(exc, AccessError):
        return "access_denied"
    if getattr(exc, "code", None) in {"run_store_unavailable", "cancelled"}:
        return exc.code
    import psycopg
    from botocore.exceptions import BotoCoreError, ClientError
    from .embeddings import VectorUnavailableError
    from .reranking import RerankerUnavailableError
    from .service import CorpusChangedError, CorpusIntegrityError
    from .storage import StorageError

    for kind, reason in (
        (CorpusChangedError, "corpus_changed"), (CorpusIntegrityError, "evidence_integrity_failed"),
        (VectorUnavailableError, "vector_unavailable"), (RerankerUnavailableError, "reranker_unavailable"),
        ((StorageError, psycopg.Error, BotoCoreError, ClientError, ConnectionError), "dependency_unavailable"),
        (KeyError, "paper_not_found"), (ValueError, "invalid_request"),
    ):
        if isinstance(exc, kind):
            return reason
    return "internal_error"
