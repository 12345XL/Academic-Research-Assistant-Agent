"""Request-local control flow. No persistence, scheduling or model-quality claims."""
from __future__ import annotations

import time
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
}


class AnswerRun:
    """Only permitted edges can run; publication also requires every prior gate.

    The generator-abstention path rechecks the corpus but cannot publish claims.
    Reuse retrieval's trace ID once available; early failures still have an ID.
    All times are elapsed wall time, not CPU time or provider billing time.
    """

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._started = clock()
        self._stage_started = self._started
        self._current = None
        self._finished = False
        self.trace_id = uuid.uuid4().hex
        self.stages = {name: {"name": name, "status": "not_run", "latency_ms": None} for name in STAGES}

    def _close_stage(self, now, status):
        if self._current is not None:
            self.stages[self._current].update(
                status=status, latency_ms=round((now - self._stage_started) * 1000, 3))

    def enter(self, name):
        if self._finished or name not in NEXT[self._current]:
            raise RuntimeError("Invalid answer stage transition")
        if name == "publish" and any(self.stages[s]["status"] != "completed" for s in STAGES[:5]):
            raise RuntimeError("Cannot publish without completed verification gates")
        now = self._clock()
        self._close_stage(now, "completed")
        self._current, self._stage_started = name, now
        self.stages[name]["status"] = "running"

    def finish(self, reason):
        if self._finished or self._current is None or reason not in TERMINAL_STATES:
            raise RuntimeError("Invalid answer termination")
        if reason == "published" and self._current != "publish":
            raise RuntimeError("Cannot finish before publication")
        now = self._clock()
        state = TERMINAL_STATES[reason]
        # Abstention after a successful revision check is a completed check,
        # even though the run deliberately publishes no answer.
        completed = state == "completed" or (reason == "generator_abstained" and self._current == "publication_check")
        self._close_stage(now, "completed" if completed else "failed" if state == "failed" else "stopped")
        self._finished = True
        return {"trace_id": self.trace_id, "state": state, "reason": reason,
                "terminal_stage": self._current, "latency_ms": round((now - self._started) * 1000, 3),
                "stages": [dict(self.stages[name]) for name in STAGES]}


def exception_reason(exc):
    """Allowlisted codes only; never copy exception messages/credentials into runs."""
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
