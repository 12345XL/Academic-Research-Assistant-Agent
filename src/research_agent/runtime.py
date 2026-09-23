"""Bounded request execution, with cooperative checks and an API wait deadline.

Blocking work is never killed unsafely. A timed-out worker keeps its concurrency
slot until it exits; late results cannot publish and are not automatically retried.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
import threading
import time


class RunStopped(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RunLimits:
    deadline_seconds: float = 100
    max_model_calls: int = 4
    max_prompt_chars: int = 48_000
    max_completion_tokens: int = 2048
    max_repairs: int = 1

    def __post_init__(self):
        if (isinstance(self.deadline_seconds, bool) or not math.isfinite(self.deadline_seconds)
                or not 0 < self.deadline_seconds <= 600):
            raise ValueError("Invalid run deadline")
        for value, low, high in ((self.max_model_calls, 0, 4), (self.max_prompt_chars, 1, 200_000),
                                 (self.max_completion_tokens, 1, 4096), (self.max_repairs, 0, 1)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid run limit")

    @classmethod
    def from_env(cls):
        try:
            return cls(float(os.getenv("RESEARCH_RUN_DEADLINE_SECONDS", "100")),
                       int(os.getenv("RESEARCH_MAX_MODEL_CALLS", "4")),
                       int(os.getenv("RESEARCH_MAX_PROMPT_CHARS", "48000")),
                       int(os.getenv("RESEARCH_MAX_COMPLETION_TOKENS", "2048")),
                       int(os.getenv("RESEARCH_MAX_REPAIRS", "1")))
        except (ValueError, TypeError):
            raise ValueError("Invalid Harness configuration") from None

    def for_request(self, allow_repair=False):
        return replace(self, max_model_calls=min(self.max_model_calls, 4 if allow_repair else 2),
                       max_repairs=self.max_repairs if allow_repair else 0)


def model_messages(system, payload, schema):
    return [{"role": "system", "content": system + "\nReturn JSON only, conforming to this schema: "
             + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


class RunControl:
    ALLOWED_TOOLS = frozenset({"retrieve", "generate", "verify", "publication_check"})

    def __init__(self, limits=None, *, authorize=None, cancelled=None, clock=time.monotonic):
        self.limits = limits or RunLimits().for_request()
        self._clock, self._started = clock, clock()
        self._authorize = authorize or (lambda: None)
        self._cancelled = cancelled or (lambda: False)
        self._stop_code = None
        self.lock = threading.RLock()
        self.budget = {"model_calls": 0, "completion_tokens_reserved": 0,
                       "reported_prompt_tokens": 0, "reported_completion_tokens": 0,
                       "usage_unknown_calls": 0}

    def remaining(self):
        return max(0, self.limits.deadline_seconds - (self._clock() - self._started))

    def check(self):
        with self.lock:
            if self._stop_code:
                raise RunStopped(self._stop_code)
            if self.remaining() <= 0:
                self._stop_code = "deadline_exceeded"
            elif self._cancelled():
                self._stop_code = "cancelled"
            if self._stop_code:
                raise RunStopped(self._stop_code)
            self._authorize()
            # Authentication/cancellation checks can involve I/O too. Never
            # start a model request using a deadline checked before that I/O.
            if self.remaining() <= 0:
                self._stop_code = "deadline_exceeded"
                raise RunStopped(self._stop_code)

    def stop(self, code):
        with self.lock:
            self._stop_code = self._stop_code or code

    def invoke(self, name, function):
        if name not in self.ALLOWED_TOOLS:
            raise RunStopped("tool_not_allowed")
        self.check()
        value = function()
        self.check()
        return value

    def reserve_model(self, system, payload, schema):
        self.check()
        # Exact serialized request-message character count; NOT model tokens.
        chars = len(json.dumps(model_messages(system, payload, schema), ensure_ascii=False))
        with self.lock:
            if chars > self.limits.max_prompt_chars:
                raise RunStopped("input_budget_exceeded")
            if self.budget["model_calls"] >= self.limits.max_model_calls:
                raise RunStopped("call_budget_exceeded")
            self.budget["model_calls"] += 1
            self.budget["completion_tokens_reserved"] += self.limits.max_completion_tokens
            self.budget["usage_unknown_calls"] += 1
        return chars

    def account(self, meta):
        with self.lock:
            usage = meta.get("usage", {})
            if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                self.budget["usage_unknown_calls"] -= 1
                self.budget["reported_prompt_tokens"] += usage["prompt_tokens"]
                self.budget["reported_completion_tokens"] += usage["completion_tokens"]

    def snapshot(self):
        with self.lock:
            return {"limits": asdict(self.limits), "budget": dict(self.budget)}


class AnswerJob:
    """One retained slot even if the HTTP waiter times out/disconnects."""
    def __init__(self, function):
        self.done = threading.Event()
        self.result, self.error = None, None
        self.thread = threading.Thread(target=self._work, args=(function,), daemon=True)

    def _work(self, function):
        try:
            self.result = function()
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()

    def start(self):
        self.thread.start()
