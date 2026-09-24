"""Explicit feedback on a server-bound published answer; never model instructions."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def target_digest(target):
    return hashlib.sha256(json.dumps(target, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_target(run_id, question, paper_id, draft, evidence, prompt_version, language):
    cited = {ref["chunk_id"] for claim in draft["claims"] for ref in claim["evidence"]}
    return {"schema_version": 1, "run_id": run_id, "paper_id": paper_id, "question": question,
            "answer": deepcopy(draft), "prompt_version": prompt_version, "language": language,
            "evidence": [{"chunk_id": row["chunk_id"], "version": row.get("version"),
                          "source": row.get("source"), "section_index": row.get("section_index"),
                          "paragraph_index": row.get("paragraph_index"),
                          "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()}
                         for row in evidence if row["chunk_id"] in cited]}


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    rating: Literal["helpful", "problem"]
    note: str = Field(default="", max_length=2000)
    expected_revision: int = Field(ge=0, le=2_147_483_647)
    target: dict

    @field_validator("note")
    @classmethod
    def safe_note(cls, value):
        if any(ord(char) < 32 and char not in "\n\t\r" for char in value):
            raise ValueError("Unsupported control character")
        return value.strip()

    @field_validator("target")
    @classmethod
    def bounded_target(cls, value):
        try:
            serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if len(serialized.encode()) > 512_000:
                raise ValueError("Snapshot too large")
        except (TypeError, UnicodeError, RecursionError):
            raise ValueError("Invalid feedback snapshot") from None
        return value


class FeedbackConflictError(RuntimeError):
    def __init__(self, reason="revision_conflict"):
        self.reason = reason
        super().__init__("回答快照不匹配或反馈版本已变化，请重新读取记录")


def next_feedback(run, current, body: FeedbackInput):
    """Both stores enforce provenance, ownership upstream, and optimistic updates."""
    target_hash = target_digest(body.target)
    expected_hash = run["snapshot"].get("generation", {}).get("feedback_sha256")
    if (run["state"] != "completed" or run["reason"] != "published" or not expected_hash
            or target_hash != expected_hash or body.target.get("run_id") != run["run_id"]
            or body.target.get("paper_id") != run["paper_id"]):
        raise FeedbackConflictError("target_mismatch")
    # Safe retry after an uncertain response: no new revision for identical data.
    if current and current["target_sha256"] == target_hash and current["rating"] == body.rating and current["note"] == body.note:
        return deepcopy(current)
    if body.expected_revision != (current["revision"] if current else 0):
        raise FeedbackConflictError()
    now = datetime.now(timezone.utc).isoformat()
    return {"run_id": run["run_id"], "revision": body.expected_revision + 1,
            "rating": body.rating, "note": body.note, "target_sha256": target_hash,
            "target": deepcopy(body.target), "created_at": current["created_at"] if current else now,
            "updated_at": now}
