"""Small server-owned bearer policy; no client-provided identities or grants.

Keep the policy under an ignored directory such as ``.local/access-policy.json``.
Its shape is ``{"tokens": {"<token sha256>": {"id": "alice",
"allowed_paper_ids": ["paper-1"]}}}``. A sole ``"*"`` grants the full corpus.
The file is read on every authentication/revalidation so revocation is immediate
at the next gate. Principal deliberately contains no token or credential hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Literal, Mapping


class AccessError(RuntimeError):
    """Only fixed, safe messages cross the API boundary."""

    def __init__(self, code: str, status_code: int, message: str):
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def _configuration_error() -> AccessError:
    return AccessError("access_policy_unavailable", 503, "访问策略暂不可用，请检查服务端配置")


def _authentication_error() -> AccessError:
    return AccessError("authentication_required", 401, "身份凭证无效、缺失或已撤销")


@dataclass(frozen=True)
class Principal:
    principal_id: str
    mode: Literal["local_public", "bearer_policy"]
    allowed_paper_ids: frozenset[str] | None

    def allows(self, paper_id: str) -> bool:
        return self.allowed_paper_ids is None or paper_id in self.allowed_paper_ids

    def public_status(self) -> dict:
        return {
            "principal_id": self.principal_id,
            "mode": self.mode,
            "allowed_paper_ids": ["*"] if self.allowed_paper_ids is None
            else sorted(self.allowed_paper_ids),
        }


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate policy field")
        value[key] = item
    return value


class AccessPolicy:
    def __init__(self, policy_file: Path | None = None):
        self._policy_file = policy_file

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AccessPolicy:
        values = os.environ if env is None else env
        if "RESEARCH_ACCESS_POLICY_FILE" not in values:
            return cls()
        filename = values["RESEARCH_ACCESS_POLICY_FILE"].strip()
        if not filename:
            # A configured-but-empty path must never turn auth off silently.
            raise _configuration_error()
        return cls(Path(filename).expanduser())

    @property
    def mode(self) -> Literal["local_public", "bearer_policy"]:
        return "local_public" if self._policy_file is None else "bearer_policy"

    def _read_policy(self) -> dict[str, Principal]:
        try:
            # Bound accidental/untrusted configuration size as well as parsing.
            with self._policy_file.open("rb") as stream:
                raw = stream.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ValueError("Policy too large")
            document = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(document, dict) or set(document) != {"tokens"}:
                raise ValueError("Invalid policy schema")
            records = document["tokens"]
            if not isinstance(records, dict):
                raise ValueError("Invalid tokens")
            principals = {}
            for digest, record in records.items():
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Invalid digest")
                if not isinstance(record, dict) or set(record) != {"id", "allowed_paper_ids"}:
                    raise ValueError("Invalid principal")
                identity, papers = record["id"], record["allowed_paper_ids"]
                if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,99}", identity):
                    raise ValueError("Invalid principal id")
                if not isinstance(papers, list) or any(
                    not isinstance(paper, str) or not paper.strip() or paper != paper.strip()
                    or len(paper) > 150 or any(ord(char) < 32 for char in paper)
                    for paper in papers
                ):
                    raise ValueError("Invalid paper scope")
                if len(set(papers)) != len(papers) or ("*" in papers and papers != ["*"]):
                    raise ValueError("Ambiguous paper scope")
                principals[digest] = Principal(identity, "bearer_policy",
                    None if papers == ["*"] else frozenset(papers))
            return principals
        except (OSError, ValueError, TypeError, RecursionError):
            raise _configuration_error() from None

    def authenticate(self, authorization: str | None) -> Principal:
        if self._policy_file is None:
            # This mode is explicitly a shared local corpus, not authentication.
            return Principal("local-user", "local_public", None)
        records = self._read_policy()
        if not isinstance(authorization, str) or len(authorization) > 8192:
            raise _authentication_error()
        parts = authorization.split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1] or any(
            ord(char) < 33 or ord(char) > 126 for char in parts[1]
        ):
            raise _authentication_error()
        digest = hashlib.sha256(parts[1].encode("ascii")).hexdigest()
        principal = records.get(digest)
        if principal is None:
            raise _authentication_error()
        return principal

    def require_paper(self, principal: Principal, paper_id: str) -> None:
        """Check this authenticated snapshot; use revalidate at later gates."""
        if not principal.allows(paper_id):
            raise AccessError("paper_access_denied", 403, "当前身份无权访问这篇论文")

    def revalidate(self, principal: Principal, paper_id: str, authorization: str | None) -> Principal:
        """Re-read authority; a stale Principal is never sufficient to publish."""
        current = self.authenticate(authorization)
        if current.principal_id != principal.principal_id or current.mode != principal.mode:
            raise _authentication_error()
        self.require_paper(current, paper_id)
        return current
