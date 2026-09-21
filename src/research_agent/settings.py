"""Explicit infrastructure configuration. Secrets never appear in repr/errors."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    s3_endpoint_url: str
    s3_access_key_id: str = field(repr=False)
    s3_secret_access_key: str = field(repr=False)
    s3_bucket: str = "research-papers"
    s3_region: str = "us-east-1"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        required = ("DATABASE_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")
        missing = [key for key in required if not values.get(key, "").strip()]
        if missing:
            raise ConfigurationError("Missing configuration: " + ", ".join(missing))
        endpoint = values.get("S3_ENDPOINT_URL", "http://127.0.0.1:8333").rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ConfigurationError("S3_ENDPOINT_URL must be an HTTP(S) service URL without credentials")
        database = values["DATABASE_URL"]
        if not database.startswith(("postgresql://", "postgres://")):
            raise ConfigurationError("DATABASE_URL must be a PostgreSQL URL")
        return cls(database, endpoint, values["S3_ACCESS_KEY_ID"],
                   values["S3_SECRET_ACCESS_KEY"], values.get("S3_BUCKET", "research-papers"),
                   values.get("S3_REGION", "us-east-1"))
