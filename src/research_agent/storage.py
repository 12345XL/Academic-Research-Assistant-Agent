"""PostgreSQL fact storage and S3 immutable derived-document storage."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import boto3
import psycopg
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from psycopg.rows import dict_row

from .paper_metadata import RESEARCH_DIRECTIONS
from .settings import Settings

IMPORT_LOCK_ID = 71420520920
MIGRATION_LOCK_ID = 71420520921


class StorageError(RuntimeError):
    """Public error messages intentionally exclude endpoints and credentials."""


class ImportBusyError(StorageError):
    pass


def serializable(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(row, default=str))


def paper_with_metadata(row: dict[str, Any]) -> dict[str, Any]:
    direction = row["research_direction"]
    return {
        **row["paper_payload"],
        "arxiv_submitted_at": row["arxiv_submitted_at"].isoformat()
        if row["arxiv_submitted_at"] else None,
        "journal_ref": row["arxiv_journal_ref"] or None,
        "arxiv_primary_category": row["arxiv_primary_category"],
        "arxiv_categories": row["arxiv_categories"] or [],
        "arxiv_pdf_url": row["arxiv_pdf_url"],
        "research_direction": direction,
        "research_direction_label": RESEARCH_DIRECTIONS.get(direction),
        "ccf_venue": row["ccf_venue"],
        "ccf_level": row["ccf_level"],
        "ccf_catalog_url": row["ccf_catalog_url"],
    }


class Repository:
    def __init__(self, settings: Settings):
        self.settings = settings

    def connect(self, *, autocommit: bool = False) -> psycopg.Connection:
        try:
            return psycopg.connect(self.settings.database_url, row_factory=dict_row,
                                   connect_timeout=5, autocommit=autocommit)
        except psycopg.Error:
            raise StorageError("PostgreSQL is unavailable; check local service and configuration") from None

    def migrate(self, migrations_dir: Path | None = None) -> list[str]:
        directory = migrations_dir or Path(__file__).resolve().parent / "migrations"
        files = sorted(directory.glob("[0-9][0-9][0-9]_*.sql"))
        if not files:
            raise StorageError("No database migration files were found")
        applied = []
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                         "(name text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())")
            existing = {r["name"]: r["sha256"] for r in conn.execute("SELECT name, sha256 FROM schema_migrations")}
            for path in files:
                data = path.read_bytes()
                checksum = hashlib.sha256(data).hexdigest()
                if path.name in existing:
                    if existing[path.name] != checksum:
                        raise StorageError("An applied migration was modified; add a new migration instead")
                    continue
                conn.execute(data.decode("utf-8"))
                conn.execute("INSERT INTO schema_migrations(name,sha256) VALUES (%s,%s)", (path.name, checksum))
                applied.append(path.name)
        return applied

    @contextmanager
    def import_lock(self) -> Iterator[psycopg.Connection]:
        # Session-scoped lock is automatically released if the process dies.
        with self.connect(autocommit=True) as conn:
            acquired = conn.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (IMPORT_LOCK_ID,)).fetchone()
            if not acquired["acquired"]:
                raise ImportBusyError("Another corpus import is running; retry when it completes")
            try:
                conn.execute("UPDATE ingestion_jobs SET status='failed',finished_at=now(), "
                             "error_code='interrupted_process' WHERE status='running'")
                conn.execute("UPDATE stored_objects o SET state='orphaned' FROM ingestion_jobs j "
                             "WHERE o.created_by_job=j.job_id AND o.state='staged' AND j.status='failed'")
                yield conn
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (IMPORT_LOCK_ID,))

    def load_corpus(self) -> tuple[list[dict], list[dict]]:
        _, papers, paragraphs = self.load_snapshot()
        return papers, paragraphs

    def load_snapshot(self) -> tuple[int, list[dict], list[dict]]:
        with self.connect() as conn:
            # Both reads must observe the same publication even if import commits between them.
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            revision = conn.execute("SELECT revision FROM corpus_state WHERE singleton").fetchone()["revision"]
            papers = [r["paper_payload"] for r in conn.execute(
                "SELECT v.paper_payload FROM papers p JOIN paper_versions v ON v.version_id=p.current_version_id "
                "WHERE p.in_current_corpus AND v.state='active' AND v.source='qasper' ORDER BY p.paper_id")]
            paragraphs = [r["paragraph_payload"] for r in conn.execute(
                "SELECT x.paragraph_payload FROM papers p JOIN paper_versions v ON v.version_id=p.current_version_id "
                "JOIN paragraphs x ON x.version_id=v.version_id WHERE p.in_current_corpus AND v.state='active' AND v.source='qasper' "
                "ORDER BY p.paper_id,x.ordinal")]
        return revision, papers, paragraphs

    def load_paper_snapshot(self, paper_id):
        with self.connect() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            revision = conn.execute("SELECT revision FROM corpus_state WHERE singleton").fetchone()["revision"]
            rows = conn.execute("SELECT v.paper_payload,v.version_id FROM papers p JOIN paper_versions v "
                "ON v.version_id=p.current_version_id WHERE p.paper_id=%s AND p.in_current_corpus AND v.state='active'",
                (paper_id,)).fetchall()
            chunks = [] if not rows else [r["paragraph_payload"] for r in conn.execute(
                "SELECT paragraph_payload FROM paragraphs WHERE version_id=%s ORDER BY ordinal", (rows[0]["version_id"],))]
        return revision, [r["paper_payload"] for r in rows], chunks

    def get_pdf_object(self, paper_id):
        with self.connect() as conn:
            row = conn.execute("SELECT o.object_key,o.sha256,o.size_bytes,o.bucket FROM pdf_documents d "
                "JOIN stored_objects o ON o.object_key=d.original_object_key JOIN papers p USING(paper_id) "
                "WHERE d.paper_id=%s AND p.in_current_corpus AND o.state='published'", (paper_id,)).fetchone()
        return dict(row) if row else None

    def revision(self) -> int:
        with self.connect() as conn:
            return conn.execute("SELECT revision FROM corpus_state WHERE singleton").fetchone()["revision"]

    def get_chunks(self, chunk_ids: list[str], paper_id: str) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        with self.connect() as conn:
            rows = conn.execute("SELECT x.chunk_id,x.paragraph_payload FROM papers p JOIN paper_versions v "
                                "ON v.version_id=p.current_version_id JOIN paragraphs x ON x.version_id=v.version_id "
                                "WHERE p.paper_id=%s AND p.in_current_corpus AND v.state='active' "
                                "AND x.chunk_id=ANY(%s)", (paper_id, chunk_ids)).fetchall()
        return {r["chunk_id"]: r["paragraph_payload"] for r in rows}

    def upsert_paper_metadata(self, rows: list[dict]) -> None:
        """Refresh browsing fields without changing the published retrieval corpus."""
        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO paper_metadata(paper_id,arxiv_submitted_at,arxiv_journal_ref,arxiv_doi,"
                    "arxiv_primary_category,arxiv_categories,arxiv_pdf_url,research_direction,"
                    "ccf_venue,ccf_level,ccf_catalog_url,metadata_source,metadata_checked_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(paper_id) DO UPDATE SET "
                    "arxiv_submitted_at=EXCLUDED.arxiv_submitted_at,"
                    "arxiv_journal_ref=EXCLUDED.arxiv_journal_ref,arxiv_doi=EXCLUDED.arxiv_doi,"
                    "arxiv_primary_category=EXCLUDED.arxiv_primary_category,"
                    "arxiv_categories=EXCLUDED.arxiv_categories,arxiv_pdf_url=EXCLUDED.arxiv_pdf_url,"
                    "research_direction=EXCLUDED.research_direction,"
                    "ccf_venue=EXCLUDED.ccf_venue,ccf_level=EXCLUDED.ccf_level,"
                    "ccf_catalog_url=EXCLUDED.ccf_catalog_url,metadata_source=EXCLUDED.metadata_source,"
                    "metadata_checked_at=EXCLUDED.metadata_checked_at",
                    [
                        (r["paper_id"], r["arxiv_submitted_at"], r["journal_ref"], r["doi"],
                         r["primary_category"], r["categories"], r["pdf_url"], r["research_direction"],
                         r.get("ccf_venue"), r.get("ccf_level"), r.get("ccf_catalog_url"),
                         r["metadata_source"], r["metadata_checked_at"])
                        for r in rows
                    ],
                )

    def list_papers(self, q: str = "", limit: int = 20, offset: int = 0, sort: str = "id_asc",
                    direction: str = "all", allowed_paper_ids: list[str] | None = None) -> dict:
        _page_bounds(limit, offset)
        orders = {
            "id_asc": "p.paper_id ASC",
            "id_desc": "p.paper_id DESC",
            "title_asc": "lower(v.title) ASC, p.paper_id ASC",
            "title_desc": "lower(v.title) DESC, p.paper_id ASC",
            "submitted_newest": "m.arxiv_submitted_at DESC NULLS LAST, p.paper_id ASC",
            "submitted_oldest": "m.arxiv_submitted_at ASC NULLS LAST, p.paper_id ASC",
            "ccf_best": "CASE m.ccf_level WHEN 'A' THEN 0 WHEN 'B' THEN 1 "
                        "WHEN 'C' THEN 2 ELSE 3 END ASC, m.arxiv_submitted_at DESC NULLS LAST, p.paper_id ASC",
        }
        if sort not in orders:
            raise ValueError("Unsupported paper sort")
        if direction != "all" and direction not in RESEARCH_DIRECTIONS:
            raise ValueError("Unsupported research direction")
        # strpos makes %, _ and backslash literal search text rather than SQL wildcards.
        join = ("FROM papers p JOIN paper_versions v ON v.version_id=p.current_version_id "
                "LEFT JOIN paper_metadata m ON m.paper_id=p.paper_id ")
        clause = ("p.in_current_corpus AND v.state='active' AND strpos(lower(v.title),lower(%s))>0 "
                  "AND (%s='all' OR m.research_direction=%s)")
        parameters = (q, direction, direction)
        if allowed_paper_ids is not None:
            # Scope both rows and total in SQL, before pagination. [] denies all.
            clause += " AND p.paper_id=ANY(%s)"
            parameters += (allowed_paper_ids,)
        with self.connect() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            total = conn.execute("SELECT count(*) AS n " + join + "WHERE " + clause,
                                 parameters).fetchone()["n"]
            rows = conn.execute(
                "SELECT v.paper_payload,m.arxiv_submitted_at,m.arxiv_journal_ref,m.arxiv_primary_category,"
                "m.arxiv_categories,m.arxiv_pdf_url,m.research_direction,m.ccf_venue,"
                "m.ccf_level,m.ccf_catalog_url " + join +
                "WHERE " + clause + " ORDER BY " + orders[sort] + " LIMIT %s OFFSET %s",
                (*parameters, limit, offset),
            )
            items = [paper_with_metadata(row) for row in rows]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def get_paper(self, paper_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT v.paper_payload,m.arxiv_submitted_at,m.arxiv_journal_ref,m.arxiv_primary_category,"
                "m.arxiv_categories,m.arxiv_pdf_url,m.research_direction,m.ccf_venue,"
                "m.ccf_level,m.ccf_catalog_url FROM papers p JOIN paper_versions v "
                "ON v.version_id=p.current_version_id LEFT JOIN paper_metadata m ON m.paper_id=p.paper_id "
                "WHERE p.paper_id=%s AND p.in_current_corpus AND v.state='active'",
                (paper_id,),
            ).fetchone()
        if not row:
            return None
        return paper_with_metadata(row)

    def get_paragraphs(self, paper_id: str, limit: int = 20, offset: int = 0) -> dict:
        _page_bounds(limit, offset)
        base = ("FROM papers p JOIN paper_versions v ON v.version_id=p.current_version_id "
                "JOIN paragraphs x ON x.version_id=v.version_id WHERE p.paper_id=%s "
                "AND p.in_current_corpus AND v.state='active' ")
        with self.connect() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            total = conn.execute("SELECT count(*) AS n " + base, (paper_id,)).fetchone()["n"]
            rows = conn.execute("SELECT x.paragraph_payload " + base + "ORDER BY x.ordinal LIMIT %s OFFSET %s",
                                (paper_id, limit, offset))
            items = [r["paragraph_payload"] for r in rows]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def summary(self, allowed_paper_ids: list[str] | None = None) -> dict:
        with self.connect() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            row = conn.execute("SELECT revision,manifest_sha256,paper_count,paragraph_count,published_at "
                               "FROM corpus_state WHERE singleton").fetchone()
            if allowed_paper_ids is None:
                uploaded = conn.execute("SELECT count(DISTINCT p.paper_id) AS papers,count(x.chunk_id) AS chunks "
                    "FROM papers p JOIN paper_versions v ON v.version_id=p.current_version_id "
                    "JOIN paragraphs x ON x.version_id=v.version_id WHERE p.in_current_corpus AND v.state='active' AND v.source='pdf'").fetchone()
                row = {**row, "paper_count": row["paper_count"] + uploaded["papers"],
                       "paragraph_count": row["paragraph_count"] + uploaded["chunks"]}
                objects = conn.execute("SELECT state,count(*) AS count FROM stored_objects GROUP BY state").fetchall()
                metadata = conn.execute(
                    "SELECT count(*) FILTER (WHERE arxiv_submitted_at IS NOT NULL) AS submitted_dates,"
                    "count(*) FILTER (WHERE ccf_level IS NOT NULL) AS ccf_venues,"
                    "count(*) FILTER (WHERE research_direction IS NOT NULL) AS categorized FROM paper_metadata"
                ).fetchone()
            else:
                join = ("FROM papers p JOIN paper_versions v ON v.version_id=p.current_version_id ")
                clause = "WHERE p.in_current_corpus AND v.state='active' AND p.paper_id=ANY(%s)"
                parameters = (allowed_paper_ids,)
                counts = conn.execute(
                    "SELECT count(DISTINCT p.paper_id) AS paper_count,count(x.chunk_id) AS paragraph_count "
                    + join + "LEFT JOIN paragraphs x ON x.version_id=v.version_id " + clause, parameters,
                ).fetchone()
                # The corpus fingerprint includes inaccessible papers; do not expose it.
                row = {**row, **counts, "manifest_sha256": None}
                objects = conn.execute(
                    "SELECT o.state,count(DISTINCT o.object_key) AS count " + join
                    + "LEFT JOIN pdf_documents d ON d.paper_id=p.paper_id "
                    + "JOIN stored_objects o ON o.object_key=v.object_key OR o.object_key=d.original_object_key " + clause + " GROUP BY o.state",
                    parameters,
                ).fetchall()
                metadata = conn.execute(
                    "SELECT count(*) FILTER (WHERE m.arxiv_submitted_at IS NOT NULL) AS submitted_dates,"
                    "count(*) FILTER (WHERE m.ccf_level IS NOT NULL) AS ccf_venues,"
                    "count(*) FILTER (WHERE m.research_direction IS NOT NULL) AS categorized " + join
                    + "LEFT JOIN paper_metadata m ON m.paper_id=p.paper_id " + clause, parameters,
                ).fetchone()
        states = {r["state"]: r["count"] for r in objects}
        return {**serializable(row), "papers": row["paper_count"], "paragraphs": row["paragraph_count"],
                "objects": states.get("published", 0), "object_states": states,
                "metadata_dates": metadata["submitted_dates"], "metadata_ccf": metadata["ccf_venues"],
                "metadata_categorized": metadata["categorized"]}

    def list_jobs(self, limit: int = 20) -> list[dict]:
        _page_bounds(limit, 0)
        with self.connect() as conn:
            rows = conn.execute("SELECT job_id,source,manifest_sha256,status,started_at,finished_at,result,error_code "
                                "FROM ingestion_jobs ORDER BY started_at DESC LIMIT %s", (limit,)).fetchall()
        return [{"id": str(r["job_id"]), "status": r["status"], "created_at": str(r["started_at"]),
                 "finished_at": str(r["finished_at"]) if r["finished_at"] else None,
                 "papers": r["result"].get("papers", 0), "paragraphs": r["result"].get("paragraphs", 0),
                 "objects": r["result"].get("objects_verified", 0), "error": r["error_code"],
                 "reused": r["result"].get("outcome") == "skipped"} for r in rows]

    def get_paper_object(self, paper_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT o.object_key,o.bucket,o.sha256,o.size_bytes,o.content_type,"
                               "v.version_id,v.dataset_version FROM papers p JOIN paper_versions v "
                               "ON v.version_id=p.current_version_id JOIN stored_objects o ON o.object_key=v.object_key "
                               "WHERE p.paper_id=%s AND p.in_current_corpus AND v.state='active'", (paper_id,)).fetchone()
        return serializable(row) if row else None


def _page_bounds(limit: int, offset: int) -> None:
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("limit must be 1..100 and offset must be nonnegative")


class S3ObjectStore:
    def __init__(self, settings: Settings, client: Any = None):
        self.bucket = settings.s3_bucket
        self.region = settings.s3_region
        self.client = client or boto3.client(
            "s3", endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id, aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          connect_timeout=5, read_timeout=30, retries={"max_attempts": 2}))

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if not _not_found(exc):
                raise StorageError("Object storage bucket cannot be accessed") from None
            args = {"Bucket": self.bucket}
            if self.region != "us-east-1":
                args["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            try:
                self.client.create_bucket(**args)
            except (BotoCoreError, ClientError):
                raise StorageError("Object storage bucket could not be created") from None
        except BotoCoreError:
            raise StorageError("Object storage service is unavailable") from None

    def head(self, key: str) -> dict | None:
        try:
            row = self.client.head_object(Bucket=self.bucket, Key=key)
            return {"object_key": key, "size_bytes": row["ContentLength"],
                    "sha256": row.get("Metadata", {}).get("sha256"),
                    "content_type": row.get("ContentType", "application/octet-stream")}
        except ClientError as exc:
            if _not_found(exc):
                return None
            raise StorageError("Object metadata could not be read") from None
        except BotoCoreError:
            raise StorageError("Object storage service is unavailable") from None

    def read_bytes(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except (BotoCoreError, ClientError):
            raise StorageError("Stored document could not be read") from None

    def put_verified(self, key: str, content: bytes, checksum: str, content_type="application/json") -> bool:
        """Return True when uploaded/repaired; verify actual bytes, not S3 ETag."""
        if hashlib.sha256(content).hexdigest() != checksum:
            raise StorageError("Document checksum does not match its content")
        current = self.head(key)
        if current and current["size_bytes"] == len(content):
            if hashlib.sha256(self.read_bytes(key)).hexdigest() == checksum:
                return False
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=content,
                                   ContentType=content_type, Metadata={"sha256": checksum})
        except (BotoCoreError, ClientError):
            raise StorageError("Document could not be stored") from None
        if hashlib.sha256(self.read_bytes(key)).hexdigest() != checksum:
            raise StorageError("Stored document failed the read-back checksum check")
        return True

    def health(self) -> dict:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return {"status": "ready", "bucket": self.bucket}
        except (BotoCoreError, ClientError):
            return {"status": "unavailable", "bucket": self.bucket}


def _not_found(exc: ClientError) -> bool:
    return str(exc.response.get("Error", {}).get("Code")) in {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}
