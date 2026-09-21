"""Publish a QASPER corpus only after its immutable objects are verified."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from psycopg.types.json import Jsonb

from .dataset import load_papers, load_paragraphs
from .storage import Repository, S3ObjectStore, StorageError

PAPER_FIELDS = ("paper_id", "title", "abstract", "split", "source", "version")
PARAGRAPH_FIELDS = ("paper_id", "title", "split", "source", "version", "chunk_id",
                    "section_name", "section_index", "paragraph_index", "text", "text_sha256")
FORBIDDEN_FIELDS = {"qas", "questions", "answers", "annotations", "evidence", "unanswerable", "gold"}
FORMAT_VERSION = "research-paper-json-v1"


class CorpusValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PaperDocument:
    paper: dict
    paragraphs: list[dict]
    content: bytes
    sha256: str
    object_key: str
    version_id: uuid.UUID


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_documents(papers: list[dict], paragraphs: list[dict]) -> tuple[str, list[PaperDocument]]:
    """Strict corpus allowlist: no evaluation questions/answers can be stored."""
    if not papers:
        raise CorpusValidationError("Corpus must contain at least one paper")
    clean_papers = {}
    for row in papers:
        paper = _project(row, PAPER_FIELDS)
        if paper["paper_id"] in clean_papers:
            raise CorpusValidationError("Duplicate paper ID in corpus")
        _validate_base(paper)
        if not isinstance(paper["abstract"], str):
            raise CorpusValidationError("Paper abstract must be text")
        clean_papers[paper["paper_id"]] = paper
    grouped = defaultdict(list)
    seen_chunks = set()
    for row in paragraphs:
        chunk = _project(row, PARAGRAPH_FIELDS)
        paper = clean_papers.get(chunk["paper_id"])
        if paper is None:
            raise CorpusValidationError("Paragraph refers to an unknown paper")
        if any(chunk[k] != paper[k] for k in ("title", "split", "source", "version")):
            raise CorpusValidationError("Paragraph metadata does not match its paper")
        if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"] or chunk["chunk_id"] in seen_chunks:
            raise CorpusValidationError("Paragraph IDs must be nonempty and unique")
        seen_chunks.add(chunk["chunk_id"])
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise CorpusValidationError("Paragraph text must be nonempty")
        if not isinstance(chunk["section_name"], str):
            raise CorpusValidationError("Paragraph section name must be text")
        if type(chunk["section_index"]) is not int or chunk["section_index"] < -1:
            raise CorpusValidationError("Paragraph section index must be an integer >= -1")
        if type(chunk["paragraph_index"]) is not int or chunk["paragraph_index"] < 0:
            raise CorpusValidationError("Paragraph index must be a nonnegative integer")
        if hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest() != chunk["text_sha256"]:
            raise CorpusValidationError("Paragraph text checksum mismatch")
        grouped[chunk["paper_id"]].append(chunk)
    documents = []
    for paper_id, paper in sorted(clean_papers.items()):
        chunks = sorted(grouped[paper_id], key=lambda c: (c["section_index"], c["paragraph_index"], c["chunk_id"]))
        if not chunks:
            raise CorpusValidationError("Each paper must contain a searchable paragraph")
        content = canonical_json({"format": FORMAT_VERSION, "paper": paper, "paragraphs": chunks})
        checksum = hashlib.sha256(content).hexdigest()
        documents.append(PaperDocument(paper, chunks, content, checksum,
                         f"derived/qasper/{checksum[:2]}/{checksum}.json",
                         uuid.uuid5(uuid.NAMESPACE_URL, f"qasper:{paper_id}:{checksum}")))
    manifest = hashlib.sha256(canonical_json({"format": FORMAT_VERSION, "documents": [
        {"paper_id": d.paper["paper_id"], "sha256": d.sha256} for d in documents]})).hexdigest()
    return manifest, documents


def _project(row: dict, fields: tuple[str, ...]) -> dict:
    if FORBIDDEN_FIELDS.intersection(row):
        raise CorpusValidationError("Evaluation labels are forbidden in online corpus records")
    if set(row) != set(fields):
        raise CorpusValidationError("Corpus fields do not match the approved schema")
    return {field: row[field] for field in fields}


def _validate_base(paper: dict) -> None:
    if not all(isinstance(paper[k], str) and paper[k].strip() for k in ("paper_id", "title", "source", "version")):
        raise CorpusValidationError("Paper ID, title, source and dataset version must be nonempty text")
    if paper["source"] != "qasper" or paper["version"] != "qasper-v0.3" or paper["split"] not in {"train", "dev"}:
        raise CorpusValidationError("This importer accepts only QASPER v0.3 train/dev; test remains held out")


def ingest_qasper(repo: Repository, store: S3ObjectStore, data_dir: Path) -> dict:
    """Serialized import. Previous publication stays readable during object IO."""
    # Validate before side effects. Only these two files are opened.
    manifest, documents = build_documents(load_papers(data_dir / "papers.jsonl"),
                                          load_paragraphs(data_dir / "paragraphs.jsonl"))
    job_id = uuid.uuid4()
    with repo.import_lock() as conn:
        conn.execute("INSERT INTO ingestion_jobs(job_id,source,manifest_sha256,status) VALUES (%s,'qasper',%s,'running')",
                     (job_id, manifest))
        try:
            store.ensure_bucket()
            # Stage metadata first: even a crash between upload and publication is auditable.
            with conn.transaction():
                wrong_bucket = conn.execute("SELECT count(*) AS n FROM stored_objects WHERE bucket<>%s", (store.bucket,)).fetchone()["n"]
                if wrong_bucket:
                    raise StorageError("Configured bucket differs from existing stored objects; use an explicit migration")
                with conn.cursor() as cursor:
                    cursor.executemany("INSERT INTO stored_objects(object_key,bucket,sha256,size_bytes,content_type,state,created_by_job) "
                                       "VALUES (%s,%s,%s,%s,'application/json','staged',%s) "
                                       "ON CONFLICT(object_key) DO UPDATE SET state=CASE WHEN stored_objects.state='published' "
                                       "THEN 'published' ELSE 'staged' END,created_by_job=CASE WHEN stored_objects.state='published' "
                                       "THEN stored_objects.created_by_job ELSE excluded.created_by_job END",
                                       [(d.object_key, store.bucket, d.sha256, len(d.content), job_id) for d in documents])
            with ThreadPoolExecutor(max_workers=8) as executor:
                uploaded = sum(executor.map(lambda d: store.put_verified(d.object_key, d.content, d.sha256), documents))
            state = conn.execute("SELECT * FROM corpus_state WHERE singleton").fetchone()
            if state["manifest_sha256"] == manifest:
                existing_manifest, _ = build_documents(*repo.load_corpus())
                if existing_manifest != manifest:
                    raise StorageError("Published corpus integrity check failed; inspect database content before retrying")
                result = {"outcome": "skipped", "job_id": str(job_id), "manifest_sha256": manifest,
                          "revision": state["revision"], "papers": len(documents),
                          "paragraphs": sum(len(d.paragraphs) for d in documents), "objects_written": uploaded,
                          "objects_verified": len(documents)}
                with conn.transaction():
                    conn.execute("UPDATE stored_objects SET state='published' WHERE object_key=ANY(%s)",
                                 ([d.object_key for d in documents],))
                    _complete_job(conn, job_id, result)
                return result
            with conn.transaction():
                revision = _publish(conn, documents, manifest, job_id)
                result = {"outcome": "imported", "job_id": str(job_id), "manifest_sha256": manifest,
                          "revision": revision, "papers": len(documents),
                          "paragraphs": sum(len(d.paragraphs) for d in documents), "objects_written": uploaded,
                          "objects_verified": len(documents)}
                _complete_job(conn, job_id, result)
            return result
        except Exception as exc:
            # Never retain arbitrary exception text, DSNs or SDK request headers.
            error_code = "storage_failure" if isinstance(exc, StorageError) else "publication_failure"
            try:
                with conn.transaction():
                    conn.execute("UPDATE ingestion_jobs SET status='failed',finished_at=now(),error_code=%s WHERE job_id=%s",
                                 (error_code, job_id))
                    conn.execute("UPDATE stored_objects SET state='orphaned' WHERE created_by_job=%s AND state='staged'", (job_id,))
            except Exception:
                # If PostgreSQL itself is down, next lock holder marks stale running jobs failed.
                pass
            raise StorageError(f"Corpus import failed ({error_code}); previous publication is unchanged. Job: {job_id}") from None


def _complete_job(conn, job_id: uuid.UUID, result: dict) -> None:
    conn.execute("UPDATE ingestion_jobs SET status='completed',finished_at=now(),result=%s WHERE job_id=%s",
                 (Jsonb(result), job_id))


def _publish(conn, documents: list[PaperDocument], manifest: str, job_id: uuid.UUID) -> int:
    """All SQL below participates in the caller's one publication transaction."""
    with conn.cursor() as cursor:
        cursor.executemany("INSERT INTO papers(paper_id) VALUES (%s) ON CONFLICT DO NOTHING",
                           [(d.paper["paper_id"],) for d in documents])
        cursor.executemany("INSERT INTO paper_versions(version_id,paper_id,dataset_version,source,split,title,abstract,"
                           "object_key,content_sha256,state,paper_payload) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'archived',%s) "
                           "ON CONFLICT(version_id) DO NOTHING", [
                               (d.version_id, d.paper["paper_id"], d.paper["version"], d.paper["source"], d.paper["split"],
                                d.paper["title"], d.paper["abstract"], d.object_key, d.sha256, Jsonb(d.paper)) for d in documents])
        cursor.executemany("INSERT INTO paragraphs(version_id,chunk_id,ordinal,section_index,paragraph_index,section_name,text,"
                           "text_sha256,paragraph_payload) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", [
                               (d.version_id, p["chunk_id"], i, p["section_index"], p["paragraph_index"], p["section_name"],
                                p["text"], p["text_sha256"], Jsonb(p)) for d in documents for i, p in enumerate(d.paragraphs)])
        # Reused historical versions must be as trustworthy as newly inserted ones.
        ids = [d.version_id for d in documents]
        stored_papers = [row["paper_payload"] for row in conn.execute(
            "SELECT paper_payload FROM paper_versions WHERE version_id=ANY(%s)", (ids,))]
        stored_paragraphs = [row["paragraph_payload"] for row in conn.execute(
            "SELECT paragraph_payload FROM paragraphs WHERE version_id=ANY(%s)", (ids,))]
        actual_manifest, _ = build_documents(stored_papers, stored_paragraphs)
        if actual_manifest != manifest:
            raise StorageError("Stored version integrity check failed; publication was rolled back")
        conn.execute("UPDATE paper_versions SET state='archived' WHERE state='active'")
        conn.execute("UPDATE papers SET in_current_corpus=false")
        cursor.executemany("UPDATE paper_versions SET state='active' WHERE version_id=%s", [(d.version_id,) for d in documents])
        cursor.executemany("UPDATE papers SET current_version_id=%s,in_current_corpus=true WHERE paper_id=%s",
                           [(d.version_id, d.paper["paper_id"]) for d in documents])
        conn.execute("UPDATE stored_objects SET state='published' WHERE object_key=ANY(%s)", ([d.object_key for d in documents],))
        row = conn.execute("UPDATE corpus_state SET revision=revision+1,manifest_sha256=%s,paper_count=%s,paragraph_count=%s,"
                           "published_at=now(),published_by_job=%s WHERE singleton RETURNING revision",
                           (manifest, len(documents), sum(len(d.paragraphs) for d in documents), job_id)).fetchone()
    return row["revision"]
