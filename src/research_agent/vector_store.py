"""Rebuildable, version-bound pgvector collections with atomic readiness."""
from __future__ import annotations

import hashlib
import time

from psycopg.types.json import Jsonb

from .embeddings import CONFIG, VectorUnavailableError, collection_id, vector_literal
from .service import CorpusChangedError, CorpusIntegrityError

BUILD_LOCK_ID = 71420520922


class VectorStore:
    def __init__(self, repository):
        self.repository = repository

    def status(self, revision: int) -> dict:
        with self.repository.connect() as conn:
            row = conn.execute("SELECT collection_id,state,paragraph_count,window_count,config "
                               "FROM vector_collections WHERE collection_id=%s",
                               (collection_id(revision),)).fetchone()
        return dict(row) if row else {"state": "missing"}

    def search(self, vector, paper_id: str, revision: int, top_k: int) -> list[dict]:
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        with self.repository.connect() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            row = conn.execute("SELECT state,config FROM vector_collections WHERE collection_id=%s",
                               (collection_id(revision),)).fetchone()
            if not row or row["state"] != "ready" or row["config"] != CONFIG:
                raise VectorUnavailableError("当前语料的向量索引未就绪，请完成索引构建")
            current = conn.execute("SELECT revision FROM corpus_state WHERE singleton").fetchone()["revision"]
            if current != revision:
                raise CorpusChangedError("语料版本发生变化，请重试")
            rows = conn.execute("""
                SELECT e.chunk_id,e.text_sha256,min(e.embedding <=> %s::vector) AS distance
                FROM papers p JOIN paper_versions v ON v.version_id=p.current_version_id
                JOIN paragraph_vectors e ON e.version_id=v.version_id
                WHERE p.paper_id=%s AND p.in_current_corpus AND v.state='active'
                    AND e.collection_id=%s
                GROUP BY e.chunk_id,e.text_sha256 ORDER BY distance,e.chunk_id LIMIT %s
            """, (vector_literal(vector), paper_id, collection_id(revision), top_k)).fetchall()
        return [{"chunk_id": r["chunk_id"], "text_sha256": r["text_sha256"],
                 "score": 1 - r["distance"]} for r in rows]

    def build(self, encoder, batch_size: int = 128, progress=None) -> dict:
        if not 1 <= batch_size <= 512:
            raise ValueError("batch_size must be between 1 and 512")
        started = time.perf_counter()
        with self.repository.connect(autocommit=True) as lock:
            if not lock.execute("SELECT pg_try_advisory_lock(%s) AS ok", (BUILD_LOCK_ID,)).fetchone()["ok"]:
                raise VectorUnavailableError("另一个向量索引任务正在运行")
            try:
                return self._build_locked(encoder, batch_size, progress, started)
            finally:
                lock.execute("SELECT pg_advisory_unlock(%s)", (BUILD_LOCK_ID,))

    def _build_locked(self, encoder, batch_size, progress, started):
        with self.repository.connect() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            revision = conn.execute("SELECT revision FROM corpus_state WHERE singleton").fetchone()["revision"]
            cid = collection_id(revision)
            paragraphs = conn.execute("""
                SELECT x.version_id,x.chunk_id,x.text,x.text_sha256 FROM papers p
                JOIN paper_versions v ON v.version_id=p.current_version_id
                JOIN paragraphs x ON x.version_id=v.version_id
                WHERE p.in_current_corpus AND v.state='active' ORDER BY p.paper_id,x.ordinal
            """).fetchall()
            if not paragraphs:
                raise VectorUnavailableError("请先导入论文正文")
            conn.execute("INSERT INTO vector_collections(collection_id,corpus_revision,config,state) "
                         "VALUES (%s,%s,%s,'building') ON CONFLICT DO NOTHING", (cid, revision, Jsonb(CONFIG)))
            state = conn.execute("SELECT * FROM vector_collections WHERE collection_id=%s", (cid,)).fetchone()
            if state["config"] != CONFIG:
                raise VectorUnavailableError("向量配置不一致")
            if state["state"] == "ready":
                return {"outcome": "reused", "collection_id": cid, "corpus_revision": revision,
                        "paragraphs": state["paragraph_count"], "windows": state["window_count"]}
            completed = {(str(r["version_id"]), r["chunk_id"]) for r in conn.execute(
                "SELECT DISTINCT version_id,chunk_id FROM paragraph_vectors WHERE collection_id=%s", (cid,))}
        pending = [p for p in paragraphs if (str(p["version_id"]), p["chunk_id"]) not in completed]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            if self.repository.revision() != revision:
                raise CorpusChangedError("构建期间语料换版，请重新运行索引任务")
            for p in batch:
                if hashlib.sha256(p["text"].encode()).hexdigest() != p["text_sha256"]:
                    raise CorpusIntegrityError("数据库正文哈希不符")
            encoded = encoder.encode_paragraphs([p["text"] for p in batch])
            if len(encoded) != len(batch) or any(not windows for windows in encoded):
                raise ValueError("Encoder returned incomplete paragraph vectors")
            with self.repository.connect() as conn:
                # A batch commits every window of each paragraph or none; retry skips only complete paragraphs.
                with conn.cursor() as cursor:
                    cursor.executemany("INSERT INTO paragraph_vectors VALUES (%s,%s,%s,%s,%s,%s::vector)", [
                        (cid, p["version_id"], p["chunk_id"], index, p["text_sha256"], vector_literal(vector))
                        for p, windows in zip(batch, encoded) for index, vector in enumerate(windows)])
            if progress:
                progress(min(start + batch_size, len(pending)) + len(completed), len(paragraphs))
        with self.repository.connect() as conn:
            # Share-lock publication row so an importer cannot publish between check and readiness commit.
            current = conn.execute("SELECT revision FROM corpus_state WHERE singleton FOR SHARE").fetchone()["revision"]
            if current != revision:
                raise CorpusChangedError("构建期间语料换版，请重新运行索引任务")
            counts = conn.execute("SELECT count(DISTINCT (version_id,chunk_id)) AS paragraphs,count(*) AS windows "
                                  "FROM paragraph_vectors WHERE collection_id=%s", (cid,)).fetchone()
            if counts["paragraphs"] != len(paragraphs):
                raise VectorUnavailableError("索引覆盖不完整")
            conn.execute("UPDATE vector_collections SET state='ready',paragraph_count=%s,window_count=%s,ready_at=now() "
                         "WHERE collection_id=%s", (counts["paragraphs"], counts["windows"], cid))
        return {"outcome": "built", "collection_id": cid, "corpus_revision": revision,
                **counts, "resumed_paragraphs": len(completed),
                "elapsed_seconds": round(time.perf_counter() - started, 2), "config": CONFIG}
