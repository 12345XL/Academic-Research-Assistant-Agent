from __future__ import annotations

import hashlib
import time
import uuid
import threading
from pathlib import Path
from typing import Any

from .dataset import load_papers, load_paragraphs
from .retrieval import BM25Index


class EvidenceService:
    """Loads only public corpus files, never questions.jsonl or gold answers."""

    def __init__(self, data_dir: Path):
        papers = load_papers(data_dir / "papers.jsonl")
        self.papers = {p["paper_id"]: p for p in papers}
        self.index = BM25Index(load_paragraphs(data_dir / "paragraphs.jsonl"))
        self.data_dir = data_dir

    @classmethod
    def from_records(cls, papers: list[dict], paragraphs: list[dict]) -> "EvidenceService":
        """Reuse exactly the P1 ranking algorithm over a database snapshot."""
        service = cls.__new__(cls)
        service.papers = {p["paper_id"]: p for p in papers}
        service.index = BM25Index(paragraphs)
        service.data_dir = None
        return service

    def retrieve(self, query: str, paper_id: str, top_k: int = 5) -> dict[str, Any]:
        if paper_id not in self.papers:
            raise KeyError(paper_id)
        if not query.strip():
            raise ValueError("query cannot be empty")
        started = time.perf_counter()
        hits = self.index.search(query, paper_id, top_k)
        citations = []
        for rank, hit in enumerate(hits, 1):
            p = hit.paragraph
            citations.append({
                "rank": rank, "chunk_id": p["chunk_id"], "paper_id": p["paper_id"],
                "title": p["title"], "section_name": p["section_name"],
                "section_index": p["section_index"], "paragraph_index": p["paragraph_index"],
                "text": p["text"], "score": round(hit.score, 6),
                "text_sha256": p["text_sha256"],
                "source": p["source"], "version": p["version"],
            })
        return {
            "trace_id": uuid.uuid4().hex,
            "mode": "evidence_only", "scope": "specified_paper",
            "query": query, "paper_id": paper_id, "top_k": top_k,
            "status": "evidence_found" if citations else "no_lexical_match",
            "citations": citations,
            "notice": "仅返回 BM25 检索原文，尚未生成或验证答案；未命中不代表论文无法回答。",
            "trace": {
                "retriever": "bm25", "k1": self.index.k1, "b": self.index.b,
                "corpus_paragraphs": self.index.n,
                "paper_paragraphs": len(self.index.paper_chunks.get(paper_id, set())),
                "returned": len(citations),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "model_calls": 0,
            },
        }


class CorpusChangedError(RuntimeError):
    """Publishing raced both attempts; client may retry with the current corpus."""


class CorpusIntegrityError(RuntimeError):
    """A committed database fact does not match its recorded content hash."""


class PersistentEvidenceService:
    """A rebuildable BM25 cache; PostgreSQL remains the source of returned facts.

    Every request checks the published revision in PostgreSQL. A disconnected
    database cannot silently turn this service into an out-of-date file backend.
    S3 is used for source downloads, not required to read committed paragraphs.
    """

    def __init__(self, repository):
        self.repository = repository
        self._revision = None
        self._evidence = None
        self._lock = threading.RLock()

    def _ensure_snapshot(self):
        revision = self.repository.revision()
        if self._evidence is None or revision != self._revision:
            actual_revision, papers, paragraphs = self.repository.load_snapshot()
            self._evidence = EvidenceService.from_records(papers, paragraphs)
            self._revision = actual_revision

    def retrieve(self, query: str, paper_id: str, top_k: int = 5) -> dict[str, Any]:
        started = time.perf_counter()
        with self._lock:
            for _ in range(2):
                self._ensure_snapshot()
                result = self._evidence.retrieve(query, paper_id, top_k)
                citations = result["citations"]
                facts = self.repository.get_chunks([item["chunk_id"] for item in citations], paper_id)
                if any(
                    item["chunk_id"] in facts
                    and hashlib.sha256(facts[item["chunk_id"]]["text"].encode("utf-8")).hexdigest()
                    != facts[item["chunk_id"]]["text_sha256"]
                    for item in citations
                ):
                    raise CorpusIntegrityError("数据库证据内容与哈希不一致")
                changed = any(item["chunk_id"] not in facts or
                              facts[item["chunk_id"]]["text_sha256"] != item["text_sha256"]
                              for item in citations)
                if changed or self.repository.revision() != self._revision:
                    self._evidence = None
                    continue
                for item in citations:
                    fact = facts[item["chunk_id"]]
                    for field in ("title", "text", "text_sha256", "section_name", "section_index",
                                  "paragraph_index", "source", "version"):
                        item[field] = fact[field]
                result["trace"].update(storage="postgres", corpus_revision=str(self._revision), fact_check="database")
                result["trace"]["bm25_latency_ms"] = result["trace"]["latency_ms"]
                result["trace"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                return result
        raise CorpusChangedError("语料正在更新，请重试检索")
