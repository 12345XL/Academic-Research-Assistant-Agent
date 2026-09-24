from __future__ import annotations

import hashlib
import math
import time
import uuid
import threading
from pathlib import Path
from typing import Any

from .dataset import load_papers, load_paragraphs
from .retrieval import BM25Index, reciprocal_rank_fusion


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

    def __init__(self, repository, encoder=None, vector_store=None, reranker=None, *, paper_id=None):
        self.repository = repository
        self._paper_id = paper_id
        self._revision = None
        self._evidence = None
        self._lock = threading.RLock()
        self._encoder = encoder
        self._vector_store = vector_store
        self._reranker = reranker

    def _candidates(self, query, paper_id, top_k, mode, query_vector, rrf_constant=60, dense_weight=0.5):
        if paper_id not in self._evidence.papers:
            raise KeyError(paper_id)
        if mode == "bm25":
            return self._evidence.retrieve(query, paper_id, top_k)
        from .embeddings import LocalEncoder, CONFIG, VectorUnavailableError, collection_id
        from .vector_store import VectorStore
        self._vector_store = self._vector_store or VectorStore(self.repository)
        if self._vector_store.status(self._revision).get("state") != "ready":
            raise VectorUnavailableError("当前语料的向量索引未就绪，请完成索引构建")
        self._encoder = self._encoder or LocalEncoder()
        pool = max(20, top_k)
        if mode == "hybrid":
            result = self._evidence.retrieve(query, paper_id, pool)
        else:
            result = {"trace_id": uuid.uuid4().hex, "mode": "evidence_only", "scope": "specified_paper",
                      "query": query, "paper_id": paper_id, "citations": [],
                      "trace": {"k1": self._evidence.index.k1, "b": self._evidence.index.b,
                                "corpus_paragraphs": self._evidence.index.n,
                                "paper_paragraphs": len(self._evidence.index.paper_chunks.get(paper_id, set())),
                                "latency_ms": 0}}
        vector_started = time.perf_counter()
        vector = query_vector if query_vector is not None else self._encoder.encode_queries([query])[0]
        vector_hits = self._vector_store.search(vector, paper_id, self._revision, pool)
        vector_latency = (time.perf_counter() - vector_started) * 1000
        bm25 = {item["chunk_id"]: item for item in result["citations"]}
        dense = {item["chunk_id"]: item for item in vector_hits}
        for item in vector_hits:
            cached = self._evidence.index.paragraphs.get(item["chunk_id"])
            if cached is None or cached["paper_id"] != paper_id or cached["text_sha256"] != item["text_sha256"]:
                raise CorpusIntegrityError("向量候选与当前证据不一致")
        ranked = (reciprocal_rank_fusion([list(bm25), list(dense)], top_k, rrf_constant,
                                          [2 * (1 - dense_weight), 2 * dense_weight])
                  if mode == "hybrid" else [(r["chunk_id"], r["score"]) for r in vector_hits[:top_k]])
        citations = []
        for rank, (chunk_id, score) in enumerate(ranked, 1):
            paragraph = self._evidence.index.paragraphs[chunk_id]
            citations.append({**{field: paragraph[field] for field in (
                "chunk_id", "paper_id", "title", "section_name", "section_index", "paragraph_index",
                "text", "text_sha256", "source", "version")}, "rank": rank, "score": round(score, 8),
                "retrieval_scores": {"bm25": bm25.get(chunk_id, {}).get("score") if mode == "hybrid" else None,
                                     "cosine": dense.get(chunk_id, {}).get("score")}})
        result.update(top_k=top_k, citations=citations, status="evidence_found" if citations else "no_evidence",
                      notice="仅返回检索原文；相似度或融合分数不是置信度，尚未生成或验证答案。")
        result["trace"].update(retriever=mode, candidate_pool=pool, rrf_constant=rrf_constant if mode == "hybrid" else None,
                               dense_weight=dense_weight if mode == "hybrid" else None,
                               vector_collection=collection_id(self._revision), embedding=CONFIG,
                               vector_latency_ms=round(vector_latency, 3), model_calls=0 if query_vector is not None else 1,
                               query_embedding_precomputed=query_vector is not None, returned=len(citations),
                               score_kind="rrf" if mode == "hybrid" else "cosine",
                               bm25_candidates=len(bm25) if mode == "hybrid" else 0, dense_candidates=len(dense))
        return result

    def _ensure_snapshot(self):
        revision = self.repository.revision()
        if self._evidence is None or revision != self._revision:
            actual_revision, papers, paragraphs = (self.repository.load_paper_snapshot(self._paper_id)
                if self._paper_id else self.repository.load_snapshot())
            self._evidence = EvidenceService.from_records(papers, paragraphs)
            self._revision = actual_revision

    def retrieve(self, query: str, paper_id: str, top_k: int = 5, mode: str = "bm25", *,
                 query_vector=None, rerank: bool = False, rrf_constant: int = 60, dense_weight: float = 0.5) -> dict[str, Any]:
        if (type(rrf_constant) is not int or not 1 <= rrf_constant <= 1000
                or type(dense_weight) not in (int, float) or not math.isfinite(dense_weight)
                or not 0 <= dense_weight <= 1):
            raise ValueError("Invalid RRF constant or dense weight")
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if not 1 <= top_k <= 50 or not query.strip():
            raise ValueError("Invalid query or top_k")
        if self._paper_id and (mode != "bm25" or rerank):
            raise ValueError("上传 PDF 当前仅支持 BM25，不支持向量、混合或重排")
        started = time.perf_counter()
        with self._lock:
            for _ in range(2):
                self._ensure_snapshot()
                result = self._candidates(query, paper_id, max(20, top_k) if rerank else top_k, mode, query_vector, rrf_constant, dense_weight)
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
                result["trace"]["rerank_enabled"] = rerank
                if rerank:
                    from .reranking import CONFIG as RERANK_CONFIG, LocalReranker, apply_reranking
                    self._reranker = self._reranker or LocalReranker()
                    rerank_started = time.perf_counter()
                    # Only checked database facts reach the cross-encoder, never cached candidate text or gold.
                    scores, stats = self._reranker.score(query, [item["text"] for item in citations])
                    reranked = apply_reranking(citations, scores, top_k)
                    if self.repository.revision() != self._revision:
                        self._evidence = None
                        continue
                    result.update(citations=reranked, top_k=top_k,
                                  notice="已对召回候选进行本地模型重排；分数不是概率，尚未生成或验证答案。")
                    result["trace"].update(
                        reranker=RERANK_CONFIG, reranker_stats=stats, score_kind="cross_encoder_logit",
                        rerank_candidates=[item["chunk_id"] for item in citations],
                        rerank_candidate_count=len(citations), returned=len(reranked),
                        rerank_latency_ms=round((time.perf_counter() - rerank_started) * 1000, 3),
                        model_calls=result["trace"]["model_calls"] + (1 if citations else 0))
                result["trace"].update(storage="postgres", corpus_revision=str(self._revision), fact_check="database")
                result["trace"]["bm25_latency_ms"] = result["trace"]["latency_ms"]
                result["trace"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                return result
        raise CorpusChangedError("语料正在更新，请重试检索")
