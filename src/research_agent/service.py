from __future__ import annotations

import time
import uuid
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
