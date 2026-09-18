"""Small, inspectable Okapi BM25 baseline over paper paragraphs.

The corpus-wide document frequencies are computed from paragraph text only.
Question text, answers and annotated evidence never enter the index.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


def tokenize(text: str) -> list[str]:
    """English baseline; keep scientific identifiers and numbers as lexical tokens."""
    return re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())


@dataclass(frozen=True)
class Hit:
    paragraph: dict[str, Any]
    score: float


class BM25Index:
    def __init__(self, paragraphs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be in [0, 1]")
        self.k1, self.b = k1, b
        self.paragraphs = {p["chunk_id"]: dict(p) for p in paragraphs}
        if len(self.paragraphs) != len(paragraphs):
            raise ValueError("Duplicate chunk_id in corpus")
        self.paper_chunks: dict[str, set[str]] = defaultdict(set)
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.lengths: dict[str, int] = {}
        for chunk_id, paragraph in self.paragraphs.items():
            self.paper_chunks[paragraph["paper_id"]].add(chunk_id)
            counts = Counter(tokenize(paragraph["text"]))
            self.lengths[chunk_id] = sum(counts.values())
            for term, tf in counts.items():
                self.postings[term][chunk_id] = tf
        self.n = len(paragraphs)
        self.avgdl = sum(self.lengths.values()) / self.n if self.n else 1.0
        self.avgdl = self.avgdl or 1.0

    def search(self, query: str, paper_id: str, top_k: int = 5) -> list[Hit]:
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        allowed = self.paper_chunks.get(paper_id, set())
        scores: dict[str, float] = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            posting = self.postings.get(term, {})
            df = len(posting)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for chunk_id in allowed.intersection(posting):
                tf = posting[chunk_id]
                norm = 1 - self.b + self.b * self.lengths[chunk_id] / self.avgdl
                scores[chunk_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
        return [Hit(self.paragraphs[cid], scores[cid]) for cid in ranked]
