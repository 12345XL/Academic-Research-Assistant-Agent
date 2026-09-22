"""Local cross-encoder reranking, after candidate fact checks and before publication."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from .embeddings import token_windows

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CONFIG = {"model": MODEL_ID, "revision": MODEL_REVISION, "max_tokens": 512,
          "max_query_tokens": 128, "overlap_tokens": 64, "max_windows": 256,
          "max_candidates": 50, "aggregation": "max-logit", "query_prefix": "",
          "preprocessing_version": 1}


class RerankerUnavailableError(RuntimeError):
    pass


def model_directory() -> Path:
    return Path(os.getenv("RESEARCH_RERANKER_DIR", ".local/models/ms-marco-MiniLM-L6-v2"))


def pair_windows(query_ids: list[int], passage_ids: list[int], cls: int, sep: int):
    if not 1 <= len(query_ids) <= CONFIG["max_query_tokens"]:
        raise ValueError("重排问题需要 1–128 个 tokenizer token，请缩短问题")
    budget = CONFIG["max_tokens"] - len(query_ids) - 3
    for window in token_windows(passage_ids, budget, CONFIG["overlap_tokens"]):
        yield ([cls, *query_ids, sep, *window, sep],
               [0] * (len(query_ids) + 2) + [1] * (len(window) + 1))


def apply_reranking(citations: list[dict], scores: list[float], top_k: int) -> list[dict]:
    if len(scores) != len(citations) or any(not math.isfinite(score) for score in scores):
        raise RerankerUnavailableError("重排模型输出不完整或分数无效")
    if len({item["chunk_id"] for item in citations}) != len(citations):
        raise RerankerUnavailableError("重排候选 ID 重复")
    ranked = sorted(zip(citations, scores), key=lambda pair: (-pair[1], pair[0]["chunk_id"]))[:top_k]
    return [{**item, "rank": rank, "score": round(score, 6),
             "candidate_rank": item["rank"], "candidate_score": item["score"]}
            for rank, (item, score) in enumerate(ranked, 1)]


class LocalReranker:
    def __init__(self, device: str | None = None):
        self.device = device or os.getenv("RESEARCH_RERANKER_DEVICE", "cpu")
        self._lock = threading.RLock()
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            directory = model_directory()
            expected = json.loads((Path(__file__).parent / "model_manifests/minilm-reranker.json").read_text())
            if expected["model"] != MODEL_ID or expected["revision"] != MODEL_REVISION:
                raise ValueError("Model identity mismatch")
            for filename, checksum in expected["sha256"].items():
                if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != checksum:
                    raise ValueError("Model checksum mismatch")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(4)
            self._tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False, use_safetensors=True)
            self._model = model.to(self.device).eval()
        except (ImportError, OSError, ValueError, KeyError, RuntimeError):
            raise RerankerUnavailableError("本地重排模型未就绪，请运行准备脚本并检查推理设备") from None

    def score(self, query: str, texts: list[str]) -> tuple[list[float], dict]:
        if len(texts) > CONFIG["max_candidates"]:
            raise ValueError("重排候选超过 50 段预算")
        if not texts:
            return [], {"windows": 0, "batches": 0, "load_ms": 0, "inference_ms": 0}
        with self._lock:
            started = time.perf_counter()
            cold = self._model is None
            self._load()
            loaded = time.perf_counter()
            query_ids = self._tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"]
            passage_ids = self._tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            pairs, owners = [], []
            for index, ids in enumerate(passage_ids):
                for pair in pair_windows(query_ids, ids, self._tokenizer.cls_token_id, self._tokenizer.sep_token_id):
                    pairs.append(pair)
                    owners.append(index)
                    if len(pairs) > CONFIG["max_windows"]:
                        raise ValueError("正文超过单次重排 256 个窗口预算，请减少返回证据数量")
            import torch
            scores = [-math.inf] * len(texts)
            try:
                for start in range(0, len(pairs), 16):
                    batch = pairs[start:start + 16]
                    width = max(len(ids) for ids, _ in batch)
                    inputs = {
                        "input_ids": [ids + [self._tokenizer.pad_token_id] * (width - len(ids)) for ids, _ in batch],
                        "attention_mask": [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch],
                        "token_type_ids": [types + [0] * (width - len(types)) for _, types in batch],
                    }
                    tensors = {key: torch.tensor(value, device=self.device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        logits = self._model(**tensors).logits.reshape(-1).cpu().float().tolist()
                    if len(logits) != len(batch) or any(not math.isfinite(x) for x in logits):
                        raise RerankerUnavailableError("重排模型返回无效分数")
                    for owner, logit in zip(owners[start:start + 16], logits):
                        scores[owner] = max(scores[owner], logit)
            except RuntimeError:
                raise RerankerUnavailableError("重排推理失败，请检查本机模型资源与设备") from None
            return scores, {"windows": len(pairs), "batches": math.ceil(len(pairs) / 16),
                            "load_ms": round((loaded - started) * 1000, 3) if cold else 0,
                            "inference_ms": round((time.perf_counter() - loaded) * 1000, 3),
                            "device": self.device}
