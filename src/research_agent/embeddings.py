"""Pinned local BGE encoder. No model download is allowed on an API request."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from pathlib import Path

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CONFIG = {
    "model": MODEL_ID, "revision": MODEL_REVISION, "dimensions": 384,
    "pooling": "normalized-cls", "max_tokens": 512, "window_tokens": 510,
    "overlap_tokens": 64, "paragraph_aggregation": "max-cosine",
    "query_prefix": QUERY_PREFIX, "preprocessing_version": 1,
}
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
MODEL_SHA256 = {
    "config.json": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    "model.safetensors": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


class VectorUnavailableError(RuntimeError):
    """Configuration, model or published vector collection is unavailable."""


def model_directory() -> Path:
    return Path(os.getenv("RESEARCH_EMBEDDING_DIR", ".local/models/bge-small-en-v1.5"))


def collection_id(revision: int) -> str:
    return hashlib.sha256(json.dumps({"revision": revision, "config": CONFIG},
                                     sort_keys=True).encode()).hexdigest()


def token_windows(ids: list[int], size: int = 510, overlap: int = 64) -> list[list[int]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid window size or overlap")
    windows = []
    for start in range(0, max(1, len(ids)), size - overlap):
        windows.append(ids[start:start + size])
        if start + size >= len(ids):
            break
    return windows


def vector_literal(values) -> str:
    values = [float(v) for v in values]
    if len(values) != 384 or not all(math.isfinite(v) for v in values) or not any(values):
        raise ValueError("Expected a nonzero finite 384-dimensional vector")
    return "[" + ",".join(str(v) for v in values) + "]"


class LocalEncoder:
    def __init__(self, device: str | None = None):
        self.device = device or os.getenv("RESEARCH_EMBEDDING_DEVICE", "cpu")
        self._lock = threading.RLock()
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        directory = model_directory()
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["model"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
                raise ValueError("Model identity mismatch")
            for filename in MODEL_FILES:
                actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                if actual != manifest["sha256"][filename] or actual != MODEL_SHA256[filename]:
                    raise ValueError("Model file checksum mismatch")
            import torch
            from transformers import AutoModel, AutoTokenizer
            torch.set_num_threads(4)
            self._tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                             use_safetensors=True)
            self._model = model.to(self.device).eval()
        except (ImportError, OSError, ValueError, KeyError, RuntimeError):
            raise VectorUnavailableError("本地向量模型未就绪，请运行模型准备脚本并检查推理设备") from None

    def _encode_ids(self, windows: list[list[int]], batch_size: int = 32) -> list[list[float]]:
        import torch
        result = []
        for start in range(0, len(windows), batch_size):
            sequences = [[self._tokenizer.cls_token_id, *ids, self._tokenizer.sep_token_id]
                         for ids in windows[start:start + batch_size]]
            width = max(map(len, sequences))
            batch = {
                "input_ids": torch.tensor([ids + [self._tokenizer.pad_token_id] * (width - len(ids))
                                           for ids in sequences], device=self.device),
                "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids))
                                                for ids in sequences], device=self.device),
            }
            with torch.inference_mode():
                output = self._model(**batch).last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(output, p=2, dim=1)
            result.extend(normalized.cpu().float().tolist())
        return result

    def encode_paragraphs(self, texts: list[str]) -> list[list[list[float]]]:
        with self._lock:
            self._load()
            ids = self._tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            grouped = [token_windows(tokens) for tokens in ids]
            vectors = self._encode_ids([window for windows in grouped for window in windows])
            result, offset = [], 0
            for windows in grouped:
                result.append(vectors[offset:offset + len(windows)])
                offset += len(windows)
            return result

    def encode_queries(self, queries: list[str]) -> list[list[float]]:
        with self._lock:
            self._load()
            ids = self._tokenizer([QUERY_PREFIX + q for q in queries], add_special_tokens=False,
                                  truncation=False)["input_ids"]
            if any(len(tokens) > 510 for tokens in ids):
                raise ValueError("问题超出向量模型的 510 token 输入限制，请缩短问题")
            return self._encode_ids(ids)
