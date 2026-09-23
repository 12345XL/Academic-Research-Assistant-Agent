"""Local research workbench. P1 files remain an explicit compatibility mode."""
from __future__ import annotations

import os
import hashlib
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .service import CorpusChangedError, CorpusIntegrityError, EvidenceService, PersistentEvidenceService
from .embeddings import VectorUnavailableError
from .reranking import RerankerUnavailableError
from .generation import AnswerService, GenerationSettings, GenerationBusyError

PaperSort = Literal["id_asc", "id_desc", "title_asc", "title_desc",
                    "submitted_newest", "submitted_oldest", "ccf_best"]
ResearchDirection = Literal["all", "nlp", "machine_learning", "information_retrieval",
                            "artificial_intelligence", "speech_audio", "computer_vision_multimedia",
                            "social_computing", "human_computer_interaction", "robotics", "other"]


class RetrievalRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=150)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["bm25", "dense", "hybrid"] = "bm25"
    rerank: bool = Field(default=False, strict=True)

    rrf_constant: int = Field(default=60, ge=1, le=1000, strict=True)
    dense_weight: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False, strict=True)

    @field_validator("query", "paper_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class AnswerRequest(RetrievalRequest):
    language: Literal["zh", "en"] = "zh"
    profile: Literal["v1", "v2"] = "v1"


def create_app(data_dir: Path | None = None, settings=None, generation_settings=None, model_client=None) -> FastAPI:
    # Explicit file or infrastructure settings stay independent of developer secrets.
    generation_settings = generation_settings or (GenerationSettings.from_env()
        if data_dir is None and settings is None else GenerationSettings())
    answers = {p: AnswerService(generation_settings, model_client, profile=p) for p in ("v1", "v2")}
    # One gate for both profiles, not one paid request per profile.
    answers["v2"]._lock = answers["v1"]._lock
    directory = data_dir or Path(os.getenv("RESEARCH_DATA_DIR", "data/processed"))
    persistent = settings is not None or (data_dir is None and bool(os.getenv("DATABASE_URL")))
    # Explicit data_dir keeps P1 evaluation/tests independent of developer secrets.
    if persistent:
        from .settings import Settings
        from .storage import Repository, S3ObjectStore, StorageError
        settings = settings or Settings.from_env()
        repository = Repository(settings)
        objects = S3ObjectStore(settings)
    else:
        repository, objects = None, None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if persistent:
            app.state.evidence = PersistentEvidenceService(repository)
        else:
            try:
                app.state.evidence = EvidenceService(directory)
            except FileNotFoundError:
                app.state.evidence = None
        yield

    app = FastAPI(
        title="科研论文助手Agent · 论文证据工作台", version=__version__, lifespan=lifespan,
        description="指定论文内的 BM25、向量与 RRF 原文证据检索；本机公共语料；可选 DeepSeek 生成与引用核验。",
    )

    def failure(status_code, detail, exc):
        content = {"detail": detail}
        if getattr(exc, "answer_run", None):
            content["run"] = exc.answer_run
        if getattr(exc, "answer_generation", None):
            content["generation"] = exc.answer_generation
        return JSONResponse(status_code=status_code, content=content)

    @app.exception_handler(GenerationBusyError)
    async def generation_busy(request: Request, exc: GenerationBusyError):
        return JSONResponse(status_code=409, content={"detail": "已有回答正在运行，请完成后再试"})

    @app.exception_handler(CorpusChangedError)
    async def changed_corpus(request: Request, exc: CorpusChangedError):
        return failure(409, "语料正在更新，请稍后重试检索", exc)

    @app.exception_handler(CorpusIntegrityError)
    async def invalid_corpus(request: Request, exc: CorpusIntegrityError):
        return failure(502, "论文证据校验失败，请检查数据并重新导入", exc)

    @app.exception_handler(VectorUnavailableError)
    async def vector_unavailable(request: Request, exc: VectorUnavailableError):
        return failure(503, str(exc), exc)

    @app.exception_handler(RerankerUnavailableError)
    async def reranker_unavailable(request: Request, exc: RerankerUnavailableError):
        return failure(503, str(exc), exc)

    @app.exception_handler(Exception)
    async def unexpected_failure(request: Request, exc: Exception):
        return failure(500, "本次请求未能完成，请稍后重试或检查服务记录", exc)

    if persistent:
        import psycopg
        from botocore.exceptions import BotoCoreError, ClientError

        async def dependency_unavailable(request: Request, exc: Exception):
            return failure(503, "存储服务暂不可用，请检查数据与存储页面后重试", exc)

        # Never leak DSNs, credentials, bucket internals or driver stack traces.
        app.add_exception_handler(psycopg.Error, dependency_unavailable)
        app.add_exception_handler(BotoCoreError, dependency_unavailable)
        app.add_exception_handler(ClientError, dependency_unavailable)
        app.add_exception_handler(StorageError, dependency_unavailable)

    def service() -> EvidenceService:
        evidence = app.state.evidence
        if evidence is None:
            raise HTTPException(503, "数据未准备好，请先运行 scripts/prepare_qasper.py")
        return evidence

    @app.get("/health")
    def health():
        if persistent:
            try:
                counts = repository.summary()
                return {"status": "ready", "stage": "P2B-3", "version": __version__, "papers": counts["papers"]}
            except Exception:
                return {"status": "database_unavailable", "stage": "P2B-3", "version": __version__, "papers": 0}
        evidence = app.state.evidence
        return {"status": "ready" if evidence else "data_missing", "stage": "P1",
                "version": __version__, "papers": len(evidence.papers) if evidence else 0}

    @app.get("/api/v1/system")
    def system():
        if not persistent:
            evidence = app.state.evidence
            return {"stage": "P1", "mode": "files", "database": {"status": "not_configured"},
                    "object_store": {"status": "not_configured", "provider": "none", "bucket": ""},
                    "corpus": {"papers": len(evidence.papers) if evidence else 0,
                               "paragraphs": evidence.index.n if evidence else 0, "objects": 0},
                    "generation": generation_settings.public_status(),
                    "capabilities": {"generation": generation_settings.configured, "pdf_upload": False}}
        try:
            counts = repository.summary()
            database_status = "ready"
        except Exception:
            counts, database_status = {"papers": 0, "paragraphs": 0, "objects": 0}, "unavailable"
        try:
            storage_health = objects.health()
            storage_status = storage_health.get("status", "unavailable")
        except Exception:
            storage_status = "unavailable"
        from .vector_store import VectorStore
        try:
            vector_status = VectorStore(repository).status(repository.revision())
        except Exception:
            vector_status = {"state": "unavailable"}
        return {"stage": "P2B-3", "mode": "postgres", "database": {"status": database_status},
                "object_store": {"status": storage_status, "provider": "S3-compatible", "bucket": settings.s3_bucket},
                "corpus": counts, "vector_index": vector_status,
                "generation": generation_settings.public_status(),
                "capabilities": {"generation": generation_settings.configured, "pdf_upload": False,
                                 "hybrid_retrieval": vector_status["state"] == "ready"}}

    @app.get("/ready")
    def ready():
        status = system()
        ok = persistent and status["database"]["status"] == status["object_store"]["status"] == "ready"
        return JSONResponse(status_code=200 if ok else 503, content=status)

    @app.get("/api/v1/ingestions")
    def ingestions(limit: int = Query(20, ge=1, le=100)):
        return {"items": repository.list_jobs(limit) if persistent else []}

    @app.get("/api/v1/papers")
    def papers(q: str = Query(default="", max_length=200), limit: int = Query(20, ge=1, le=100),
               offset: int = Query(0, ge=0),
               sort: PaperSort = "id_asc", direction: ResearchDirection = "all"):
        if persistent:
            return repository.list_papers(q=q, limit=limit, offset=offset, sort=sort, direction=direction)
        items = list(service().papers.values())
        if q:
            items = [p for p in items if q.lower() in p["title"].lower()]
        if direction != "all":
            items = [p for p in items if p.get("research_direction") == direction]
        if sort == "id_desc":
            items.sort(key=lambda p: p["paper_id"], reverse=True)
        elif sort in ("title_asc", "title_desc"):
            items.sort(key=lambda p: (p["title"].casefold(), p["paper_id"]),
                       reverse=sort == "title_desc")
        elif sort in ("submitted_newest", "submitted_oldest"):
            # P1 compatibility files contain modern arXiv IDs but no timestamp.
            # YYMM + sequence is a deterministic month/order approximation only.
            items.sort(key=lambda p: p["paper_id"],
                       reverse=sort == "submitted_newest")
        elif sort == "ccf_best":
            items.sort(key=lambda p: ({"A": 0, "B": 1, "C": 2}.get(p.get("ccf_level"), 3),
                                      p["paper_id"]))
        else:
            items.sort(key=lambda p: p["paper_id"])
        return {"total": len(items), "items": items[offset:offset + limit]}

    @app.get("/api/v1/papers/{paper_id}")
    def paper(paper_id: str):
        item = repository.get_paper(paper_id) if persistent else service().papers.get(paper_id)
        if item is None:
            raise HTTPException(404, "论文不存在")
        return item

    @app.get("/api/v1/papers/{paper_id}/paragraphs")
    def paragraphs(paper_id: str, limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)):
        paper(paper_id)
        if persistent:
            return repository.get_paragraphs(paper_id, limit=limit, offset=offset)
        items = [p for p in service().index.paragraphs.values() if p["paper_id"] == paper_id]
        items.sort(key=lambda p: (p["section_index"], p["paragraph_index"]))
        return {"total": len(items), "items": items[offset:offset + limit]}

    @app.get("/api/v1/papers/{paper_id}/source")
    def source(paper_id: str):
        paper(paper_id)
        if not persistent:
            raise HTTPException(409, "文件模式未接入对象存储")
        metadata = repository.get_paper_object(paper_id)
        if metadata is None:
            raise HTTPException(404, "该论文没有可用的结构化原文")
        body = objects.read_bytes(metadata["object_key"])
        if hashlib.sha256(body).hexdigest() != metadata["sha256"] or len(body) != metadata["size_bytes"]:
            raise HTTPException(502, "原文文件校验失败，请检查存储或重新导入")
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id) + ".json"
        return Response(body, media_type="application/json", headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-SHA256": metadata["sha256"], "Cache-Control": "no-store",
        })

    @app.post("/api/v1/retrieve")
    def retrieve(body: RetrievalRequest):
        try:
            if persistent:
                return service().retrieve(body.query, body.paper_id, body.top_k, mode=body.mode, rerank=body.rerank,
                                          rrf_constant=body.rrf_constant, dense_weight=body.dense_weight)
            if body.mode != "bm25" or body.rerank:
                raise HTTPException(409, "向量、混合与重排检索需要 PostgreSQL 模式及相应模型/索引")
            return service().retrieve(body.query, body.paper_id, body.top_k)
        except KeyError:
            raise HTTPException(404, "论文不存在") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/v1/answer")
    def answer(body: AnswerRequest):
        if not persistent and (body.mode != "bm25" or body.rerank):
            raise HTTPException(409, "向量、混合与重排检索需要 PostgreSQL 模式及相应模型/索引")
        try:
            return answers[body.profile].answer(service(), body.query, body.paper_id, body.top_k, body.mode, body.rerank,
                                                language=body.language, rrf_constant=body.rrf_constant, dense_weight=body.dense_weight)
        except KeyError as exc:
            return failure(404, "论文不存在", exc)
        except ValueError as exc:
            return failure(422, "输入超出模型或检索限制，请缩短问题后重试", exc)

    return app


app = create_app()
