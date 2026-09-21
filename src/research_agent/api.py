"""Local research workbench. P1 files remain an explicit compatibility mode."""
from __future__ import annotations

import os
import hashlib
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .service import CorpusChangedError, CorpusIntegrityError, EvidenceService, PersistentEvidenceService


class RetrievalRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=150)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query", "paper_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


def create_app(data_dir: Path | None = None, settings=None) -> FastAPI:
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
        description="指定论文内的原文证据检索。P2A 接入 PostgreSQL 与 S3；当前仅本机公共语料，无生成式回答。",
    )

    @app.exception_handler(CorpusChangedError)
    async def changed_corpus(request: Request, exc: CorpusChangedError):
        return JSONResponse(status_code=409, content={"detail": "语料正在更新，请稍后重试检索"})

    @app.exception_handler(CorpusIntegrityError)
    async def invalid_corpus(request: Request, exc: CorpusIntegrityError):
        return JSONResponse(status_code=502, content={"detail": "论文证据校验失败，请检查数据并重新导入"})

    if persistent:
        import psycopg
        from botocore.exceptions import BotoCoreError, ClientError

        async def dependency_unavailable(request: Request, exc: Exception):
            return JSONResponse(status_code=503, content={"detail": "存储服务暂不可用，请检查数据与存储页面后重试"})

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
                return {"status": "ready", "stage": "P2A", "version": __version__, "papers": counts["papers"]}
            except Exception:
                return {"status": "database_unavailable", "stage": "P2A", "version": __version__, "papers": 0}
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
                    "capabilities": {"generation": False, "pdf_upload": False}}
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
        return {"stage": "P2A", "mode": "postgres", "database": {"status": database_status},
                "object_store": {"status": storage_status, "provider": "S3-compatible", "bucket": settings.s3_bucket},
                "corpus": counts, "capabilities": {"generation": False, "pdf_upload": False}}

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
               offset: int = Query(0, ge=0)):
        if persistent:
            return repository.list_papers(q=q, limit=limit, offset=offset)
        items = sorted(service().papers.values(), key=lambda p: p["paper_id"])
        if q:
            items = [p for p in items if q.lower() in p["title"].lower()]
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
            return service().retrieve(body.query, body.paper_id, body.top_k)
        except KeyError:
            raise HTTPException(404, "论文不存在") from None

    return app


app = create_app()
