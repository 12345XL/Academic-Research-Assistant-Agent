"""Local, public-corpus P1 API. Private uploads/authentication arrive in later stages."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .service import EvidenceService


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


def create_app(data_dir: Path | None = None) -> FastAPI:
    directory = data_dir or Path(os.getenv("RESEARCH_DATA_DIR", "data/processed"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.evidence = EvidenceService(directory)
        except FileNotFoundError:
            app.state.evidence = None
        yield

    app = FastAPI(
        title="科研论文助手Agent · P1 证据检索", version=__version__, lifespan=lifespan,
        description="指定论文内的原文检索基线；不是生成式问答。默认仅本机使用公共 QASPER 语料。",
    )

    def service() -> EvidenceService:
        evidence = app.state.evidence
        if evidence is None:
            raise HTTPException(503, "数据未准备好，请先运行 scripts/prepare_qasper.py")
        return evidence

    @app.get("/health")
    def health():
        evidence = app.state.evidence
        return {"status": "ready" if evidence else "data_missing", "stage": "P1",
                "version": __version__, "papers": len(evidence.papers) if evidence else 0}

    @app.get("/api/v1/papers")
    def papers(q: str = Query(default="", max_length=200), limit: int = Query(20, ge=1, le=100),
               offset: int = Query(0, ge=0)):
        items = sorted(service().papers.values(), key=lambda p: p["paper_id"])
        if q:
            items = [p for p in items if q.lower() in p["title"].lower()]
        return {"total": len(items), "items": items[offset:offset + limit]}

    @app.get("/api/v1/papers/{paper_id}")
    def paper(paper_id: str):
        item = service().papers.get(paper_id)
        if item is None:
            raise HTTPException(404, "论文不存在")
        return item

    @app.post("/api/v1/retrieve")
    def retrieve(body: RetrievalRequest):
        try:
            return service().retrieve(body.query, body.paper_id, body.top_k)
        except KeyError:
            raise HTTPException(404, "论文不存在") from None

    return app


app = create_app()
