"""Local research workbench. P1 files remain an explicit compatibility mode."""
from __future__ import annotations

import os
import hashlib
import re
import asyncio
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .service import CorpusChangedError, CorpusIntegrityError, EvidenceService, PersistentEvidenceService
from .embeddings import VectorUnavailableError
from .reranking import RerankerUnavailableError
from .generation import AnswerService, GenerationSettings, GenerationBusyError
from .harness import AnswerRun, exception_reason
from .runtime import AnswerJob, RunControl, RunLimits, RunStopped
from .access import AccessError, AccessPolicy
from .run_store import MemoryRunStore, PostgresRunStore, RunStoreError, RunStoreConflictError
from .feedback import FeedbackInput, FeedbackConflictError
from .pdf_parser import PdfError, MAX_BYTES
from .pdf_ingestion import ingest_pdf

PaperSort = Literal["id_asc", "id_desc", "title_asc", "title_desc",
                    "submitted_newest", "submitted_oldest", "ccf_best"]
ResearchDirection = Literal["all", "nlp", "machine_learning", "information_retrieval",
                            "artificial_intelligence", "speech_audio", "computer_vision_multimedia",
                            "social_computing", "human_computer_interaction", "robotics", "other"]


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    run_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    allow_repair: bool = Field(default=False, strict=True)


def create_app(data_dir: Path | None = None, settings=None, generation_settings=None, model_client=None,
               access_policy=None, run_store=None, run_limits=None) -> FastAPI:
    explicit = data_dir is not None or settings is not None
    policy = access_policy or (AccessPolicy() if explicit else AccessPolicy.from_env())
    limits = run_limits or (RunLimits() if explicit else RunLimits.from_env())
    dispatch_lock = threading.Lock()
    upload_lock = threading.Lock()
    # Explicit file or infrastructure settings stay independent of developer secrets.
    generation_settings = generation_settings or (GenerationSettings.from_env()
        if data_dir is None and settings is None else GenerationSettings())
    answers = {p: AnswerService(generation_settings, model_client, profile=p, limits=limits) for p in ("v1", "v2")}
    # One gate for both profiles, not one paid request per profile.
    answers["v2"]._lock = answers["v1"]._lock
    directory = data_dir or Path(os.getenv("RESEARCH_DATA_DIR", "data/processed"))
    persistent = settings is not None or (data_dir is None and bool(os.getenv("DATABASE_URL")))
    # Explicit data_dir keeps P1 evaluation/tests independent of developer secrets.
    if persistent:
        from .settings import Settings
        from .storage import Repository, S3ObjectStore, StorageError, ImportBusyError
        settings = settings or Settings.from_env()
        repository = Repository(settings)
        objects = S3ObjectStore(settings)
    else:
        repository, objects = None, None

    runs = run_store or (PostgresRunStore(repository) if persistent else MemoryRunStore())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runs.start()
        app.state.runs = runs
        app.state.answer_active = dispatch_lock.locked
        if persistent:
            app.state.evidence = PersistentEvidenceService(repository)
        else:
            try:
                app.state.evidence = EvidenceService(directory)
            except FileNotFoundError:
                app.state.evidence = None
        try:
            yield
        finally:
            runs.close()

    app = FastAPI(
        title="科研论文助手Agent · 论文证据工作台", version=__version__, lifespan=lifespan,
        description="指定论文内的证据检索与受控问答；支持服务端论文授权、运行预算和持久记录。未配置授权策略时为本机公共语料模式。",
    )

    def failure(status_code, detail, exc):
        content = {"detail": detail}
        if getattr(exc, "answer_run", None):
            content["run"] = exc.answer_run
        if getattr(exc, "answer_generation", None):
            content["generation"] = exc.answer_generation
        return JSONResponse(status_code=status_code, content=content)

    def identity(request: Request):
        return policy.authenticate(request.headers.get("authorization"))

    def scope(principal):
        return None if principal.allowed_paper_ids is None else sorted(principal.allowed_paper_ids)

    def scoped_result(result, principal):
        # Ranking uses a shared index, but its global size is not a paper grant.
        if principal.allowed_paper_ids is not None:
            result["trace"] = {key: value for key, value in result["trace"].items()
                               if key not in {"corpus_paragraphs", "corpus_revision", "vector_collection"}}
        return result

    @app.exception_handler(AccessError)
    async def access_failure(request: Request, exc: AccessError):
        return failure(exc.status_code, str(exc), exc)

    @app.exception_handler(RunStoreError)
    async def run_store_failure(request: Request, exc: RunStoreError):
        return failure(409 if isinstance(exc, RunStoreConflictError) or exc.code == "cancelled" else 503,
                       "运行记录冲突，请刷新后重试" if isinstance(exc, RunStoreConflictError) else "运行记录暂不可用，已停止本次操作", exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo feedback/snapshots or malformed Unicode in validation errors.
        if request.url.path.startswith("/api/v1/runs/") and request.url.path.endswith("/feedback"):
            return JSONResponse(status_code=422, content={"detail": "反馈格式无效，请检查评价、说明长度与回答快照"},
                                headers={"Cache-Control": "no-store"})
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(FeedbackConflictError)
    async def feedback_conflict(request: Request, exc: FeedbackConflictError):
        return JSONResponse(status_code=409, content={"detail": str(exc), "reason": exc.reason})

    @app.exception_handler(RunStopped)
    async def run_stopped(request: Request, exc: RunStopped):
        return failure(408 if exc.code == "deadline_exceeded" else 409, "本次运行已停止，请查看运行记录", exc)

    @app.exception_handler(PdfError)
    async def pdf_error(request: Request, exc: PdfError):
        return JSONResponse(status_code=exc.status, content={"detail": str(exc), "reason": exc.code},
                            headers={"Cache-Control": "no-store"})

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

        async def import_busy(request: Request, exc: ImportBusyError):
            return JSONResponse(status_code=409, content={"detail": "已有论文导入任务正在运行，请稍后重试"})
        app.add_exception_handler(ImportBusyError, import_busy)

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
        if policy.mode == "bearer_policy":
            return {"status": "running", "access_mode": policy.mode, "stage": "P3", "version": __version__}
        if persistent:
            try:
                counts = repository.summary()
                return {"status": "ready", "stage": "P3", "version": __version__, "papers": counts["papers"]}
            except Exception:
                return {"status": "database_unavailable", "stage": "P3", "version": __version__, "papers": 0}
        evidence = app.state.evidence
        return {"status": "ready" if evidence else "data_missing", "stage": "P1",
                "version": __version__, "papers": len(evidence.papers) if evidence else 0}

    @app.get("/api/v1/system")
    def system(principal=Depends(identity)):
        harness_status = {"limits": {**limits.__dict__}, "persistence": "postgres" if persistent else "memory"}
        access_status = principal.public_status()
        if not persistent:
            evidence = app.state.evidence
            return {"stage": "P1", "mode": "files", "access": access_status, "harness": harness_status, "database": {"status": "not_configured"},
                    "object_store": {"status": "not_configured", "provider": "none", "bucket": ""},
                    "corpus": {"papers": sum(principal.allows(pid) for pid in evidence.papers) if evidence else 0,
                               "paragraphs": sum(principal.allows(p["paper_id"]) for p in evidence.index.paragraphs.values()) if evidence else 0, "objects": 0},
                    "generation": generation_settings.public_status(),
                    "capabilities": {"generation": generation_settings.configured, "pdf_upload": False}}
        try:
            counts = repository.summary(allowed_paper_ids=scope(principal))
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
        if principal.allowed_paper_ids is not None:
            vector_status = {"state": vector_status["state"]}
        return {"stage": "P3", "mode": "postgres", "access": access_status, "harness": harness_status, "database": {"status": database_status},
                "object_store": {"status": storage_status, "provider": "S3-compatible", "bucket": settings.s3_bucket if principal.allowed_paper_ids is None else ""},
                "corpus": counts, "vector_index": vector_status,
                "generation": generation_settings.public_status(),
                "capabilities": {"generation": generation_settings.configured, "pdf_upload": principal.mode == "local_public",
                                 "hybrid_retrieval": vector_status["state"] == "ready"}}

    @app.get("/ready")
    def ready(principal=Depends(identity)):
        status = system(principal)
        ok = persistent and status["database"]["status"] == status["object_store"]["status"] == "ready"
        return JSONResponse(status_code=200 if ok else 503, content=status)

    @app.get("/api/v1/ingestions")
    def ingestions(limit: int = Query(20, ge=1, le=100), principal=Depends(identity)):
        return {"items": repository.list_jobs(limit) if persistent and principal.mode == "local_public" else []}

    @app.get("/api/v1/papers")
    def papers(q: str = Query(default="", max_length=200), limit: int = Query(20, ge=1, le=100),
               offset: int = Query(0, ge=0),
               sort: PaperSort = "id_asc", direction: ResearchDirection = "all", principal=Depends(identity)):
        if persistent:
            return repository.list_papers(q=q, limit=limit, offset=offset, sort=sort, direction=direction, allowed_paper_ids=scope(principal))
        items = [p for p in service().papers.values() if principal.allows(p["paper_id"])]
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

    @app.post("/api/v1/uploads/pdf")
    async def upload_pdf(request: Request, principal=Depends(identity)):
        if not persistent or principal.mode != "local_public":
            raise HTTPException(403, "PDF 上传仅在本机公共存储模式开放，暂不提供私有上传")
        if request.headers.get("content-type", "").split(";")[0].lower() != "application/pdf":
            raise HTTPException(415, "请上传 PDF 文件")
        if not upload_lock.acquire(blocking=False):
            raise HTTPException(409, "已有 PDF 正在导入，请稍后重试")
        worker_started = False
        try:
            async def read_body():
                data = bytearray()
                async for chunk in request.stream():
                    data.extend(chunk)
                    if len(data) > MAX_BYTES:
                        raise PdfError("size_limit", "PDF 文件不得超过 10 MiB", 413)
                return bytes(data)
            try:
                data = await asyncio.wait_for(read_body(), 60)
            except TimeoutError:
                raise HTTPException(408, "文件上传超时，请稍后重试") from None
            filename = unquote(request.headers.get("x-pdf-filename", "uploaded.pdf"))
            def execute():
                try:
                    return ingest_pdf(repository, objects, data, filename)
                finally:
                    upload_lock.release()
            # The worker owns the lock until publication really stops, even if the browser disconnects.
            job = AnswerJob(execute)
            job.start(); worker_started = True
            while not job.done.is_set():
                await asyncio.sleep(0.05)
            if job.error: raise job.error
            return JSONResponse(content=job.result, headers={"Cache-Control": "no-store"})
        finally:
            if not worker_started: upload_lock.release()

    def evidence_for(paper_id, mode="bm25", rerank=False):
        if persistent and paper_id.startswith("pdf-"):
            if mode != "bm25" or rerank:
                raise HTTPException(409, "上传 PDF 当前仅支持 BM25 词法检索，未构建向量索引或启用重排")
            return PersistentEvidenceService(repository, paper_id=paper_id)
        return service()

    @app.get("/api/v1/papers/{paper_id}/pdf")
    def original_pdf(paper_id: str, request: Request, principal=Depends(identity)):
        paper(paper_id, principal)
        metadata = repository.get_pdf_object(paper_id) if persistent else None
        if metadata is None: raise HTTPException(404, "该论文没有上传的原始 PDF")
        body = objects.read_bytes(metadata["object_key"])
        if len(body) != metadata["size_bytes"] or hashlib.sha256(body).hexdigest() != metadata["sha256"]:
            raise HTTPException(502, "原 PDF 校验失败，请重新上传同一文件修复")
        policy.revalidate(principal, paper_id, request.headers.get("authorization"))
        return Response(body, media_type="application/pdf", headers={
            "Content-Disposition": f'attachment; filename="{paper_id}.pdf"',
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "X-Content-SHA256": metadata["sha256"]})

    @app.get("/api/v1/papers/{paper_id}")
    def paper(paper_id: str, principal=Depends(identity)):
        policy.require_paper(principal, paper_id)
        item = repository.get_paper(paper_id) if persistent else service().papers.get(paper_id)
        if item is None:
            raise HTTPException(404, "论文不存在")
        return item

    @app.get("/api/v1/papers/{paper_id}/paragraphs")
    def paragraphs(paper_id: str, limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0), principal=Depends(identity)):
        paper(paper_id, principal)
        if persistent:
            return repository.get_paragraphs(paper_id, limit=limit, offset=offset)
        items = [p for p in service().index.paragraphs.values() if p["paper_id"] == paper_id]
        items.sort(key=lambda p: (p["section_index"], p["paragraph_index"]))
        return {"total": len(items), "items": items[offset:offset + limit]}

    @app.get("/api/v1/papers/{paper_id}/source")
    def source(paper_id: str, request: Request, principal=Depends(identity)):
        paper(paper_id, principal)
        if not persistent:
            raise HTTPException(409, "文件模式未接入对象存储")
        metadata = repository.get_paper_object(paper_id)
        if metadata is None:
            raise HTTPException(404, "该论文没有可用的结构化原文")
        body = objects.read_bytes(metadata["object_key"])
        if hashlib.sha256(body).hexdigest() != metadata["sha256"] or len(body) != metadata["size_bytes"]:
            raise HTTPException(502, "原文文件校验失败，请检查存储或重新导入")
        policy.revalidate(principal, paper_id, request.headers.get("authorization"))
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id) + ".json"
        return Response(body, media_type="application/json", headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-SHA256": metadata["sha256"], "Cache-Control": "no-store",
        })

    @app.post("/api/v1/retrieve")
    def retrieve(body: RetrievalRequest, request: Request, principal=Depends(identity)):
        policy.require_paper(principal, body.paper_id)
        try:
            if persistent:
                result = evidence_for(body.paper_id, body.mode, body.rerank).retrieve(body.query, body.paper_id, body.top_k, mode=body.mode, rerank=body.rerank,
                                          rrf_constant=body.rrf_constant, dense_weight=body.dense_weight)
                policy.revalidate(principal, body.paper_id, request.headers.get("authorization"))
                return scoped_result(result, principal)
            if body.mode != "bm25" or body.rerank:
                raise HTTPException(409, "向量、混合与重排检索需要 PostgreSQL 模式及相应模型/索引")
            result = service().retrieve(body.query, body.paper_id, body.top_k)
            policy.revalidate(principal, body.paper_id, request.headers.get("authorization"))
            return scoped_result(result, principal)
        except KeyError:
            raise HTTPException(404, "论文不存在") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/v1/runs")
    def recent_runs(limit: int = Query(20, ge=1, le=50), paper_id: str | None = None, principal=Depends(identity)):
        if paper_id:
            policy.require_paper(principal, paper_id)
        return {"items": runs.list(principal.principal_id, limit,
                                  allowed_paper_ids=[paper_id] if paper_id else scope(principal))}

    def owned_run(run_id, principal):
        record = runs.get(run_id, principal.principal_id)
        if record is None:
            raise HTTPException(404, "运行记录不存在")
        policy.require_paper(principal, record["paper_id"])
        return record

    @app.get("/api/v1/runs/{run_id}")
    def read_run(run_id: str, principal=Depends(identity)):
        return owned_run(run_id, principal)

    @app.post("/api/v1/runs/{run_id}/cancel")
    def cancel_run(run_id: str, principal=Depends(identity)):
        owned_run(run_id, principal)
        return runs.request_cancel(run_id, principal.principal_id)

    @app.get("/api/v1/runs/{run_id}/feedback")
    def read_feedback(run_id: str, request: Request, principal=Depends(identity)):
        record = owned_run(run_id, principal)
        feedback = runs.get_feedback(run_id, principal.principal_id)
        policy.revalidate(principal, record["paper_id"], request.headers.get("authorization"))
        return JSONResponse(content={"feedback": feedback}, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/runs/{run_id}/feedback")
    def save_feedback(run_id: str, body: FeedbackInput, request: Request, principal=Depends(identity)):
        record = owned_run(run_id, principal)
        policy.revalidate(principal, record["paper_id"], request.headers.get("authorization"))
        feedback = runs.save_feedback(run_id, principal.principal_id, body)
        policy.revalidate(principal, record["paper_id"], request.headers.get("authorization"))
        return JSONResponse(content={"feedback": feedback}, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/answer")
    async def answer(body: AnswerRequest, request: Request, principal=Depends(identity)):
        policy.require_paper(principal, body.paper_id)
        evidence = evidence_for(body.paper_id, body.mode, body.rerank)
        if not persistent and (body.mode != "bm25" or body.rerank):
            raise HTTPException(409, "向量、混合与重排检索需要 PostgreSQL 模式及相应模型/索引")
        if not dispatch_lock.acquire(blocking=False):
            raise GenerationBusyError()
        run_id = body.run_id or uuid.uuid4().hex
        trace = {}
        try:
            await asyncio.to_thread(runs.create, run_id, principal.principal_id, body.paper_id,
                                    {k: v for k, v in body.model_dump().items() if k not in {"query", "paper_id", "run_id"}})
            def cancelled():
                record = runs.get(run_id, principal.principal_id)
                if record is None:
                    raise RunStopped("run_store_unavailable")
                return record["cancel_requested"]
            control = RunControl(limits.for_request(body.allow_repair),
                authorize=lambda: policy.revalidate(principal, body.paper_id, request.headers.get("authorization")),
                cancelled=cancelled)
            def observe(snapshot):
                runs.save(run_id, principal.principal_id,
                          {"run": snapshot, "generation": trace, "access": {"principal_id": principal.principal_id, "mode": principal.mode}})
            run = AnswerRun(control=control, trace_id=run_id, observer=observe)
            def execute():
                try:
                    return answers[body.profile].answer(evidence, body.query, body.paper_id, body.top_k,
                        body.mode, body.rerank, language=body.language, rrf_constant=body.rrf_constant,
                        dense_weight=body.dense_weight, allow_repair=body.allow_repair, run=run, trace=trace)
                finally:
                    dispatch_lock.release()
            job = AnswerJob(execute)
            job.start()
        except BaseException:
            dispatch_lock.release()
            raise
        # The event loop keeps serving cancel/status while local or remote I/O blocks.
        def stop_waiter(reason):
            with run.lock:
                # The worker may have committed its own error just before setting
                # job.done. Let its original HTTP error/trace win that race.
                if run.snapshot()["state"] != "running":
                    return None
                return run.abort(reason)

        while not job.done.is_set():
            try:
                await asyncio.to_thread(control.check)
                if await request.is_disconnected():
                    raise RunStopped("cancelled")
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.to_thread(run.abort, "cancelled"))
                raise
            except Exception as exc:
                snapshot = await asyncio.to_thread(stop_waiter, exception_reason(exc))
                if snapshot is None:
                    await asyncio.sleep(0.01)
                    continue
                exc.answer_run = snapshot
                status_code = getattr(exc, "status_code", 503 if snapshot["reason"] == "run_store_unavailable"
                                      else 408 if snapshot["reason"] == "deadline_exceeded" else 409)
                return failure(status_code,
                               "本次运行已停止，请查看运行记录", exc)
            await asyncio.sleep(0.05)
        if job.error:
            if isinstance(job.error, KeyError):
                return failure(404, "论文不存在", job.error)
            if isinstance(job.error, ValueError):
                return failure(422, "输入超出模型或检索限制，请缩短问题后重试", job.error)
            raise job.error
        # Recheck authority before sending a response body that may contain evidence.
        policy.revalidate(principal, body.paper_id, request.headers.get("authorization"))
        return scoped_result(job.result, principal)

    return app


app = create_app()
