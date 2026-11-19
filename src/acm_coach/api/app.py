# -*- coding: utf-8 -*-
"""FastAPI 应用装配。

分层：router（HTTP 语义）→ service（业务编排）→ repository（数据访问）→ DB。
应用层只做三件事：注册路由、装中间件、把异常翻译成统一 JSON。
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from acm_coach.api.routers import admin, auth, chat, diagnosis, knowledge, me, ops
from acm_coach.config import get_settings
from acm_coach.core import metrics
from acm_coach.core.errors import AppError, Forbidden
from acm_coach.core.logging import get_logger, get_trace_id, new_trace_id, setup_logging

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from acm_coach.db import init_db
    from acm_coach.rag.index import get_retriever

    settings = get_settings()
    init_db()
    try:
        get_retriever().ensure_loaded()
    except Exception as exc:  # 索引缺失不应阻止服务启动，接口会返回"题库未就绪"
        log.warning("索引加载失败，检索将不可用：%s", exc)
    log.info("服务启动 env=%s db=%s", settings.app_env, settings.sqlalchemy_url)
    yield
    log.info("服务退出")


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="面向 ACM 程序设计竞赛的 RAG 智能陪练系统",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_prod else None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def trace_and_metrics(request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or new_trace_id()
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - started
            metrics.inc("http_requests_total", {"status": "500"})
            metrics.observe("http_request_seconds", elapsed)
            log.exception("未捕获异常 path=%s", request.url.path)
            raise
        elapsed = time.perf_counter() - started
        response.headers["x-trace-id"] = trace_id
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        metrics.inc("http_requests_total", {"status": str(response.status_code)})
        metrics.observe("http_request_seconds", elapsed, {"path": path})
        return response

    @app.exception_handler(Forbidden)
    async def forbidden_handler(request: Request, exc: Forbidden):
        """越权访问统一落 access_denied 审计（守库要求：拒绝行为必须可追溯）。

        用户身份尽力解析：令牌有效则记录 user_id/username，无效/缺失则记匿名。
        审计写入失败不得影响 403 响应本身。
        """
        metrics.inc("app_errors_total", {"code": exc.code})
        try:
            from acm_coach.core.security import decode_token
            from acm_coach.db.repositories import AuditRepository
            from acm_coach.db.session import get_session_factory

            user_id: int | None = None
            username: str | None = None
            authz = request.headers.get("authorization", "")
            if authz.lower().startswith("bearer "):
                try:
                    payload = decode_token(authz.split(" ", 1)[1].strip())
                    user_id = int(payload["sub"])
                except Exception:
                    user_id = None
            db = get_session_factory()()
            try:
                AuditRepository(db).write(
                    "access_denied", user_id=user_id, username=username,
                    resource=request.url.path, result="denied",
                    ip=request.client.host if request.client else None,
                    trace_id=get_trace_id(),
                    detail={"method": request.method, "message": exc.message},
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            log.warning("access_denied 审计写入失败 path=%s", request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message,
                     "trace_id": get_trace_id(), "details": exc.details},
        )

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        metrics.inc("app_errors_total", {"code": exc.code})
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code, "message": exc.message,
                "trace_id": get_trace_id(), "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def fallback_handler(request: Request, exc: Exception):
        log.exception("未处理异常")
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "服务内部错误",
                     "trace_id": get_trace_id()},
        )

    for module in (auth, chat, knowledge, diagnosis, ops, me, admin):
        for attr in ("router", "diag_router", "stats_router", "knowledge_admin_router"):
            r = getattr(module, attr, None)
            if r is not None:
                app.include_router(r)
    # 调试类接口默认不注册（404），仅受控环境显式开启
    if settings.enable_debug_api:
        app.include_router(ops.debug_router)

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": settings.app_name, "version": settings.app_version,
                "docs": "/docs", "health": "/healthz"}

    return app


app = create_app()
