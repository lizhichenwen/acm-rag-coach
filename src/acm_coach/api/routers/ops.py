# -*- coding: utf-8 -*-
"""运维接口：健康检查、指标、审计、抓取记录、检索质量。

健康检查分两级：
- `/healthz` 只证明进程活着（K8s liveness），不查依赖，避免依赖抖动导致误杀；
- `/readyz` 检查数据库与索引是否可用（readiness），不可用返回 503。
"""
from __future__ import annotations

from sqlalchemy import text

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from acm_coach.api.deps import DbSession, StaffUser
from acm_coach.config import get_settings
from acm_coach.core import metrics
from acm_coach.core.errors import ServiceUnavailable
from acm_coach.db.repositories import AuditRepository, OpsRepository, UserRepository
from acm_coach.rag.index import get_retriever

router = APIRouter(tags=["ops"])
# 调试类只读接口单独成组，由 app 按 ENABLE_DEBUG_API 条件注册（默认不注册→404）
debug_router = APIRouter(tags=["ops-debug"])


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz(db: DbSession):
    checks = {"database": False, "bm25": False, "vector": False}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        pass
    retriever = get_retriever()
    try:
        checks["bm25"] = retriever.ready
        checks["vector"] = bool(retriever.vector and retriever.vector.available)
    except Exception:
        pass
    if not checks["database"] or not checks["bm25"]:
        raise ServiceUnavailable("依赖未就绪", details=checks)
    return {"status": "ready", **checks}


@router.get("/version")
def version(user: StaffUser):
    settings = get_settings()
    return {"name": settings.app_name, "version": settings.app_version,
            "env": settings.app_env}


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics(user: StaffUser):
    return PlainTextResponse(metrics.render_prometheus(),
                             media_type="text/plain; version=0.0.4")


@router.get("/api/v1/ops/overview")
def ops_overview(db: DbSession, user: StaffUser):
    from acm_coach.db.repositories import KnowledgeRepository

    retriever = get_retriever()
    return {
        "knowledge": KnowledgeRepository(db).count_by(),
        "retrieval": OpsRepository(db).retrieval_stats(),
        "runtime": metrics.snapshot(),
        "index": {
            "documents": retriever.doc_count,
            "vector": retriever.vector.count() if retriever.vector else 0,
            "users": UserRepository(db).count(),
        },
    }


@router.get("/api/v1/ops/crawls")
def crawl_runs(db: DbSession, user: StaffUser, limit: int = 20):
    rows = OpsRepository(db).list_crawls(limit)
    return [{
        "source": r.source, "status": r.status, "total": r.total,
        "inserted": r.inserted, "updated": r.updated, "skipped": r.skipped,
        "error": r.error,
        "started_at": r.started_at.strftime("%Y-%m-%d %H:%M"),
        "finished_at": r.finished_at.strftime("%Y-%m-%d %H:%M") if r.finished_at else None,
    } for r in rows]


@router.get("/api/v1/ops/audit")
def audit_logs(db: DbSession, user: StaffUser, limit: int = 50):
    return [{
        "id": a.id, "username": a.username, "action": a.action, "resource": a.resource,
        "result": a.result, "ip": a.ip, "trace_id": a.trace_id,
        "created_at": a.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    } for a in AuditRepository(db).recent(limit)]


@router.get("/api/v1/ops/retrieval")
def retrieval_logs(db: DbSession, user: StaffUser):
    return OpsRepository(db).retrieval_stats(24)


@debug_router.get("/api/v1/ops/config")
def config_view(user: StaffUser):
    """只暴露白名单标量参数，永不返回密钥类字段。"""
    s = get_settings()
    return {
        "top_k": s.top_k, "recall_k": s.recall_k, "rrf_k": s.rrf_k,
        "bm25_k1": s.bm25_k1, "bm25_b": s.bm25_b,
        "strong_threshold": s.strong_threshold, "weak_threshold": s.weak_threshold,
        "enable_rerank": s.enable_rerank, "embed_model": s.embed_model,
        "embed_text_mode": s.embed_text_mode, "type_boost": s.type_boost,
        "enable_bm25_field_boost": s.enable_bm25_field_boost,
        "bm25_title_weight": s.bm25_title_weight, "bm25_tags_weight": s.bm25_tags_weight,
    }
