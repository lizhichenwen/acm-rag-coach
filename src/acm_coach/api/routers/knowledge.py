# -*- coding: utf-8 -*-
"""知识库接口：浏览、详情、录入、下架、重建索引、练习标记。

写操作全部限 admin（auditor 只读），每次变更写审计。
录入后不立即重建全量索引（8000 条重建代价高），而是标记"索引待更新"，
由 admin 触发后台重建——这是有意的取舍，避免单条录入把接口拖成分钟级。
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Query, Request

from acm_coach.api.deps import AdminUser, CurrentUser, DbSession, StaffUser, audit
from acm_coach.api.schemas import (
    KnowledgeCreateRequest, KnowledgeDetail, KnowledgeItemResponse, PageResponse,
)
from acm_coach.core.errors import Conflict, NotFound
from acm_coach.db.repositories import KnowledgeRepository, PracticeRepository
from acm_coach.rag.index import Hit, get_retriever
from acm_coach.services import knowledge_service

router = APIRouter(prefix="/api/v1", tags=["knowledge"])


def _item(row) -> KnowledgeItemResponse:
    return KnowledgeItemResponse(
        id=row.id, title=row.title, difficulty=row.difficulty, tags=row.tags,
        domain=row.domain, source=row.source, url=row.url, rating=row.rating,
    )


@router.get("/knowledge", response_model=PageResponse)
def list_knowledge(db: DbSession, user: CurrentUser,
                   q: str = "", domain: str = "", difficulty: str = "",
                   source: str = "", page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
    rows, total = KnowledgeRepository(db).search(
        q, domain, difficulty, source, page, size,
        include_restricted=user.role in ("admin", "auditor"),
    )
    return PageResponse(total=total, page=page, size=size, items=[_item(r) for r in rows])


@router.get("/knowledge/{item_id}", response_model=KnowledgeDetail)
def get_knowledge(item_id: str, db: DbSession, user: CurrentUser):
    row = KnowledgeRepository(db).get_visible(
        item_id, include_restricted=user.role in ("admin", "auditor"))
    if row is None:
        # 不存在与无权可见统一 404：403 会泄露"该条目存在"
        raise NotFound("条目不存在")
    similar: list[Hit] = get_retriever().keyword_search(f"{row.title} {row.tags}", 6)
    return KnowledgeDetail(
        id=row.id, title=row.title, difficulty=row.difficulty, tags=row.tags,
        domain=row.domain, source=row.source, url=row.url, rating=row.rating,
        content=row.content, updated_at=row.updated_at.strftime("%Y-%m-%d %H:%M"),
        similar=[_item_like(h) for h in similar if h.id != row.id][:4],
    )


def _item_like(hit: Hit) -> KnowledgeItemResponse:
    return KnowledgeItemResponse(
        id=hit.id, title=hit.title, difficulty=hit.difficulty, tags=hit.tags,
        domain=hit.domain, source=hit.source, url=hit.url,
    )


@router.post("/knowledge", response_model=KnowledgeItemResponse, status_code=201)
def create_knowledge(payload: KnowledgeCreateRequest, request: Request,
                     db: DbSession, user: AdminUser):
    items = [{
        "title": payload.title, "difficulty": payload.difficulty, "tags": payload.tags,
        "domain": payload.domain, "description": payload.description,
        "solution": payload.solution, "pitfalls": payload.pitfalls, "source": "curated",
    }]
    records = knowledge_service.to_records(items)
    if not records:
        raise Conflict("入库校验未通过：请检查字段完整性与正文长度（40-2400 字）")
    inserted, updated = KnowledgeRepository(db).upsert_many(records)
    audit(db, request, "knowledge.create", user=user, resource="knowledge",
          resource_id=records[0]["id"], detail={"title": payload.title})
    db.commit()
    row = KnowledgeRepository(db).get(records[0]["id"])
    return _item(row)


@router.delete("/knowledge/{item_id}", status_code=204)
def delete_knowledge(item_id: str, request: Request, db: DbSession, user: AdminUser):
    repo = KnowledgeRepository(db)
    if not repo.soft_delete(item_id):
        raise NotFound("条目不存在")
    audit(db, request, "knowledge.delete", user=user, resource="knowledge", resource_id=item_id)
    db.commit()


@router.post("/knowledge/reindex", status_code=202)
def reindex(background: BackgroundTasks, request: Request, db: DbSession, user: AdminUser):
    """后台重建索引（约 8000 条，CPU 环境需要数分钟）。"""
    from acm_coach.cli import _load_all
    from acm_coach.db.session import session_scope
    from acm_coach.services.index_service import build_index

    def _job() -> None:
        with session_scope() as job_db:
            build_index(_load_all(job_db))

    background.add_task(_job)
    audit(db, request, "knowledge.reindex", user=user, resource="index")
    db.commit()
    return {"status": "accepted", "message": "索引重建已在后台开始"}


@router.post("/practice/{item_id}", status_code=201)
def mark_practiced(item_id: str, db: DbSession, user: CurrentUser):
    PracticeRepository(db).mark(user.id, item_id)
    db.commit()
    return {"status": "ok", "knowledge_id": item_id}


@router.get("/practice")
def list_practiced(db: DbSession, user: CurrentUser):
    return PracticeRepository(db).list(user.id)


@router.get("/knowledge/meta/summary")
def knowledge_summary(db: DbSession, user: StaffUser):
    return KnowledgeRepository(db).count_by()
