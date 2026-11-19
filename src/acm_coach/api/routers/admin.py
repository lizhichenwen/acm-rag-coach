# -*- coding: utf-8 -*-
"""管理员接口：用户管理 + 知识批量导入 + 入库审核状态机。

安全约束（docs/05 §3.3 / §3.5）：
- 全部要求 admin 角色（auditor 只读，不得写入）；
- 禁止修改自己的角色、禁止停用自己（防自提权与自锁）；
- 每次写操作写审计，角色变更事件码 admin.role_change / admin.user_disable。
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from acm_coach.api.deps import AdminUser, DbSession, audit
from acm_coach.api.schemas import (
    AdminUserResponse, KnowledgeCreateRequest, UserRoleUpdate,
)
from acm_coach.core.errors import Conflict, Forbidden, NotFound
from acm_coach.db.repositories import KnowledgeRepository, UserRepository
from acm_coach.services import knowledge_service

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# 知识批量导入/审核挂在 knowledge 前缀下，便于与现有知识接口同组
knowledge_admin_router = APIRouter(prefix="/api/v1", tags=["knowledge"])

VALID_STATUSES = {"draft", "reviewing", "published", "retired"}


def _admin_user(row) -> AdminUserResponse:
    return AdminUserResponse(
        id=row.id, username=row.username, role=row.role,
        is_active=row.is_active,
        created_at=row.created_at.strftime("%Y-%m-%d %H:%M") if row.created_at else None,
        last_login_at=row.last_login_at.strftime("%Y-%m-%d %H:%M") if row.last_login_at else None,
    )


@router.get("/users", response_model=list[AdminUserResponse])
def list_users(db: DbSession, user: AdminUser,
               limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    return [_admin_user(row) for row in UserRepository(db).list_all(limit, offset)]


@router.patch("/users/{user_id}/role", response_model=AdminUserResponse)
def change_role(user_id: int, payload: UserRoleUpdate, request: Request,
                db: DbSession, user: AdminUser):
    if user_id == user.id:
        raise Forbidden("不能修改自己的角色")
    repo = UserRepository(db)
    target = repo.by_id(user_id)
    if target is None:
        raise NotFound("用户不存在")
    before = target.role
    repo.set_role(target, payload.role)
    # 角色变更立即使旧令牌失效，防止旧 role 继续生效（docs/05 §3.5 垂直越权防护）
    repo.bump_token_version(target)
    audit(db, request, "admin.role_change", user=user, resource="user", resource_id=user_id,
          detail={"before": before, "after": payload.role, "target": target.username})
    db.commit()
    return _admin_user(target)


@router.post("/users/{user_id}/disable", response_model=AdminUserResponse)
def disable_user(user_id: int, request: Request, db: DbSession, user: AdminUser):
    if user_id == user.id:
        raise Forbidden("不能停用自己")
    repo = UserRepository(db)
    target = repo.by_id(user_id)
    if target is None:
        raise NotFound("用户不存在")
    repo.set_active(target, False)
    repo.bump_token_version(target)
    audit(db, request, "admin.user_disable", user=user, resource="user", resource_id=user_id,
          detail={"target": target.username})
    db.commit()
    return _admin_user(target)


@knowledge_admin_router.post("/knowledge/import")
def import_knowledge(items: list[KnowledgeCreateRequest], request: Request,
                     db: DbSession, user: AdminUser,
                     status: str = Query("published")):
    """批量导入精讲条目（admin）。单条不通过门禁则跳过，返回成功/跳过统计。"""
    if status not in VALID_STATUSES:
        raise Conflict(f"status 必须是 {sorted(VALID_STATUSES)} 之一")
    raw = [{
        "title": it.title, "difficulty": it.difficulty, "tags": it.tags,
        "domain": it.domain, "description": it.description,
        "solution": it.solution, "pitfalls": it.pitfalls, "source": "curated",
        "status": status,
    } for it in items]
    records = knowledge_service.to_records(raw)
    inserted, updated = KnowledgeRepository(db).upsert_many(records)
    audit(db, request, "knowledge.import", user=user, resource="knowledge",
          detail={"submitted": len(items), "accepted": len(records),
                  "inserted": inserted, "updated": updated, "status": status})
    db.commit()
    return {"submitted": len(items), "accepted": len(records),
            "skipped": len(items) - len(records),
            "inserted": inserted, "updated": updated}


@knowledge_admin_router.post("/knowledge/{item_id}/review")
def review_knowledge(item_id: str, request: Request, db: DbSession, user: AdminUser,
                     status: str = Query(...)):
    """入库审核状态流转：draft -> reviewing -> published / retired。"""
    if status not in VALID_STATUSES:
        raise Conflict(f"status 必须是 {sorted(VALID_STATUSES)} 之一")
    repo = KnowledgeRepository(db)
    row = repo.get(item_id)
    if row is None:
        raise NotFound("条目不存在")
    before = row.status
    repo.set_status(item_id, status)
    audit(db, request, "knowledge.review", user=user, resource="knowledge",
          resource_id=item_id, detail={"before": before, "after": status})
    db.commit()
    return {"id": item_id, "status": status, "before": before}
