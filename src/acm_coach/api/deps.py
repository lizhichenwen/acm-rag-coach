# -*- coding: utf-8 -*-
"""FastAPI 依赖注入：数据库会话、当前用户、角色校验、客户端 IP。

鉴权失败一律 401 且不区分"用户不存在/口令错误"，防止账号枚举；
越权一律 403；资源不存在与他人资源一律 404（不泄露存在性）。
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from acm_coach.core.errors import Forbidden, Unauthorized
from acm_coach.core.security import TokenError, decode_token
from acm_coach.db.models import User
from acm_coach.db.repositories import AuditRepository, UserRepository
from acm_coach.db.session import get_session_factory


def get_db() -> Iterator[Session]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


DbSession = Annotated[Session, Depends(get_db)]


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "-"


def get_current_user(
    db: DbSession,
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise Unauthorized("缺少访问令牌")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except TokenError as exc:
        raise Unauthorized(str(exc)) from exc

    user = UserRepository(db).by_id(int(payload["sub"]))
    if user is None or not user.is_active:
        raise Unauthorized("账号不存在或已禁用")
    # 改密/封禁后旧令牌立即失效
    if int(payload.get("ver", 0)) != int(user.token_version):
        raise Unauthorized("登录状态已失效，请重新登录")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_roles(*roles: str):
    def _guard(user: CurrentUser) -> User:
        if user.role not in roles:
            raise Forbidden("当前角色无权访问该资源")
        return user

    return _guard


AdminUser = Annotated[User, Depends(require_roles("admin"))]
StaffUser = Annotated[User, Depends(require_roles("admin", "auditor"))]


def audit(db: Session, request: Request, action: str, *, user: User | None = None,
          resource: str | None = None, resource_id=None, result: str = "ok",
          detail: dict | None = None) -> None:
    from acm_coach.core.logging import get_trace_id

    AuditRepository(db).write(
        action, user_id=user.id if user else None,
        username=user.username if user else None,
        resource=resource, resource_id=resource_id, result=result,
        ip=client_ip(request), trace_id=get_trace_id(), detail=detail,
    )
