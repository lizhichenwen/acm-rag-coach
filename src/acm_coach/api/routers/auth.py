# -*- coding: utf-8 -*-
"""认证接口：注册、登录、当前用户、改密。

安全要点：
- 登录失败不区分"用户不存在"与"口令错误"，统一 401；
- 同一用户名连续失败 5 次锁定 5 分钟（进程内滑动窗口）；
- 登录成功/失败都写审计；
- 返回体经 `UserResponse` 过滤，不泄露哈希、token_version、is_active。
"""
from __future__ import annotations


from fastapi import APIRouter, Request

from acm_coach.api.deps import DbSession, CurrentUser, audit
from acm_coach.api.schemas import (
    ChangePasswordRequest, LoginRequest, RegisterRequest, TokenResponse, UserResponse,
)
from acm_coach.core.errors import Conflict, RateLimited, Unauthorized
from acm_coach.core.security import (
    create_access_token, hash_password, login_throttle, password_strength, verify_password,
)
from acm_coach.db.repositories import UserRepository

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _to_response(user) -> UserResponse:
    return UserResponse(
        id=user.id, username=user.username, role=user.role,
        created_at=user.created_at.strftime("%Y-%m-%d %H:%M") if user.created_at else None,
    )


@router.post("/register", response_model=UserResponse, status_code=201)
def register(payload: RegisterRequest, request: Request, db: DbSession):
    repo = UserRepository(db)
    if repo.by_username(payload.username):
        raise Conflict("用户名已被占用")
    ok, reason = password_strength(payload.password)
    if not ok:
        raise Conflict(reason)
    user = repo.create(payload.username, hash_password(payload.password), "student")
    # 自注册用户口令由本人设定，无强制改密需求
    user.must_change_password = False
    audit(db, request, "auth.register", user=user, resource="user", resource_id=user.id)
    db.commit()
    return _to_response(user)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: DbSession):
    locked, retry = login_throttle.is_locked(payload.username)
    if locked:
        audit(db, request, "auth.login", result="locked", detail={"username": payload.username})
        db.commit()
        raise RateLimited(f"登录失败次数过多，请 {retry} 秒后再试")

    repo = UserRepository(db)
    user = repo.by_username(payload.username)
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        login_throttle.record_failure(payload.username)
        audit(db, request, "auth.login", result="fail", detail={"username": payload.username})
        db.commit()
        raise Unauthorized("用户名或口令错误")

    login_throttle.reset(payload.username)
    repo.touch_login(user)
    token, expires_at = create_access_token(user.id, user.username, user.role, user.token_version)
    audit(db, request, "auth.login", user=user, result="ok")
    db.commit()
    return TokenResponse(access_token=token, expires_at=expires_at, user=_to_response(user),
                         must_change_password=bool(user.must_change_password))


@router.get("/me", response_model=UserResponse)
def me(user: CurrentUser):
    return _to_response(user)


@router.post("/change-password", status_code=204)
def change_password(payload: ChangePasswordRequest, request: Request, db: DbSession,
                    user: CurrentUser):

    repo = UserRepository(db)
    if not verify_password(payload.old_password, user.password_hash):
        audit(db, request, "auth.change_password", user=user, result="fail")
        db.commit()
        raise Unauthorized("原口令错误")
    ok, reason = password_strength(payload.new_password)
    if not ok:
        raise Conflict(reason)
    user.password_hash = hash_password(payload.new_password)
    repo.clear_must_change_password(user)  # 改密后清除强制标志
    repo.bump_token_version(user)  # 旧令牌立即失效，强制重新登录
    audit(db, request, "auth.change_password", user=user, result="ok")
    db.commit()
