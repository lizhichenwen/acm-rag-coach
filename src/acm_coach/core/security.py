# -*- coding: utf-8 -*-
"""认证与口令安全。

口令：PBKDF2-HMAC-SHA256 + 每用户随机盐，迭代次数可配（默认 60 万，
与 docs/05 安全方案一致；单次校验本机实测 ~258ms），
比对用 `hmac.compare_digest` 防时序侧信道。

会话：无状态 JWT（HS256），claims 含 sub / role / exp / iat / jti，
登出通过与 `token_version` 比对实现（用户改密后旧 token 立即失效）。
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any

import jwt

from acm_coach.config import get_settings

ALGORITHM = "HS256"


# ---------------- 口令 ----------------
def hash_password(password: str, iterations: int | None = None) -> str:
    iterations = iterations or get_settings().pbkdf2_iterations
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, expected = (stored or "").split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(digest.hex(), expected)
    except (ValueError, TypeError):
        return False


def password_strength(password: str) -> tuple[bool, str]:
    """最小口令策略：≥8 位，含字母与数字。返回 (是否通过, 原因)。"""
    if len(password or "") < 8:
        return False, "口令至少 8 位"
    if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        return False, "口令需同时包含字母和数字"
    return True, ""


# ---------------- JWT ----------------
@dataclass(frozen=True)
class TokenPayload:
    user_id: int
    username: str
    role: str
    jti: str
    expires_at: int


def _secret() -> str:
    return get_settings().resolve_secret_key()


def create_access_token(user_id: int, username: str, role: str, version: int = 0,
                        expire_minutes: int | None = None) -> tuple[str, int]:
    """签发访问令牌。

    claims 只保留鉴权必需的字段：`sub/role/ver/iat/nbf/exp/jti/iss`。
    不放 username——展示名一律从数据库取，避免改名后令牌内信息过期，
    也少一处可被用来做账号枚举的泄露面。
    """
    settings = get_settings()
    expire_minutes = expire_minutes or settings.jwt_expire_minutes
    now = int(time.time())
    exp = now + expire_minutes * 60
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "ver": version,
        "iat": now,
        "nbf": now,
        "exp": exp,
        "jti": uuid.uuid4().hex,
        "iss": "acm-coach",
    }
    token = jwt.encode(payload, _secret(), algorithm=settings.jwt_algorithm)
    return token, exp


class TokenError(Exception):
    pass


def decode_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(
            token, _secret(), algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]}, issuer="acm-coach",
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("令牌已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("令牌无效") from exc


# ---------------- 行为日志脱敏（PIPL 最小化原则） ----------------
def fingerprint_query(query: str) -> str:
    """检索日志只存查询指纹，不落原文。

    retrieval_logs 用于运维统计（命中率/耗时分布），业务上不需要还原用户问了什么；
    原文属于用户行为轨迹，落库前脱敏：``sha256 前 16 位 + 原文长度``，
    既能区分不同查询（排障时对比两次请求是否同文），又不可逆推出原文。
    """
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}#len={len(query)}"


# ---------------- 登录失败限流（进程内滑动窗口） ----------------
class LoginThrottle:
    """轻量防爆破：按用户名累计失败次数，超限锁定一段时间。

    单机内存实现，满足毕设部署形态；多实例部署应换 Redis（见 docs/05）。
    """

    def __init__(self) -> None:
        self._fails: dict[str, list[float]] = {}

    def _prune(self, key: str, window: int) -> list[float]:
        now = time.time()
        items = [t for t in self._fails.get(key, []) if now - t < window]
        self._fails[key] = items
        return items

    def is_locked(self, key: str) -> tuple[bool, int]:
        settings = get_settings()
        items = self._prune(key, settings.login_lock_seconds)
        if len(items) >= settings.login_max_failures:
            retry = int(settings.login_lock_seconds - (time.time() - items[0]))
            return True, max(retry, 1)
        return False, 0

    def record_failure(self, key: str) -> None:
        self._fails.setdefault(key, []).append(time.time())

    def reset(self, key: str) -> None:
        self._fails.pop(key, None)


login_throttle = LoginThrottle()
