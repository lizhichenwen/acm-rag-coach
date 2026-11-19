# -*- coding: utf-8 -*-
"""统一异常体系。

约定：业务代码只抛 `AppError` 子类，HTTP 状态码与错误码由异常自身携带，
API 层统一转换成 `{"code": "...", "message": "...", "trace_id": "..."}`，
避免每个接口各写一套 try/except，也避免把堆栈泄露给前端。
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None,
                 status_code: int | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.details = details or {}


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


class RateLimited(AppError):
    status_code = 429
    code = "rate_limited"


class UpstreamError(AppError):
    status_code = 502
    code = "upstream_error"


class ServiceUnavailable(AppError):
    status_code = 503
    code = "service_unavailable"
