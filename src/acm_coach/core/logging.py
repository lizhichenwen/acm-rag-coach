# -*- coding: utf-8 -*-
"""结构化日志。

设计目标：
1. 每条日志带 `trace_id`，一次请求内的检索、生成、DB 调用可串起来排查；
2. 开发态人类可读，生产态单行 JSON（便于 Loki / ELK 采集）；
3. 敏感字段（api_key / password / token）在落盘前脱敏。
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from acm_coach.config import get_settings

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")

SENSITIVE_KEYS = ("api_key", "apikey", "password", "token", "secret", "authorization")
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id.get()


def set_trace_id(tid: str) -> None:
    _trace_id.set(tid or "-")


def mask(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}***{value[-keep:]}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": get_trace_id(),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            if any(s in key.lower() for s in SENSITIVE_KEYS):
                value = mask(str(value))
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = get_trace_id()
        return super().format(record)


_configured = False


def setup_logging(level: str | None = None, json_mode: bool | None = None) -> None:
    """进程级初始化，幂等。"""
    global _configured
    settings = get_settings()
    level = (level or settings.log_level).upper()
    json_mode = settings.log_json if json_mode is None else json_mode

    root = logging.getLogger()
    root.setLevel(level)
    if _configured:
        return

    # 去掉 uvicorn 默认 handler，统一走我们的格式
    for name in ("uvicorn", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    stream = logging.StreamHandler(sys.stdout)
    if json_mode:
        stream.setFormatter(JsonFormatter())
    else:
        stream.setFormatter(
            HumanFormatter("%(asctime)s %(levelname)-7s [%(trace_id)s] %(name)s: %(message)s", "%H:%M:%S")
        )
    root.addHandler(stream)

    log_dir = Path(settings.data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    """业务事件日志：event 是稳定的机器可读标识（如 retrieval.done）。"""
    logger.info(event, extra={"event": event, **fields})
