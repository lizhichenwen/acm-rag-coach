# -*- coding: utf-8 -*-
"""数据库引擎与会话管理。

SQLite 作为毕设部署形态足够，但必须开 WAL：默认 rollback journal 在
"读会话 + 写消息"并发时会 SQLITE_BUSY。开启 WAL 后读写互不阻塞。
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger
from acm_coach.db.models import Base, User
from acm_coach.core.security import hash_password

log = get_logger("db")

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    from sqlalchemy import create_engine

    settings = get_settings()
    url = settings.sqlalchemy_url
    kwargs: dict = {"future": True, "echo": False}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 15}
    _engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _sqlite_pragma(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=15000")
            cur.close()

    _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """脚本/后台任务用：自动提交、异常回滚、始终关闭。"""
    db = get_session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db(seed_admin: bool = True) -> None:
    """建表 + 轻量迁移 + 首次初始化管理员。幂等，可重复调用。"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    # 迁移必须无条件执行：CLI/脚本常以 seed_admin=False 调用，
    # 若放在管理员初始化之后，新增列永远不会补上（曾因此报 no such column）。
    _ensure_columns(engine)
    _ensure_audit_guards(engine)
    if not seed_admin:
        return

    settings = get_settings()
    with session_scope() as db:
        exists = db.query(User).filter(User.username == settings.admin_username).first()
        if exists:
            return
        # 禁止默认口令：未配置则生成一次性随机口令，仅打印一次，强制首登改密
        import secrets

        initial_password = settings.admin_password or secrets.token_urlsafe(12)
        db.add(User(
            username=settings.admin_username,
            password_hash=hash_password(initial_password),
            role="admin",
            must_change_password=True,  # 一次性口令，首登强制改密
        ))
        if not settings.admin_password:
            log.warning("初始管理员 %s 的一次性口令：%s（仅本次打印，请登录后立即修改）",
                        settings.admin_username, initial_password)
        else:
            log.warning("已按 ADMIN_PASSWORD 创建管理员 %s，建议登录后立即改密", settings.admin_username)


def _ensure_audit_guards(engine: Engine) -> None:
    """审计表 append-only：数据库层拒绝 UPDATE / DELETE。

    应用层没有提供改审计的接口，但"没提供接口"不等于"改不了"——
    任何能拿到 sqlite 连接的人都能改。触发器把这条约束下沉到存储层。
    """
    with engine.connect() as conn:
        conn.execute(text(
            """CREATE TRIGGER IF NOT EXISTS audit_logs_no_update
               BEFORE UPDATE ON audit_logs
               BEGIN SELECT RAISE(ABORT, 'audit_logs is append-only'); END"""
        ))
        conn.execute(text(
            """CREATE TRIGGER IF NOT EXISTS audit_logs_no_delete
               BEFORE DELETE ON audit_logs
               BEGIN SELECT RAISE(ABORT, 'audit_logs is append-only'); END"""
        ))
        conn.commit()


def _ensure_columns(engine: Engine) -> None:
    """SQLite 轻量迁移：缺列则 ALTER TABLE 补齐。"""
    additions = {
        "users": {"token_version": "INTEGER DEFAULT 0", "last_login_at": "DATETIME",
                  "must_change_password": "BOOLEAN DEFAULT 0"},
        "chat_sessions": {"message_count": "INTEGER DEFAULT 0", "is_deleted": "BOOLEAN DEFAULT 0"},
        "messages": {"trace_json": "TEXT", "latency_ms": "INTEGER"},
        "knowledge_items": {
            "status": "TEXT DEFAULT 'published'",
            "visibility": "TEXT DEFAULT 'internal'",
            "sensitivity": "TEXT DEFAULT 'L2'",
            "retention_days": "INTEGER DEFAULT 0",
            "meta_json": "TEXT",
        },
    }
    with engine.connect() as conn:
        for table, columns in additions.items():
            existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            if not existing:
                continue
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    log.info("迁移：%s 增加列 %s", table, name)
        conn.commit()


def reset_engine() -> None:
    """测试用：丢弃引擎，强制按新的 DATABASE_URL 重建。"""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
