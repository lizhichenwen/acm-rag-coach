# -*- coding: utf-8 -*-
"""安全合规能力测试：查询脱敏、检索日志清除、type_boost 静默失效护栏、越权审计。"""
from __future__ import annotations

import pytest

from acm_coach.core.security import fingerprint_query
from acm_coach.evaluation.runner import _verify_type_boost


class TestFingerprintQuery:
    def test_format_stable(self):
        fp = fingerprint_query("01背包内层循环为什么倒序")
        assert fp.startswith("sha256:")
        assert fp.endswith("#len=13")
        # 同一查询指纹恒定（排障时可对比两次请求是否同文）
        assert fingerprint_query("01背包内层循环为什么倒序") == fp

    def test_no_raw_text(self):
        q = "我的代码哪里错了"
        fp = fingerprint_query(q)
        assert q not in fp

    def test_distinguishes_queries(self):
        assert fingerprint_query("背包") != fingerprint_query("最短路")


class _StubRetriever:
    """_verify_type_boost 只用到 .docs，构造轻量替身避免加载真实索引。"""

    def __init__(self, docs):
        self.docs = docs


class _StubDoc:
    def __init__(self, source):
        self.source = source


class TestVerifyTypeBoost:
    def test_passes_with_curated(self):
        docs = {"a": _StubDoc("curated"), "b": _StubDoc("cf")}
        _verify_type_boost(_StubRetriever(docs))  # 不抛异常即通过

    def test_fails_without_curated(self):
        docs = {"a": _StubDoc("cf"), "b": _StubDoc("atcoder")}
        with pytest.raises(SystemExit, match="静默失效"):
            _verify_type_boost(_StubRetriever(docs))


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    """独立临时库：不碰真实 data/acm_coach.db。

    get_settings 带 lru_cache，必须清缓存后临时库的 DATABASE_URL 才生效。
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from acm_coach.config import get_settings
    from acm_coach.db import session as sess

    get_settings.cache_clear()
    sess.reset_engine()
    sess.init_db(seed_admin=False)
    yield sess
    sess.reset_engine()
    get_settings.cache_clear()


class TestPurgeUserLogs:
    def test_purge_only_target_user(self, temp_db):
        from acm_coach.db.models import RetrievalLog
        from acm_coach.db.repositories import OpsRepository

        with temp_db.session_scope() as db:
            db.add_all([
                RetrievalLog(user_id=1, query="sha256:a#len=1", level="strong"),
                RetrievalLog(user_id=1, query="sha256:b#len=2", level="none"),
                RetrievalLog(user_id=2, query="sha256:c#len=3", level="weak"),
            ])
        with temp_db.session_scope() as db:
            repo = OpsRepository(db)
            assert repo.purge_user_logs(1) == 2
        with temp_db.session_scope() as db:
            remaining = db.query(RetrievalLog).all()
            assert len(remaining) == 1
            assert remaining[0].user_id == 2


class TestAccessDeniedAudit:
    def test_forbidden_writes_audit(self, temp_db, monkeypatch):
        """student 访问 staff 接口 → 403 + access_denied 落审计。"""
        monkeypatch.setenv("APP_ENV", "test")
        from fastapi.testclient import TestClient

        from acm_coach.api.app import create_app
        from acm_coach.core.security import create_access_token
        from acm_coach.db.models import AuditLog, User
        from acm_coach.core.security import hash_password

        with temp_db.session_scope() as db:
            db.add(User(username="stu", password_hash=hash_password("Passw0rd!x"), role="student"))

        client = TestClient(create_app())
        token, _ = create_access_token(1, "stu", "student", version=0)
        resp = client.get("/api/v1/ops/retrieval",
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

        with temp_db.session_scope() as db:
            rows = db.query(AuditLog).filter(AuditLog.action == "access_denied").all()
            assert len(rows) == 1
            assert rows[0].result == "denied"
            assert rows[0].user_id == 1
