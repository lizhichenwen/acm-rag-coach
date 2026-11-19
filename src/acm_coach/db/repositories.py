# -*- coding: utf-8 -*-
"""仓储层（Repository）。

职责边界：只做"数据怎么取、怎么写"，不含业务规则，不直接抛 HTTP 异常。
服务层通过仓储访问数据库，便于单测时用内存库替换、也避免 SQL 散落在路由里。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from acm_coach.db.models import (
    AuditLog, ChatSession, CrawlRun, DiagResult, DiagSubmission, KnowledgeItem,
    Message, MessageFeedback, PracticeMark, RetrievalLog, User, UserSkill, utcnow,
)


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def by_username(self, username: str) -> User | None:
        return self.db.scalar(select(User).where(User.username == username))

    def by_id(self, user_id: int) -> User | None:
        return self.db.get(User, user_id)

    def create(self, username: str, password_hash: str, role: str = "student") -> User:
        user = User(username=username, password_hash=password_hash, role=role)
        self.db.add(user)
        self.db.flush()
        return user

    def touch_login(self, user: User) -> None:
        user.last_login_at = utcnow()

    def bump_token_version(self, user: User) -> None:
        user.token_version += 1

    def list_all(self, limit: int = 100, offset: int = 0) -> list[User]:
        return list(self.db.scalars(
            select(User).order_by(User.id).limit(limit).offset(offset)
        ))

    def set_role(self, user: User, role: str) -> None:
        user.role = role

    def set_active(self, user: User, active: bool) -> None:
        user.is_active = active

    def clear_must_change_password(self, user: User) -> None:
        user.must_change_password = False

    def count(self) -> int:
        return self.db.scalar(select(func.count()).select_from(User)) or 0


class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(self, user_id: int, title: str = "新会话") -> ChatSession:
        obj = ChatSession(user_id=user_id, title=title[:120])
        self.db.add(obj)
        self.db.flush()
        return obj

    def get_session(self, session_id: int, user_id: int, *, include_deleted: bool = False) -> ChatSession | None:
        stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        if not include_deleted:
            stmt = stmt.where(ChatSession.is_deleted.is_(False))
        return self.db.scalar(stmt)

    def list_sessions(self, user_id: int, limit: int = 50) -> list[ChatSession]:
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id, ChatSession.is_deleted.is_(False))
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt))

    def delete_session(self, session: ChatSession) -> None:
        session.is_deleted = True

    def add_message(self, session_id: int, role: str, content: str, **fields) -> Message:
        msg = Message(session_id=session_id, role=role, content=content, **fields)
        self.db.add(msg)
        sess = self.db.get(ChatSession, session_id)
        if sess:
            sess.message_count = (sess.message_count or 0) + 1
            sess.updated_at = utcnow()
            if role == "user" and sess.message_count <= 1:
                sess.title = (content or "新会话").strip().replace("\n", " ")[:40]
        self.db.flush()
        return msg

    def list_messages(self, session_id: int, limit: int = 200) -> list[Message]:
        stmt = select(Message).where(Message.session_id == session_id).order_by(Message.id).limit(limit)
        return list(self.db.scalars(stmt))

    def recent_messages(self, session_id: int, turns: int = 3) -> list[Message]:
        stmt = (
            select(Message).where(Message.session_id == session_id)
            .order_by(Message.id.desc()).limit(max(turns, 1) * 2 + 2)
        )
        return list(reversed(list(self.db.scalars(stmt))))

    def last_matched_titles(self, session_id: int) -> list[str]:
        stmt = (
            select(Message)
            .where(Message.session_id == session_id, Message.role == "assistant", Message.matched.is_(True))
            .order_by(Message.id.desc()).limit(1)
        )
        msg = self.db.scalar(stmt)
        if not msg or not msg.sources_json:
            return []
        try:
            return [s.get("title", "") for s in json.loads(msg.sources_json) if s.get("title")]
        except (ValueError, AttributeError):
            return []

    def upsert_feedback(self, message_id: int, user_id: int, rating: int, comment: str | None) -> MessageFeedback:
        obj = self.db.scalar(
            select(MessageFeedback).where(
                MessageFeedback.message_id == message_id, MessageFeedback.user_id == user_id
            )
        )
        if obj:
            obj.rating, obj.comment = rating, comment
        else:
            obj = MessageFeedback(message_id=message_id, user_id=user_id, rating=rating, comment=comment)
            self.db.add(obj)
        self.db.flush()
        return obj

    # ---- 统计 ----
    def stats_for_user(self, user_id: int) -> dict:
        total = self.db.scalar(
            select(func.count()).select_from(Message)
            .join(ChatSession, Message.session_id == ChatSession.id)
            .where(ChatSession.user_id == user_id, Message.role == "user")
        ) or 0
        sessions = self.db.scalar(
            select(func.count()).select_from(ChatSession)
            .where(ChatSession.user_id == user_id, ChatSession.is_deleted.is_(False))
        ) or 0
        matched = self.db.scalar(
            select(func.count()).select_from(Message)
            .join(ChatSession, Message.session_id == ChatSession.id)
            .where(ChatSession.user_id == user_id, Message.role == "assistant", Message.matched.is_(True))
        ) or 0
        marks = self.db.scalar(select(func.count()).select_from(PracticeMark).where(PracticeMark.user_id == user_id)) or 0
        avg_latency = self.db.scalar(
            select(func.avg(Message.latency_ms))
            .join(ChatSession, Message.session_id == ChatSession.id)
            .where(ChatSession.user_id == user_id, Message.latency_ms.is_not(None))
        )
        return {
            "questions": total,
            "sessions": sessions,
            "matched": matched,
            "mastered": marks,
            "match_rate": round(matched / total, 3) if total else 0.0,
            "avg_latency_ms": int(avg_latency or 0),
        }

    def intent_distribution(self, user_id: int) -> dict[str, int]:
        stmt = (
            select(Message.intent, func.count())
            .join(ChatSession, Message.session_id == ChatSession.id)
            .where(ChatSession.user_id == user_id, Message.role == "user", Message.intent.is_not(None))
            .group_by(Message.intent)
        )
        return {row[0]: row[1] for row in self.db.execute(stmt)}


class KnowledgeRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert_many(self, items: list[dict]) -> tuple[int, int]:
        """按 id 幂等写入，返回 (新增, 更新)。"""
        if not items:
            return 0, 0
        ids = [it["id"] for it in items]
        existing = {
            row.id: row
            for row in self.db.scalars(select(KnowledgeItem).where(KnowledgeItem.id.in_(ids)))
        }
        inserted = updated = 0
        for item in items:
            row = existing.get(item["id"])
            if row is None:
                self.db.add(KnowledgeItem(**item))
                inserted += 1
            else:
                changed = False
                for key, value in item.items():
                    if key != "id" and getattr(row, key, None) != value:
                        setattr(row, key, value)
                        changed = True
                if changed:
                    updated += 1
        self.db.flush()
        return inserted, updated

    def get(self, item_id: str) -> KnowledgeItem | None:
        return self.db.get(KnowledgeItem, item_id)

    def search(self, keyword: str = "", domain: str = "", difficulty: str = "",
               source: str = "", page: int = 1, size: int = 20,
               include_restricted: bool = False) -> tuple[list[KnowledgeItem], int]:
        # 四条读路径必须同一组可见性条件，否则"检索封住了、列表没封"就是绕过口
        stmt = select(KnowledgeItem).where(
            KnowledgeItem.is_deleted.is_(False),
            KnowledgeItem.status == "published",
        )
        if not include_restricted:
            stmt = stmt.where(KnowledgeItem.visibility != "restricted")
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(or_(KnowledgeItem.title.like(like), KnowledgeItem.tags.like(like)))
        if domain:
            stmt = stmt.where(KnowledgeItem.domain == domain)
        if difficulty:
            stmt = stmt.where(KnowledgeItem.difficulty == difficulty)
        if source:
            stmt = stmt.where(KnowledgeItem.source == source)
        total = self.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = list(self.db.scalars(stmt.order_by(KnowledgeItem.updated_at.desc())
                                    .offset((page - 1) * size).limit(size)))
        return rows, total

    def get_visible(self, item_id: str, include_restricted: bool = False) -> KnowledgeItem | None:
        """详情接口专用：不可见一律返回 None（路由层转 404，避免泄露"存在性"）。"""
        row = self.db.get(KnowledgeItem, item_id)
        if row is None or row.is_deleted or row.status != "published":
            return None
        if not include_restricted and row.visibility == "restricted":
            return None
        return row

    def all_for_index(self) -> list[KnowledgeItem]:
        stmt = select(KnowledgeItem).where(KnowledgeItem.is_deleted.is_(False))
        return list(self.db.scalars(stmt))

    def count_by(self) -> dict:
        total = self.db.scalar(
            select(func.count()).select_from(KnowledgeItem).where(KnowledgeItem.is_deleted.is_(False))
        ) or 0
        by_source = {
            row[0]: row[1] for row in self.db.execute(
                select(KnowledgeItem.source, func.count())
                .where(KnowledgeItem.is_deleted.is_(False)).group_by(KnowledgeItem.source)
            )
        }
        by_domain = {
            row[0]: row[1] for row in self.db.execute(
                select(KnowledgeItem.domain, func.count())
                .where(KnowledgeItem.is_deleted.is_(False)).group_by(KnowledgeItem.domain)
            )
        }
        return {"total": total, "by_source": by_source, "by_domain": by_domain}

    def tag_distribution(self, limit: int = 15) -> list[tuple[str, int]]:
        rows = self.db.execute(
            select(KnowledgeItem.tags).where(KnowledgeItem.is_deleted.is_(False))
        ).scalars()
        counter: dict[str, int] = {}
        for tags in rows:
            for tag in (tags or "").split(","):
                tag = tag.strip()
                if tag:
                    counter[tag] = counter.get(tag, 0) + 1
        return sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:limit]

    def set_status(self, item_id: str, status: str) -> bool:
        """审核状态流转（draft/reviewing/published/retired）。返回是否命中。"""
        row = self.db.get(KnowledgeItem, item_id)
        if row is None:
            return False
        row.status = status
        self.db.flush()
        return True

    def soft_delete(self, item_id: str) -> bool:
        row = self.get(item_id)
        if not row:
            return False
        row.is_deleted = True
        return True


class PracticeRepository:
    def __init__(self, db: Session):
        self.db = db

    def mark(self, user_id: int, knowledge_id: str) -> None:
        exists = self.db.scalar(
            select(PracticeMark).where(
                PracticeMark.user_id == user_id, PracticeMark.knowledge_id == knowledge_id
            )
        )
        if not exists:
            self.db.add(PracticeMark(user_id=user_id, knowledge_id=knowledge_id))

    def list(self, user_id: int) -> list[dict]:
        rows = self.db.scalars(
            select(PracticeMark).where(PracticeMark.user_id == user_id).order_by(PracticeMark.created_at.desc())
        )
        out = []
        for row in rows:
            item = self.db.get(KnowledgeItem, row.knowledge_id)
            out.append({
                "knowledge_id": row.knowledge_id,
                "title": item.title if item else "(已删除)",
                "difficulty": item.difficulty if item else "",
                "tags": item.tags if item else "",
                "marked_at": row.created_at.strftime("%Y-%m-%d %H:%M"),
            })
        return out


class DiagnosisRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_submission(self, user_id: int, problem_title: str, code: str,
                          verdict: str, language: str) -> DiagSubmission:
        obj = DiagSubmission(user_id=user_id, problem_title=(problem_title or "")[:4000],
                             code=code, verdict=verdict, language=language)
        self.db.add(obj)
        self.db.flush()
        return obj

    def save_result(self, **fields) -> DiagResult:
        obj = DiagResult(**fields)
        self.db.add(obj)
        self.db.flush()
        return obj

    def history(self, user_id: int, limit: int = 30) -> list[dict]:
        stmt = (
            select(DiagResult, DiagSubmission)
            .join(DiagSubmission, DiagResult.submission_id == DiagSubmission.id)
            .where(DiagResult.user_id == user_id)
            .order_by(DiagResult.id.desc()).limit(limit)
        )
        out = []
        for result, sub in self.db.execute(stmt):
            out.append({
                "id": result.id,
                "problem_title": sub.problem_title,
                "verdict": sub.verdict,
                "cause_code": result.cause_code,
                "cause_label": result.cause_label,
                "points": json.loads(result.points_json or "[]"),
                "confidence": result.confidence,
                "engine": result.engine,
                "created_at": result.created_at.strftime("%Y-%m-%d %H:%M"),
            })
        return out

    def update_skills(self, user_id: int, points: list[str], accepted: bool) -> None:
        """按诊断结果更新知识点掌握度（任务书「重点研究的问题 4」）。

        口径：初始 0.3，钳制在 [0, 1]；`accepted=True` 表示本次判题结果为 AC，
        该知识点已掌握 → 掌握度上调；非 AC 表示仍出错 → 掌握度下调。
        每次诊断命中的知识点 attempts + 1，供画像页展示练习频次。

        注意：调用方传的是 `verdict == "AC"`，不是"是否命中错因"。
        旧版参数名 hit 语义含糊，且增减方向写反（AC 反而扣 0.12、做错反而加 0.06），
        会让「知识点画像」与「薄弱点推荐」整体失真，2026-11-12 修正。
        """
        from acm_coach.diagnosis.knowledge import KP_BY_ID

        for pid in points:
            row = self.db.scalar(
                select(UserSkill).where(UserSkill.user_id == user_id, UserSkill.point_id == pid)
            )
            if row is None:
                name = KP_BY_ID.get(pid, {}).get("name", pid)
                # 列默认值（mastery=0.3 / attempts=0）要等 flush 才生效，
                # 新建行上直接 += 会因 None 抛 TypeError，这里显式赋初值。
                row = UserSkill(user_id=user_id, point_id=pid, point_name=name,
                                mastery=0.3, attempts=0)
                self.db.add(row)
            # 历史行可能有 NULL，统一兜底，避免一次脏数据让诊断接口 500
            attempts = row.attempts or 0
            mastery = row.mastery if row.mastery is not None else 0.3
            row.attempts = attempts + 1
            row.mastery = round(max(0.0, min(1.0, mastery + (0.12 if accepted else -0.06))), 3)

    def skills(self, user_id: int) -> list[dict]:
        rows = self.db.scalars(
            select(UserSkill).where(UserSkill.user_id == user_id).order_by(UserSkill.mastery)
        )
        return [
            {"point_id": r.point_id, "point_name": r.point_name, "mastery": r.mastery, "attempts": r.attempts}
            for r in rows
        ]


class AuditRepository:
    def __init__(self, db: Session):
        self.db = db

    def write(self, action: str, *, user_id: int | None = None, username: str | None = None,
              resource: str | None = None, resource_id: str | None = None,
              result: str = "ok", ip: str | None = None, trace_id: str | None = None,
              detail: dict | None = None) -> None:
        self.db.add(AuditLog(
            user_id=user_id, username=username, action=action, resource=resource,
            resource_id=str(resource_id) if resource_id is not None else None,
            result=result, ip=ip, trace_id=trace_id,
            detail_json=json.dumps(detail, ensure_ascii=False) if detail else None,
        ))

    def recent(self, limit: int = 100) -> list[AuditLog]:
        return list(self.db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)))


class OpsRepository:
    """爬取任务与检索日志。"""

    def __init__(self, db: Session):
        self.db = db

    def start_crawl(self, source: str) -> CrawlRun:
        run = CrawlRun(source=source, status="running")
        self.db.add(run)
        self.db.flush()
        return run

    def finish_crawl(self, run_id: int, *, status: str, total: int = 0, inserted: int = 0,
                     updated: int = 0, skipped: int = 0, error: str | None = None) -> None:
        run = self.db.get(CrawlRun, run_id)
        if not run:
            return
        run.status, run.total, run.inserted = status, total, inserted
        run.updated, run.skipped, run.error = updated, skipped, (error or "")[:490] or None
        run.finished_at = utcnow()

    def list_crawls(self, limit: int = 20) -> list[CrawlRun]:
        return list(self.db.scalars(select(CrawlRun).order_by(CrawlRun.id.desc()).limit(limit)))

    def log_retrieval(self, **fields) -> None:
        self.db.add(RetrievalLog(**fields))

    def retrieval_stats(self, hours: int = 24) -> dict:
        since = datetime.utcnow() - timedelta(hours=hours)
        rows = list(self.db.scalars(select(RetrievalLog).where(RetrievalLog.created_at >= since)))
        if not rows:
            return {"count": 0, "hit_rate": 0.0, "strong": 0, "weak": 0, "none": 0, "p95_ms": 0}
        levels = {"strong": 0, "weak": 0, "none": 0}
        for r in rows:
            levels[r.level] = levels.get(r.level, 0) + 1
        costs = sorted(r.cost_ms for r in rows)
        return {
            "count": len(rows),
            "hit_rate": round(1 - levels["none"] / len(rows), 3),
            "strong": levels["strong"], "weak": levels["weak"], "none": levels["none"],
            "p95_ms": costs[int(len(costs) * 0.95) - 1] if costs else 0,
        }

    def purge_retrieval_logs(self, keep_days: int = 30) -> int:
        cutoff = datetime.utcnow() - timedelta(days=keep_days)
        result = self.db.execute(delete(RetrievalLog).where(RetrievalLog.created_at < cutoff))
        return result.rowcount or 0

    def purge_user_logs(self, user_id: int) -> int:
        """删除指定用户的全部检索日志（PIPL 删除权：注销账号时随账号一并清除）。"""
        result = self.db.execute(delete(RetrievalLog).where(RetrievalLog.user_id == user_id))
        return result.rowcount or 0
