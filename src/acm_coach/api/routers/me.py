# -*- coding: utf-8 -*-
"""个人数据导出与删除（PIPL 知情权/可携带权/删除权的技术落点）。

导出范围：本人会话、消息、诊断记录、练习标记。
**不导出**：retrieval_logs（含他人不可见的行为轨迹，且已脱敏后仅用于运维统计）。
"""
from __future__ import annotations


from fastapi import APIRouter, Request

from acm_coach.api.deps import CurrentUser, DbSession, audit
from acm_coach.db.models import (
    ChatSession, DiagResult, DiagSubmission, Message, MessageFeedback,
    PracticeMark, RetrievalLog, User, UserSkill,
)

router = APIRouter(prefix="/api/v1", tags=["me"])


@router.get("/export/my-data")
def export_my_data(request: Request, db: DbSession, user: CurrentUser):
    sessions = db.query(ChatSession).filter(ChatSession.user_id == user.id).all()
    payload = {
        "user": {"id": user.id, "username": user.username, "role": user.role,
                 "created_at": user.created_at.isoformat() if user.created_at else None},
        "sessions": [
            {
                "id": s.id, "title": s.title,
                "messages": [
                    {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
                    for m in db.query(Message).filter(Message.session_id == s.id).order_by(Message.id)
                ],
            }
            for s in sessions
        ],
        "diagnosis": [
            {"problem": d.problem_title, "verdict": d.verdict,
             "created_at": d.created_at.isoformat()}
            for d in db.query(DiagSubmission).filter(DiagSubmission.user_id == user.id)
        ],
        "practice": [
            {"knowledge_id": p.knowledge_id, "created_at": p.created_at.isoformat()}
            for p in db.query(PracticeMark).filter(PracticeMark.user_id == user.id)
        ],
    }
    audit(db, request, "data.export", user=user, resource="user", resource_id=user.id,
          detail={"sessions": len(sessions)})
    db.commit()
    return payload


@router.delete("/me")
def delete_my_account(request: Request, db: DbSession, user: CurrentUser):
    """注销账号（PIPL 删除权）：显式按依赖顺序清除本人全部数据。

    不依赖 SQLite 外键级联（PRAGMA 未必开启），逐表删除；
    retrieval_logs 虽已脱敏，仍随账号一并清除。
    审计保留 user_id/username 快照，账号本身删除后仍可追溯"谁在何时注销"。
    """
    uid = user.id
    session_ids = [s.id for s in db.query(ChatSession).filter(ChatSession.user_id == uid)]
    counts: dict[str, int] = {}
    if session_ids:
        msg_ids = [m.id for m in db.query(Message).filter(Message.session_id.in_(session_ids))]
        if msg_ids:
            counts["feedbacks"] = db.query(MessageFeedback).filter(
                MessageFeedback.message_id.in_(msg_ids)).delete(synchronize_session=False)
            counts["messages"] = db.query(Message).filter(
                Message.id.in_(msg_ids)).delete(synchronize_session=False)
        counts["sessions"] = db.query(ChatSession).filter(
            ChatSession.id.in_(session_ids)).delete(synchronize_session=False)
    counts["diag_results"] = db.query(DiagResult).filter(DiagResult.user_id == uid).delete(synchronize_session=False)
    counts["diag_submissions"] = db.query(DiagSubmission).filter(DiagSubmission.user_id == uid).delete(synchronize_session=False)
    counts["practice_marks"] = db.query(PracticeMark).filter(PracticeMark.user_id == uid).delete(synchronize_session=False)
    counts["user_skills"] = db.query(UserSkill).filter(UserSkill.user_id == uid).delete(synchronize_session=False)
    counts["retrieval_logs"] = db.query(RetrievalLog).filter(RetrievalLog.user_id == uid).delete(synchronize_session=False)

    audit(db, request, "data.delete", user=user, resource="user", resource_id=uid, detail=counts)
    db.query(User).filter(User.id == uid).delete(synchronize_session=False)
    db.commit()
    return {"deleted": True, "purged": counts}
