# -*- coding: utf-8 -*-
"""对话接口：会话管理 + SSE 流式问答 + 反馈。

流式实现要点：
- 用 `text/event-stream`，事件分 `meta / token / done` 三类，前端按事件名分发；
- 生成器内部自建数据库会话：FastAPI 的依赖注入会话在流式响应开始前就可能关闭，
  复用会出现 "Session is closed"；
- 设置 `X-Accel-Buffering: no`，避免 Nginx 缓冲导致"一次性吐出"。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from acm_coach.api.deps import CurrentUser, DbSession, audit
from acm_coach.api.schemas import (
    ChatRequest, FeedbackRequest, MessageResponse, SessionCreateRequest, SessionResponse,
)
from acm_coach.core.errors import NotFound
from acm_coach.db.repositories import ChatRepository
from acm_coach.db.session import session_scope
from acm_coach.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _session_response(s) -> SessionResponse:
    return SessionResponse(
        id=s.id, title=s.title, message_count=s.message_count or 0,
        created_at=s.created_at.strftime("%Y-%m-%d %H:%M"),
        updated_at=s.updated_at.strftime("%Y-%m-%d %H:%M"),
    )


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(payload: SessionCreateRequest, db: DbSession, user: CurrentUser):
    session = ChatRepository(db).create_session(user.id, payload.title)
    db.commit()
    return _session_response(session)


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(db: DbSession, user: CurrentUser, limit: int = 50):
    rows = ChatRepository(db).list_sessions(user.id, min(limit, 100))
    return [_session_response(s) for s in rows]


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: int, request: Request, db: DbSession, user: CurrentUser):
    repo = ChatRepository(db)
    session = repo.get_session(session_id, user.id)
    if session is None:
        raise NotFound("会话不存在")
    repo.delete_session(session)
    audit(db, request, "chat.delete_session", user=user, resource="session", resource_id=session_id)
    db.commit()


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
def list_messages(session_id: int, db: DbSession, user: CurrentUser):
    repo = ChatRepository(db)
    if repo.get_session(session_id, user.id) is None:
        raise NotFound("会话不存在")
    out = []
    for m in repo.list_messages(session_id):
        try:
            sources = json.loads(m.sources_json) if m.sources_json else []
        except ValueError:
            sources = []
        out.append(MessageResponse(
            id=m.id, role=m.role, content=m.content, intent=m.intent,
            matched=bool(m.matched), sources=sources, latency_ms=m.latency_ms,
            created_at=m.created_at.strftime("%Y-%m-%d %H:%M"),
        ))
    return out


@router.post("/sessions/{session_id}/messages")
def send_message(session_id: int, payload: ChatRequest, request: Request,
                 db: DbSession, user: CurrentUser):
    repo = ChatRepository(db)
    session = repo.get_session(session_id, user.id)
    if session is None:
        raise NotFound("会话不存在")
    audit(db, request, "chat.ask", user=user, resource="session", resource_id=session_id)
    db.commit()

    if not payload.stream:
        return ChatService(db).answer(user, session, payload.prompt)

    user_id, prompt = user.id, payload.prompt

    def event_stream():
        with session_scope() as stream_db:
            stream_session = ChatRepository(stream_db).get_session(session_id, user_id)
            if stream_session is None:
                yield "event: error\ndata: {\"message\": \"会话不存在\"}\n\n"
                return
            service = ChatService(stream_db)
            try:
                for event in service.stream(user, stream_session, prompt):
                    yield (
                        f"event: {event['event']}\n"
                        f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                    )
            except Exception as exc:  # 流内异常必须转成事件，否则前端只看到断流
                payload = {"message": f"生成失败：{type(exc).__name__}"}
                yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@router.post("/messages/{message_id}/feedback", status_code=204)
def feedback(message_id: int, payload: FeedbackRequest, db: DbSession, user: CurrentUser):
    ChatRepository(db).upsert_feedback(message_id, user.id, payload.rating, payload.comment)
    db.commit()
