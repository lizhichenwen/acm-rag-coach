# -*- coding: utf-8 -*-
"""代码诊断接口与统计接口。"""
from __future__ import annotations

from fastapi import APIRouter, Request

from acm_coach.api.deps import CurrentUser, DbSession, audit
from acm_coach.api.schemas import DiagnosisRequest, DiagnosisResponse, StatsResponse
from acm_coach.db.repositories import (
    ChatRepository, DiagnosisRepository, KnowledgeRepository, PracticeRepository,
)
from acm_coach.diagnosis.engine import diagnose
from acm_coach.services.intent import INTENT_LABELS
from acm_coach.services.llm import LLMClient

diag_router = APIRouter(prefix="/api/v1/diagnosis", tags=["diagnosis"])
stats_router = APIRouter(prefix="/api/v1/stats", tags=["stats"])


@diag_router.post("", response_model=DiagnosisResponse)
def submit(payload: DiagnosisRequest, request: Request, db: DbSession, user: CurrentUser):
    repo = DiagnosisRepository(db)
    submission = repo.create_submission(user.id, payload.problem_title, payload.code,
                                        payload.verdict, payload.language)
    result = diagnose(payload.code, payload.verdict, payload.problem_title, LLMClient(),
                      language=payload.language)

    import json

    row = repo.save_result(
        submission_id=submission.id, user_id=user.id,
        cause_code=result["cause_code"], cause_label=result["cause_label"],
        points_json=json.dumps(result["points"], ensure_ascii=False),
        evidence_json=json.dumps(result["evidence"], ensure_ascii=False),
        suggestions_json=json.dumps(result["suggestions"], ensure_ascii=False),
        confidence=float(result["confidence"]), engine=result["engine"],
    )
    # 只有 AC 视为该知识点已掌握；非 AC 表示该知识点仍需强化（影响薄弱点推荐）
    repo.update_skills(user.id, result["points"], accepted=payload.verdict == "AC")
    audit(db, request, "diagnosis.submit", user=user, resource="submission",
          resource_id=submission.id)
    db.commit()
    return DiagnosisResponse(
        id=row.id, cause_code=result["cause_code"], cause_label=result["cause_label"],
        points=result["points"], evidence=result["evidence"],
        suggestions=result["suggestions"], confidence=result["confidence"],
        engine=result["engine"], features=result.get("features", {}),
    )


@diag_router.get("/history")
def history(db: DbSession, user: CurrentUser, limit: int = 30):
    return DiagnosisRepository(db).history(user.id, min(limit, 100))


@diag_router.get("/skills")
def skills(db: DbSession, user: CurrentUser):
    return DiagnosisRepository(db).skills(user.id)


@stats_router.get("/overview", response_model=StatsResponse)
def overview(db: DbSession, user: CurrentUser):
    chat = ChatRepository(db)
    stats = chat.stats_for_user(user.id)
    intents = chat.intent_distribution(user.id)
    knowledge = KnowledgeRepository(db)
    return StatsResponse(
        **stats,
        intents={INTENT_LABELS.get(k, k): v for k, v in intents.items()},
        top_tags=[[k, v] for k, v in knowledge.tag_distribution(12)],
    )


@stats_router.get("/knowledge")
def knowledge_stats(db: DbSession, user: CurrentUser):
    knowledge = KnowledgeRepository(db)
    return {
        "counts": knowledge.count_by(),
        "top_tags": knowledge.tag_distribution(20),
        "mastered": len(PracticeRepository(db).list(user.id)),
    }
