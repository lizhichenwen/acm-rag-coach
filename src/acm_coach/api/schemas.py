# -*- coding: utf-8 -*-
"""API 出入参模型（Pydantic）。

统一约定：
- 请求体一律 `*Request`，响应体 `*Response`；
- 分页响应带 `total / page / size`，前端不必猜；
- 任何返回给前端的字段都经过 `response_model` 过滤，避免 ORM 字段泄露。
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ---------------- 认证 ----------------
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=64)

    @field_validator("username")
    @classmethod
    def _username_charset(cls, v: str) -> str:
        cleaned = v.strip()
        if not all(c.isalnum() or c in "_-" for c in cleaned):
            raise ValueError("用户名只能包含字母、数字、下划线和短横线")
        return cleaned


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: int
    user: "UserResponse"
    must_change_password: bool = False


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: str | None = None


class AdminUserResponse(BaseModel):
    """管理视角的用户视图：仍是字段白名单，不回传哈希/令牌版本等内部状态。"""
    id: int
    username: str
    role: str
    is_active: bool
    created_at: str | None = None
    last_login_at: str | None = None


class UserRoleUpdate(BaseModel):
    role: str

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        allowed = {"student", "admin", "auditor"}
        value = (v or "").strip().lower()
        if value not in allowed:
            raise ValueError(f"role 必须是 {sorted(allowed)} 之一")
        return value


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8, max_length=64)


# ---------------- 对话 ----------------
class SessionCreateRequest(BaseModel):
    title: str = Field(default="新会话", max_length=120)


class SessionResponse(BaseModel):
    id: int
    title: str
    message_count: int
    created_at: str
    updated_at: str


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    stream: bool = True


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    intent: str | None = None
    matched: bool = False
    sources: list[dict] = []
    latency_ms: int | None = None
    created_at: str


class SourceItem(BaseModel):
    id: str
    title: str
    difficulty: str = ""
    tags: str = ""
    domain: str = ""
    source: str = "curated"
    url: str = ""
    score: float = 0.0
    similarity: float = 0.0


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=500)

    @field_validator("rating")
    @classmethod
    def _nonzero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("rating 只能是 1（有用）或 -1（没用）")
        return v


# ---------------- 知识库 ----------------
class KnowledgeItemResponse(BaseModel):
    id: str
    title: str
    difficulty: str
    tags: str
    domain: str
    source: str
    url: str | None = None
    rating: int | None = None


class KnowledgeDetail(KnowledgeItemResponse):
    content: str
    updated_at: str
    similar: list[KnowledgeItemResponse] = []


class KnowledgeCreateRequest(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    difficulty: str = Field(default="中等")
    tags: str = Field(default="")
    domain: str = Field(default="")
    description: str = ""
    solution: str = ""
    pitfalls: str = ""


class PageResponse(BaseModel):
    total: int
    page: int
    size: int
    items: list[KnowledgeItemResponse]


# ---------------- 诊断 ----------------
class DiagnosisRequest(BaseModel):
    # 支持粘贴完整题面（前端 st.text_area 的 max_chars 与此处保持一致）
    problem_title: str = Field(default="", max_length=4000)
    code: str = Field(min_length=10, max_length=20000)
    verdict: str = Field(default="WA")
    language: str = "cpp"

    @field_validator("verdict")
    @classmethod
    def _verdict(cls, v: str) -> str:
        allowed = {"AC", "WA", "TLE", "RE", "MLE", "CE", "PE", "OLE"}
        v = (v or "WA").upper()
        if v not in allowed:
            raise ValueError(f"判题结果必须是 {sorted(allowed)} 之一")
        return v

    @field_validator("language")
    @classmethod
    def _language(cls, v: str) -> str:
        allowed = {"cpp", "python", "java"}
        value = (v or "cpp").strip().lower()
        aliases = {"c++": "cpp", "cxx": "cpp", "py": "python"}
        value = aliases.get(value, value)
        if value not in allowed:
            raise ValueError(f"language 必须是 {sorted(allowed)} 之一")
        return value


class DiagnosisResponse(BaseModel):
    id: int | None = None
    cause_code: str
    cause_label: str
    points: list[str]
    evidence: list[str]
    suggestions: list[str]
    confidence: float
    engine: str
    features: dict = {}


# ---------------- 统计 / 运营 ----------------
class StatsResponse(BaseModel):
    questions: int
    sessions: int
    matched: int
    mastered: int
    match_rate: float
    avg_latency_ms: int
    intents: dict[str, int] = {}
    top_tags: list[list] = []


class AuditItem(BaseModel):
    id: int
    username: str | None
    action: str
    resource: str | None
    result: str
    created_at: str


TokenResponse.model_rebuild()
