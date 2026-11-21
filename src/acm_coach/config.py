# -*- coding: utf-8 -*-
"""全局配置（十二要素应用 / 配置与代码分离）。

所有可变参数集中在这里，通过环境变量或 `.env` 注入，代码中不出现任何密钥常量。
`Settings` 是唯一的配置出口，其他模块只允许 `get_settings()`，便于测试时替换。
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- 应用 ----------------
    app_name: str = "ACM 智能陪练系统"
    app_version: str = "4.0.0"
    app_env: str = "dev"  # dev / test / prod
    debug: bool = False
    # 调试类只读接口（如 /api/v1/ops/config）默认不注册路由，请求返回 404。
    # 仅在受控环境显式开启，避免泄露检索/运维参数（docs/05 §3.3）。
    enable_debug_api: bool = False
    log_level: str = "INFO"
    log_json: bool = False  # 生产置 true，输出单行 JSON 便于采集
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    # ---------------- 安全 ----------------
    secret_key: str = ""  # 留空则自动生成并持久化到 data/.secret
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 120
    pbkdf2_iterations: int = 600_000
    login_max_failures: int = 5
    login_lock_seconds: int = 300
    admin_username: str = "admin"
    admin_password: str = ""  # 留空则由初始化流程生成一次性随机口令（禁止默认口令）

    # ---------------- 存储 ----------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    database_url: str = ""
    knowledge_dirs: str = "data/curated,data/knowledge"

    # ---------------- 大模型 ----------------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_timeout: int = 120
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.3
    llm_max_continue: int = 1
    history_turns: int = 3
    # 历史消息保真：单条历史的字符上限。旧版硬编码 1500 字，导致"长题面+代码"
    # 在送入模型前被整段截断，多轮追问时模型看不到代码。分两类保留：
    # history_text_cap 用于纯文本消息；含代码的消息按 code/nl 分别保留。
    history_text_cap: int = 3000
    history_code_cap: int = 6000
    history_nl_keep: int = 600

    # ---------------- 检索（召回 / 融合 / 重排 / 截断） ----------------
    enable_vector: bool = True
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    # 向量入库文本：content_only 与 v3 严格对齐（v3 只 embed content），
    # title_plus_content 是待验证的增益项（E11），默认关闭以保证基线可比。
    embed_text_mode: str = "content_only"
    rerank_model: str = "BAAI/bge-reranker-base"
    enable_rerank: bool = True
    rerank_download: bool = False
    top_k: int = 6
    recall_k: int = 24
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # BM25 字段加权（docs/04 §1.2）：title×3 / tags×2 / 正文×1。
    # 默认关闭以保持与历史索引及阈值标定的可比性；开启需重建索引并重新标定强/弱阈值。
    enable_bm25_field_boost: bool = False
    bm25_title_weight: float = 3.0
    bm25_tags_weight: float = 2.0
    # 纯词法模式下把 BM25 分值映射到 0-1 的尺度（仅影响降级模式与 G1 消融）
    bm25_score_scale: float = 8.0
    rrf_k: int = 60
    weight_bm25: float = 1.0
    weight_vector: float = 1.0
    # 精讲条目加权与配额：避免 8500 条薄条目把 120 条精讲挤出上下文
    type_boost: bool = True
    curated_weight: float = 1.15
    curated_quota_ratio: float = 0.6
    rerank_top_n: int = 24
    # 送进 cross-encoder 的文档截断长度（字符）。注意真正的开销是 tokenizer 后的
    # token 数：语料 p99 仅 423 字符（≈300 token），所以 1200 字符几乎不触发截断。
    # 2026-11-21 实测 24 对下 512→192 token 可省约 25% 延迟，
    # 且与 512 的排序 Spearman 相关 0.965（30 query 抽样），故默认降到 192。
    rerank_doc_chars: int = 1200
    # cross-encoder 的 max_length（token）。512 是模型上限，但本项目文档 p50≈260
    # token，实测 192 已覆盖头部信息且保序性 0.965；再低（≤160）相关性掉到 0.75。
    rerank_max_length: int = 192
    # 推理 batch size。实测 8/16/24/32 差异 <10%，16 为吞吐拐点。
    rerank_batch_size: int = 16
    # 重排结果 LRU 缓存容量。键为 (query, 候选 id 序列, 模型, max_length)；
    # 同一 query + 同一候选池的重复请求（答辩演示 / 热点问题）直接命中，零重排开销。
    # 0 = 关闭缓存。
    rerank_cache_size: int = 256
    # 后台预热是否连带加载重排模型：true 让首条对话不再等 3s 模型加载，
    # 代价是启动后常驻 ~1.2GB 内存。内存紧张的学生机可设 false 改为首次对话懒加载。
    warmup_reranker: bool = True
    # 相关性三档（max(向量余弦, BM25 归一化) 口径，取两路置信度较强者）
    # 2026-11-08 按 data/reports/calibration.json 回填（指纹 n=8558;bge-small-zh-v1.5;cosine，
    # 门关闭状态裸分测量）：strong = 域内查询下四分位，weak = 域外查询 p90 + 0.02；
    # 换模型/换库规模必须重新标定
    strong_threshold: float = 0.9382
    weak_threshold: float = 0.572
    rel_margin: float = 0.12
    gap_margin: float = 0.05
    context_char_budget: int = 4000
    query_rewrite: str = "auto"  # on / off / auto

    # ---------------- 爬虫 ----------------
    crawler_user_agent: str = "ACM-Coach-Bot/4.0 (+educational; contact: student@example.edu)"
    crawler_qps: float = 1.0
    crawler_timeout: int = 30
    crawler_retries: int = 3
    crawler_cache_ttl: int = 86400
    cf_api_base: str = "https://codeforces.com/api"
    atcoder_problems_url: str = "https://kenkoooo.com/atcoder/resources/problems.json"
    atcoder_models_url: str = "https://kenkoooo.com/atcoder/resources/problem-models.json"

    @field_validator("app_env")
    @classmethod
    def _check_env(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"dev", "test", "prod"}:
            raise ValueError("APP_ENV 只能是 dev/test/prod")
        return v

    # ---------------- 派生路径 ----------------
    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def taxonomy_dir(self) -> Path:
        return self.data_dir / "taxonomy"

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'acm_coach.db').as_posix()}"

    @property
    def knowledge_paths(self) -> list[Path]:
        out: list[Path] = []
        for raw in self.knowledge_dirs.split(","):
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw)
            out.append(p if p.is_absolute() else PROJECT_ROOT / p)
        return out

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    def resolve_secret_key(self) -> str:
        """优先环境变量；否则生成一次并落到 data/.secret（gitignore）。"""
        if self.secret_key:
            return self.secret_key
        path = self.data_dir / ".secret"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
        key = secrets.token_urlsafe(48)
        path.write_text(key, encoding="utf-8")
        return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for p in (settings.index_dir, settings.raw_dir, settings.taxonomy_dir):
        p.mkdir(parents=True, exist_ok=True)
    return settings
