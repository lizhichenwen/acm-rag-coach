# -*- coding: utf-8 -*-
"""混合检索引擎：BM25 词法召回 + 向量语义召回 → 加权 RRF 融合 → 重排 → 相关性截断。

检索质量的关键不在"多召"，而在"敢不敢不召回"：
题库从 77 条扩到 8000+ 条后，"勉强沾边"的条目会大量出现，直接喂给大模型必然串题。
因此这里用三档判定（strong / weak / none），weak 只留极少数并打标，none 直接返回空，
让生成层如实说"题库未收录"。
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger
from acm_coach.rag.bm25 import BM25Index
from acm_coach.rag.text import tokenize
from acm_coach.rag.vector import VectorStore

log = get_logger("rag.index")


@dataclass
class Document:
    id: str
    title: str
    content: str
    difficulty: str = "中等"
    tags: str = ""
    domain: str = "未分类"
    source: str = "curated"
    url: str = ""
    rating: int | None = None
    # 融合键：同一道题的所有切片共享一个 problem_id，RRF 之前先按它塌缩，
    # 否则长精讲被切成 5 片就会吃掉 5 份融合分数，把其他题挤出去。
    problem_id: str = ""
    # 合规控制：检索出口 fail-closed，未发布或受限条目不进候选池
    status: str = "published"
    visibility: str = "internal"
    sensitivity: str = "L2"
    is_deleted: bool = False

    def __post_init__(self) -> None:
        if not self.problem_id:
            self.problem_id = self.id.split("#")[0]


@dataclass
class Hit:
    id: str
    title: str
    content: str
    difficulty: str
    tags: str
    domain: str
    source: str
    url: str
    problem_id: str = ""
    score: float = 0.0          # 最终排序分（0-1）
    similarity: float = 0.0     # 语义/词法置信度（用于三档判定）
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None
    matched_terms: list[str] = field(default_factory=list)
    rank_bm25: int | None = None
    rank_vector: int | None = None

    def to_source(self) -> dict:
        return {
            "id": self.id, "title": self.title, "difficulty": self.difficulty,
            "tags": self.tags, "domain": self.domain, "source": self.source,
            "url": self.url, "score": round(self.score, 4),
            "similarity": round(self.similarity, 4),
        }


@dataclass
class RetrievalResult:
    hits: list[Hit]
    level: str = "none"          # strong / weak / none
    query: str = ""
    mode: str = "direct"
    recall: int = 0
    cost_ms: int = 0
    strategy: str = "hybrid"     # hybrid / bm25-only
    meta: dict = field(default_factory=dict)

    @property
    def best_similarity(self) -> float:
        return max((h.similarity for h in self.hits), default=0.0)

    def sources(self) -> list[dict]:
        return [h.to_source() for h in self.hits]

    def to_dict(self) -> dict:
        return {
            "level": self.level, "query": self.query, "mode": self.mode,
            "recall": self.recall, "final": len(self.hits), "cost_ms": self.cost_ms,
            "strategy": self.strategy, "best_similarity": round(self.best_similarity, 4),
            **self.meta,
        }


class HybridRetriever:
    """检索门面：对上层只暴露 search()，内部隐藏索引加载与多路融合细节。"""

    def __init__(self, index_path: Path | None = None, vector: VectorStore | None = None):
        settings = get_settings()
        self.settings = settings
        self.index_path = index_path or (settings.index_dir / "bm25.pkl")
        self.docs: dict[str, Document] = {}
        self.bm25: BM25Index | None = None
        self.vector = vector
        self._loaded = False
        self._load_lock = threading.Lock()
        # 仅在实际加载完成后置位；并发调用方在锁上等待，绝不读到半加载状态。
        self._load_event = threading.Event()

    # ---------------- 装配 ----------------
    def attach(self, docs: dict[str, Document], bm25: BM25Index, vector: VectorStore | None) -> None:
        self.docs, self.bm25, self.vector = docs, bm25, vector
        self._loaded = True
        self._load_event.set()

    def ensure_loaded(self) -> None:
        """确保索引就绪。后台预热与首个请求并发时：只加载一次，且请求会
        在锁/事件上等待预热真正完成，绝不读到半加载的空索引。"""
        if self._load_event.is_set():
            return
        with self._load_lock:
            if self._load_event.is_set():
                return
            bm25 = BM25Index.load(self.index_path)
            if bm25 is None:
                log.warning("未找到 BM25 索引（%s），请先执行 `python -m acm_coach.cli index`", self.index_path)
            else:
                self.bm25 = bm25
                if self.settings.enable_vector:
                    self.vector = VectorStore(self.settings.chroma_dir, self.settings.embed_model,
                                              enabled=True)
                # 文档正文从 BM25 索引的同批快照读取，避免每次检索打数据库
                from acm_coach.rag.docstore import load_docstore

                self.docs = load_docstore(self.settings.index_dir)
            self._loaded = True
            self._load_event.set()  # 加载完成后才通知等待方

    @property
    def ready(self) -> bool:
        self.ensure_loaded()
        has_lexical = self.bm25 is not None
        has_vector = bool(self.vector and self.vector.available)
        # 任一路可用即可服务：消融时会把其中一路置空，不能因此判定"未就绪"
        return bool(self.docs) and (has_lexical or has_vector)

    @property
    def doc_count(self) -> int:
        self.ensure_loaded()
        return len(self.docs)

    # ---------------- 检索 ----------------
    def _visible(self, doc: Document) -> bool:
        """检索出口的合规闸门：未发布 / 受限 / 敏感度过高一律不进候选。

        放在召回入口而不是生成前，是因为"先取回再剔除"意味着泄露已经发生
        （日志、trace、调试页面都可能留下痕迹）。
        """
        return (not doc.is_deleted and doc.status == "published"
                and doc.visibility != "restricted")

    def search(self, query: str, queries: list[str] | None = None,
               top_k: int | None = None, recall_k: int | None = None,
               domain: str | None = None, mode: str = "direct") -> RetrievalResult:
        settings = self.settings
        top_k = top_k or settings.top_k
        recall_k = recall_k or settings.recall_k
        start = time.perf_counter()

        if not self.ready or not query.strip():
            return RetrievalResult(hits=[], level="none", query=query, mode=mode,
                                   cost_ms=0, strategy="empty")

        queries = [q for q in (queries or [query]) if q and q.strip()][:3]
        bm25_ranks: dict[str, int] = {}
        vec_ranks: dict[str, int] = {}
        hits: dict[str, Hit] = {}   # 键是 problem_id，不是切片 id

        # ---- 路 1：BM25（多查询取并集，同题多切片只记最优名次一次） ----
        for q in queries:
            seen: set[str] = set()
            position = 0
            for doc_id, score, terms in (self.bm25.search(q, recall_k) if self.bm25 else []):
                doc = self.docs.get(doc_id)
                if doc is None or doc.problem_id in seen or not self._visible(doc):
                    continue
                seen.add(doc.problem_id)
                position += 1
                hit = hits.get(doc.problem_id)
                if hit is None:
                    hit = self._make_hit(doc)
                    hits[doc.problem_id] = hit
                bm25_ranks.setdefault(doc.problem_id, position)
                hit.bm25_score = max(hit.bm25_score, score)
                hit.matched_terms = list(dict.fromkeys(hit.matched_terms + terms))[:8]

        # ---- 路 2：向量（仅主查询，控延迟） ----
        vec_hits: list[tuple[str, float]] = []
        if self.vector and self.vector.available:
            vec_hits = self.vector.query(queries[0], recall_k)
        seen_vec: set[str] = set()
        position = 0
        for doc_id, similarity in vec_hits:
            doc = self.docs.get(doc_id)
            if doc is None or doc.problem_id in seen_vec or not self._visible(doc):
                continue
            seen_vec.add(doc.problem_id)
            position += 1
            hit = hits.get(doc.problem_id)
            if hit is None:
                hit = self._make_hit(doc)
                hits[doc.problem_id] = hit
            vec_ranks[doc.problem_id] = position
            hit.vector_score = similarity

        # ---- 元数据过滤（可解释的硬约束） ----
        # 注意："未分类"必须放行——"不知道属于哪类" ≠ "不属于这类"，
        # 否则 AtCoder 的 1,510 条无标签语料在领域检索时会集体失联。
        if domain:
            filtered = {k: v for k, v in hits.items()
                        if v.domain == domain or v.domain in ("未分类", "")}
            hits = filtered or hits

        # ---- 融合：加权 RRF ----
        k = settings.rrf_k
        for pid, hit in hits.items():
            score = 0.0
            if pid in bm25_ranks:
                score += settings.weight_bm25 / (k + bm25_ranks[pid])
                hit.rank_bm25 = bm25_ranks[pid]
            if pid in vec_ranks:
                score += settings.weight_vector / (k + vec_ranks[pid])
                hit.rank_vector = vec_ranks[pid]
            # 精讲加权：8500 条薄条目会天然占据头部，120 条精讲需要一点结构性倾斜
            if settings.type_boost and hit.source == "curated":
                score *= settings.curated_weight
            hit.rrf_score = score

        ranked = sorted(hits.values(), key=lambda h: h.rrf_score, reverse=True)[:settings.rerank_top_n]

        # ---- 重排（跟随本检索器自身的 settings，消融分组之间互不影响） ----
        from acm_coach.rag.rerank import rerank, rerank_backend

        ranked = rerank(query, ranked, settings)

        # ---- 归一化 + 置信度 ----
        self._normalize(ranked, query)

        # ---- 相关性自适应截断 ----
        chosen, level, quota_applied = self._select(ranked, top_k)
        cost = (time.perf_counter() - start) * 1000
        result = RetrievalResult(
            hits=chosen, level=level, query=query, mode=mode,
            recall=len(ranked), cost_ms=round(cost),
            strategy="hybrid" if vec_hits else "bm25-only",
            meta={
                "bm25_candidates": len(bm25_ranks),
                "vector_candidates": len(vec_ranks),
                "rerank": bool(ranked and ranked[0].rerank_score is not None),
                # cross-encoder / fusion-fallback / off —— 用于确认 G4 真的走了精排
                "rerank_backend": rerank_backend(),
                "quota_applied": quota_applied,
                "queries": queries,
            },
        )
        log.info("retrieval.done query=%r strategy=%s recall=%d final=%d level=%s best=%.3f cost=%.0fms",
                 query[:40], result.strategy, result.recall, len(chosen), level,
                 result.best_similarity, cost, extra={"event": "retrieval.done"})
        return result

    # ---------------- 内部 ----------------
    @staticmethod
    def _make_hit(doc: Document) -> Hit:
        return Hit(id=doc.id, title=doc.title, content=doc.content,
                   difficulty=doc.difficulty, tags=doc.tags, domain=doc.domain,
                   source=doc.source, url=doc.url, problem_id=doc.problem_id)

    def _normalize(self, ranked: list[Hit], query: str) -> None:
        """把各路分数统一到 0-1，供三档阈值判定：

        similarity = max(余弦相似度, BM25 归一化分)，取两路置信度的**较强者**。
        排序由 RRF 负责（两路互补），门控只需回答"有没有任何一路 strongly 支持这条"：
        - 只取向量分会把"BM25 第 1 但余弦一般"的正确条目挡在 cap 外
          （实测反例：「01背包内层循环为什么必须倒序」的 01背包 条目被 0.816 的
          cap 误杀，而 零钱兑换 以 0.936 留下——语义近邻的词法错配）；
        - 域外查询两路都低，max 不会放宽"未收录"判定（见 calibrate 的域外分布）。
        BM25 归一化用 1 - exp(-bm25/scale) 单调映射，不用"除以查询理论上界"——
        长查询里大量未命中词会把分母撑大（实测反例见 docs/04）。
        """
        scale = self.settings.bm25_score_scale
        for hit in ranked:
            bm25_norm = round(1.0 - math.exp(-hit.bm25_score / max(scale, 1e-6)), 4)
            hit.similarity = max(hit.vector_score, bm25_norm)
            if hit.rerank_score is not None:
                hit.score = hit.rerank_score
            else:
                hit.score = hit.similarity

        # 重排分数归一化（cross-encoder 输出是无界 logits）
        scored = [h.score for h in ranked]
        if scored and (max(scored) > 1.0 or min(scored) < 0.0):
            lo, hi = min(scored), max(scored)
            span = (hi - lo) or 1.0
            for hit in ranked:
                hit.score = round((hit.score - lo) / span, 4)

    def _select(self, ranked: list[Hit], top_k: int) -> tuple[list[Hit], str, bool]:
        settings = self.settings
        if not ranked:
            return [], "none", False
        best = max(h.similarity for h in ranked)
        if best < settings.weak_threshold:
            return [], "none", False

        threshold = settings.weak_threshold if best < settings.strong_threshold else settings.strong_threshold
        cap = max(threshold, best - settings.rel_margin) if best >= settings.strong_threshold else threshold

        kept: list[Hit] = []
        prev: float | None = None
        for hit in sorted(ranked, key=lambda h: h.score, reverse=True):
            if hit.similarity < cap:
                continue
            if prev is not None and (prev - hit.score) > settings.gap_margin and len(kept) >= 2:
                break  # 分数断层：后面的属于"沾边"，截断
            kept.append(hit)
            prev = hit.score
            if len(kept) >= top_k:
                break

        level = "strong" if best >= settings.strong_threshold else "weak"
        if level == "weak":
            kept = kept[:2]

        quota_applied = False
        if (settings.type_boost and level == "strong" and len(kept) >= 3
                and all(h.source != "curated" for h in kept[:5])):
            for candidate in ranked:
                if candidate.source == "curated" and candidate.similarity >= best * settings.curated_quota_ratio:
                    if candidate not in kept:
                        kept.insert(2, candidate)
                        kept = kept[:top_k]
                        quota_applied = True
                    break
        return kept, level, quota_applied

    def keyword_search(self, text: str, top_k: int = 10) -> list[Hit]:
        """给"相似题推荐"用的轻量接口（纯词法，毫秒级）。"""
        if not self.ready:
            return []
        out = []
        for doc_id, score, terms in self.bm25.search(text, top_k) if self.bm25 else []:
            doc = self.docs.get(doc_id)
            if doc:
                out.append(Hit(id=doc.id, title=doc.title, content=doc.content,
                               difficulty=doc.difficulty, tags=doc.tags, domain=doc.domain,
                               source=doc.source, url=doc.url, bm25_score=score,
                               matched_terms=terms))
        return out

    def tokens_of(self, text: str) -> list[str]:
        return tokenize(text)


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def set_retriever(retriever: HybridRetriever | None) -> None:
    """测试注入用。"""
    global _retriever
    _retriever = retriever


# ---------------- 后台预热 ----------------
# 冷启动最慢的一段是 `import sentence_transformers`（本机实测 11.3s），
# 若放在 lifespan 里同步执行，healthz 要等 15s 才起来，前端登录页一直转圈。
# 改为：healthz 立即返回，重活在 daemon 线程里做；首个检索请求会经
# ensure_loaded() 的锁等待预热完成，因此不会拿到半加载的空索引。
_warmup_started = False
_warmup_lock = threading.Lock()
_warmup_done = threading.Event()


def warmup_status() -> dict:
    """非阻塞地报告预热进度，供就绪探针与运维查看。

    index_done 才是 readiness 的依据（readyz 等它）；reranker 预热失败不影响可用性。
    """
    retriever_ready = _retriever is not None and _retriever._load_event.is_set()
    return {"started": _warmup_started,
            "index_done": retriever_ready,
            "reranker_done": _warmup_done.is_set()}


def start_warmup() -> threading.Thread | None:
    """启动后台预热（幂等）。返回线程对象便于测试 join。"""
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return None
        _warmup_started = True

    def _run() -> None:
        started = time.perf_counter()
        try:
            get_retriever().ensure_loaded()
            log.info("后台预热完成：索引就绪，耗时 %.1fs", time.perf_counter() - started)
        except Exception as exc:  # 预热失败不致命，首个请求会再试一次
            log.warning("后台预热失败（%s），检索将按需重试", exc)
        finally:
            # 可选：一并预热重排模型，让首条对话不再等 3s 加载。
            settings = get_settings()
            if settings.enable_rerank and getattr(settings, "warmup_reranker", True):
                try:
                    from acm_coach.rag.rerank import get_reranker
                    get_reranker(settings)
                    log.info("后台预热完成：重排模型就绪，累计 %.1fs", time.perf_counter() - started)
                except Exception as exc:
                    log.warning("重排模型预热失败：%s", exc)
            _warmup_done.set()

    t = threading.Thread(target=_run, name="rag-warmup", daemon=True)
    t.start()
    return t


def doc_to_dict(doc: Document) -> dict:
    return asdict(doc)
