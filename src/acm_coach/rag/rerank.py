# -*- coding: utf-8 -*-
"""重排层：cross-encoder 精排，模型缺席时退化为"语义+词面"融合打分。

为什么重排值得多花一次计算：
向量召回是双塔（query 与 doc 各自编码），只能粗筛；cross-encoder 把
(query, doc) 拼在一起过一遍，能识别"问的是内层为什么倒序"与"倒序写法"之间的细粒度匹配。
代价是 O(n) 次前向，所以只对 RRF 头部 rerank_top_n 条做。

三种后端（由调用方 settings.enable_rerank 决定，见 rerank_backend()）：
  cross-encoder   —— 模型已加载，真实精排（G4 的目标状态）；
  fusion-fallback —— 开启了重排但模型未缓存/加载失败，退化为
                     0.55×语义 + 0.30×词面 + 0.15×词法的线性融合；
  off             —— 未开启重排，原样返回，保留 RRF/词法/向量通道顺序（G1/G2/G3）。

消融隔离（2026-11-21 修复）：
本模块必须用**调用方传入的 settings**，不能用 get_settings() 的全局单例。
评测 runner 会为每个消融分组深拷贝一份 settings（G1/G2/G3 关重排、G4 开重排），
若这里读全局配置，模型一旦缓存，G1/G2/G3 也会被静默重排，消融对比失真。
"""
from __future__ import annotations

import math
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from acm_coach.config import Settings, get_settings
from acm_coach.core.logging import get_logger
from acm_coach.rag.text import tokenize

log = get_logger("rag.rerank")

_MODEL = None
_MODEL_TRIED = False
_MODEL_KEY: tuple[str, bool] | None = None
_LAST_BACKEND = "off"

# (query, 候选 id 序列, 模型名, max_length) -> {id: 归一化分}
# LRU：命中时零 cross-encoder 推理，专治答辩反复问同一题、热点问题。
_CACHE: OrderedDict[tuple, dict[str, float]] = OrderedDict()
_CACHE_LOCK = Lock()
_CACHE_HIT = 0
_CACHE_MISS = 0


def _model_cached(model_name: str) -> bool:
    if not model_name:
        return False
    if Path(model_name).exists():
        return True
    root = Path(os.getenv("HF_HOME") or os.getenv("SENTENCE_TRANSFORMERS_HOME")
                or (Path.home() / ".cache" / "huggingface"))
    snapshots = root / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    try:
        return snapshots.exists() and any(snapshots.iterdir())
    except OSError:
        return False


def get_reranker(settings: Settings | None = None):
    """按调用方的配置加载 cross-encoder；未开启重排时直接返回 None。

    _MODEL_TRIED 必须在 enable_rerank 判定**之后**才置位：否则同一进程里
    先跑 G1（关重排）会把"已尝试"锁死，后面的 G4（开重排）永远拿不到模型。
    """
    global _MODEL, _MODEL_TRIED, _MODEL_KEY
    settings = settings or get_settings()
    if not settings.enable_rerank:
        return None
    key = (settings.rerank_model, settings.rerank_download)
    if _MODEL_TRIED:
        return _MODEL if _MODEL_KEY == key else None
    _MODEL_TRIED = True
    _MODEL_KEY = key
    if not _model_cached(settings.rerank_model) and not settings.rerank_download:
        log.info("重排模型 %s 未缓存，使用融合打分", settings.rerank_model)
        return None
    try:
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder(settings.rerank_model,
                              local_files_only=not settings.rerank_download)
        log.info("重排模型加载成功：%s", settings.rerank_model)
    except Exception as exc:
        log.warning("重排模型加载失败（%s），使用融合打分", exc)
        _MODEL = None
    return _MODEL


def rerank_backend() -> str:
    """最近一次 rerank() 实际采用的后端：cross-encoder / fusion-fallback / off。"""
    return _LAST_BACKEND


def cache_stats() -> dict:
    """缓存命中统计，供 /ops 与文档核对优化是否生效。"""
    total = _CACHE_HIT + _CACHE_MISS
    return {"size": len(_CACHE), "hits": _CACHE_HIT, "misses": _CACHE_MISS,
            "hit_rate": round(_CACHE_HIT / total, 4) if total else 0.0}


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def lexical_coverage(query: str, content: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    d = set(tokenize(content))
    return len(q & d) / len(q)


def rerank(query: str, hits: list, settings: Settings | None = None) -> list:
    """按调用方配置重排；关闭重排时原样返回，保持上游通道顺序。"""
    global _LAST_BACKEND
    if not hits:
        return hits
    settings = settings or get_settings()

    if not settings.enable_rerank:
        _LAST_BACKEND = "off"
        for hit in hits:
            hit.rerank_score = None
        return hits

    global _CACHE_HIT, _CACHE_MISS
    model = get_reranker(settings)
    if model is not None:
        # 缓存键必须包含候选 id 序列——同一 query 在不同召回状态下候选可能不同。
        max_len = getattr(settings, "rerank_max_length", 192)
        cache_on = getattr(settings, "rerank_cache_size", 256) > 0
        key = (query, tuple(h.id for h in hits), settings.rerank_model, max_len)
        cached: dict[str, float] | None = None
        if cache_on:
            with _CACHE_LOCK:
                if key in _CACHE:
                    cached = _CACHE.pop(key)  # LRU：命中后移到队尾
                    _CACHE[key] = cached
                    _CACHE_HIT += 1
        if cached is not None:
            for hit in hits:
                hit.rerank_score = cached.get(hit.id)
            _LAST_BACKEND = "cross-encoder"
            return sorted(hits, key=lambda h: (h.rerank_score or 0.0), reverse=True)
        if cache_on:
            _CACHE_MISS += 1
        try:
            pairs = [(query, h.content[:settings.rerank_doc_chars]) for h in hits]
            scores = model.predict(pairs, max_length=max_len,
                                   batch_size=getattr(settings, "rerank_batch_size", 16),
                                   show_progress_bar=False)
            score_map: dict[str, float] = {}
            for hit, raw in zip(hits, scores):
                # logits -> 0-1，sigmoid 更稳，避免 min-max 被单条极值拉伸
                hit.rerank_score = round(1.0 / (1.0 + math.exp(-float(raw))), 4)
                score_map[hit.id] = hit.rerank_score
            if cache_on:
                with _CACHE_LOCK:
                    _CACHE[key] = score_map
                    cap = settings.rerank_cache_size
                    while len(_CACHE) > cap:
                        _CACHE.popitem(last=False)
            _LAST_BACKEND = "cross-encoder"
            return sorted(hits, key=lambda h: h.rerank_score, reverse=True)
        except Exception as exc:
            log.warning("重排推理失败，回退融合打分：%s", exc)

    _LAST_BACKEND = "fusion-fallback"
    bm25_ceiling = max((h.bm25_score for h in hits), default=1.0) or 1.0
    for hit in hits:
        semantic = hit.vector_score if hit.vector_score > 0 else min(1.0, hit.bm25_score / bm25_ceiling)
        overlap = lexical_coverage(query, hit.content)
        lexical = min(1.0, hit.bm25_score / bm25_ceiling)
        hit.rerank_score = round(0.55 * semantic + 0.30 * overlap + 0.15 * lexical, 4)
    return sorted(hits, key=lambda h: h.rerank_score, reverse=True)
