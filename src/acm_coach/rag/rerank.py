# -*- coding: utf-8 -*-
"""重排层：cross-encoder 精排，模型缺席时退化为"语义+词面"融合打分。

为什么重排值得多花一次计算：
向量召回是双塔（query 与 doc 各自编码），只能粗筛；cross-encoder 把
(query, doc) 拼在一起过一遍，能识别"问的是内层为什么倒序"与"倒序写法"之间的细粒度匹配。
代价是 O(n) 次前向，所以只对 RRF 头部 rerank_top_n 条做。

跨域降级：bge-reranker-base 未缓存且未允许联网时，用
  0.55×语义相似度 + 0.30×词面覆盖率 + 0.15×词法分数
的线性融合替代，保证任何环境都能出结果。
"""
from __future__ import annotations

import os
from pathlib import Path

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger
from acm_coach.rag.text import tokenize

log = get_logger("rag.rerank")

_MODEL = None
_MODEL_TRIED = False


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


def get_reranker():
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    settings = get_settings()
    if not settings.enable_rerank:
        return None
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


def lexical_coverage(query: str, content: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    d = set(tokenize(content))
    return len(q & d) / len(q)


def rerank(query: str, hits: list) -> list:
    if not hits:
        return hits

    model = get_reranker()
    if model is not None:
        try:
            pairs = [(query, h.content[:1200]) for h in hits]
            scores = model.predict(pairs, show_progress_bar=False)
            for hit, raw in zip(hits, scores):
                # logits -> 0-1，sigmoid 更稳，避免 min-max 被单条极值拉伸
                hit.rerank_score = round(1.0 / (1.0 + pow(2.718281828, -float(raw))), 4)
            return sorted(hits, key=lambda h: h.rerank_score, reverse=True)
        except Exception as exc:
            log.warning("重排推理失败，回退融合打分：%s", exc)

    bm25_ceiling = max((h.bm25_score for h in hits), default=1.0) or 1.0
    for hit in hits:
        semantic = hit.vector_score if hit.vector_score > 0 else min(1.0, hit.bm25_score / bm25_ceiling)
        overlap = lexical_coverage(query, hit.content)
        lexical = min(1.0, hit.bm25_score / bm25_ceiling)
        hit.rerank_score = round(0.55 * semantic + 0.30 * overlap + 0.15 * lexical, 4)
    return sorted(hits, key=lambda h: h.rerank_score, reverse=True)
