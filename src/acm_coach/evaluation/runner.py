# -*- coding: utf-8 -*-
"""离线检索评测：数据集构建 + 消融实验。

为什么不追求"人工标注 300 条"的完美评测集：
本科毕设的时间预算下，更可靠的做法是**规则化构造 + 可复核**：
- query 从真实语料条目派生（三种问法模板），分布可查；
- 相关性用"目标条目 + 同题标签 Jaccard≥0.5 的条目"自动标注，
  规则写在代码里，任何人都能复算；
- 指标实现复用 `evaluation/metrics.py`，与 v3 基线同一份代码。

消融配置：
  g1 = BM25 单路
  g2 = 向量单路
  g3 = 混合（RRF，无重排）
  g4 = 混合 + 重排
"""
from __future__ import annotations

import json
import random
import time

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger
from acm_coach.evaluation.metrics import aggregate
from acm_coach.rag.index import HybridRetriever
from acm_coach.rag.text import tokenize

log = get_logger("eval")

TEMPLATES = {
    "solution": ["「{title}」这道题的思路是什么", "{title} 怎么做，卡住了", "{title} 的正解方向"],
    "mistake": ["{title} 我总是 WA，为什么", "{title} 写挂了怎么排查", "{title} 常见错误有哪些"],
    "concept": ["{tags} 这类问题怎么分析", "{tags} 相关的题怎么入手"],
}

JACCARD_THRESHOLD = 0.5
MAX_RELEVANT_PER_QUERY = 8


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_dataset(max_queries: int = 150, seed: int = 42) -> list[dict]:
    from acm_coach.rag.docstore import load_docstore

    settings = get_settings()
    docs = load_docstore(settings.index_dir)
    if not docs:
        return []

    curated = [d for d in docs.values() if d.source in ("curated", "manual")]
    crawled = [d for d in docs.values() if d.source in ("cf", "atcoder")]
    rng = random.Random(seed)
    rng.shuffle(curated)
    rng.shuffle(crawled)

    # 一半精讲、一半爬取，避免评测集只覆盖"有题解"的条目
    picks = (curated[: max_queries // 2] + crawled[: max_queries - max_queries // 2])
    picks = [d for d in picks if d.title]

    title_tokens = {d.id: set(tokenize(f"{d.title} {d.tags}")) for d in docs.values()}
    dataset = []
    for doc in picks:
        kind = rng.choice(list(TEMPLATES))
        template = rng.choice(TEMPLATES[kind])
        tags = doc.tags.split(",")[0] if doc.tags else doc.domain
        query = template.format(title=doc.title, tags=tags)

        target_tokens = title_tokens[doc.id]
        qrels = {doc.id: 2}
        count = 0
        for other_id, tokens in title_tokens.items():
            if other_id == doc.id or count >= MAX_RELEVANT_PER_QUERY:
                continue
            if _jaccard(target_tokens, tokens) >= JACCARD_THRESHOLD:
                qrels[other_id] = 1
                count += 1
        dataset.append({"query": query, "qtype": kind, "qrels": qrels, "target": doc.id})

    rel_counts = sorted(len(q) for q in (d["qrels"] for d in dataset))
    if rel_counts:
        log.info("评测集：%d 条，每 query 相关条目 中位数=%d 最大=%d",
                 len(dataset), rel_counts[len(rel_counts) // 2], rel_counts[-1])
    return dataset


def _retriever_variant(base: HybridRetriever, mode: str, rerank: bool) -> HybridRetriever:
    """按消融配置裁剪检索器的两路能力（共享已加载的索引，避免重复加载）。"""
    r = HybridRetriever(index_path=base.index_path)
    r.docs, r.bm25, r.vector = base.docs, base.bm25, base.vector
    r._loaded = True
    r._load_event.set()  # 共享已加载索引，勿触发重复加载
    r.settings = base.settings.model_copy(deep=True)
    if mode == "bm25":
        r.vector = None
    elif mode == "vector":
        r.bm25 = None
    r.settings.enable_rerank = rerank
    # 消融不看阈值截断，只比排序质量：置零后不做三档裁剪
    r.settings.strong_threshold = 0.0
    r.settings.weak_threshold = 0.0
    r.settings.rel_margin = 1.0
    r.settings.gap_margin = 1.0
    # 排序深度放宽到 100，支撑 recall@100（生产默认 24 只够看头部精度）
    r.settings.rerank_top_n = 100
    return r


CONFIGS = {
    "g1": ("bm25", False),
    "g2": ("vector", False),
    "g3": ("hybrid", False),
    "g4": ("hybrid", True),
}


def _verify_type_boost(base: HybridRetriever) -> None:
    """检优的静默失效护栏：type_boost 挂在 source=="curated" 上。

    若配置开启加权、但候选池里 curated 条目数为 0（例如 source 枚举被改动），
    加权会静默失效——消融结果将失去可比性。verify 模式直接报错退出。
    """
    settings = get_settings()
    if not settings.type_boost:
        return
    curated_n = sum(1 for d in base.docs.values() if d.source == "curated")
    if curated_n == 0:
        raise SystemExit(
            "VERIFY FAILED: type_boost=true 但候选集中 source=='curated' 的条目数为 0，"
            "curated 加权静默失效。请检查 source 枚举是否被改动（应为 curated/cf/atcoder/manual）。"
        )
    log.info("verify 通过：type_boost 生效，候选池 curated 条目 %d 个", curated_n)


def run_evaluation(configs: list[str], max_queries: int = 150, rebuild: bool = False,
                   verify: bool = False) -> dict:
    base = HybridRetriever()
    base.ensure_loaded()
    if not base.ready:
        return {"error": "索引未就绪，请先执行 index"}
    if verify:
        _verify_type_boost(base)

    dataset = build_dataset(max_queries)
    if not dataset:
        return {"error": "语料为空"}

    report: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_size": len(base.docs),
        "vector_coverage": (base.vector.count() if base.vector else 0),
        "queries": len(dataset),
        "semantics": {
            "jaccard_threshold": JACCARD_THRESHOLD,
            "rel_scale": "2=目标条目, 1=同题标签相似, 0=未标注",
            "gate": "off（消融只比排序，不做三档截断）",
        },
        "configs": {},
    }

    for name in configs:
        if name not in CONFIGS:
            continue
        mode, rerank = CONFIGS[name]
        retriever = _retriever_variant(base, mode, rerank)
        records = []
        for item in dataset:
            started = time.perf_counter()
            result = retriever.search(item["query"], top_k=100)
            latency = (time.perf_counter() - started) * 1000
            records.append({
                "ranked": [h.id for h in result.hits],
                "qrels": item["qrels"],
                "latency_ms": latency,
                "level": result.level,
                "qtype": item["qtype"],
            })
        report["configs"][name] = aggregate(records)
        log.info("评测完成 %s: %s", name, report["configs"][name])

    out_dir = get_settings().data_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
