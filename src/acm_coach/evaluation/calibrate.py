# -*- coding: utf-8 -*-
"""阈值标定：用真实语料统计"最优相似度"分布，据此设定三档阈值。

为什么不能拍脑袋：
- 余弦相似度的绝对值依赖模型与语料，bge-small-zh 上"相关"通常 0.55-0.75，
  换 bge-m3 会整体上移；
- 库从 77 条涨到 8000+ 条后，最近邻天然变远，同一阈值会从"过严"变"过松"。

所以阈值必须由**分布分位数**决定，并把"库规模 + 模型 + 度量"写成标定指纹，
换了任何一个都必须重新标定。
"""
from __future__ import annotations

import json
import random
import statistics
import time

from acm_coach.config import get_settings
from acm_coach.rag.index import HybridRetriever, get_retriever

# 域外问题：用来估计"不该命中时"的相似度上界
OUT_OF_SCOPE = [
    "红烧肉怎么做好吃", "明天会下雨吗", "怎么准备英语六级", "线性代数的秩是什么",
    "简历怎么写", "今天天气如何", "推荐一部科幻电影", "感冒了吃什么药",
]


def calibrate(sample_size: int = 120) -> dict:
    retriever = get_retriever()
    retriever.ensure_loaded()
    if not retriever.ready:
        return {"error": "索引未就绪"}
    # 测量必须在"门关闭"的裸分数上进行：若沿用当前阈值，域外查询会先被门截成
    # 空结果（best_similarity=0），域外分布被污染，标定陷入循环论证。
    retriever.settings = retriever.settings.model_copy(deep=True)
    retriever.settings.strong_threshold = 0.0
    retriever.settings.weak_threshold = 0.0
    retriever.settings.rel_margin = 1.0
    retriever.settings.gap_margin = 1.0

    curated = [d for d in retriever.docs.values() if d.source in ("curated", "manual")]
    rng = random.Random(7)
    rng.shuffle(curated)
    in_scope = [f"{d.title} 怎么做" for d in curated[:sample_size]]

    in_scores = []
    for q in in_scope:
        result = retriever.search(q, top_k=3)
        in_scores.append(round(result.best_similarity, 4))

    out_scores = []
    for q in OUT_OF_SCOPE:
        result = retriever.search(q, top_k=3)
        out_scores.append(round(result.best_similarity, 4))

    def pct(values: list[float], p: float) -> float:
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round(p / 100 * len(ordered))) - 1))
        return round(ordered[idx], 4)

    strong = pct(in_scores, 25)      # 相关问题的下四分位：低于它说明把握不大
    weak = pct(out_scores, 90) + 0.02  # 域外问题的上沿：高于它才敢说"相关"
    if weak >= strong:               # 分布重叠时的保守取法
        weak = round(strong - 0.05, 4)

    report = {
        "corpus_size": len(retriever.docs),
        "vector_coverage": retriever.vector.count() if retriever.vector else 0,
        "embed_model": get_settings().embed_model,
        "metric": "cosine" if retriever.vector and retriever.vector.available else "bm25-exp",
        "sampled_at": time.strftime("%Y-%m-%d %H:%M"),
        "in_scope": {
            "n": len(in_scores), "p10": pct(in_scores, 10), "p25": pct(in_scores, 25),
            "median": round(statistics.median(in_scores), 4),
            "p75": pct(in_scores, 75), "max": max(in_scores),
        },
        "out_of_scope": {
            "n": len(out_scores), "median": round(statistics.median(out_scores), 4),
            "p90": pct(out_scores, 90), "max": max(out_scores),
        },
        "recommended": {"strong_threshold": strong, "weak_threshold": weak},
        "fingerprint": f"n={len(retriever.docs)};model={get_settings().embed_model};metric=cosine",
    }
    path = get_settings().data_dir / "reports" / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
