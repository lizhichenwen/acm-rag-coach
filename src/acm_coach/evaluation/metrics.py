# -*- coding: utf-8 -*-
"""检索评测指标（单一事实来源）。

v3 的评测脚本与 v4 必须调用同一份实现，否则"提升 X%"来自指标口径差异而非算法。
所有函数只依赖 `(ranked_ids, qrels, k)`，不依赖任何检索实现。

NDCG 口径（论文需写明，否则不可复现）：
- gain = 2^rel - 1，rel ∈ {0,1,2}
- discount = 1 / log2(i + 1)，i 从 1 开始
- IDCG 取该 query 全部标注相关项的理想排序，结果不足 k 条不补零
- 无相关项的 query 不参与 NDCG 平均（分母是"有正例的 query 数"）
"""
from __future__ import annotations

import math
from statistics import mean


def hit_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    return 1.0 if relevant & set(ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float | None:
    """召回率：top-k 命中的相关条目数 / 该 query 全部相关条目数。

    无正例的 query 返回 None，不参与平均（与 NDCG 口径一致）。
    """
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return None
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    for i, doc_id in enumerate(ranked[:k], 1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float | None:
    gains = [qrels.get(doc_id, 0) for doc_id in ranked[:k]]
    dcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(gains, 1))
    ideal = sorted((r for r in qrels.values() if r > 0), reverse=True)[:k]
    if not ideal:
        return None
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100 * len(ordered))) - 1))
    return round(ordered[idx], 2)


def aggregate(records: list[dict], k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """records 每项：{ranked, qrels, latency_ms, level, qtype}"""
    out: dict = {"queries": len(records)}
    if not records:
        return out
    for k in k_values:
        out[f"hit@{k}"] = round(mean(hit_at_k(r["ranked"], r["qrels"], k) for r in records), 4)
    out["mrr@10"] = round(mean(mrr_at_k(r["ranked"], r["qrels"], 10) for r in records), 4)
    recalls = [recall_at_k(r["ranked"], r["qrels"], 100) for r in records]
    valid_recalls = [v for v in recalls if v is not None]
    out["recall@100"] = round(mean(valid_recalls), 4) if valid_recalls else None
    ndcgs = [ndcg_at_k(r["ranked"], r["qrels"], 10) for r in records]
    valid = [v for v in ndcgs if v is not None]
    out["ndcg@10"] = round(mean(valid), 4) if valid else None

    latencies = [r["latency_ms"] for r in records]
    out["latency_p50_ms"] = percentile(latencies, 50)
    out["latency_p95_ms"] = percentile(latencies, 95)

    levels = {"strong": 0, "weak": 0, "none": 0}
    for r in records:
        levels[r.get("level", "none")] = levels.get(r.get("level", "none"), 0) + 1
    out["empty_rate"] = round(levels["none"] / len(records), 4)
    out["level_distribution"] = levels

    by_type: dict[str, dict] = {}
    for r in records:
        by_type.setdefault(r.get("qtype", "unknown"), []).append(r)
    out["by_type"] = {
        qtype: {
            "n": len(items),
            "hit@5": round(mean(hit_at_k(i["ranked"], i["qrels"], 5) for i in items), 4),
            "mrr@10": round(mean(mrr_at_k(i["ranked"], i["qrels"], 10) for i in items), 4),
        }
        for qtype, items in by_type.items()
    }
    return out
