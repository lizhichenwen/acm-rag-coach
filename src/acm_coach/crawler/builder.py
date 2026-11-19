# -*- coding: utf-8 -*-
"""爬取条目的正文合成。

核心约束：**只写元数据能支撑的话**。
爬取数据只有 标题/难度/标签/通过人数，没有题面、没有题解，因此正文只允许描述
"这题考什么、属于哪个难度档、该拿什么题打底"，不允许出现具体复杂度、具体数据结构
选择或代码片段——那些属于题解职责，由 curated 语料承担。

切片规则（与 docs/03 一致）：≤600 字符整条入库；601-1200 按【】段落切两块；
再长则父子结构（父块存摘要，子块存明细）。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from acm_coach.config import get_settings
from acm_coach.crawler.sources import RawProblem
from acm_coach.crawler.taxonomy import ATCODER_SUFFIX_HINT, Taxonomy, atcoder_tier, cf_tier

TIER_ADVICE = {
    "入门": "适合作为该知识点的第一道练习题：先把暴力写法跑对，再考虑优化。",
    "简单": "适合巩固基本实现：重点检查边界条件与常数因子，确保复杂度达标。",
    "中等": "训练建模能力：先把题意翻译成标准模型，再选工具，不要一上来就写代码。",
    "较难": "需要在标准模型上做变形，或组合两个知识点；建议先复习同标签的中档题。",
    "困难": "适合专项突破：建议先做同标签的中档题打底，再回来看这道题的切入点。",
    "未知": "平台未提供难度标注，建议按通过人数估计其相对难度。",
}


def entry_id(p: RawProblem) -> str:
    prefix = "CF" if p.platform == "codeforces" else "AT"
    return f"{prefix}-{p.problem_id}"


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_content(p: RawProblem, taxonomy: Taxonomy, similar: list[RawProblem]) -> tuple[str, dict]:
    if p.platform == "codeforces":
        tier5, display = cf_tier(p.rating)
        domain, points = taxonomy.main_domain(p.tags)
        zh_tags = taxonomy.zh_tags(p.tags)[:6]
        hints = taxonomy.hints(p.tags, limit=2)
        difficulty_line = f"{p.rating}（{tier5}）" if p.rating else "未标注"
        platform_line = "Codeforces"
    else:
        tier5, display = atcoder_tier(p.rating)
        domain, points = _atcoder_domain(p)
        zh_tags = [f"AtCoder {p.contest.split('_')[0].upper() if p.contest else ''}".strip()]
        hints = [ATCODER_SUFFIX_HINT.get(p.index.lower(), "")] if p.index else []
        difficulty_line = f"{p.rating}（{tier5}）" if p.rating else "未知（平台未提供）"
        platform_line = "AtCoder"

    lines = [
        f"题目：{p.title}",
        f"平台：{platform_line}｜编号：{p.problem_id}｜难度：{difficulty_line}",
        f"标签：{'、'.join(t for t in zh_tags if t) or '未标注'}",
    ]
    if hints:
        lines.append(f"考点拆解：{' '.join(h for h in hints if h)}")
    if points:
        lines.append(f"关联知识点：{'、'.join(points)}")
    lines.append(f"训练定位：{TIER_ADVICE[tier5]}")
    if p.solved:
        lines.append(f"通过人数：约 {p.solved} 人（可作难度参照）")
    if similar:
        names = "；".join(f"{s.problem_id} {s.title}" for s in similar[:2])
        lines.append(f"同类练习：{names}")
    lines.append("说明：本条为平台元数据加工的考点提示，不含题面与题解，具体解法请结合精讲条目。")
    lines.append(f"原题链接：{p.url}")

    content = "\n".join(lines)
    meta = {
        "domain": domain, "difficulty": display, "tier5": tier5,
        "tags": ",".join(zh_tags[:6]), "rating": p.rating,
    }
    return content, meta


def _atcoder_domain(p: RawProblem) -> tuple[str, list[str]]:
    """AtCoder 无标签体系：按比赛类型与题号推断领域，全部标注为推断值。"""
    contest = (p.contest or "").lower()
    if contest.startswith("abc"):
        return "基础与实现", ["模拟与实现"]
    if contest.startswith("arc"):
        return "算法思想", ["贪心", "构造策略"]
    if contest.startswith("agc"):
        return "算法思想", ["构造策略"]
    return "未分类", []


def build_records(problems: list[RawProblem], taxonomy: Taxonomy,
                  crawl_batch: str = "") -> list[dict]:
    """把原始题目转成入库契约；同域近难度题互相作为"同类练习"。

    ``crawl_batch`` 由 CLI 传入，与 crawl_runs 一同写入 meta_json 溯源字段。
    """
    from acm_coach.services.knowledge_service import quality_gate

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 先算主域，便于配对同类题
    by_domain: dict[str, list[RawProblem]] = {}
    for p in problems:
        domain = taxonomy.main_domain(p.tags)[0] if p.platform == "codeforces" else _atcoder_domain(p)[0]
        by_domain.setdefault(domain, []).append(p)
    for pool in by_domain.values():
        pool.sort(key=lambda x: (x.rating or 0))

    records: list[dict] = []
    for p in problems:
        domain = taxonomy.main_domain(p.tags)[0] if p.platform == "codeforces" else _atcoder_domain(p)[0]
        pool = by_domain.get(domain, [])
        similar: list[RawProblem] = []
        if pool and p.rating:
            neighbors = [x for x in pool if x.problem_id != p.problem_id and x.rating]
            neighbors.sort(key=lambda x: abs((x.rating or 0) - p.rating))
            similar = neighbors[:2]

        content, meta = build_content(p, taxonomy, similar)
        item = {
            "title": p.title[:200], "content": content, "tags": meta["tags"],
            "difficulty": meta["difficulty"], "domain": meta["domain"],
            "rating": p.rating, "source": "cf" if p.platform == "codeforces" else "atcoder",
            "url": p.url,
        }
        if not quality_gate(item, content):
            continue
        records.append({
            "id": entry_id(p),
            "title": item["title"],
            "difficulty": meta["difficulty"],
            "rating": p.rating,
            "tags": meta["tags"],
            "domain": meta["domain"],
            "source": item["source"],
            "url": p.url,
            "content": content,
            "content_hash": content_hash(content),
            "chunk_index": 0,
            "quality_score": 1.0,
            # 合规控制字段与溯源契约：爬取条目统一"内部已发布/公开可见/L1 低敏"
            "status": "published",
            "visibility": "public",
            "sensitivity": "L1",
            "retention_days": 0,
            "meta_json": json.dumps({
                "entry_type": "hint",
                "entry_type_label": "考点提示",
                "content_origin": "metadata",
                "processed_by": "crawler.builder.build_records/v4",
                "owner_type": "platform",
                "fetched_at": fetched_at,
                "crawl_batch": crawl_batch,
                "license": "Codeforces-Terms" if p.platform == "codeforces" else "AtCoder-Terms",
                "mirror": "official-api" if p.platform == "codeforces" else "kenkoooo",
                "source_url": p.url or "",
            }, ensure_ascii=False),
        })
    return records


def crawl_summary(records: list[dict]) -> dict:
    from collections import Counter

    return {
        "total": len(records),
        "by_source": dict(Counter(r["source"] for r in records)),
        "by_domain": dict(Counter(r["domain"] for r in records)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in records)),
    }
