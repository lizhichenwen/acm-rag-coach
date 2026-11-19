# -*- coding: utf-8 -*-
"""生成 Codeforces 官方 tag 的全量实测分布（供论文与分类体系引用）。

口径：
- 数据源：/api/problemset.problems 全量返回（含未评级题目）
- 计数单位：题目数（同一题多个 tag 会分别计入各 tag）
- 附带每 tag 的 rating 五档分布，用于论证"哪些知识点爬取覆盖不足、必须靠精讲补"
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acm_coach.config import get_settings
from acm_coach.crawler.client import HttpClient
from acm_coach.crawler.taxonomy import cf_tier


def main() -> None:
    settings = get_settings()
    payload = HttpClient().get_json(f"{settings.cf_api_base}/problemset.problems")
    problems = payload["result"]["problems"]
    stats = payload["result"].get("problemStatistics", [])
    solved = {(s["contestId"], s["index"]): s.get("solvedCount", 0) for s in stats}

    tag_counter: Counter = Counter()
    tag_tiers: dict[str, Counter] = defaultdict(Counter)
    ratings = []
    for p in problems:
        rating = p.get("rating")
        if rating:
            ratings.append(rating)
        tier, _ = cf_tier(rating)
        for tag in p.get("tags", []):
            tag_counter[tag] += 1
            tag_tiers[tag][tier] += 1

    ratings.sort()
    total = len(problems)
    summary = {
        "source": "Codeforces /api/problemset.problems",
        "total_problems": total,
        "rated_problems": len(ratings),
        "unrated_problems": total - len(ratings),
        "rating_median": ratings[len(ratings) // 2] if ratings else None,
        "rating_p25": ratings[len(ratings) // 4] if ratings else None,
        "rating_p75": ratings[len(ratings) * 3 // 4] if ratings else None,
        "tags": {
            tag: {
                "count": count,
                "tiers": {k: tag_tiers[tag].get(k, 0) for k in ("入门", "简单", "中等", "较难", "困难", "未知")},
            }
            for tag, count in tag_counter.most_common()
        },
        "under_30_tags": [t for t, c in tag_counter.items() if c < 30],
    }
    out = settings.taxonomy_dir / "cf_tag_stats.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}")
    print(f"题目总数 {total}，有 rating {len(ratings)}，中位 {summary['rating_median']}")
    print("TOP10:", [(t, c) for t, c in tag_counter.most_common(10)])
    print("不足30条:", summary["under_30_tags"])


if __name__ == "__main__":
    main()
