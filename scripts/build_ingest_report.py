# -*- coding: utf-8 -*-
"""生成交付用数据统计报告（供论文与三份设计文档统一引用）。

输出：
- data/reports/ingest_report.json：按来源/领域/难度/标签的条数，语料漏斗
- data/reports/ingest_report.md：可直接贴进论文的表格
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "acm_coach.db"


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"数据库不存在：{DB}")
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "select source, domain, difficulty, tags, rating from knowledge_items where is_deleted=0"
    ).fetchall()

    by_source = Counter(r[0] for r in rows)
    by_domain = Counter(r[1] for r in rows)
    by_difficulty = Counter(r[2] for r in rows)
    tag_counter: Counter = Counter()
    for r in rows:
        for tag in (r[3] or "").split(","):
            tag = tag.strip()
            if tag:
                tag_counter[tag] += 1

    source_domain: dict[str, Counter] = {}
    for src, domain, _diff, _tags, _rating in rows:
        source_domain.setdefault(src, Counter())[domain] += 1

    runs = conn.execute(
        "select source, status, total, inserted, updated, skipped, started_at from crawl_runs order by id desc limit 5"
    ).fetchall()

    report = {
        "total": len(rows),
        "by_source": dict(by_source),
        "by_domain": dict(by_domain),
        "by_difficulty": dict(by_difficulty),
        "difficulty_ratio": {k: round(v / len(rows), 4) for k, v in by_difficulty.items()},
        "top_tags": tag_counter.most_common(20),
        "source_domain": {k: dict(v) for k, v in source_domain.items()},
        "crawl_runs": [
            {"source": r[0], "status": r[1], "total": r[2], "inserted": r[3],
             "updated": r[4], "skipped": r[5], "started_at": r[6]} for r in runs
        ],
    }

    out_dir = ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 语料入库统计", "",
        f"- 总条数：**{report['total']}**",
        f"- 来源：{report['by_source']}",
        f"- 难度分布：{report['by_difficulty']}（比例 {report['difficulty_ratio']}）", "",
        "## 领域分布", "", "| 领域 | 条数 |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(by_domain.items(), key=lambda kv: -kv[1])]
    lines += ["", "## 标签 TOP20", "", "| 标签 | 条数 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in tag_counter.most_common(20)]
    (out_dir / "ingest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"total": report["total"], "by_source": report["by_source"],
                      "by_difficulty": report["by_difficulty"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
