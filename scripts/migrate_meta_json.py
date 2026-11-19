# -*- coding: utf-8 -*-
"""把历史 knowledge_items.meta_json 回填为 docs/05 §6.7 的溯源契约（幂等）。

背景：8,558 条语料是在 meta_json 列加入之前入库的，实测全为 NULL；
新数据由 knowledge_service._meta_contract / crawler.builder 按契约写入。
本脚本负责历史数据，使答辩演示时"随便抽一条都能对上契约"。

契约 8 键：entry_type / content_origin / processed_by / owner_type /
fetched_at / crawl_batch / license / mirror（另附 entry_type_label、source_url 备查）。
mirror 是通道维度：atcoder→kenkoooo、cf→official-api、人工自撰→n/a(self_written)。

注意：source 列保持 curated/cf/atcoder 不动——它是检索 type_boost 的承重契约，
"人工精讲 vs 平台元数据"的语义由 content_origin / owner_type 承载。
历史数据无法追溯真实批次，crawl_batch 如实标 "legacy"，不编造。
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "acm_coach.db"
CONTRACT_KEYS = {
    "entry_type", "content_origin", "processed_by", "owner_type",
    "fetched_at", "crawl_batch", "license", "mirror",
}
MIRROR_VALUES = {"official-api", "kenkoooo", "n/a(self_written)", "unknown"}


def curated_forms() -> dict[str, str]:
    """从原始语料重建 curated 形态（题解/竞赛元知识/知识点卡）。"""
    forms: dict[str, str] = {}
    for path, card in ((ROOT / "data/curated/acm_knowledge.txt", False),
                       (ROOT / "data/curated/knowledge_cards.txt", True)):
        if not path.exists():
            continue
        for block in path.read_text(encoding="utf-8").split("\n---\n"):
            tags = title = ""
            for line in block.strip().splitlines():
                if line.startswith("题目："):
                    title = line[3:].strip()
                elif line.startswith("标签："):
                    tags = line[3:].strip()
            if not title:
                continue
            if card:
                forms[title] = "知识点卡"
            elif any(k in tags for k in ("竞赛策略", "赛制", "训练方法")):
                forms[title] = "竞赛元知识"
            else:
                forms[title] = "题解"
    return forms


ENUM = {"题解": "solution", "知识点卡": "concept",
        "竞赛元知识": "strategy", "考点提示": "hint"}


def main() -> None:
    forms = curated_forms()
    con = sqlite3.connect(DB)
    rows = con.execute(
        "select id, title, source, url, created_at, meta_json from knowledge_items"
    ).fetchall()
    stats: Counter[str] = Counter()
    unknown_curated: list[str] = []

    license_values = {"internal-only", "Codeforces-Terms", "AtCoder-Terms",
                      "CC-BY-NC-SA-4.0"}
    for item_id, title, source, url, created_at, meta_json in rows:
        try:
            old = json.loads(meta_json) if meta_json else {}
        except (ValueError, TypeError):
            old = {}
        # 幂等：新写入路径的数据（processed_by 以 /v4 结尾且枚举合法）不再覆盖，
        # 避免重复运行把 to_records/builder 的溯源标记抹成 backfill。
        well_formed = (CONTRACT_KEYS <= set(old)
                       and old.get("license") in license_values
                       and old.get("content_origin") in {"self_written", "metadata"}
                       and old.get("owner_type") in {"user", "platform"}
                       and old.get("entry_type") in {"solution", "concept", "strategy", "hint"}
                       and old.get("mirror") in MIRROR_VALUES
                       and old.get("fetched_at") not in {"", None}
                       and old.get("crawl_batch") not in {"", None})
        if well_formed and (
                old.get("processed_by", "").endswith("/v4")
                or old.get("processed_by") == "migrate_meta_json/v4-backfill"):
            stats["unchanged"] += 1
            stats[f"entry_type:{old['entry_type']}"] += 1
            continue
        curated = source in ("curated", "manual")
        label = old.get("entry_type_label") or (
            forms.get(title) or ("题解" if curated else "考点提示")
        )
        if curated and title not in forms and label == "题解":
            unknown_curated.append(title)
        # 取值严格沿用 docs/05 §6.7 枚举；历史数据无抓取时间戳，按契约
        # "不得用 created_at 冒充 fetched_at"，如实标 unknown 并附说明。
        meta = {
            **old,
            "entry_type": ENUM.get(label, old.get("entry_type", "hint")),
            "entry_type_label": label,
            "content_origin": "self_written" if curated else "metadata",
            "processed_by": "migrate_meta_json/v4-backfill",
            "owner_type": "user" if curated else "platform",
            # 不保留历史旧值：首轮回填曾用 created_at 冒充、写过中文 license，
            # 这里一律归一，避免"看起来有值但口径是错的"。
            "fetched_at": "n/a(self_written)" if curated else "unknown",
            "crawl_batch": "n/a(self_written)" if curated else "legacy",
            "license": ("internal-only" if curated
                        else "Codeforces-Terms" if source == "cf" else "AtCoder-Terms"),
            "mirror": ("n/a(self_written)" if curated
                       else "official-api" if source == "cf"
                       else "kenkoooo" if source == "atcoder" else "unknown"),
            "source_url": old.get("source_url") or (url or ""),
            "fetch_time_note": "历史入库未记录抓取时间（人工自撰条目本无抓取语义）；按 docs/05 §6.7 不得用 created_at 冒充",
        }
        meta.pop("license_note", None)  # 契约键名统一为 license
        # 合规控制列也按来源归一：历史行全是迁移默认值 internal/L2，
        # 与"人工精讲内部 L2 / 平台元数据公开 L1"的新写入契约不一致。
        visibility, sensitivity = ("internal", "L2") if curated else ("public", "L1")
        con.execute(
            "update knowledge_items set meta_json = ?, status = 'published', "
            "visibility = ?, sensitivity = ?, retention_days = 0 where id = ?",
            (json.dumps(meta, ensure_ascii=False), visibility, sensitivity, item_id),
        )
        stats["updated"] += 1
        stats[f"entry_type:{meta['entry_type']}"] += 1
    con.commit()

    # 校验：所有行非 NULL 且契约键齐全
    bad = 0
    for (m,) in con.execute("select meta_json from knowledge_items"):
        try:
            missing = CONTRACT_KEYS - set(json.loads(m or "{}"))
        except ValueError:
            missing = CONTRACT_KEYS
        if missing:
            bad += 1
    con.close()
    print(json.dumps({"total": len(rows), "forms_in_curated_files": len(forms),
                      "unknown_curated_titles": unknown_curated[:10],
                      "stats": dict(stats), "rows_missing_contract_keys": bad},
                     ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(f"FAILED: {bad} rows still missing contract keys")


if __name__ == "__main__":
    main()
