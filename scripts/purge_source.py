# -*- coding: utf-8 -*-
"""按来源清除语料（站方/权利人要求删除时的技术兜底）。

用法：
    python scripts/purge_source.py --source cf --dry-run
    python scripts/purge_source.py --source cf --yes

执行内容：
1. 统计该来源条目数并打印样本（dry-run 时到此为止）；
2. 硬删除 knowledge_items 中该来源的行（不是软删——合规删除要求数据真的不在）；
3. 删除对应向量（按 id 逐条 delete，Chroma 无 where-by-metadata 的批量删）；
4. 重写 docstore 与 BM25 索引；
5. 写 PURGE_SOURCE 审计（含删除条数快照），审计表 append-only 不可改。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acm_coach.config import get_settings
from acm_coach.db import init_db, session_scope
from acm_coach.db.models import KnowledgeItem
from acm_coach.db.repositories import AuditRepository, KnowledgeRepository
from acm_coach.rag.vector import VectorStore
from acm_coach.services.index_service import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="cf / atcoder / curated / manual")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    init_db(seed_admin=False)

    with session_scope() as db:
        rows = db.query(KnowledgeItem).filter(KnowledgeItem.source == args.source).all()
        ids = [r.id for r in rows]
        print(f"来源 {args.source}：{len(ids)} 条")
        for r in rows[:5]:
            print(f"  - {r.id} {r.title}")
        if args.dry_run:
            print("dry-run：未删除任何数据")
            return
        if not args.yes:
            print("需要 --yes 确认执行删除")
            return

        store = VectorStore(settings.chroma_dir, settings.embed_model,
                            enabled=settings.enable_vector)
        if store.available:
            store.delete_ids(ids)
            print(f"向量库已删除 {len(ids)} 条（当前 {store.count()} 条）")

        for r in rows:
            db.delete(r)
        AuditRepository(db).write(
            "purge_source", resource="knowledge", resource_id=args.source,
            detail={"source": args.source, "deleted": len(ids),
                    "sample": ids[:20], "vector_deleted": store.available},
        )
        print("结构化表已删除，审计已写入 PURGE_SOURCE")

    with session_scope() as db:
        remaining = KnowledgeRepository(db).all_for_index()
    stats = build_index(remaining, with_vector=False)
    print("索引已重建：", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
