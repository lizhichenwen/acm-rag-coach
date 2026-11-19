# -*- coding: utf-8 -*-
"""冒烟测试：语料 → 入库 → 建索引 → 检索，验证主链路是否通。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acm_coach.config import get_settings
from acm_coach.db import init_db, session_scope
from acm_coach.db.repositories import KnowledgeRepository
from acm_coach.services import knowledge_service
from acm_coach.services.index_service import build_index
from acm_coach.rag.index import set_retriever

settings = get_settings()
print("data_dir:", settings.data_dir)
init_db(seed_admin=False)

items = knowledge_service.load_directory(settings.knowledge_paths)
records = knowledge_service.to_records(items)
print("parsed:", len(items), "records:", len(records))

with session_scope() as db:
    ins, upd = KnowledgeRepository(db).upsert_many(records)
    print("upsert inserted/updated:", ins, upd)
    rows = KnowledgeRepository(db).all_for_index()

stats = build_index(rows, with_vector=False)
print("index:", stats)

set_retriever(None)
from acm_coach.rag.index import get_retriever

r = get_retriever()
print("ready:", r.ready, "docs:", r.doc_count)
res = r.search("01背包内层循环为什么必须倒序")
print("level:", res.level, "recall:", res.recall, "cost:", res.cost_ms, "ms")
for h in res.hits:
    print(f"  {h.score:.3f} sim={h.similarity:.3f} {h.title} | {h.tags}")
