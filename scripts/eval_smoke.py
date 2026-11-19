# -*- coding: utf-8 -*-
"""检索质量抽查：向量 / 词法 / 混合三种模式在同一批问题上的表现。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acm_coach.config import get_settings
from acm_coach.db import session_scope
from acm_coach.db.repositories import KnowledgeRepository
from acm_coach.rag.index import HybridRetriever
from acm_coach.services.index_service import build_index

QUERIES = [
    "01背包内层循环为什么必须倒序",
    "两数之和怎么做到 O(n)",
    "线段树区间加和区间求和怎么写",
    "并查集路径压缩是什么",
    "KMP 的 next 数组含义",
    "这道题 TLE 了怎么优化常数",
    "ICPC 罚时怎么算",
    "网络流最大流和最小割什么关系",
]

settings = get_settings()
with session_scope() as db:
    rows = KnowledgeRepository(db).all_for_index()

t0 = time.perf_counter()
stats = build_index(rows, with_vector=True, reset_vector=True)
print("index:", stats, f"总耗时 {time.perf_counter() - t0:.1f}s")

r = HybridRetriever()
r.ensure_loaded()
print("vector available:", bool(r.vector and r.vector.available),
      "vectors:", r.vector.count() if r.vector else 0)

for q in QUERIES:
    t = time.perf_counter()
    res = r.search(q)
    cost = (time.perf_counter() - t) * 1000
    print(f"\nQ: {q}  [{res.level}] {res.strategy} {cost:.0f}ms")
    for h in res.hits[:3]:
        print(f"   {h.score:.3f} sim={h.similarity:.3f} vec={h.vector_score:.3f} "
              f"bm25={h.bm25_score:.2f} | {h.title}")
