# -*- coding: utf-8 -*-
"""核心纯函数单测：分词、BM25、查询构造、指标、门禁。

原则：不依赖网络、不依赖向量模型、不依赖数据库——毫秒级可跑。
"""
from __future__ import annotations

import math

import pytest

from acm_coach.evaluation.metrics import aggregate, hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from acm_coach.rag.bm25 import BM25Index
from acm_coach.rag.query import build_query, extract_code_signals, is_followup, split_code
from acm_coach.rag.text import expand_query, tokenize
from acm_coach.services.knowledge_service import compose_curated_content, to_records


class TestTokenizer:
    def test_chinese_tokens(self):
        tokens = tokenize("01背包的内层循环为什么倒序")
        assert tokens
        assert any("背包" in t or "循环" in t for t in tokens)

    def test_stopwords_removed(self):
        assert "的" not in tokenize("这个问题的解法")

    def test_synonym_expansion(self):
        expanded = expand_query("这题 TLE 了")
        assert "超时" in expanded

    def test_empty(self):
        assert tokenize("") == []


class TestBM25:
    def test_rank_relevant_first(self):
        docs = {
            "a": "01背包 动态规划 内层循环 倒序 每种物品只用一次",
            "b": "并查集 路径压缩 连通块",
            "c": "线段树 区间求和 懒标记",
        }
        index = BM25Index.build(docs)
        hits = index.search("01背包 内层循环 倒序", top_k=3)
        assert hits[0][0] == "a"

    def test_persistence_roundtrip(self, tmp_path):
        index = BM25Index.build({"a": "二分查找", "b": "贪心"})
        path = tmp_path / "bm25.pkl"
        index.save(path)
        loaded = BM25Index.load(path)
        assert loaded is not None
        assert loaded.size == 2
        assert loaded.search("二分", 1)[0][0] == "a"

    def test_empty_query(self):
        index = BM25Index.build({"a": "x"})
        assert index.search("", 5) == []


class TestQueryBuilder:
    def test_code_stripped(self):
        prompt = "这段代码为什么 WA\n```cpp\nint main(){return 0;}\n```"
        info = build_query(prompt)
        assert "return 0" not in info["query"]
        assert info["is_code"] is False

    def test_followup_uses_anchor(self):
        history = [{"role": "user", "content": "01背包怎么推状态转移"}]
        info = build_query("那内层为什么倒序", history)
        assert info["is_followup"] is True
        assert "背包" in info["anchor"]
        assert info["mode"] == "anchor"

    def test_code_signal(self):
        features = extract_code_signals("for(int i=0;i<n;i++) dp[i]=max(dp[i],dp[i-1]);")
        assert any("动态规划" in f for f in features)

    def test_split_code_keeps_chinese_comment(self):
        nl, code = split_code("// 状态转移\nint dp[100];\nfor(int i=0;i<n;i++) dp[i]=0;\n为什么错")
        assert "为什么错" in nl
        assert "dp[100]" in code

    def test_blank_input_falls_back(self):
        info = build_query("   ")
        assert info["query"]


class TestMetrics:
    def test_hit_at_k(self):
        assert hit_at_k(["a", "b"], {"a": 2}, 1) == 1.0
        assert hit_at_k(["b", "a"], {"a": 2}, 1) == 0.0

    def test_mrr(self):
        assert mrr_at_k(["b", "a"], {"a": 2}, 10) == 0.5

    def test_ndcg_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 10) == 1.0

    def test_ndcg_ignores_query_without_relevant(self):
        assert ndcg_at_k(["a"], {}, 10) is None

    def test_recall_at_k(self):
        qrels = {"a": 2, "b": 1, "c": 1}
        assert recall_at_k(["a", "x", "b"], qrels, 3) == pytest.approx(2 / 3)
        assert recall_at_k(["a", "b", "c"], qrels, 100) == 1.0
        assert recall_at_k(["x"], {}, 100) is None  # 无正例不参与平均

    def test_aggregate_has_recall100(self):
        records = [{"ranked": ["a", "b"], "qrels": {"a": 2, "b": 1}, "latency_ms": 5.0,
                    "level": "strong", "qtype": "solution"}]
        assert aggregate(records)["recall@100"] == 1.0

    def test_aggregate_empty(self):
        assert aggregate([])["queries"] == 0

    def test_aggregate_shape(self):
        records = [{"ranked": ["a"], "qrels": {"a": 2}, "latency_ms": 12.0,
                    "level": "strong", "qtype": "solution"}]
        result = aggregate(records)
        assert result["hit@1"] == 1.0
        assert result["ndcg@10"] == 1.0
        assert result["empty_rate"] == 0.0
        assert math.isclose(result["latency_p50_ms"], 12.0)


class TestIngestGate:
    def _item(self, **over):
        item = {
            "title": "测试题", "difficulty": "中等", "tags": "动态规划",
            "description": "描述" * 10, "solution": "思路" * 30, "pitfalls": "边界",
            "source": "curated",
        }
        item.update(over)
        return item

    def test_min_length_rejected(self):
        short = self._item(description="短", solution="短", pitfalls="")
        assert to_records([short]) == []

    def test_valid_item_accepted(self):
        records = to_records([self._item()])
        assert len(records) == 1
        assert records[0]["domain"] in {
            "动态规划", "图论", "数据结构", "数学", "字符串", "算法思想", "基础与实现"
        }

    def test_statement_copy_rejected(self):
        long_item = self._item(solution="输入格式：第一行一个整数 n，接下来 n 行每行两个整数表示边。" * 10)
        assert to_records([long_item]) == []

    def test_crawled_entry_length_capped(self):
        crawled = self._item(source="cf", entry_type="考点提示", solution="内容" * 400)
        assert to_records([crawled]) == []

    def test_compose_keeps_raw(self):
        raw = "题目：X\n难度：简单\n标签：a\n描述：b\n题解思路：c\n常见错误：d"
        assert compose_curated_content({"title": "X", "raw": raw}) == raw
