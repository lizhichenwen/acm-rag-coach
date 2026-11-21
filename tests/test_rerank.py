# -*- coding: utf-8 -*-
"""重排后端单测：消融隔离 + 降级路径，全程离线、不加载真实模型。

锁定的两个回归点：
1. enable_rerank=False 时绝不重排——G1/G2/G3 必须保留上游通道顺序；
2. 同一进程里先跑关重排的分组，不能把 _MODEL_TRIED 锁死导致 G4 拿不到模型。
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from acm_coach.rag import rerank as rerank_mod
from acm_coach.rag.rerank import get_reranker, rerank, rerank_backend


def _hits() -> list:
    return [
        SimpleNamespace(id="a", content="普通条目 动态规划", rerank_score=None,
                        bm25_score=3.0, vector_score=0.1),
        SimpleNamespace(id="b", content="01背包 内层循环 倒序", rerank_score=None,
                        bm25_score=1.0, vector_score=0.9),
    ]


def _settings(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(enable_rerank=enabled,
                           rerank_model="BAAI/bge-reranker-base",
                           rerank_download=False, rerank_doc_chars=1200,
                           rerank_max_length=192, rerank_batch_size=16,
                           rerank_cache_size=256)


def _reset(monkeypatch):
    monkeypatch.setattr(rerank_mod, "_MODEL", None)
    monkeypatch.setattr(rerank_mod, "_MODEL_TRIED", False)
    monkeypatch.setattr(rerank_mod, "_MODEL_KEY", None)
    monkeypatch.setattr(rerank_mod, "_LAST_BACKEND", "off")


def _fake_cross_encoder(monkeypatch, predict_fn):
    """用轻量假模块顶替 sentence_transformers，CI 不装重型依赖也能跑。"""
    import types

    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = type("CrossEncoder", (), {
        "__init__": lambda self, model_name, local_files_only=True: None,
        "predict": lambda self, pairs, show_progress_bar=False, **kw: predict_fn(pairs),
    })
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)


class TestRerankBackend:
    def test_disabled_keeps_original_order(self, monkeypatch):
        _reset(monkeypatch)
        hits = _hits()
        out = rerank("倒序", hits, _settings(False))
        assert [h.content for h in out] == ["普通条目 动态规划", "01背包 内层循环 倒序"]
        assert all(h.rerank_score is None for h in out)
        assert rerank_backend() == "off"
        assert get_reranker(_settings(False)) is None

    def test_enabled_without_model_uses_fusion_fallback(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(rerank_mod, "_model_cached", lambda name: False)
        hits = _hits()
        out = rerank("倒序 背包", hits, _settings(True))
        assert rerank_backend() == "fusion-fallback"
        assert all(h.rerank_score is not None for h in out)
        assert len(out) == 2

    def test_cache_hit_skips_model_predict(self, monkeypatch):
        """同一 query + 同一候选池，第二次调用不应再触发模型推理。"""
        _reset(monkeypatch)
        rerank_mod.clear_cache()
        monkeypatch.setattr(rerank_mod, "_model_cached", lambda name: True)

        calls = []

        def predict(pairs):
            calls.append(len(pairs))
            return [5.0 if "倒序" in doc else -1.0 for _q, doc in pairs]

        _fake_cross_encoder(monkeypatch, predict)
        s = _settings(True)
        rerank("倒序", _hits(), s)
        rerank("倒序", _hits(), s)
        assert len(calls) == 1, "第二次应命中缓存，不再调用 predict"
        assert rerank_mod.cache_stats()["hits"] == 1

    def test_off_call_does_not_lock_model_loading(self, monkeypatch):
        """回归：旧实现把 _MODEL_TRIED 在 enable_rerank 判定前置位，G4 永远拿不到模型。"""
        _reset(monkeypatch)
        monkeypatch.setattr(rerank_mod, "_model_cached", lambda name: True)

        _fake_cross_encoder(monkeypatch,
                            lambda pairs: [5.0 if "倒序" in doc else -1.0 for _q, doc in pairs])

        # 先跑一个关重排的分组（G1/G2/G3 的语义）
        assert get_reranker(_settings(False)) is None
        assert rerank_mod._MODEL_TRIED is False

        # 再跑开重排的分组，必须仍有机会加载模型
        hits = _hits()
        out = rerank("倒序", hits, _settings(True))
        assert rerank_backend() == "cross-encoder"
        assert out[0].content == "01背包 内层循环 倒序"
        assert out[0].rerank_score > out[1].rerank_score
