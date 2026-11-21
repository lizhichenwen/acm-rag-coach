# -*- coding: utf-8 -*-
"""启动链路单测：后台预热不阻塞 healthz，且并发请求不会读到半加载索引。

不依赖真实模型/索引（那要 13s），用人工阻塞的假 BM25 模拟加载过程。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from acm_coach.rag import index as index_mod
from acm_coach.rag.index import HybridRetriever


class TestEnsureLoadedConcurrency:
    def test_concurrent_callers_wait_for_full_load(self, monkeypatch):
        """加载未完成时，并发调用方必须在锁上等待，而不是拿到空 docs 就返回。"""
        started = threading.Event()
        release = threading.Event()

        class FakeBM25:
            @staticmethod
            def load(_path):
                started.set()
                release.wait(5)  # 模拟 11s 的 sentence_transformers import
                return object()

        monkeypatch.setattr(index_mod, "BM25Index", FakeBM25)
        monkeypatch.setattr(index_mod, "load_docstore", lambda _p: None) if hasattr(index_mod, "load_docstore") else None
        # load_docstore 是函数内 import，这里 patch 住 sys.modules 里的来源
        import acm_coach.rag.docstore as docstore
        monkeypatch.setattr(docstore, "load_docstore", lambda _p: {"doc1": object()})

        s = SimpleNamespace(index_dir=Path("/fake"), enable_vector=False,
                            chroma_dir=Path("/fake"), embed_model="x")
        r = HybridRetriever.__new__(HybridRetriever)
        r.settings = s
        r.index_path = Path("/fake/bm25.pkl")
        r.docs, r.bm25, r.vector = {}, None, None
        r._loaded = False
        r._load_lock = threading.Lock()
        r._load_event = threading.Event()

        results = []

        def first():
            r.ensure_loaded()
            results.append(("first", len(r.docs)))

        def second():
            r.ensure_loaded()
            results.append(("second", len(r.docs)))

        t1 = threading.Thread(target=first)
        t1.start()
        assert started.wait(2), "加载线程应已启动"
        t2 = threading.Thread(target=second)
        t2.start()
        time.sleep(0.1)
        assert len(results) == 0, "加载未完成时，两个调用方都应阻塞等待"
        release.set()
        t1.join(3)
        t2.join(3)
        assert sorted(results) == [("first", 1), ("second", 1)], "两方都应看到完整加载结果"

    def test_loaded_event_set_by_attach(self):
        """attach 注入索引后 ensure_loaded 必须直接返回，不重复加载。"""
        r = HybridRetriever.__new__(HybridRetriever)
        r._load_event = threading.Event()
        r._load_lock = threading.Lock()
        r.attach({"a": object()}, object(), None)
        assert r._load_event.is_set()
