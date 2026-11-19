# -*- coding: utf-8 -*-
"""BM25 倒排索引（词法召回）。

为什么自己实现而不用 rank_bm25：
1. 需要持久化（8k 条文档每次启动重建要 2-3 秒，且占内存）；
2. 需要中文分词与停用词策略可控；
3. 需要输出"可解释的命中词"，用于检索日志与论文分析。

公式：score(q,d) = Σ idf(t) · tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))
  idf(t) = ln(1 + (N-df+0.5)/(df+0.5))
"""
from __future__ import annotations

import math
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from acm_coach.core.logging import get_logger
from acm_coach.rag.text import tokenize

log = get_logger("rag.bm25")

INDEX_VERSION = 3  # v3: 支持字段加权（title×3/tags×2/正文×1），旧缓存需重建


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    ids: list[str] = field(default_factory=list)
    doc_len: dict[str, int] = field(default_factory=dict)
    postings: dict[str, dict[str, int]] = field(default_factory=dict)  # term -> {doc_id: tf}
    avgdl: float = 0.0
    built_at: float = 0.0

    # ---------------- 构建 ----------------
    @classmethod
    def build(cls, documents: dict[str, str], k1: float = 1.5, b: float = 0.75,
              field_docs: dict[str, dict[str, str]] | None = None,
              title_weight: float = 0.0, tags_weight: float = 0.0) -> "BM25Index":
        """构建索引。

        `field_docs` 提供每篇文档的 {title, tags, content}，配合权重实现字段加权
        （docs/04 §1.2：title×3、tags×2、正文×1）。权重为 0 时退化为纯 `documents`
        文本，与历史行为一致。加权用"字段文本按权重重复"实现，正文字典序不变，
        只对命中词频加权，零成本且可解释。
        """
        index = cls(k1=k1, b=b)
        postings: dict[str, dict[str, int]] = {}
        total_len = 0
        weighted = bool(field_docs) and (title_weight > 0 or tags_weight > 0)
        for doc_id, text in documents.items():
            if weighted:
                fields = (field_docs or {}).get(doc_id, {})
                repeats = [text]
                tw = int(round(title_weight))
                sw = int(round(tags_weight))
                if tw > 0 and fields.get("title"):
                    repeats.extend([fields["title"]] * tw)
                if sw > 0 and fields.get("tags"):
                    repeats.extend([fields["tags"]] * sw)
                text = " ".join(repeats)
            tokens = tokenize(text)
            tf = Counter(tokens)
            index.ids.append(doc_id)
            index.doc_len[doc_id] = len(tokens)
            total_len += len(tokens)
            for term, count in tf.items():
                postings.setdefault(term, {})[doc_id] = count
        index.postings = postings
        index.avgdl = (total_len / len(documents)) if documents else 0.0
        index.built_at = time.time()
        log.info("BM25 索引构建完成：%d 篇 / %d 词 / avgdl=%.1f",
                 len(index.ids), len(postings), index.avgdl)
        return index

    # ---------------- 查询 ----------------
    def _idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        n = max(len(self.ids), 1)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 24) -> list[tuple[str, float, list[str]]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores: dict[str, float] = {}
        matched_terms: dict[str, list[str]] = {}
        for term in tokens:
            docs = self.postings.get(term)
            if not docs:
                continue
            idf = self._idf(term)
            for doc_id, tf in docs.items():
                dl = self.doc_len.get(doc_id, 0) or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / denom
                matched_terms.setdefault(doc_id, []).append(term)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(doc_id, round(score, 4), matched_terms.get(doc_id, [])[:6]) for doc_id, score in ranked]

    def max_possible_score(self, query: str) -> float:
        """查询侧理论上界，用于把 BM25 分数归一化到 0-1（解释性用）。"""
        tokens = set(tokenize(query))
        return sum(self._idf(t) * (self.k1 + 1) for t in tokens) or 1.0

    # ---------------- 持久化 ----------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION, "k1": self.k1, "b": self.b, "ids": self.ids,
            "doc_len": self.doc_len, "postings": self.postings,
            "avgdl": self.avgdl, "built_at": self.built_at,
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index | None":
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") != INDEX_VERSION:
                log.info("BM25 索引版本不匹配，忽略缓存")
                return None
            return cls(k1=payload["k1"], b=payload["b"], ids=payload["ids"],
                       doc_len=payload["doc_len"], postings=payload["postings"],
                       avgdl=payload["avgdl"], built_at=payload["built_at"])
        except Exception as exc:  # 损坏则重建
            log.warning("BM25 索引加载失败（%s），将重建", exc)
            return None

    @property
    def size(self) -> int:
        return len(self.ids)
