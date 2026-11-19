# -*- coding: utf-8 -*-
"""向量检索（ChromaDB 持久化 + bge 中文向量）。

两个工程约束：
1. **不联网**：模型只从本地缓存加载（`local_files_only=True`），国内直连 HuggingFace
   会长时间重试并卡住页面，因此模型缺失时直接降级为纯词法检索，而不是等待超时；
2. **可降级**：`available=False` 时上层照常工作，只是召回退化为 BM25 单路。
"""
from __future__ import annotations

from pathlib import Path

from acm_coach.core.logging import get_logger

log = get_logger("rag.vector")


class VectorStore:
    def __init__(self, path: Path, model_name: str, collection: str = "acm_knowledge",
                 enabled: bool = True):
        self.path = path
        self.model_name = model_name
        self.collection_name = collection
        self.enabled = enabled
        self.available = False
        self._client = None
        self._collection = None
        self._model = None
        if enabled:
            try:
                self._init()
            except Exception as exc:
                log.warning("向量检索不可用，降级为 BM25 单路：%s", exc)

    # ---------------- 初始化 ----------------
    def _init(self) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, local_files_only=True,
                                          device="cpu")
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"},
        )
        self.available = True
        log.info("向量库就绪：%s，现有 %d 条", self.model_name, self._collection.count())

    # ---------------- 写入 ----------------
    def upsert(self, records: list[dict], batch_size: int = 64) -> int:
        if not self.available:
            return 0
        written = 0
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            docs = [r["content"] for r in batch]
            embeddings = self._model.encode(docs, normalize_embeddings=True,
                                            show_progress_bar=False).tolist()
            self._collection.upsert(
                ids=[r["id"] for r in batch],
                documents=docs,
                embeddings=embeddings,
                metadatas=[{
                    "title": r.get("title", "")[:200],
                    "difficulty": r.get("difficulty", ""),
                    "tags": r.get("tags", "")[:200],
                    "domain": r.get("domain", ""),
                    "source": r.get("source", ""),
                    "url": r.get("url") or "",
                } for r in batch],
            )
            written += len(batch)
        return written

    # ---------------- 查询 ----------------
    def query(self, text: str, top_k: int = 24) -> list[tuple[str, float]]:
        """返回 [(doc_id, cosine_similarity)]，按相似度降序。"""
        if not self.available or not text.strip() or self._collection.count() == 0:
            return []
        embedding = self._model.encode([text], normalize_embeddings=True,
                                       show_progress_bar=False).tolist()
        result = self._collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, self._collection.count()),
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for doc_id, distance in zip(ids, distances):
            # cosine space 下 distance = 1 - cos_sim
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            hits.append((doc_id, round(similarity, 4)))
        return hits

    def count(self) -> int:
        if not self.available:
            return 0
        return self._collection.count()

    def existing_ids(self) -> set[str]:
        """已入库的 id 集合。用于增量构建：中断后重跑只补缺失部分。"""
        if not self.available:
            return set()
        try:
            data = self._collection.get(include=[])
            return set(data.get("ids") or [])
        except Exception:
            return set()

    def delete_ids(self, ids: list[str], batch_size: int = 200) -> int:
        """按 id 删除（合规清除用）。返回删除条数。"""
        if not self.available or not ids:
            return 0
        deleted = 0
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            try:
                self._collection.delete(ids=batch)
                deleted += len(batch)
            except Exception:
                continue
        return deleted

    def reset(self) -> None:
        if not self.available:
            return
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"},
        )
