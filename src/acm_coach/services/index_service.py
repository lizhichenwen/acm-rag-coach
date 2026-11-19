# -*- coding: utf-8 -*-
"""索引构建：把数据库里的知识条目编成 BM25 倒排 + 向量库 + 文档快照。

三个索引产物必须同源同批：
- `bm25.pkl`：词法倒排
- `docstore.jsonl`：检索时用的文档快照（避免检索时回表）
- `chroma/`：向量索引（可选，缺失时自动降级为纯词法）

构建过程幂等：upsert 语义，重复执行只会更新，不会产生重复条目。
"""
from __future__ import annotations

import time

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger, log_event
from acm_coach.db.models import KnowledgeItem
from acm_coach.rag.bm25 import BM25Index
from acm_coach.rag.docstore import save_docstore
from acm_coach.rag.index import Document
from acm_coach.rag.vector import VectorStore

log = get_logger("index")


def documents_from_rows(rows: list[KnowledgeItem]) -> dict[str, Document]:
    docs: dict[str, Document] = {}
    for row in rows:
        docs[row.id] = Document(
            id=row.id, title=row.title, content=row.content,
            difficulty=row.difficulty, tags=row.tags or "", domain=row.domain or "未分类",
            source=row.source or "manual", url=row.url or "", rating=row.rating,
            problem_id=row.id.split("#")[0],
            status=getattr(row, "status", "published") or "published",
            visibility=getattr(row, "visibility", "internal") or "internal",
            sensitivity=getattr(row, "sensitivity", "L2") or "L2",
        )
    return docs


def embed_text(doc: Document, mode: str = "content_only") -> str:
    """向量入库文本。

    `content_only` 与 v3 严格对齐（v3 只 embed content，title 仅在 metadata），
    这样"同 query 同文档的余弦相似度"才能与 v3 基线逐条比对；
    `title_plus_content` 是待验证的增益项（E11），会改变整个向量空间，不可混用。
    """
    if mode == "title_plus_content":
        return f"{doc.title}\n{doc.content}"
    return doc.content


def build_index(rows: list[KnowledgeItem], *, with_vector: bool = True,
                reset_vector: bool = False, max_vectors: int | None = None) -> dict:
    settings = get_settings()
    t0 = time.perf_counter()
    docs = documents_from_rows(rows)
    if not docs:
        return {"documents": 0, "bm25_terms": 0, "vector": 0, "cost_ms": 0}

    field_docs = {k: {"title": v.title, "tags": v.tags, "content": v.content}
                  for k, v in docs.items()}
    bm25 = BM25Index.build(
        {k: f"{v.title} {v.tags} {v.content}" for k, v in docs.items()},
        k1=settings.bm25_k1, b=settings.bm25_b,
        field_docs=field_docs if settings.enable_bm25_field_boost else None,
        title_weight=settings.bm25_title_weight if settings.enable_bm25_field_boost else 0.0,
        tags_weight=settings.bm25_tags_weight if settings.enable_bm25_field_boost else 0.0,
    )
    bm25.save(settings.index_dir / "bm25.pkl")
    save_docstore(settings.index_dir, docs)

    written = 0
    if with_vector and settings.enable_vector:
        store = VectorStore(settings.chroma_dir, settings.embed_model, enabled=True)
        if reset_vector:
            store.reset()
        if store.available:
            existing = set() if reset_vector else store.existing_ids()
            todo = [d for d in docs.values() if d.id not in existing]
            if max_vectors is not None:
                todo = todo[:max_vectors]
            records = [{
                "id": d.id, "content": embed_text(d, settings.embed_text_mode),
                "title": d.title, "difficulty": d.difficulty, "tags": d.tags,
                "domain": d.domain, "source": d.source, "url": d.url,
            } for d in todo]
            written = store.upsert(records)
            log.info("向量增量构建：已有 %d，本次补 %d，剩余 %d",
                     len(existing), written, max(0, len(docs) - len(existing) - written))

    cost = (time.perf_counter() - t0) * 1000
    log_event(log, "index.built", documents=len(docs), terms=len(bm25.postings),
              vectors=written, cost_ms=round(cost))
    return {
        "documents": len(docs), "bm25_terms": len(bm25.postings),
        "vector": written, "cost_ms": round(cost),
    }
