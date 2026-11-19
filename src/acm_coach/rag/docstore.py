# -*- coding: utf-8 -*-
"""文档快照：检索时不再回表查数据库。

写索引时把 Document 一并落成 JSONL（每行一条），
好处是可读、可 diff、可直接给论文附录做样本核查。
"""
from __future__ import annotations

import json
from pathlib import Path

from acm_coach.core.logging import get_logger
from acm_coach.rag.index import Document

log = get_logger("rag.docstore")

FILENAME = "docstore.jsonl"


def save_docstore(index_dir: Path, docs: dict[str, Document]) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / FILENAME
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for doc in docs.values():
            fh.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def load_docstore(index_dir: Path) -> dict[str, Document]:
    path = index_dir / FILENAME
    if not path.exists():
        return {}
    docs: dict[str, Document] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                docs[payload["id"]] = Document(**payload)
            except (ValueError, TypeError):
                continue
    log.info("文档快照加载完成：%d 篇", len(docs))
    return docs
