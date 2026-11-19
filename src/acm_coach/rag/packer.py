# -*- coding: utf-8 -*-
"""上下文打包：把召回结果压进模型的 token 预算。

三个约束：
1. **同题去重**：一题多切片只保留最相关的一条，否则 6 个名额被同一题占满；
2. **预算控制**：按字符预算贪心装入（中文 1 字≈1 token 量级，留出安全边际）；
3. **可溯源**：每条资料带稳定编号 [1][2]…，与返回给前端的 sources 一一对应。
"""
from __future__ import annotations

from acm_coach.rag.index import Hit

PER_ITEM_CHAR_CAP = 900


def dedupe(hits: list[Hit]) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for hit in hits:
        key = (hit.title or hit.id).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def pack(hits: list[Hit], budget: int = 4000) -> tuple[list[Hit], str]:
    """返回（实际装入的条目, 参考资料文本）。"""
    if not hits:
        return [], "（题库未检索到相关内容）"

    selected: list[Hit] = []
    used = 0
    for hit in dedupe(hits):
        snippet = hit.content[:PER_ITEM_CHAR_CAP]
        cost = len(snippet) + len(hit.title) + 40
        if selected and used + cost > budget:
            break
        selected.append(hit)
        used += cost

    lines: list[str] = []
    for idx, hit in enumerate(selected, 1):
        meta = f"难度：{hit.difficulty or '未标注'}；标签：{hit.tags or '未标注'}"
        if hit.domain and hit.domain != "未分类":
            meta += f"；领域：{hit.domain}"
        if hit.source != "curated":
            meta += f"；来源：{hit.source}"
        body = hit.content[:PER_ITEM_CHAR_CAP]
        lines.append(f"[{idx}] {hit.title}（{meta}）\n{body}")
    return selected, "\n\n".join(lines)


def citation_hint(selected: list[Hit]) -> str:
    if not selected:
        return ""
    titles = "；".join(f"[{i}] {h.title}" for i, h in enumerate(selected, 1))
    return f"可用引用：{titles}"
