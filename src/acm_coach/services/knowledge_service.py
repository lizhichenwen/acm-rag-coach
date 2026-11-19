# -*- coding: utf-8 -*-
"""知识入库：把多来源语料解析成统一的 KnowledgeItem 契约。

语料分两类，`source` 字段是二者的唯一边界：
- `curated`：人工精讲（题解/知识点卡/竞赛元知识），可以直接作为题解依据；
- `cf` / `atcoder`：爬取的题目元数据，只能提供"考点—难度—相似题"定位，
  生成层会据此降级引用话术，禁止当成题解使用。

格式契约（curated txt）：
    题目：xxx
    难度：简单|中等|困难
    标签：a,b,c
    描述：...
    题解思路：...
    常见错误：...
    ---
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from acm_coach.core.logging import get_logger

log = get_logger("knowledge")

DIFFICULTY_SET = {"简单", "中等", "困难"}
REQUIRED_FIELDS = ("题目", "难度", "标签", "描述", "题解思路")

# 拒绝入库的特征：整篇题面/官方题解转载（版权与合规红线）
_REJECT_PATTERNS = (
    re.compile(r"样例输入|样例输出|sample input|sample output", re.I),
    re.compile(r"输入格式[:：].{80,}", re.S),
    re.compile(r"输出格式[:：].{80,}", re.S),
)
MAX_CONTENT_CHARS = 2400
MIN_CONTENT_CHARS = 40


def content_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def slug_id(title: str, prefix: str = "KP") -> str:
    """稳定 id：同一标题永远得到同一个 id，保证 upsert 幂等。"""
    digest = hashlib.sha1(title.strip().encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def parse_curated_text(text: str) -> list[dict]:
    blocks = re.split(r"\n\s*---\s*\n", text)
    items: list[dict] = []
    for block in blocks:
        block = block.strip()
        if not block or "题目：" not in block:
            continue
        fields: dict[str, str] = {}
        current = None
        for line in block.splitlines():
            m = re.match(r"^([\u4e00-\u9fff]{2,6})[:：]\s*(.*)$", line.strip())
            if m and m.group(1) in {"题目", "难度", "标签", "描述", "题解思路", "常见错误", "来源", "链接", "领域"}:
                current = m.group(1)
                fields[current] = m.group(2)
            elif current:
                fields[current] = f"{fields[current]}\n{line.strip()}".strip()
        if not all(k in fields for k in REQUIRED_FIELDS):
            log.debug("跳过不完整条目：%s", fields.get("题目", "?"))
            continue
        items.append({
            "title": fields["题目"].strip(),
            "difficulty": fields["难度"].strip(),
            "tags": fields["标签"].strip(),
            "description": fields.get("描述", "").strip(),
            "solution": fields.get("题解思路", "").strip(),
            "pitfalls": fields.get("常见错误", "").strip(),
            "domain": fields.get("领域", "").strip(),
            "source": "curated",
            "url": fields.get("链接", "").strip(),
            # 形态判定：竞赛元知识（赛制/策略类）与题解分开，供 meta_json 契约枚举映射
            "entry_type": (
                "竞赛元知识" if any(k in fields.get("标签", "") for k in ("竞赛策略", "赛制", "训练方法"))
                else "题解"
            ),
            # 保留原始文本：与 v3 入库内容逐字一致，保证"v3 vs v4 向量一致性"
            # 的前置校验不被格式化差异污染
            "raw": block,
        })
    return items


def compose_curated_content(item: dict) -> str:
    if item.get("raw"):
        return item["raw"]
    parts = [f"题目：{item['title']}"]
    parts.append(f"难度：{item['difficulty']}｜标签：{item['tags']}")
    if item.get("description"):
        parts.append(f"题意：{item['description']}")
    if item.get("solution"):
        parts.append(f"解题思路：{item['solution']}")
    if item.get("pitfalls"):
        parts.append(f"常见错误：{item['pitfalls']}")
    return "\n".join(parts)


# 条目形态中文标签 → 溯源契约枚举（docs/05 §6.7）
ENTRY_TYPE_ENUM = {
    "题解": "solution",
    "知识点卡": "concept",
    "竞赛元知识": "strategy",
    "考点提示": "hint",
}


def _meta_contract(item: dict, source: str, entry_label: str) -> dict:
    """meta_json 溯源契约 7 键（+ 中文形态标签与来源链接备查）。

    取值严格沿用 docs/05 §6.7「团队唯一准」的枚举，不自造第五套命名：
    - content_origin: self_written / metadata
    - owner_type: user / platform
    - license: internal-only / Codeforces-Terms / AtCoder-Terms
    - entry_type: solution / concept / strategy / hint（hint 为 02 文档扩展的考点提示）

    source 列维持 curated/cf/atcoder 不动（检索 type_boost 承重契约，见 docs/04），
    "人工精讲 vs 平台元数据"的语义由 content_origin / owner_type 承载。
    """
    curated_side = source in ("curated", "manual")
    default_license = ("internal-only" if curated_side
                       else "Codeforces-Terms" if source == "cf" else "AtCoder-Terms")
    # mirror：通道/镜像维度（docs/05 §6.7 第 8 键），与 source/license 不可合并——
    # 权利人主张删除时要回答"该停哪个通道"，只看 atcoder 是不够的。
    default_mirror = ("n/a(self_written)" if curated_side
                      else "official-api" if source == "cf"
                      else "kenkoooo" if source == "atcoder" else "unknown")
    return {
        "entry_type": ENTRY_TYPE_ENUM.get(entry_label, "hint"),
        "entry_type_label": entry_label,
        "content_origin": "self_written" if curated_side else "metadata",
        "processed_by": "knowledge_service.to_records/v4",
        "owner_type": "user" if curated_side else "platform",
        # 人工自撰条目没有"抓取时间/批次"语义：写显式 n/a 而非留空或用 created_at 冒充
        "fetched_at": item.get("fetched_at") or ("n/a(self_written)" if curated_side else "unknown"),
        "crawl_batch": item.get("crawl_batch") or ("n/a(self_written)" if curated_side else "unknown"),
        "license": item.get("license") or item.get("license_note") or default_license,
        "mirror": item.get("mirror") or default_mirror,
        "source_url": item.get("url", ""),
    }


def to_records(items: list[dict], prefix: str = "KP") -> list[dict]:
    """转换成入库契约（与 KnowledgeItem 列一一对应）。"""
    records: list[dict] = []
    for item in items:
        content = item.get("content") or compose_curated_content(item)
        if not quality_gate(item, content):
            continue
        domain = item.get("domain") or infer_domain(item.get("tags", ""))
        source = item.get("source", "manual")
        records.append({
            "id": item.get("id") or slug_id(item["title"], prefix),
            "title": item["title"][:250],
            "difficulty": item["difficulty"] if item["difficulty"] in DIFFICULTY_SET else "中等",
            "rating": item.get("rating"),
            "tags": item.get("tags", "")[:250],
            "domain": domain,
            "source": source,
            "url": (item.get("url") or None),
            "content": content,
            "content_hash": content_hash(content),
            "chunk_index": int(item.get("chunk_index", 0)),
            "quality_score": float(item.get("quality_score", 1.0)),
            # 合规字段：默认已发布、内部可见、L2 敏感级（人工精讲可能含未公开题解）
            "status": item.get("status", "published"),
            "visibility": item.get("visibility", "internal" if source == "curated" else "public"),
            "sensitivity": item.get("sensitivity", "L2" if source == "curated" else "L1"),
            "retention_days": int(item.get("retention_days", 0)),
            "meta_json": json.dumps(
                _meta_contract(item, source,
                               item.get("entry_type") or ("题解" if source == "curated" else "考点提示")),
                ensure_ascii=False,
            ),
        })
    return records


def quality_gate(item: dict, content: str) -> bool:
    """入库门禁：字段完整性 / 长度分档 / 版权特征。

    长度按条目形态分档：爬取类（考点提示）上限 500 字符——2400 字符足以塞下一段
    完整题面，而正则只能拦住带"样例输入"标记的，等于给"只存元数据"留了口子。
    """
    if not item.get("title") or not item.get("tags"):
        return False
    entry_type = item.get("entry_type") or ("题解" if item.get("source") in (None, "curated", "manual") else "考点提示")
    max_chars = 500 if entry_type == "考点提示" else MAX_CONTENT_CHARS
    if not (MIN_CONTENT_CHARS <= len(content) <= max_chars):
        log.warning("门禁拦截（长度越界 %d/%d）：%s", len(content), max_chars, item.get("title"))
        return False
    for pattern in _REJECT_PATTERNS:
        if pattern.search(content):
            log.warning("门禁拦截（疑似题面转载）：%s", item.get("title"))
            return False
    return True


DOMAIN_RULES = (
    ("动态规划", ("动态规划", "dp", "背包", "状态压缩")),
    ("图论", ("图论", "最短路", "并查集", "拓扑", "最小生成树", "网络流", "连通")),
    ("数据结构", ("数据结构", "线段树", "树状数组", "堆", "栈", "队列", "字典树", "平衡树")),
    ("数学", ("数论", "组合", "概率", "博弈", "矩阵", "几何", "多项式")),
    ("字符串", ("字符串", "kmp", "后缀", "哈希", "trie", "回文")),
    ("算法思想", ("二分", "贪心", "分治", "排序", "双指针", "搜索", "构造")),
)


def infer_domain(tags: str) -> str:
    lowered = (tags or "").lower()
    for domain, keywords in DOMAIN_RULES:
        if any(k in lowered for k in keywords):
            return domain
    return "基础与实现"


def load_directory(paths: list[Path]) -> list[dict]:
    """加载目录下所有 .txt / .jsonl / .json 语料。"""
    items: list[dict] = []
    for base in paths:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix == ".txt":
                parsed = parse_curated_text(path.read_text(encoding="utf-8"))
                # 知识点卡文件单独标记形态：与题解/竞赛元知识一起归一到枚举 concept
                if "card" in path.stem.lower():
                    for item in parsed:
                        item["entry_type"] = "知识点卡"
                items.extend(parsed)
            elif path.suffix == ".jsonl":
                import json

                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except ValueError:
                                continue
    log.info("语料加载：%d 条（来自 %s）", len(items), [str(p) for p in paths])
    return items
