# -*- coding: utf-8 -*-
"""标签体系与难度归一化。

两个必须注意的口径问题（都会让统计结论失真）：
1. **Codeforces 返回的 tags 是字母序，不是语义主次**。按首个 tag 定域会把
   `["dfs and similar","graphs"]` 判成"算法思想"、`["2-sat","graphs"]` 才判成"图论"，
   导致图论占比掉到 2.2%。因此主域用"最具体域优先"的显式优先级表派生。
2. **AtCoder 约一半题目没有 difficulty**（problems.json 无该字段，需 join
   problem-models.json，且新题缺失）。缺失一律记 `未知` 并标记来源，不得猜测。
"""
from __future__ import annotations

import json
from pathlib import Path

from acm_coach.core.logging import get_logger

log = get_logger("crawler.taxonomy")

# 最具体域优先：越靠前越具体，命中即定域
DOMAIN_PRIORITY = (
    "计算几何", "字符串", "图论", "动态规划", "数据结构", "数学", "竞赛技巧", "算法思想", "基础与实现",
)

# 兜底映射：data/taxonomy/cf_tag_map.json 缺失时仍可运行
FALLBACK_TAG_MAP: dict[str, dict] = {
    "implementation": {"zh": "模拟与实现", "domain": "基础与实现", "points": ["按题意模拟"], "hint": "按题面步骤逐步模拟，注意边界与格式。"},
    "brute force": {"zh": "暴力枚举", "domain": "基础与实现", "points": ["枚举与剪枝"], "hint": "数据范围通常很小，枚举全部可能并验证。"},
    "greedy": {"zh": "贪心", "domain": "算法思想", "points": ["贪心策略正确性"], "hint": "每步取局部最优，重点在证明策略不会错过最优解。"},
    "binary search": {"zh": "二分查找", "domain": "算法思想", "points": ["二分查找", "二分答案"], "hint": "关注单调性与边界收缩方式。"},
    "sortings": {"zh": "排序", "domain": "算法思想", "points": ["排序", "结构体排序"], "hint": "排序后把无序问题转成有序扫描。"},
    "two pointers": {"zh": "双指针", "domain": "算法思想", "points": ["双指针"], "hint": "两个指针同向或相向移动，均摊 O(n)。"},
    "dp": {"zh": "动态规划", "domain": "动态规划", "points": ["状态设计", "转移方程"], "hint": "先定义状态含义，再写转移与边界。"},
    "graphs": {"zh": "图论", "domain": "图论", "points": ["建图", "遍历"], "hint": "先把题意抽象成图，再选遍历或最短路。"},
    "dfs and similar": {"zh": "深度优先搜索", "domain": "图论", "points": ["DFS", "回溯"], "hint": "递归搜索状态空间，注意剪枝与回溯还原。"},
    "shortest paths": {"zh": "最短路", "domain": "图论", "points": ["Dijkstra", "Bellman-Ford"], "hint": "按边权正负选择算法。"},
    "dsu": {"zh": "并查集", "domain": "图论", "points": ["并查集"], "hint": "维护连通性，路径压缩后近似 O(1)。"},
    "trees": {"zh": "树", "domain": "数据结构", "points": ["树的遍历", "LCA"], "hint": "树上问题先想子树与路径两类视角。"},
    "data structures": {"zh": "数据结构", "domain": "数据结构", "points": ["线段树", "树状数组"], "hint": "需要动态区间信息时考虑树状结构。"},
    "strings": {"zh": "字符串", "domain": "字符串", "points": ["字符串匹配"], "hint": "注意预处理与失配跳转。"},
    "math": {"zh": "数学", "domain": "数学", "points": ["数论", "组合计数"], "hint": "先推公式再写代码，注意取模与溢出。"},
    "number theory": {"zh": "数论", "domain": "数学", "points": ["整除", "同余"], "hint": "关注 gcd/取模/逆元三类工具。"},
    "combinatorics": {"zh": "组合数学", "domain": "数学", "points": ["组合计数"], "hint": "计数问题先想是否可拆成独立子问题。"},
    "geometry": {"zh": "计算几何", "domain": "计算几何", "points": ["向量运算", "凸包"], "hint": "用向量叉积判断方向，注意浮点误差。"},
    "flows": {"zh": "网络流", "domain": "图论", "points": ["最大流", "最小割"], "hint": "把约束建成容量，答案常与最小割对应。"},
    "games": {"zh": "博弈论", "domain": "数学", "points": ["SG 函数"], "hint": "先分析必败态，再考虑 SG 合并。"},
    "interactive": {"zh": "交互题", "domain": "竞赛技巧", "points": ["交互协议"], "hint": "严格控制询问次数，注意刷新输出缓冲。"},
    "constructive algorithms": {"zh": "构造", "domain": "算法思想", "points": ["构造策略"], "hint": "从奇偶性、边界或归纳角度构造可行解。"},
    "divide and conquer": {"zh": "分治", "domain": "算法思想", "points": ["分治"], "hint": "把问题拆成规模减半的子问题再合并。"},
    "bitmasks": {"zh": "位运算", "domain": "数学", "points": ["状态压缩"], "hint": "用二进制位表示集合，n 通常不超过 20。"},
    "meet-in-the-middle": {"zh": "折半搜索", "domain": "算法思想", "points": ["折半枚举"], "hint": "枚举规模 n/2 再合并两侧结果。"},
}

ATCODER_SUFFIX_HINT = {
    "a": "通常为读入与最基础的模拟，用来热身。",
    "b": "基础实现题，注意边界与格式。",
    "c": "常见为贪心或简单数据结构。",
    "d": "需要组合一个标准算法或数据结构。",
    "e": "需要组合多个知识点或较复杂的状态设计。",
    "f": "难题，通常需要较深的建模能力。",
}


class Taxonomy:
    def __init__(self, base_dir: Path | None = None):
        self.map: dict[str, dict] = dict(FALLBACK_TAG_MAP)
        self.tree: list[dict] = []
        if base_dir and base_dir.exists():
            self._load(base_dir)

    def _load(self, base_dir: Path) -> None:
        map_path = base_dir / "cf_tag_map.json"
        tree_path = base_dir / "tree.json"
        try:
            if map_path.exists():
                raw = json.loads(map_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    # 深合并：保留兜底项，覆盖为人工核验后的正式映射
                    for key, value in raw.items():
                        self.map[key.strip().lower()] = value
            if tree_path.exists():
                raw_tree = json.loads(tree_path.read_text(encoding="utf-8"))
                # tree.json 顶层是数组，不是对象
                self.tree = raw_tree if isinstance(raw_tree, list) else raw_tree.get("children", [])
            log.info("标签体系加载：%d 个 tag / %d 个领域", len(self.map), len(self.tree))
        except (ValueError, OSError) as exc:
            log.warning("标签体系加载失败，使用内置兜底映射：%s", exc)

    # ---------------- 域与知识点 ----------------
    def main_domain(self, tags: list[str]) -> tuple[str, list[str]]:
        """返回（主域, 同域知识点）。按最具体域优先，而不是 tag 顺序。"""
        domains: set[str] = set()
        points: dict[str, list[str]] = {}
        for tag in tags:
            meta = self.map.get(tag.strip().lower())
            if not meta:
                continue
            domain = meta.get("domain", "基础与实现")
            domains.add(domain)
            points.setdefault(domain, []).extend(meta.get("points", []))
        for domain in DOMAIN_PRIORITY:
            if domain in domains:
                deduped = list(dict.fromkeys(points.get(domain, [])))[:3]
                return domain, deduped
        return "基础与实现", []

    def zh_tags(self, tags: list[str]) -> list[str]:
        out = []
        for tag in tags:
            meta = self.map.get(tag.strip().lower())
            out.append(meta.get("zh", tag) if meta else tag)
        return out

    def hints(self, tags: list[str], limit: int = 2) -> list[str]:
        out = []
        for tag in tags:
            meta = self.map.get(tag.strip().lower())
            if meta and meta.get("hint"):
                out.append(meta["hint"])
            if len(out) >= limit:
                break
        return out


# ---------------- 难度归一化 ----------------
TIERS = ("入门", "简单", "中等", "较难", "困难")
DISPLAY = ("简单", "中等", "困难")


def cf_tier(rating: int | None) -> tuple[str, str]:
    """返回（五档, 三档展示）。"""
    if rating is None:
        return "未知", "中等"
    if rating < 1200:
        return "入门", "简单"
    if rating < 1400:
        return "简单", "简单"
    if rating < 1700:
        return "中等", "中等"
    if rating < 2000:
        return "较难", "中等"
    return "困难", "困难"


def atcoder_tier(difficulty: int | None) -> tuple[str, str]:
    if difficulty is None:
        return "未知", "中等"
    if difficulty < 400:
        return "入门", "简单"
    if difficulty < 800:
        return "简单", "简单"
    if difficulty < 1200:
        return "中等", "中等"
    if difficulty < 1600:
        return "较难", "中等"
    return "困难", "困难"
