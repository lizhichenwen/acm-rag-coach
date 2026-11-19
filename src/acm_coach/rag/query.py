# -*- coding: utf-8 -*-
"""查询构造：把"学生怎么说"翻译成"检索怎么查"。

三类典型失败（都在 v3 日志里出现过）：
1. 学生整段粘代码 → 代码 token 落进向量空间的无意义区域，召回全跑偏；
2. 多轮追问"那内层为什么倒序" → 丢了"01 背包"这个锚点，检索漂移；
3. 口语缩写（TLE / 板子 / 对拍）→ 与题库书面语不匹配，召回率低。

对策：剥离代码噪声 → 补话题锚点 → 抽代码算法信号 → 同义词扩展 → 控制长度。
规则版零延迟、可单测；追问场景可选让大模型改写一次（auto 模式）。
"""
from __future__ import annotations

import re

from acm_coach.rag.text import expand_query

_FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
_CJK = re.compile(r"[\u4e00-\u9fff]")
_CODE_TOKENS = (";", "{", "}", "#include", "->", "::", "++", "--", "+=", "==",
                "&&", "scanf", "printf", "cin", "cout", "dp[", "std::", "for(", "while(")

# 代码 → 算法术语（用于代码类提问的第二路召回）
CODE_SIGNALS = (
    (r"dp\s*\[", "动态规划"),
    (r"max\s*\(\s*dp|min\s*\(\s*dp", "动态规划 状态转移"),
    (r"priority_queue|heapq|make_heap", "堆 优先队列"),
    (r"lower_bound|upper_bound|binary_search", "二分查找"),
    (r"unordered_map|unordered_set|map\s*<|dict\s*\(", "哈希表"),
    (r"vector\s*<\s*vector|vector\s*<\s*pair", "图 邻接表 建图"),
    (r"queue\s*<|deque", "广度优先搜索 BFS"),
    (r"dfs\s*\(|vis\s*\[", "深度优先搜索 回溯"),
    (r"parent\s*\[|fa\s*\[|find\s*\(", "并查集"),
    (r"0x3f3f3f3f|INF|dist\s*\[", "最短路"),
    (r"gcd|lcm|1e9\s*\+\s*7|mod", "数论 取模 逆元"),
    (r"sort\s*\(", "排序 贪心"),
    (r"stack\s*<", "栈 单调栈"),
    (r"pre\s*\[|sum\s*\[|prefix", "前缀和"),
    (r"nxt\s*\[|fail\s*\[|kmp", "KMP 字符串匹配"),
    (r"trie|son\s*\[", "字典树"),
    (r"while\s*\(\s*l\s*<\s*r", "二分答案"),
)

_FOLLOWUP_HINTS = ("为什么", "为啥", "那", "这", "它", "其中", "上面", "刚才", "还是",
                   "如果", "换成", "改成", "反过来", "再", "继续", "呢", "吗", "意思",
                   "具体", "举个例子", "不懂", "没懂", "哪里", "怎么改", "对不对", "这样")
_SMALLTALK = re.compile(
    r"^(你好|您好|hi|hello|在吗|谢谢|多谢|感谢|再见|拜拜|嗯+|哦+|好的|ok)[啊呀呢。！!.~]*$", re.I)

MAX_QUERY_CHARS = 160


def _is_code_line(line: str) -> bool:
    s = line.strip()
    if not s or _CJK.search(s):
        return False
    if s in ("}", "{", "};"):
        return True
    return any(tok in s for tok in _CODE_TOKENS)


def split_code(text: str) -> tuple[str, str]:
    text = text or ""
    fenced = _FENCE.findall(text)
    body = _FENCE.sub(" ", text)
    kept, code_lines = [], []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        (code_lines if _is_code_line(stripped) else kept).append(stripped)
    if len(code_lines) >= 2:
        return " ".join(kept), "\n".join(fenced + ["\n".join(code_lines)])
    return " ".join(kept + code_lines), "\n".join(fenced)


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def looks_like_code(text: str) -> bool:
    nl, code = split_code(text)
    return bool(code.strip()) and len(nl) < 6


def extract_code_signals(code: str) -> list[str]:
    found: list[str] = []
    for pattern, term in CODE_SIGNALS:
        if re.search(pattern, code or "") and term not in found:
            found.append(term)
    return found


def is_followup(text: str) -> bool:
    t = squash(text)
    if not t:
        return True
    low = t.lower()
    if len(t) <= 24 and any(h in low for h in _FOLLOWUP_HINTS):
        return True
    return len(t) <= 8


def topic_anchor(history: list[dict] | None, last_titles: list[str] | None = None,
                 max_turns: int = 2) -> str:
    parts: list[str] = []
    for title in (last_titles or [])[:1]:
        if title:
            parts.append(title)
    used = 0
    for msg in reversed(history or []):
        if msg.get("role") != "user":
            continue
        nl, _ = split_code(msg.get("content") or "")
        if not nl or _SMALLTALK.match(nl):
            continue
        parts.append(nl[:50])
        used += 1
        if used >= max_turns:
            break
    return squash(" ".join(parts))[:90]


def build_query(prompt: str, history: list[dict] | None = None,
                last_titles: list[str] | None = None) -> dict:
    raw = prompt or ""
    nl, code = split_code(raw)
    signals = extract_code_signals(code)
    followup = is_followup(nl)
    anchor = topic_anchor(history, last_titles)

    pieces: list[str] = []
    mode = "direct"
    if nl:
        pieces.append(nl)
    else:
        mode = "code"
    if (followup or not nl) and anchor:
        pieces.append(anchor)
        if mode != "code":
            mode = "anchor"
    if signals:
        pieces.append(" ".join(signals[:3]))
        if mode == "direct":
            mode = "rule"

    query = squash(" ".join(pieces))[:MAX_QUERY_CHARS]
    if not query:
        query = "代码 常见错误 调试"
    query = expand_query(query)

    queries = [query]
    if signals and nl:
        alt = squash(" ".join(signals[:2]))
        if alt and alt != query:
            queries.append(alt)

    return {
        "query": query, "queries": queries, "mode": mode, "anchor": anchor,
        "signals": signals, "is_followup": followup, "is_code": looks_like_code(raw),
    }


REWRITE_PROMPT = (
    "你是 ACM 题库检索系统的查询改写器。\n"
    "当前对话话题：{anchor}\n"
    "学生新提问：{prompt}\n"
    "请改写为一条可独立检索的查询：保留算法术语与题目名，补全省略的主语，去掉寒暄与代码。"
    "只输出查询本身，不超过 30 字。"
)
