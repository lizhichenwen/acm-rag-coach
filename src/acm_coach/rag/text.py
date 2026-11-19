# -*- coding: utf-8 -*-
"""检索用文本处理：归一化、分词、词表统计。

中文分词优先用 jieba（保留专有名词如"并查集""树状数组"），
在 jieba 不可用的环境下退化为"汉字二元组 + 英文单词"，保证部署环境再瘦也能跑。
两套分词的结果都只影响召回，不影响上层接口。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

try:  # 可选依赖
    import jieba  # type: ignore

    jieba.setLogLevel(logging.WARNING)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - 环境相关
    _HAS_JIEBA = False

_CJK_BLOCK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[a-z][a-z0-9_+#.-]*|\d+")

# 检索停用词：只保留有区分度的词
STOPWORDS = frozenset({
    "的", "了", "和", "与", "是", "在", "有", "我", "你", "他", "它", "这", "那",
    "怎么", "如何", "什么", "为什么", "可以", "是否", "一下", "一个", "怎样",
    "请问", "麻烦", "帮", "帮我", "想要", "应该", "需要", "问题", "时候",
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "for", "in", "on",
})

# 领域同义词：把学生口语扩展到题库用词（只作用于查询侧）
SYNONYMS = {
    "tle": "超时 时间复杂度 优化",
    "wa": "答案错误 边界 调试",
    "re": "运行错误 数组越界 递归",
    "mle": "内存超限 空间复杂度",
    "ce": "编译错误 语法",
    "板子": "模板 代码框架",
    "套题": "训练 题单",
    "不会": "思路 讲解",
    "卡住": "思路 提示",
    "为什么": "原因 分析",
    "快读": "输入优化 常数",
    "对拍": "随机数据 验证",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _cjk_bigrams(chunk: str) -> list[str]:
    if len(chunk) == 1:
        return [chunk]
    return [chunk[i:i + 2] for i in range(len(chunk) - 1)]


def tokenize(text: str) -> list[str]:
    """返回检索词序列（已小写、去停用词）。"""
    text = normalize(text).lower()
    if not text:
        return []
    tokens: list[str] = []

    if _HAS_JIEBA:
        for piece in jieba.lcut(text):  # type: ignore[attr-defined]
            piece = piece.strip()
            if not piece:
                continue
            if _CJK_BLOCK.fullmatch(piece):
                if len(piece) > 1 and piece not in STOPWORDS:
                    tokens.append(piece)
                elif len(piece) == 1 and piece not in STOPWORDS:
                    tokens.append(piece)
            else:
                tokens.extend(t for t in _LATIN.findall(piece) if t not in STOPWORDS)
    else:
        for chunk in _CJK_BLOCK.findall(text):
            tokens.extend(t for t in _cjk_bigrams(chunk) if t not in STOPWORDS)
        tokens.extend(t for t in _LATIN.findall(text) if t not in STOPWORDS)

    # 英文缩写统一小写已在上方完成；过滤纯标点
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


def expand_query(text: str) -> str:
    """把口语缩写扩成标准术语，提升召回的稳定性。"""
    lowered = normalize(text).lower()
    extra = [syn for key, syn in SYNONYMS.items() if key in lowered]
    if not extra:
        return text
    return f"{text} {' '.join(extra)}"


def term_freq(tokens: list[str]) -> Counter:
    return Counter(tokens)


def jieba_enabled() -> bool:
    return _HAS_JIEBA
