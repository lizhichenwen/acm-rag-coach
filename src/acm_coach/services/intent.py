# -*- coding: utf-8 -*-
"""意图识别（规则版）。

为什么规则而不是分类模型：
1. 意图集合小且边界清晰，规则可达 95%+ 准确率，且完全可解释；
2. 零延迟、零成本，答辩现场可断言"这一层不依赖任何外部服务"；
3. 误判代价低——意图只影响提示词策略，不影响检索（检索对所有非问候意图都执行）。

五类意图：greeting / chitchat / code_request / algorithm / strategy
"""
from __future__ import annotations

import re

GREETINGS = ("你好", "您好", "hi", "hello", "在吗", "在不在", "早上好", "晚上好",
             "谢谢", "多谢", "感谢", "再见", "拜拜")

CODE_REQUEST_PATTERNS = (
    "直接给我代码", "给我完整代码", "写出完整代码", "帮我写代码", "代码发我", "直接写",
    "帮我写", "完整代码", "答案给我", "直接给答案", "抄一下", "write the code",
    "give me the code", "把代码给我",
)

ALGORITHM_KEYWORDS = (
    "tle", "超时", "复杂度", "时间复杂度", "空间复杂度", "怎么优化", "为什么错", "wa", "re",
    "mle", "思路", "题解", "算法", "数据结构", "动态规划", "背包", "二分", "贪心", "并查集",
    "图论", "最短路", "拓扑", "哈希", "双指针", "滑动窗口", "单调栈", "单调队列", "前缀和",
    "线段树", "树状数组", "递归", "回溯", "dfs", "bfs", "排序", "字符串", "数组", "链表",
    "栈", "队列", "acm", "竞赛", "蓝桥", "leetcode", "力扣", "网络流", "后缀数组", "2-sat",
    "博弈", "概率", "构造", "交互题", "fft", "ntt", "树链剖分", "莫队", "分块",
)

STRATEGY_KEYWORDS = (
    "训练计划", "怎么练", "如何提升", "上分", "rating", "icpc", "ccpc", "赛制", "罚时",
    "对拍", "补题", "复盘", "队伍分工", "训练路线", "板子", "模板怎么整理", "比赛策略",
    "区域赛", "省赛", "蓝桥杯", "时间分配", "读题",
)

# 代码复盘类：学生把自己写好的代码贴出来，问它"对不对/能不能看懂/复杂度多少"。
# 这类问题必须落到 algorithm（而非 chitchat），否则提示词会禁止模型读题、读代码。
CODE_REVIEW_PATTERNS = (
    r"(看|读|检查|分析|讲|解释|看懂|读懂|看完).{0,6}(代码|这段|上面|刚才)",
    r"(代码|这段|上面|刚才).{0,8}(对不对|对吗|有没有问题|哪里错|什么意思|怎么理解|能看懂|看不懂)",
    r"(复杂度|时间复杂度|空间复杂度).{0,6}(多少|是|怎么算|对吗)",
    r"我(刚|写|做|贴|AC|ac)(完|的|了)?.{0,6}(代码|题|思维题)",
    # "刚才那份/上面那段"＋技术词：针对已有代码的反驳或追问
    r"(刚才|上面|这份|那段).{0,20}(逻辑|条件|开头|调用|写法|gcd|lcm|复杂度|sort|排序)",
)

ALGORITHM_PATTERNS = (
    r"为什么(会)?(错|超时|wa|re)", r"怎么(写|做|优化|改)", r"如何(写|做|优化|分析)",
    r"(这|那)道?题", r"(求|计算).*(最大|最小|个数|方案|路径)", r"(n|N)\s*\*\s*(log|logN)",
)

INTENT_LABELS = {
    "greeting": "问候/自我介绍",
    "chitchat": "闲聊/学习方法",
    "code_request": "索要完整代码",
    "algorithm": "算法题目/知识点",
    "strategy": "竞赛策略/训练规划",
}


def classify_intent(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "chitchat"
    low = t.lower()

    if any(p in low for p in CODE_REQUEST_PATTERNS):
        return "code_request"
    if len(t) <= 12 and any(g in low for g in GREETINGS):
        return "greeting"
    if len(t) <= 12 and any(g in low for g in ("你是谁", "你叫什么", "介绍一下你", "能做什么", "有什么功能")):
        return "greeting"
    if any(k in low for k in STRATEGY_KEYWORDS):
        return "strategy"
    if any(k in low for k in ALGORITHM_KEYWORDS):
        return "algorithm"
    for pattern in CODE_REVIEW_PATTERNS:
        if re.search(pattern, t, flags=re.IGNORECASE):
            return "algorithm"
    for pattern in ALGORITHM_PATTERNS:
        if re.search(pattern, t, flags=re.IGNORECASE):
            return "algorithm"
    # 兜底：出现"代码/这段"却没匹配上面任何模式，也视作算法追问而非闲聊
    if any(k in low for k in ("代码", "这段", "上面那段", "刚才那段")):
        return "algorithm"
    return "chitchat"
