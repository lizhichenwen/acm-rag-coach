# -*- coding: utf-8 -*-
"""对话编排回归测试：代码读不全 / 意图误判问题。

针对线上现象："你能否看懂我的代码"被判成闲聊，且历史里的代码被 [:1500]
从头部截断，导致模型回答"我没看完"。本测试锁死修复后的行为。
"""
from __future__ import annotations

from acm_coach.services.chat_service import (
    INTENT_GUIDANCE,
    SYSTEM_PROMPT,
    build_messages,
    keep_history_content,
)
from acm_coach.services.intent import classify_intent


class TestCodeIntent:
    """代码类追问必须落到 algorithm，不能落到 chitchat。"""

    def test_code_review_questions_are_algorithm(self):
        for q in (
            "你能否看懂我的代码",
            "帮我看看这段代码对不对",
            "这段代码复杂度是多少",
            "我刚刚做了一道思维题，代码是对的",
        ):
            assert classify_intent(q) == "algorithm", q

    def test_code_followup_with_tech_terms_is_algorithm(self):
        # 截图里的原话：针对刚才那份代码的反驳
        q = ("如果是刚才那份，我确实看过开头——gcd 写出来但没调用，"
             "后面逻辑和题目条件对不上，你说的是错的，考虑思维题我这是对的")
        assert classify_intent(q) == "algorithm"

    def test_other_intents_unchanged(self):
        assert classify_intent("直接给我代码") == "code_request"
        assert classify_intent("你好") == "greeting"
        assert classify_intent("训练计划怎么安排") == "strategy"
        assert classify_intent("01背包内层循环为什么倒序") == "algorithm"
        assert classify_intent("谢谢") == "greeting"


class TestHistoryFidelity:
    """历史消息里的代码不能被截断丢弃。"""

    SAMPLE_CODE = "\n".join([
        "#include <bits/stdc++.h>",
        "using namespace std;",
        "void solve(){",
        "int n,cnt=1,ans=1;cin>>n;",
        "vector<int> a(n),b(n+1);",
        "for(int i=0;i<n;i++) cin>>a[i];",
        "sort(a.begin(),a.end());",
        "for(int i=1;i<n;i++){",
        "if(a[i]!=a[i-1]){cnt++;ans=1;}else ans++;",
        "}",
        "cout<<cnt<<'\\n';",
        "}",
    ])

    def test_short_message_kept_verbatim(self):
        msg = "你好，这是短消息"
        assert keep_history_content(msg) == msg

    def test_long_code_message_keeps_whole_code(self):
        title = "小T和她的朋友小U在看天文台看星星。" * 220
        msg = f"{title}\n{self.SAMPLE_CODE}"
        out = keep_history_content(msg)
        assert "void solve()" in out
        assert "sort(a.begin(),a.end())" in out
        assert "cout<<cnt" in out
        assert "【学生代码】" in out
        assert len(out) < len(msg)  # 确实压缩了冗余题面
        assert len(out) <= 6000 + 1400  # 代码上限 + 自然语言上下尾

    def test_long_text_message_keeps_tail(self):
        msg = "开头的话" + ("正文" * 2000) + "【结论在末尾】"
        out = keep_history_content(msg)
        assert "【结论在末尾】" in out
        assert len(out) <= 3000

    def test_build_messages_keeps_code_in_history(self):
        title = "小T和她的朋友小U在看天文台看星星。" * 220
        history = [
            {"role": "user", "content": f"{title}\n{self.SAMPLE_CODE}"},
            {"role": "assistant", "content": "我先确认一下你卡在哪一步"},
        ]
        messages = build_messages(history, "你能否看懂我的代码", [], "algorithm", "none")
        history_user = [m for m in messages if m["role"] == "user"][-1]
        # 当前轮问题本身在最后一个 user block，历史代码在更早的 user message
        assert any("void solve()" in m["content"] for m in messages), "历史代码必须送达模型"
        assert "【学生本次问题】" in history_user["content"]


class TestPromptContract:
    """提示词必须明确允许代码分析，且不再一刀切禁止读题。"""

    def test_prompt_has_code_analysis_exception(self):
        assert "代码分析例外" in SYSTEM_PROMPT
        assert "学生提供的材料" in SYSTEM_PROMPT
        assert "我没看完代码" in SYSTEM_PROMPT

    def test_chitchat_no_longer_bans_topic(self):
        assert "不得回避" in INTENT_GUIDANCE["chitchat"]

    def test_algorithm_guidance_mentions_code(self):
        assert "代码" in INTENT_GUIDANCE["algorithm"]
