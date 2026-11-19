# -*- coding: utf-8 -*-
"""大模型访问层。

对外只暴露两个能力：`stream_chat()` 流式生成、`available` 是否可用。
未配置 API Key 时**不报错**，而是走"检索式作答"降级路径：
把召回资料按模板拼成结构化回答，保证系统在任何环境都能演示，也便于自动化测试。
"""
from __future__ import annotations

import re
from collections.abc import Iterator

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger

log = get_logger("llm")

CONTINUE_PROMPT = (
    "你的上一条回复被长度限制截断了。请直接从**中断的那一句**接着往下写，"
    "不要重复已输出内容，不要重新开头或打招呼。"
)


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        if self.settings.deepseek_api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.deepseek_api_key,
                                  base_url=self.settings.deepseek_base_url,
                                  timeout=self.settings.llm_timeout)

    @property
    def available(self) -> bool:
        return self._client is not None

    def stream(self, messages: list[dict], *, max_continue: int | None = None) -> Iterator[str]:
        if self._client is None:
            return
        settings = self.settings
        max_continue = settings.llm_max_continue if max_continue is None else max_continue
        convo = list(messages)
        gathered: list[str] = []
        continues = 0
        pending: list[str] = []
        is_continuing = False

        while True:
            finished = "stop"
            try:
                response = self._client.chat.completions.create(
                    model=settings.deepseek_model, messages=convo, stream=True,
                    temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens,
                    timeout=settings.llm_timeout,
                )
                for chunk in response:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta.content if choice.delta else None
                    if delta:
                        if is_continuing and sum(len(p) for p in pending) < 60:
                            pending.append(delta)
                            continue
                        if pending:
                            delta = _strip_overlap(gathered, pending) + delta
                            pending = []
                        if delta:
                            gathered.append(delta)
                            yield delta
                    if choice.finish_reason:
                        finished = choice.finish_reason
                if pending:
                    delta = _strip_overlap(gathered, pending)
                    pending = []
                    if delta:
                        gathered.append(delta)
                        yield delta
            except Exception as exc:  # 网络/鉴权/限流
                log.error("llm.call_failed type=%s err=%s", type(exc).__name__, exc)
                if not gathered:
                    yield f"\n\n⚠️ 调用大模型失败（{type(exc).__name__}），请检查网络与 API Key。"
                    return
                if continues >= max_continue:
                    yield "\n\n⚠️ 生成中断，回复「继续」可让我接着讲。"
                    return
                continues += 1
                convo = _continue_messages(messages, gathered)
                is_continuing = True
                continue

            if finished != "length":
                return
            if continues >= max_continue:
                yield "\n\n（回答较长，已达单次上限，回复「继续」我接着讲。）"
                return
            continues += 1
            convo = _continue_messages(messages, gathered)
            is_continuing = True


def _continue_messages(messages: list[dict], gathered: list[str]) -> list[dict]:
    return list(messages) + [
        {"role": "assistant", "content": "".join(gathered)[-800:]},
        {"role": "user", "content": CONTINUE_PROMPT},
    ]


def _strip_overlap(gathered: list[str], pending: list[str]) -> str:
    """续写时模型常把上一段结尾重复一遍，按最长重叠裁掉。"""
    text = "".join(pending)
    prev = "".join(gathered)
    for k in range(min(len(text), len(prev), 200), 3, -1):
        if prev.endswith(text[:k]):
            return text[k:]
    return text


# ---------------- 降级：检索式作答 ----------------
def extractive_answer(sources: list[dict], question: str, level: str) -> Iterator[str]:
    """没有大模型时的可演示回答：只摘录、不推理。"""
    if not sources:
        yield ("题库里没有检索到与这个问题相关的内容。\n\n"
               "建议你把题目链接或关键描述贴出来，我可以从考点角度帮你拆解。")
        return

    yield "【未配置大模型，以下为题库检索结果摘录】\n\n"
    for idx, src in enumerate(sources, 1):
        body = src.get("content", "")
        solution = re.search(r"解题思路[:：]\s*(.+?)(?:\n常见错误|$)", body, re.S)
        core = solution.group(1).strip() if solution else body
        yield f"**[{idx}] {src.get('title', '')}**（{src.get('difficulty', '')}｜{src.get('tags', '')}）\n"
        yield f"{core[:500]}\n\n"

    if level == "weak":
        yield "> 注：以上条目与你的问题只有弱相关，仅供参考。\n"
    yield "下一步：先自己想一下这个思路的关键一步是什么，卡在哪里告诉我，我们再往下拆。"
