# -*- coding: utf-8 -*-
"""对话编排（应用服务层）。

一次提问的完整链路：
    意图识别 → 查询构造（去代码噪声/补话题锚点/同义扩展）→ 混合检索
    → 上下文打包 → 提示编排 → 流式生成 → 落库与埋点

这层的职责是"编排"，不含检索算法、不含 SQL、不含 HTTP：
换向量库、换模型、换前端都不需要改它。
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator

from sqlalchemy.orm import Session

from acm_coach.config import get_settings
from acm_coach.core import metrics
from acm_coach.core.logging import get_logger, get_trace_id
from acm_coach.core.security import fingerprint_query
from acm_coach.db.models import ChatSession, User
from acm_coach.db.repositories import ChatRepository, OpsRepository
from acm_coach.rag.index import RetrievalResult, get_retriever
from acm_coach.rag.packer import pack
from acm_coach.rag.query import REWRITE_PROMPT, build_query, split_code
from acm_coach.services.intent import INTENT_LABELS, classify_intent
from acm_coach.services.llm import LLMClient, extractive_answer

log = get_logger("chat")

SYSTEM_PROMPT = """你是一名严格的 ACM 程序设计竞赛教练，面向本科生。

教学原则：
1. 引导优先：不直接给出可提交的完整代码，用提问、提示、反例引导学生自己想出解法。
2. 忠于资料：具体题目的解法只依据【参考资料】；资料没有的题目，明确说"题库未收录"，
   不得编造题解、复杂度或结论。
3. 来源分级：【精讲】条目可以作为题解依据；标为【考点提示】的条目只描述考查方向与
   典型套路，不得据此推断具体复杂度、数据结构选择或代码细节。
4. 可溯源：引用资料时用 [1]、[2] 标注对应编号。
5. 一次只问一个关键问题，等学生回应后再继续。
6. 语气严厉但耐心：先肯定做对的部分，再指出问题。

代码分析例外（优先级高于原则 2 与原则 5）：
- 当学生在对话中贴出自己的代码，或因代码提问（例如"这段代码对不对""复杂度多少"
  "帮我看看这段"），这段学生代码属于学生提供的材料，不是题库资料。你可以、也应当
  就代码本身逐段分析：它做了什么、逻辑是否正确、复杂度是多少、为什么能过样例。
  不得因为【参考资料】未收录该题就拒绝分析学生给出的代码，也不得说"我没看完代码"。
- 分析结论必须基于你实际看到的代码；如果本轮上下文里没有代码，才说明看不到并请学生重贴。
- 只有在"给出这道题的标准解法/官方复杂度"时才受原则 2 约束。代码分析不受此限。
- 回答代码分析类问题时先给结论，再展开依据；引导性提问放在结论之后，不超过一个。"""

INTENT_GUIDANCE = {
    "greeting": "【意图】问候/自我介绍。正常打招呼，说明你能做什么（讲解思路、分析错因、规划训练），不要硬套题目。",
    "chitchat": (
        "【意图】学习方法/闲聊。直接给建议，不主动引入具体题目；"
        "但若本轮消息里贴了代码或提到刚才的代码，按代码分析处理，不得回避。"
    ),
    "code_request": (
        "【意图】学生索要完整代码。必须拒绝直接给代码：先指出关键步骤，再给伪代码骨架或关键片段，"
        "最后用一个问题引导学生补全。"
    ),
    "algorithm": (
        "【意图】算法题目/知识点。若本轮或历史里贴了代码、或学生在问自己的代码，"
        "先就代码给出具体分析与结论（逻辑、正误、复杂度），再引导；"
        "纯思路题才按引导式教学：先确认卡在哪一步，再给提示，最后提问。"
    ),
    "strategy": "【意图】竞赛策略/训练规划。给可执行的计划（频次、题量、难度区间、复盘方式），不要泛泛而谈。",
}


def keep_history_content(content: str) -> str:
    """代码感知的历史消息保真。

    旧实现 ``content[:1500]`` 会从头截断：题面一长，代码整段就被切掉，多轮追问时
    模型只能回答"我没看完"。改为：
    - 短消息原样保留；
    - 纯文本长消息只保留尾部（对话结论在尾部）；
    - 含代码的消息拆成自然语言与代码两段，代码按上限完整保留，自然语言保留头部
      与尾部，中间的冗余描述丢弃。

    返回文本永远不超过 config 中的对应上限。
    """
    settings = get_settings()
    text_cap = settings.history_text_cap
    if len(content) <= text_cap:
        return content

    nl, code = split_code(content)
    if not code.strip():
        return content[-text_cap:]

    code_cap = settings.history_code_cap
    nl_keep = settings.history_nl_keep
    kept_code = code[:code_cap]
    if len(nl) <= 2 * nl_keep:
        kept_nl = nl
    else:
        kept_nl = f"{nl[:nl_keep]}\n…（中间省略）…\n{nl[-nl_keep:]}"
    return f"{kept_nl}\n\n【学生代码】\n{kept_code}".strip()


def format_sources(hits) -> str:
    if not hits:
        return "（题库未检索到相关内容）"
    blocks = []
    for idx, hit in enumerate(hits, 1):
        label = "精讲" if hit.source == "curated" else "考点提示"
        meta = f"难度：{hit.difficulty or '未标注'}；标签：{hit.tags or '未标注'}；类型：{label}"
        if hit.url:
            meta += f"；链接：{hit.url}"
        blocks.append(f"[{idx}] {hit.title}（{meta}）\n{hit.content[:900]}")
    return "\n\n".join(blocks)


def build_messages(history: list[dict], question: str, hits, intent: str,
                   level: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_history = max(0, get_settings().history_turns) * 2
    for msg in (history[-max_history:] if history else []):
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"],
                             "content": keep_history_content(msg["content"])})

    guidance = INTENT_GUIDANCE.get(intent, INTENT_GUIDANCE["algorithm"])
    if level == "none":
        constraint = ("【检索结果】题库中未检索到相关条目。若问的是这道题的标准解法，须先声明"
                      "“题库未收录”，再给方向性提示，不得虚构题解；"
                      "若学生贴了代码或问自己的代码，直接就代码本身分析，无需声明未收录。")
    elif level == "weak":
        constraint = "【检索结果】仅有弱相关条目，回答时须提示“以下内容与你的问题相关度有限”。"
    else:
        constraint = "【检索结果】以下为强相关条目，可直接作为回答依据。"

    user_block = (
        f"{guidance}\n\n{constraint}\n\n"
        f"【参考资料】\n{format_sources(hits)}\n\n"
        f"【学生本次问题】\n{question}"
    )
    messages.append({"role": "user", "content": user_block})
    return messages


class ChatService:
    def __init__(self, db: Session, llm: LLMClient | None = None):
        self.db = db
        self.chat = ChatRepository(db)
        self.ops = OpsRepository(db)
        self.llm = llm or LLMClient()
        self.retriever = get_retriever()

    # ---------------- 主流程 ----------------
    def stream(self, user: User, session: ChatSession, prompt: str) -> Iterator[dict]:
        """产出事件流：meta → token* → done。"""
        started = time.perf_counter()
        intent = classify_intent(prompt)
        history_rows = self.chat.recent_messages(session.id)
        history = [{"role": m.role, "content": m.content} for m in history_rows]
        last_titles = self.chat.last_matched_titles(session.id)

        qinfo = build_query(prompt, history, last_titles)
        rewritten = self._maybe_rewrite(prompt, history, last_titles, qinfo)
        queries = ([rewritten] + qinfo["queries"]) if rewritten else qinfo["queries"]
        query_text = rewritten or qinfo["query"]
        mode = "llm" if rewritten else qinfo["mode"]

        if intent == "greeting":
            result = RetrievalResult(hits=[], level="none", query=query_text, mode=mode)
        else:
            with metrics.timer("retrieval"):
                result = self.retriever.search(query_text, queries=queries, mode=mode)

        hits, packed_text = pack(result.hits, get_settings().context_char_budget)
        # pack 可能因预算丢弃条目，保持引用编号与实际上文一致
        messages = build_messages(history, prompt, hits, intent, result.level)

        self.chat.add_message(session.id, "user", prompt, intent=intent)
        yield {
            "event": "meta",
            "data": {
                "intent": intent,
                "intent_label": INTENT_LABELS.get(intent, intent),
                "level": result.level,
                "query": query_text,
                "mode": mode,
                "strategy": result.strategy,
                "cost_ms": result.cost_ms,
                "sources": [h.to_source() for h in hits],
            },
        }

        chunks: list[str] = []
        if self.llm.available:
            gen = self.llm.stream(messages)
        else:
            gen = extractive_answer([{**h.to_source(), "content": h.content} for h in hits],
                                    prompt, result.level)

        with metrics.timer("generation"):
            for delta in gen:
                chunks.append(delta)
                yield {"event": "token", "data": delta}

        answer = "".join(chunks).strip()
        latency = int((time.perf_counter() - started) * 1000)

        msg = self.chat.add_message(
            session.id, "assistant", answer,
            matched=bool(hits), intent=intent, retrieval_query=query_text[:250],
            sources_json=json.dumps([h.to_source() for h in hits], ensure_ascii=False),
            trace_json=json.dumps(result.to_dict(), ensure_ascii=False),
            latency_ms=latency,
        )
        metrics.inc("chat_requests_total", {"level": result.level})
        metrics.inc("llm_tokens_chars_total", value=len(answer))

        self.ops.log_retrieval(
            # 查询原文不落库（PIPL 最小化），只存不可逆指纹，见 core/security.py
            trace_id=get_trace_id(), user_id=user.id, query=fingerprint_query(query_text),
            mode=mode, level=result.level, recall=result.recall, final=len(hits),
            best_score=result.best_similarity, cost_ms=result.cost_ms,
            candidates_json=json.dumps([h.id for h in hits], ensure_ascii=False),
        )
        self.db.commit()

        yield {
            "event": "done",
            "data": {
                "message_id": msg.id, "latency_ms": latency,
                "matched": bool(hits), "level": result.level,
                "sources": [h.to_source() for h in hits],
            },
        }

    def answer(self, user: User, session: ChatSession, prompt: str) -> dict:
        """非流式封装（测试与批量评测用）。"""
        text, meta = [], {}
        for event in self.stream(user, session, prompt):
            if event["event"] == "token":
                text.append(event["data"])
            elif event["event"] == "meta":
                meta = event["data"]
            elif event["event"] == "done":
                meta.update(event["data"])
        return {**meta, "answer": "".join(text)}

    # ---------------- 辅助 ----------------
    def _maybe_rewrite(self, prompt: str, history: list[dict], last_titles: list[str],
                       qinfo: dict) -> str:
        settings = get_settings()
        if settings.query_rewrite == "off" or not self.llm.available:
            return ""
        if settings.query_rewrite != "on" and not qinfo["is_followup"]:
            return ""
        if not qinfo["anchor"]:
            return ""
        try:
            client = self.llm._client  # noqa: SLF001 同包内复用，避免重复建连接
            resp = client.chat.completions.create(
                model=settings.deepseek_model,
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                    anchor=qinfo["anchor"], prompt=prompt[:300])}],
                temperature=0.0, max_tokens=64, timeout=15,
            )
            text = (resp.choices[0].message.content or "").strip().splitlines()
            candidate = text[0].strip().strip("`\"'。 ") if text else ""
            if 2 <= len(candidate) <= 60 and not any(
                    w in candidate for w in ("抱歉", "无法", "作为")):
                return candidate
        except Exception as exc:
            log.debug("查询改写失败，回退规则版：%s", exc)
        return ""
