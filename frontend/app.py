# -*- coding: utf-8 -*-
"""Streamlit 前端（瘦客户端 / 答辩演示版）。

课题：面向 ACM 程序设计竞赛的 RAG 智能陪练系统的设计与实现

【本次改动范围】只改这一个文件，后端 API / 数据库 / 检索 / 配置一律不动。
前端仍然只做三件事：登录拿 token、调 HTTP 接口、渲染页面；
不 import 后端任何模块，不直接碰数据库。

【页面覆盖任务书的 9 项功能】
    题库管理      → 📚 题库管理（检索 / 筛选 / 详情 / 录入 / 下架 / 重建索引）
    检索增强问答  → 💬 智能陪练（SSE 流式 + 引用溯源 + 三档相关性门控）
    意图识别      → 💬 智能陪练（meta 事件展示意图 / 查询改写 / 检索策略）
    多轮对话      → 💬 智能陪练（会话列表 / 新建 / 切换 / 删除 / 历史回放）
    代码诊断      → 🧪 代码诊断（提交 / 错因 / 证据 / 建议 / 静态特征 / 历史）
    知识点画像    → 🎯 知识点画像（掌握度 / 尝试次数 / 错因分布）
    薄弱点推荐    → 📌 薄弱点推荐（掌握度最低 → 题库检索 → 标记已练）
    用户与权限    → 👤 账号与权限（角色矩阵 / 改密 / 数据导出 / 注销）
    数据统计      → 📊 数据统计（提问 / 命中率 / 延迟 / 意图 / 标签 / 题库构成）
    答辩演示      → 🎬 答辩演示指引（完整流程 + 环境自检 + 排障手册）

【关于「问一个问题就报红」】后端日志与数据库都证明问答本身是成功的，
红框来自旧前端的错误处理，本文件逐条修掉：
    1. 旧 stream_chat() 不检查 HTTP 状态码：401/404/422/500 的 JSON 错误体被当成
       SSE 逐行读，轻则静默空白，重则 json.loads 抛异常 → 被兜底成红框。
       现在先判状态码，并把后端统一错误体 {code,message,trace_id,details} 解析出来。
    2. 旧版把 403（学生访问 /ops/overview）显示成「请确认 API 已启动」，
       提示完全误导。现在按状态码分派：401→重新登录、403→角色不足、
       404→会话自动重建、5xx→看 trace_id、仅连接失败才提示服务未启动。
    3. 旧 session_id 缓存在浏览器里，后端换库/删会话后 GET messages 会 404；
       现在自动新建一个会话并继续，不再弹红框。
    4. SSE 解析改为按「空行分事件块」的标准做法，data 非 JSON 时按纯文本兜底，
       不再因为一个异常字符把整轮问答打断。
    5. 后端 event: error 事件被单独标注为「生成阶段错误」，与网络故障区分开。
"""
from __future__ import annotations

import html
import json
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = httpx.Timeout(120.0, connect=10.0)

st.set_page_config(
    page_title="ACM 智能陪练系统",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- 展示用常量（与后端枚举保持一致，仅用于中文标签） ----------------
ROLE_LABELS = {"student": "学生", "admin": "管理员", "auditor": "审计员"}
INTENT_LABELS = {
    "greeting": "问候 / 自我介绍",
    "chitchat": "闲聊 / 学习方法",
    "code_request": "索要完整代码",
    "algorithm": "算法题目 / 知识点",
    "strategy": "竞赛策略 / 训练规划",
}
LEVEL_LABELS = {"strong": "强相关", "weak": "弱相关", "none": "未命中（拒答）"}
VERDICTS = ["WA", "TLE", "RE", "MLE", "CE", "AC", "PE", "OLE"]
DIFFICULTIES = ["简单", "中等", "困难", "未知"]

PAGE_ICONS = {
    "智能陪练": "💬", "代码诊断": "🧪", "知识点画像": "🎯", "薄弱点推荐": "📌",
    "题库管理": "📚", "数据统计": "📊", "账号与权限": "👤", "运行状态": "🩺",
    "答辩演示指引": "🎬",
}
PAGE_DESC = {
    "智能陪练": "检索增强问答 · 意图识别 · 多轮对话 · 引用溯源",
    "代码诊断": "静态特征 + 大模型结构化判定，结论可解释可追溯",
    "知识点画像": "基于诊断记录的知识点掌握度建模",
    "薄弱点推荐": "按掌握度排序，检索薄弱知识点对应练习题",
    "题库管理": "8,558 条语料的浏览、筛选、录入与索引维护",
    "数据统计": "训练量、命中率、延迟、意图与知识点分布",
    "账号与权限": "RBAC 角色、改密、个人数据导出与注销（PIPL）",
    "运行状态": "健康检查、检索参数、抓取记录与审计日志（仅管理员/审计员）",
    "答辩演示指引": "完整演示流程、环境自检与常见报错排查",
}

QUICK_PROMPTS = [
    ("🔍 算法题引导", "01 背包内层循环为什么必须倒序？"),
    ("🙅 索要代码（看拒答）", "直接给我 01 背包的完整代码"),
    ("🔁 多轮追问", "那用一维数组怎么写？"),
    ("📅 训练规划", "帮我定一份四周的 DP 训练计划"),
]

SAMPLE_CODE = """#include <bits/stdc++.h>
using namespace std;
int main() {
    int n; cin >> n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                cout << i * j * k << endl;
    return 0;
}
"""


# ============================================================
# 样式：浅色、克制、适合投屏答辩
# ============================================================
def inject_style() -> None:
    st.markdown(
        """
<style>
  .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
  [data-testid="stSidebar"] { background: #f6f8fc; border-right: 1px solid #e3e8f2; }
  [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
  .acm-hero {
      background: linear-gradient(115deg, #1e3a8a 0%, #2563eb 55%, #3b82f6 100%);
      color: #fff; padding: 26px 30px; border-radius: 16px; margin-bottom: 20px;
      box-shadow: 0 8px 24px rgba(30,58,138,.18);
  }
  .acm-hero h1 { color:#fff; font-size: 28px; margin: 0 0 6px 0; letter-spacing:.5px; }
  .acm-hero .sub { font-size: 14px; opacity: .92; margin-bottom: 14px; }
  .acm-hero .meta { font-size: 12.5px; opacity: .82; }
  .acm-chip {
      display:inline-block; background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
      color:#fff; padding: 3px 10px; border-radius: 999px; font-size: 12px; margin: 3px 5px 3px 0;
  }
  .acm-page-head { border-bottom: 1px solid #e6ebf4; padding-bottom: 10px; margin-bottom: 16px; }
  .acm-page-head h2 { margin:0; font-size: 22px; color:#16233d; }
  .acm-page-head p { margin:5px 0 0 0; color:#64748b; font-size: 13px; }
  .acm-user {
      background:#fff; border:1px solid #e3e8f2; border-radius:12px;
      padding:12px 14px; margin-bottom:14px;
  }
  .acm-user .name { font-weight:700; color:#16233d; font-size:15px; }
  .acm-user .role { font-size:12px; color:#2563eb; background:#eff6ff;
      padding:2px 9px; border-radius:999px; margin-left:6px; }
  .acm-note {
      background:#f8fafc; border-left:3px solid #2563eb; padding:10px 14px;
      border-radius:0 8px 8px 0; color:#475569; font-size:13px; margin:10px 0;
  }
  .acm-kpi { text-align:center; padding:6px 4px; }
  .acm-kpi .v { font-size:24px; font-weight:700; color:#1e3a8a; line-height:1.2; }
  .acm-kpi .l { font-size:12px; color:#64748b; margin-top:2px; }
  .acm-src { border:1px solid #e8edf5; border-radius:10px; padding:10px 13px;
      background:#fff; margin-bottom:8px; }
  .acm-tag { display:inline-block; font-size:11px; padding:1px 8px; border-radius:999px;
      background:#eef2ff; color:#3730a3; margin-right:6px; }
  .acm-tag.warn { background:#fff7ed; color:#9a3412; }
  .acm-tag.ok { background:#ecfdf5; color:#047857; }
  .acm-tag.gray { background:#f1f5f9; color:#475569; }
  .acm-step { border:1px solid #e6ebf4; border-radius:12px; padding:12px 15px;
      margin-bottom:9px; background:#fff; }
  .acm-step .t { font-weight:700; color:#1e3a8a; font-size:14px; }
  .acm-step .d { color:#475569; font-size:13px; margin-top:4px; }
  .acm-step .m { color:#94a3b8; font-size:12px; margin-top:5px; }
  h1, h2, h3 { color:#16233d; }
  .stChatMessage { border-radius: 12px; }
</style>
""",
        unsafe_allow_html=True,
    )


def esc(value: Any) -> str:
    """拼进 unsafe_allow_html 的 HTML 时，用户输入必须转义。"""
    return html.escape(str(value if value is not None else ""), quote=True)


def safe_md(text: str) -> str:
    """把代码块之外的裸尖括号转义成实体，再交给 markdown 渲染。

    模型回答里常有 ``<bits/stdc++.h>``、``<n>`` 这类尖括号。markdown 会把代码块外的
    ``<...>`` 当 HTML 标签解析，生成不完整/嵌套异常的 DOM；流式重绘或历史回显时
    前端 React 卸载节点就会报 ``removeChild: 节点不是该节点的子节点``。
    只处理代码块（```）与行内代码（`）之外的内容，代码本身原样保留。
    """
    if not text:
        return ""
    blocks = str(text).split("```")
    for i in range(0, len(blocks), 2):  # 偶数下标：代码块之外
        inline = blocks[i].split("`")
        for j in range(0, len(inline), 2):  # 偶数下标：行内代码之外
            inline[j] = inline[j].replace("<", "&lt;").replace(">", "&gt;")
        blocks[i] = "`".join(inline)
    return "```".join(blocks)


def page_header(name: str) -> None:
    st.markdown(
        f'<div class="acm-page-head"><h2>{PAGE_ICONS[name]}  {esc(name)}</h2>'
        f'<p>{esc(PAGE_DESC[name])}</p></div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="acm-note">{esc(text)}</div>', unsafe_allow_html=True)


def kpi(col, value: Any, label: str) -> None:
    col.markdown(
        f'<div class="acm-kpi"><div class="v">{esc(value)}</div>'
        f'<div class="l">{esc(label)}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# API 客户端：统一错误体 + 精确的状态码语义
# ============================================================
class ApiError(Exception):
    """把后端的 {code,message,trace_id,details} 原样带到前端，方便定位问题。"""

    def __init__(self, status: int, message: str, code: str = "",
                 trace_id: str = "", details: Any = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.trace_id = trace_id
        self.details = details


def _parse_error_body(raw: str | bytes, status: int) -> ApiError:
    try:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    except Exception:
        text = str(raw)
    try:
        body = json.loads(text)
    except (ValueError, TypeError):
        body = {}
    code, trace_id, details, message = "", "", None, ""
    if isinstance(body, dict):
        code = str(body.get("code") or "")
        trace_id = str(body.get("trace_id") or "")
        details = body.get("details")
        message = body.get("message") or body.get("detail") or text
    elif body:
        message = str(body)
    else:
        message = text
    if not isinstance(message, str):
        message = json.dumps(message, ensure_ascii=False)
    return ApiError(status, message.strip()[:500] or f"HTTP {status}", code, trace_id, details)


def api(method: str, path: str, **kwargs) -> Any:
    """普通 JSON 请求。所有失败都抛 ApiError，由 show_api_error 统一解释。"""
    headers = dict(kwargs.pop("headers", {}) or {})
    if st.session_state.get("token") and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        with httpx.Client(base_url=API_BASE, timeout=TIMEOUT) as client:
            resp = client.request(method, path, headers=headers, **kwargs)
    except httpx.RequestError as exc:
        raise ApiError(0, f"{type(exc).__name__}: {exc}", code="network_error") from exc
    if resp.status_code >= 400:
        raise _parse_error_body(resp.text, resp.status_code)
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError:
        return resp.text


def _parse_sse_data(raw: str) -> Any:
    """后端 token 事件是 JSON 字符串，meta/done 是 JSON 对象；容错成纯文本。"""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def stream_chat(session_id: int, prompt: str):
    """SSE 逐事件产出 (event, data)。

    与旧版的两点关键差别：
      - 先检查 HTTP 状态码，非 2xx 直接还原成 ApiError，不再把错误 JSON 当事件流读；
      - 按 SSE 规范用「空行」划分事件块，而不是见到 data: 就 yield。
    """
    headers = {
        "Authorization": f"Bearer {st.session_state.token}",
        "Accept": "text/event-stream",
    }
    try:
        with httpx.Client(base_url=API_BASE, timeout=None) as client:
            with client.stream(
                "POST", f"/api/v1/sessions/{session_id}/messages",
                json={"prompt": prompt, "stream": True}, headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    raw = b"".join(resp.iter_bytes())
                    raise _parse_error_body(raw, resp.status_code)

                event: str | None = None
                buf: list[str] = []

                def flush():
                    if event and buf:
                        return (event, _parse_sse_data("\n".join(buf)))
                    return None

                for line in resp.iter_lines():
                    if not line:
                        item = flush()
                        event, buf = None, []
                        if item:
                            yield item
                        continue
                    if line.startswith(":"):  # SSE 注释/心跳
                        continue
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        buf.append(line.split(":", 1)[1].lstrip())
                item = flush()  # 流末尾没有空行时兜底
                if item:
                    yield item
    except httpx.RequestError as exc:
        raise ApiError(0, f"{type(exc).__name__}: {exc}", code="network_error") from exc


def logout() -> None:
    for key in ("token", "user", "session_id", "last_trace", "diag_result",
                "diag_error", "quick_prompt"):
        st.session_state.pop(key, None)
    st.rerun()


def show_api_error(exc: ApiError) -> None:
    """按状态码给出「人能照着做」的提示，不再一律「请确认 API 已启动」。"""
    role = (st.session_state.get("user") or {}).get("role", "未知")
    if exc.status == 0:
        st.error("🔌 连不上后端服务（网络层失败）")
        st.markdown(
            f"请求根本没有到达 `{API_BASE}`。请在项目目录执行：\n\n"
            "```bash\nvenv\\Scripts\\activate\npython -m acm_coach.cli serve\n```\n"
            "另开一个窗口执行 `streamlit run frontend/app.py`。"
        )
    elif exc.status == 401:
        st.error("🔑 登录状态已失效，需要重新登录")
        st.markdown(
            "JWT 有效期 120 分钟；改密、账号禁用或换库后旧令牌也会立即失效。"
            "**这不是后端没启动，也不是问答失败。**"
        )
        if st.button("重新登录", key="err_relogin", type="primary"):
            logout()
    elif exc.status == 403:
        st.warning("🔒 当前账号角色无权访问该资源")
        st.markdown(
            f"当前角色：`{esc(role)}`。该接口要求 `admin` 或 `auditor`。"
            "学生账号请改用「智能陪练 / 题库管理 / 数据统计」等页面；"
            "管理员操作请用 admin 账号登录。**这是权限设计生效，不是故障。**"
        )
    elif exc.status == 404:
        st.error("🔍 资源不存在或无权可见")
        st.markdown(
            "后端对「不存在」与「他人资源」统一返回 404（防枚举）。"
            "若是会话问题，智能陪练页会自动新建会话；若是条目 id，请回列表重新选择。"
        )
    elif exc.status == 409:
        st.warning("⚠️ 数据冲突：" + exc.message)
    elif exc.status == 422:
        st.error("📝 参数校验未通过：" + exc.message)
        st.markdown("检查输入长度、必填项与枚举取值（例如判题结果只能是 AC/WA/TLE/…）。")
    elif exc.status == 429:
        st.warning("⏳ 操作过于频繁：" + exc.message)
        st.markdown("同一账号连续 5 次登录失败会锁定 5 分钟，稍后再试即可。")
    elif exc.status >= 500:
        st.error("🛠 后端内部错误：" + exc.message)
        st.markdown(
            "请把下面的 trace_id 拿到 `data/logs/app.log` 里检索，可以定位到具体请求。"
            "若 /readyz 报 bm25=false，说明索引未构建。"
        )
    else:
        st.error(f"请求失败（HTTP {exc.status}）：{exc.message}")

    with st.expander("技术细节（答辩排障用）", expanded=False):
        st.json({
            "http_status": exc.status,
            "code": exc.code,
            "message": exc.message,
            "trace_id": exc.trace_id,
            "details": exc.details,
            "api_base": API_BASE,
            "role": role,
        })


# ============================================================
# 登录页
# ============================================================
HERO_HTML = """
<div class="acm-hero">
  <h1>🏆 ACM 智能陪练系统</h1>
  <div class="sub">面向 ACM 程序设计竞赛的 RAG 引导式陪练  · v4 分层架构版</div>
  <div>
    <span class="acm-chip">题库管理</span><span class="acm-chip">检索增强问答</span>
    <span class="acm-chip">意图识别</span><span class="acm-chip">多轮对话</span>
    <span class="acm-chip">代码诊断</span><span class="acm-chip">知识点画像</span>
    <span class="acm-chip">薄弱点推荐</span><span class="acm-chip">用户与权限</span>
    <span class="acm-chip">数据统计</span>
  </div>
  <div class="meta" style="margin-top:12px">
    课题：面向 ACM 程序设计竞赛的 RAG 智能陪练系统的设计与实现 
  </div>
</div>
"""


def render_login() -> None:
    st.markdown(HERO_HTML, unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("登录")
            username = st.text_input("用户名", key="login_user", placeholder="admin 或你的学生账号")
            password = st.text_input("口令", type="password", key="login_pwd")
            if st.button("登录", type="primary", width="stretch"):
                try:
                    data = api("POST", "/api/v1/auth/login",
                               json={"username": username, "password": password})
                    st.session_state.token = data["access_token"]
                    st.session_state.user = data["user"]
                    st.rerun()
                except ApiError as exc:
                    # 登录接口的 401 是「账号或口令错误」，不要套用「登录过期」话术
                    st.error(f"登录失败：{exc.message}")
                    if exc.status == 429:
                        st.caption("同账号连续 5 次失败会锁定 5 分钟。")
    with right:
        with st.container(border=True):
            st.subheader("注册学生账号")
            new_user = st.text_input("用户名", key="reg_user", placeholder="3–32 位，字母数字 _ -")
            new_pwd = st.text_input("口令（≥8 位，需含字母和数字）", type="password", key="reg_pwd")
            if st.button("注册", width="stretch"):
                try:
                    api("POST", "/api/v1/auth/register",
                        json={"username": new_user, "password": new_pwd})
                    st.success("注册成功，请在左侧登录（新账号默认角色 student）")
                except ApiError as exc:
                    st.error(f"注册失败：{exc.message}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**引导式**")
        st.caption("不给完整可提交代码，只给关键步骤与反问")
    with c2:
        st.markdown("**可溯源**")
        st.caption("回答标注 [1][2]，区分「精讲」与「考点提示」")
    with c3:
        st.markdown("**敢拒答**")
        st.caption("题库未命中时明确说明，不编造题解")
    note("演示提示：学生账号可完整走训练流程；题库录入、索引重建、运行状态页需要 admin 角色。"
         "首次管理员口令见 .env 的 ADMIN_PASSWORD，未配置时启动日志会打印一次性随机口令。")


# ============================================================
# 通用组件
# ============================================================
def ensure_session() -> int:
    if st.session_state.get("session_id"):
        return st.session_state.session_id
    data = api("POST", "/api/v1/sessions", json={"title": "新会话"})
    st.session_state.session_id = data["id"]
    return data["id"]


def is_admin() -> bool:
    return (st.session_state.get("user") or {}).get("role") == "admin"


def is_staff() -> bool:
    return (st.session_state.get("user") or {}).get("role") in ("admin", "auditor")


def render_sources(sources: list[dict]) -> None:
    """引用来源：精讲 / 考点提示分级展示，对应论文「可溯源」契约。"""
    if not sources:
        st.caption("本轮未召回题库条目")
        return
    for i, src in enumerate(sources, 1):
        curated = src.get("source") in ("curated", "manual")
        tag = "精讲" if curated else "考点提示"
        cls = "ok" if curated else "warn"
        score = src.get("similarity", src.get("score", 0))
        st.markdown(
            f'<div class="acm-src"><span class="acm-tag {cls}">{esc(tag)}</span>'
            f'**[{i}] {esc(src.get("title", ""))}** '
            f'<span class="acm-tag gray">{esc(src.get("difficulty") or "未标注难度")}</span>'
            f'<span class="acm-tag gray">相关度 {esc(score)}</span><br>'
            f'<span style="font-size:12px;color:#64748b">标签：{esc(src.get("tags") or "未标注")}'
            f' · 域：{esc(src.get("domain") or "未标注")}</span></div>',
            unsafe_allow_html=True,
        )
        if src.get("url"):
            st.caption(f"原题链接：{src['url']}")


def render_trace(meta: dict) -> None:
    """把 SSE meta/done 事件摊开：意图识别 → 查询改写 → 混合检索 → 生成。"""
    if not meta:
        return
    with st.container(border=True):
        st.markdown("**🧠 本次推理链路**（意图识别 → 查询构造 → 混合检索 → 引导式生成）")
        c1, c2, c3, c4 = st.columns(4)
        intent = meta.get("intent_label") or INTENT_LABELS.get(meta.get("intent", ""), meta.get("intent", "-"))
        kpi(c1, intent, "意图识别")
        kpi(c2, LEVEL_LABELS.get(meta.get("level", ""), "-"), "检索相关性档位")
        kpi(c3, meta.get("strategy", "-"), "召回策略")
        kpi(c4, f"{meta.get('latency_ms', meta.get('cost_ms', '-'))} ms", "本次耗时")
        extra = []
        if meta.get("query"):
            extra.append(f"检索词：{meta['query']}")
        if meta.get("mode"):
            extra.append(f"查询模式：{meta['mode']}")
        if meta.get("cost_ms"):
            extra.append(f"检索耗时：{meta['cost_ms']} ms")
        if extra:
            st.caption(" ｜ ".join(extra))
        if meta.get("level") == "none":
            st.info("题库未命中：按系统约束，模型必须声明「题库未收录」并只给方向性提示——这是防幻觉设计，不是故障。")
        elif meta.get("level") == "weak":
            st.caption("仅弱相关条目，回答中应提示「相关度有限」。")


# ============================================================
# 1. 智能陪练（RAG 问答 / 意图识别 / 多轮对话）
# ============================================================
def render_chat() -> None:
    page_header("智能陪练")
    session_id = ensure_session()

    # --- 会话管理：多轮对话的载体 ---
    bar1, bar2, bar3 = st.columns([6, 1, 1])
    sessions = api("GET", "/api/v1/sessions")
    options = {
        f"#{s['id']}  {s['title']}  ·  {s['message_count']} 条": s["id"]
        for s in sessions
    }
    labels = list(options)
    current = next((l for l, i in options.items() if i == session_id), labels[0] if labels else None)
    with bar1:
        if labels:
            chosen_label = st.selectbox(
                "当前会话", labels,
                index=labels.index(current) if current in labels else 0,
                label_visibility="collapsed",
            )
            if options[chosen_label] != session_id:
                st.session_state.session_id = options[chosen_label]
                st.session_state.pop("last_trace", None)
                st.rerun()
    with bar2:
        if st.button("➕ 新会话", width="stretch"):
            st.session_state.pop("session_id", None)
            st.session_state.pop("last_trace", None)
            st.rerun()
    with bar3:
        if st.button("🗑 删除", width="stretch"):
            api("DELETE", f"/api/v1/sessions/{session_id}")
            st.session_state.pop("session_id", None)
            st.session_state.pop("last_trace", None)
            st.toast("会话已删除")
            st.rerun()

    # --- 历史回放（会话失效时自动恢复，这是旧版报红的常见来源之一） ---
    try:
        messages = api("GET", f"/api/v1/sessions/{session_id}/messages")
    except ApiError as exc:
        if exc.status != 404:
            raise
        st.info("检测到缓存的会话在后端已不存在，已自动新建会话并继续（无需处理）。")
        st.session_state.pop("session_id", None)
        st.session_state.pop("last_trace", None)
        st.session_state.session_id = ensure_session()
        messages = []

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(safe_md(msg.get("content") or ""))
            if msg["role"] == "assistant":
                foot = []
                if msg.get("intent"):
                    foot.append("意图：" + INTENT_LABELS.get(msg["intent"], msg["intent"]))
                if msg.get("latency_ms"):
                    foot.append(f"生成 {msg['latency_ms']} ms")
                if foot:
                    st.caption(" · ".join(foot))
                if msg.get("sources"):
                    with st.expander(f"📎 引用来源（{len(msg['sources'])} 条）"):
                        render_sources(msg["sources"])

    # --- 快捷提问（演示时不用手打） ---
    st.caption("快捷提问：")
    qcols = st.columns(4)
    for idx, (label, text) in enumerate(QUICK_PROMPTS):
        if qcols[idx].button(label, width="stretch", key=f"quick_{idx}"):
            st.session_state.quick_prompt = text

    typed = st.chat_input("例如：01 背包内层循环为什么必须倒序？")
    quick = st.session_state.pop("quick_prompt", None)  # 无条件消费，避免下一轮重复发送
    prompt = typed or quick
    if not prompt:
        trace = st.session_state.get("last_trace")
        if trace:
            render_trace(trace)
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    placeholder = st.empty()
    text, meta, error_msg = "", {}, ""
    try:
        with st.spinner("检索题库 → 上下文打包 → 引导式生成…"):
            for event, data in stream_chat(session_id, prompt):
                if event == "meta":
                    meta = dict(data) if isinstance(data, dict) else {}
                elif event == "token":
                    text += data if isinstance(data, str) else str(data)
                    # 流式阶段只显示纯文本，不走 Markdown 解析：
                    # 回答里的反引号 / LaTeX / <bits/stdc++.h> 这类符号在生成到一半时
                    # 是"未闭合"状态，反复用 markdown() 重绘会让前端节点树结构跳变，
                    # 触发 React 的 removeChild 报红。生成完整后在下方一次性 markdown 渲染。
                    placeholder.text(text + " ▌")
                elif event == "done":
                    if isinstance(data, dict):
                        meta.update(data)
                elif event == "error":
                    error_msg = data.get("message", "生成失败") if isinstance(data, dict) else str(data)
    except ApiError as exc:
        show_api_error(exc)
        return

    placeholder.markdown(safe_md(text) or "（本轮没有生成文本）")
    if error_msg:
        st.error(f"生成阶段返回错误：{error_msg}")
        st.caption(
            "这是后端 SSE 的 `event: error`：流内异常被转成事件返回，不是网络问题。"
            "该分支后端不写错误日志，可查看启动终端的堆栈或 /docs 手动复现。"
        )
    if meta:
        render_trace(meta)
        st.session_state.last_trace = meta
        sources = meta.get("sources") or []
        if sources:
            with st.expander(f"📎 引用来源（{len(sources)} 条）", expanded=False):
                render_sources(sources)
        message_id = meta.get("message_id")
        if message_id:
            f1, f2, _ = st.columns([1, 1, 6])
            for btn, rating in ((f1, 1), (f2, -1)):
                label = "👍 有用" if rating == 1 else "👎 没用"
                if btn.button(label, key=f"fb_{rating}_{message_id}"):
                    try:
                        api("POST", f"/api/v1/messages/{message_id}/feedback",
                            json={"rating": rating})
                        st.toast("已记录反馈")
                    except ApiError as exc:
                        show_api_error(exc)
    elif not error_msg:
        st.warning("没有收到任何 SSE 事件：检查后端是否返回了 SSE（/docs 里 stream=true），或查看下方技术细节。")

    note("引导式教学：本系统不会直接给完整可提交代码；回答以 [1][2] 标注题库来源，"
         "「精讲」可作题解依据，「考点提示」只描述考查方向。连续追问会带上最近 3 轮上下文。")


# ============================================================
# 2. 代码诊断
# ============================================================
def kp_name_map() -> dict[str, str]:
    try:
        rows = api("GET", "/api/v1/diagnosis/skills")
        return {r["point_id"]: r["point_name"] for r in rows}
    except ApiError:
        return {}


MAX_PROBLEM_CHARS = 4000  # 与后端 DiagnosisRequest.problem_title.max_length 对齐


def render_diagnosis() -> None:
    """代码诊断页。

    渲染稳定性（修 React `removeChild` 报错）：提交结果/错误一律写进
    session_state 后 st.rerun()，由下一轮在固定占位容器里渲染；绝不在按钮
    回调里就地往表单容器插入 st.error / st.expander 或整块切换结果区，
    否则同一轮 DOM 结构剧变会让 Streamlit 的 react-dom diff 失败。
    """
    page_header("代码诊断")

    # 固定错误位：无论有没有错误，这里始终是一个 container
    error_box = st.empty()
    with error_box.container():
        err = st.session_state.get("diag_error")
        if err:
            st.error(f"诊断失败（HTTP {err['status']}）：{err['message']}")
            if err["status"] == 422:
                st.caption(f"检查代码长度（10–20000 字符）、判题结果枚举；题面最长 {MAX_PROBLEM_CHARS} 字。")
            elif err["status"] == 401:
                st.caption("登录状态已失效，请退出后重新登录再试。")
            elif err["status"] == 403:
                st.caption("当前角色无权调用诊断接口。")
            elif err["status"] >= 500:
                st.caption(f"trace_id：{err.get('trace_id') or '-'}，可在 data/logs/app.log 里检索。")

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**提交一段被判错的代码**")
            title = st.text_area(
                f"题目 / 题面（可选，最长 {MAX_PROBLEM_CHARS} 字）",
                height=110, max_chars=MAX_PROBLEM_CHARS, key="diag_title",
                placeholder="可粘贴完整题面；也可以只填标题，例如「01 背包」。题面会随诊断记录保存并作为上下文提供给诊断模型。",
            )
            c1, c2 = st.columns(2)
            verdict = c1.selectbox("判题结果", VERDICTS, key="diag_verdict")
            language = c2.selectbox("语言", ["cpp", "python", "java"], key="diag_lang")
            # 结构恒定：始终渲染一条说明，避免选 AC 时新增元素触发 DOM diff 问题
            st.caption(
                "AC 表示代码已通过：本次只做记录与复盘，不输出错因。"
                if verdict == "AC" else
                "非 AC 判题结果会走静态特征 + 结构化诊断，给出错因、证据与建议。"
            )

            def _fill_sample():
                # 回调在 widget 实例化之前执行，可以安全写入同名 session_state
                st.session_state["diag_code"] = SAMPLE_CODE

            st.button("载入演示代码（三重循环 + endl，典型 TLE）",
                      width="stretch", on_click=_fill_sample)
            code = st.text_area("代码", height=260, key="diag_code",
                                placeholder="粘贴 10–20000 个字符的源代码")

            if st.button("🔬 开始诊断", type="primary", width="stretch"):
                if len((code or "").strip()) < 10:
                    st.session_state.diag_error = {
                        "status": 422, "message": "代码太短，后端要求 10–20000 字符",
                        "trace_id": "",
                    }
                    st.session_state.pop("diag_result", None)
                elif len(title or "") > MAX_PROBLEM_CHARS:
                    st.session_state.diag_error = {
                        "status": 422,
                        "message": f"题面超过 {MAX_PROBLEM_CHARS} 字，请精简后再提交",
                        "trace_id": "",
                    }
                else:
                    try:
                        st.session_state.diag_result = api(
                            "POST", "/api/v1/diagnosis",
                            json={"problem_title": title or "", "code": code,
                                  "verdict": verdict, "language": language},
                        )
                        st.session_state.pop("diag_error", None)
                    except ApiError as exc:
                        st.session_state.diag_error = {
                            "status": exc.status, "message": exc.message,
                            "trace_id": exc.trace_id,
                        }
                        st.session_state.pop("diag_result", None)
                # 关键：状态切换交给下一轮渲染，避免同轮 DOM 剧变
                st.rerun()

            note("诊断 = 静态特征（循环嵌套 / 递归 / 边界 / IO）+ 大模型结构化判定 + 三重校验降级，"
                 "不执行代码、不接 OJ 沙箱（答辩需主动说明的已知限制）。题面会作为上下文提供给诊断模型。")

    with right:
        # 结果区同样固定在同一个占位容器里，空态与结果态不再整块替换
        result_box = st.empty()
        with result_box.container():
            result = st.session_state.get("diag_result")
            if not result:
                st.info("诊断结果会显示在这里：错因、证据、改进建议、置信度与静态特征。")
            else:
                passed = (result.get("cause_code") == "AC"
                          or "已通过" in str(result.get("cause_label", "")))
                if passed:
                    # AC 表示代码已通过，不展示任何「错因」措辞
                    st.success("本题已通过（AC）：代码正确，系统不做错因判定。")
                    st.markdown("### 已通过 · 复盘建议")
                    for i, item in enumerate(result.get("suggestions") or [], 1):
                        st.markdown(f"{i}. {item}")
                    st.caption("AC 提交不调用诊断模型、不产生错因与知识点扣分；记录仍保存在下方历史中。")
                else:
                    st.markdown(f"### {esc(result.get('cause_label', '未分类'))}")
                    c1, c2, c3 = st.columns(3)
                    kpi(c1, f"{float(result.get('confidence', 0)):.2f}", "置信度")
                    kpi(c2, result.get("engine", "-"), "判定引擎")
                    kpi(c3, result.get("cause_code", "-"), "错因编号")
                    st.progress(float(result.get("confidence", 0)))
                    names = kp_name_map()
                    points = result.get("points") or []
                    if points:
                        st.markdown("**关联知识点**")
                        st.markdown(" ".join(
                            f'<span class="acm-tag">{esc(names.get(p, p))}</span>' for p in points
                        ), unsafe_allow_html=True)
                    st.markdown("**证据**")
                    for item in result.get("evidence") or []:
                        st.markdown(f"- {item}")
                    st.markdown("**改进建议**")
                    for i, item in enumerate(result.get("suggestions") or [], 1):
                        st.markdown(f"{i}. {item}")
                with st.expander("静态特征（可解释性）"):
                    st.json(result.get("features") or {})

    st.divider()
    st.markdown("**📜 诊断历史**")
    try:
        rows = api("GET", "/api/v1/diagnosis/history", params={"limit": 30})
    except ApiError as exc:
        show_api_error(exc)
        return
    if not rows:
        st.caption("暂无记录。提交一次代码后，命中的知识点会进入「知识点画像」。")
    else:
        df = pd.DataFrame(rows)
        if "points" in df.columns:
            df["mastery_hit"] = df["points"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        else:
            df["mastery_hit"] = ""
        # 兼容修复前的历史数据：AC 行曾被模型编造错因，展示时还原为「已通过」
        if "cause_label" in df.columns and "verdict" in df.columns:
            df["cause_label"] = df.apply(
                lambda r: "已通过（AC）" if str(r["verdict"]).upper() == "AC"
                else r["cause_label"], axis=1)
        show = [c for c in ["created_at", "problem_title", "verdict", "cause_label",
                            "mastery_hit", "confidence", "engine"] if c in df.columns]
        st.dataframe(df[show], width="stretch", hide_index=True)


# ============================================================
# 3. 知识点画像
# ============================================================
def render_profile() -> None:
    page_header("知识点画像")
    try:
        skills = api("GET", "/api/v1/diagnosis/skills")
        history = api("GET", "/api/v1/diagnosis/history", params={"limit": 50})
    except ApiError as exc:
        show_api_error(exc)
        return

    note("掌握度口径：每命中一个知识点初始 0.3，之后按每次代码诊断的判题结果在 0–1 区间滑动更新，"
         "同时累计尝试次数；数值越低代表越薄弱。数据来自 /api/v1/diagnosis/skills。")
    if not skills:
        st.info("暂无知识点记录。请先到「代码诊断」提交一段代码，系统会据诊断结果建立画像。")
        return

    df = pd.DataFrame(skills)
    weak = df[df["mastery"] < 0.25]
    middle = df[(df["mastery"] >= 0.25) & (df["mastery"] <= 0.6)]
    strong = df[df["mastery"] > 0.6]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, len(df), "已建模知识点")
    kpi(c2, len(weak), "薄弱（<0.25）")
    kpi(c3, len(middle), "巩固中")
    kpi(c4, len(strong), "已掌握（>0.6）")
    st.divider()

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**掌握度分布（升序，最上方最薄弱）**")
        chart = df.sort_values("mastery").set_index("point_name")[["mastery"]]
        st.bar_chart(chart)
        st.dataframe(
            df[["point_id", "point_name", "mastery", "attempts"]].rename(columns={
                "point_id": "编号", "point_name": "知识点",
                "mastery": "掌握度", "attempts": "尝试次数",
            }),
            width="stretch", hide_index=True,
        )
    with right:
        st.markdown("**近期错因分布**")
        if history:
            # AC 提交是「已通过」，不计入错因分布
            wrong = [h for h in history if str(h.get("verdict", "")).upper() != "AC"]
            if wrong:
                causes = pd.Series([h.get("cause_label", "未分类") for h in wrong]).value_counts()
                st.bar_chart(causes)
                st.dataframe(
                    causes.rename_axis("错因").reset_index(name="次数"),
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("暂无判错记录（AC 提交不计入错因分布）")
        else:
            st.caption("暂无诊断记录")
    note("对应任务书「重点研究的问题 4」：知识点掌握度建模与个性化推荐。下一步请到「薄弱点推荐」。")


# ============================================================
# 4. 薄弱点推荐（前端编排：skills → /knowledge 检索）
# ============================================================
def render_recommend() -> None:
    page_header("薄弱点推荐")
    note("推荐逻辑（纯前端编排，不改后端）：取掌握度最低的 N 个知识点 → 用知识点名称作为关键词"
         "调用现有 /api/v1/knowledge 检索 → 去重后按难度展示 → 可一键标记已练。"
         "这样演示的是「画像驱动推荐」的完整闭环，而不需要新增后端接口。")
    top_n = st.slider("选取薄弱知识点个数", 1, 6, 3, key="rec_top_n")
    per_point = st.slider("每个知识点推荐题数", 1, 10, 4, key="rec_size")
    st.divider()

    try:
        skills = api("GET", "/api/v1/diagnosis/skills")
    except ApiError as exc:
        show_api_error(exc)
        return
    if not skills:
        st.info("画像为空：先到「代码诊断」提交代码，产生知识点掌握度后再来看推荐。")
        return

    weak = sorted(skills, key=lambda s: s["mastery"])[:top_n]
    seen: set[str] = set()
    for idx, skill in enumerate(weak):
        label = skill["point_name"]
        mastery = float(skill["mastery"])
        with st.expander(
            f"🎯 {label}　·　掌握度 {mastery:.2f}　·　尝试 {skill['attempts']} 次",
            expanded=(idx == 0),
        ):
            st.progress(1.0 - mastery, text="薄弱程度")
            try:
                data = api("GET", "/api/v1/knowledge",
                           params={"q": label, "size": per_point, "page": 1})
            except ApiError as exc:
                show_api_error(exc)
                continue
            items = [i for i in data.get("items", []) if i["id"] not in seen]
            if not items:
                st.caption("题库中没有检索到该关键词的条目（可换个知识点或减少推荐数）")
                continue
            for item in items:
                seen.add(item["id"])
                c1, c2 = st.columns([6, 1])
                with c1:
                    kind = "精讲" if item.get("source") == "curated" else "考点提示"
                    cls = "ok" if kind == "精讲" else "warn"
                    st.markdown(
                        f'<span class="acm-tag {cls}">{esc(kind)}</span>'
                        f'**{esc(item["title"])}** '
                        f'<span class="acm-tag gray">{esc(item.get("difficulty") or "未标注")}</span>'
                        f'<br><span style="font-size:12px;color:#64748b">'
                        f'{esc(item.get("tags") or "无标签")}</span>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("标记已练", key=f"rec_mark_{item['id']}"):
                        try:
                            api("POST", f"/api/v1/practice/{item['id']}")
                            st.toast("已加入练习记录")
                        except ApiError as exc:
                            show_api_error(exc)

    st.divider()
    st.markdown("**✅ 已练习条目**")
    try:
        practiced = api("GET", "/api/v1/practice")
    except ApiError as exc:
        show_api_error(exc)
        return
    if practiced:
        st.dataframe(pd.DataFrame(practiced).rename(columns={
            "title": "题目", "difficulty": "难度", "tags": "标签", "marked_at": "标记时间",
        }), width="stretch", hide_index=True)
    else:
        st.caption("还没有练习标记")


# ============================================================
# 5. 题库管理
# ============================================================
def render_knowledge() -> None:
    page_header("题库管理")
    c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
    keyword = c1.text_input("搜索标题或标签", key="kb_q", placeholder="例如：背包 / dp / 最短路")
    domain = c2.text_input("知识域", key="kb_domain", placeholder="可选")
    difficulty = c3.selectbox("难度", ["全部"] + DIFFICULTIES, key="kb_diff")
    source = c4.selectbox("来源", ["全部", "curated", "cf", "atcoder"], key="kb_src")
    p1, p2 = st.columns([1, 5])
    page_no = p1.number_input("页码", min_value=1, value=1, step=1, key="kb_page")
    size = p2.select_slider("每页条数", options=[10, 20, 50, 100], value=20, key="kb_size")

    params = {
        "q": keyword, "domain": domain,
        "difficulty": "" if difficulty == "全部" else difficulty,
        "source": "" if source == "全部" else source,
        "page": int(page_no), "size": int(size),
    }
    try:
        data = api("GET", "/api/v1/knowledge", params=params)
    except ApiError as exc:
        show_api_error(exc)
        return

    items = data.get("items", [])
    total = data.get("total", 0)
    st.caption(f"共 {total} 条 · 第 {data.get('page', page_no)} 页 · 本页 {len(items)} 条"
               "　（curated = 人工精讲，cf / atcoder = 元数据加工的考点提示）")
    if not items:
        st.info("没有匹配的条目，试试更短的关键词或清空筛选。")
    else:
        df = pd.DataFrame(items)
        cols = [c for c in ["id", "title", "difficulty", "tags", "domain", "source", "rating"] if c in df.columns]
        st.dataframe(df[cols], width="stretch", hide_index=True)

        st.markdown("**🔎 条目详情**")
        chosen = st.selectbox("选择条目", [i["id"] for i in items],
                              format_func=lambda x: next(
                                  (f'{i["title"]}  ·  {x}' for i in items if i["id"] == x), x),
                              key="kb_detail_id")
        if chosen:
            try:
                detail = api("GET", f"/api/v1/knowledge/{chosen}")
            except ApiError as exc:
                show_api_error(exc)
            else:
                with st.container(border=True):
                    st.markdown(f"### {esc(detail.get('title', ''))}")
                    st.caption(f"难度 {esc(detail.get('difficulty'))} · 标签 {esc(detail.get('tags'))} "
                               f"· 来源 {esc(detail.get('source'))} · 更新 {esc(detail.get('updated_at'))}")
                    if detail.get("url"):
                        st.markdown(f"原题：{detail['url']}")
                    st.markdown(str(detail.get("content", ""))[:2000])
                if detail.get("similar"):
                    st.markdown("**相似条目**")
                    for s in detail["similar"]:
                        st.markdown(f"- {s.get('title', '')}（{s.get('difficulty', '')}）")
                c1, c2 = st.columns([1, 5])
                if c1.button("✅ 标记已练", key=f"kb_mark_{chosen}"):
                    try:
                        api("POST", f"/api/v1/practice/{chosen}")
                        st.toast("已加入练习记录")
                    except ApiError as exc:
                        show_api_error(exc)

    st.divider()
    if is_admin():
        with st.expander("➕ 录入精讲条目（admin）"):
            with st.form("kb_create_form"):
                t = st.text_input("标题 *")
                cc1, cc2 = st.columns(2)
                diff = cc1.selectbox("难度", DIFFICULTIES[:3], key="kb_new_diff")
                dom = cc2.text_input("知识域", key="kb_new_domain")
                tags = st.text_input("标签（逗号分隔）", key="kb_new_tags")
                desc = st.text_area("题面描述", height=90, key="kb_new_desc")
                sol = st.text_area("解题思路", height=140, key="kb_new_sol")
                pit = st.text_area("常见错误", height=90, key="kb_new_pit")
                if st.form_submit_button("提交入库", type="primary"):
                    try:
                        api("POST", "/api/v1/knowledge", json={
                            "title": t, "difficulty": diff, "tags": tags, "domain": dom,
                            "description": desc, "solution": sol, "pitfalls": pit,
                        })
                        st.success("录入成功。索引为快照，需点下方「重建索引」后才会被检索到。")
                    except ApiError as exc:
                        show_api_error(exc)
            st.caption("入库门禁要求正文合计 40–2400 字；不满足会返回 409。")
        with st.expander("⚙️ 索引与下架（admin）"):
            st.caption("重建索引为后台任务（约 8,558 条，CPU 环境需数分钟），返回 202 即已开始。")
            if st.button("🔄 重建索引", key="kb_reindex"):
                try:
                    res = api("POST", "/api/v1/knowledge/reindex")
                    st.success(res.get("message", "已提交"))
                except ApiError as exc:
                    show_api_error(exc)
            del_id = st.text_input("要下架的条目 id", key="kb_del_id")
            if st.button("🗑 下架条目", key="kb_del_btn") and del_id:
                try:
                    api("DELETE", f"/api/v1/knowledge/{del_id}")
                    st.success("已下架（软删除）")
                except ApiError as exc:
                    show_api_error(exc)
    else:
        note("录入 / 下架 / 重建索引需要 admin 角色；当前账号可浏览、检索与标记练习。"
             "这是 RBAC 权限生效，不是接口故障。")


# ============================================================
# 6. 数据统计
# ============================================================
def render_stats() -> None:
    page_header("数据统计")
    try:
        data = api("GET", "/api/v1/stats/overview")
        extra = api("GET", "/api/v1/stats/knowledge")
    except ApiError as exc:
        show_api_error(exc)
        return

    cols = st.columns(6)
    values = [
        (data.get("questions", 0), "累计提问"),
        (data.get("sessions", 0), "会话数"),
        (f"{float(data.get('match_rate', 0)) * 100:.1f}%", "题库命中率"),
        (f"{data.get('avg_latency_ms', 0)} ms", "平均生成延迟"),
        (data.get("mastered", 0), "已练习条目"),
        (extra.get("counts", {}).get("total", 0), "题库总量"),
    ]
    for col, (value, label) in zip(cols, values):
        # 注意：DeltaGenerator.__enter__ 返回 None，不能用 `with ... as box`
        box = col.container(border=True)
        kpi(box, value, label)

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**意图分布**（意图识别层的可观测结果）")
        intents = data.get("intents") or {}
        if intents:
            st.bar_chart(pd.Series(intents, name="提问数"))
            st.dataframe(pd.Series(intents, name="提问数").rename_axis("意图").reset_index(),
                         width="stretch", hide_index=True)
        else:
            st.caption("暂无提问记录")
    with right:
        st.markdown("**题库标签 TOP 10**")
        tags = data.get("top_tags") or []
        if tags:
            df = pd.DataFrame(tags, columns=["标签", "题量"]).set_index("标签")
            st.bar_chart(df.head(10))
        else:
            st.caption("暂无标签数据")

    st.divider()
    st.markdown("**题库构成与练习进度**")
    counts = extra.get("counts") or {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("按来源")
        by_source = counts.get("by_source") or {}
        st.dataframe(pd.Series(by_source, name="条数").rename_axis("来源").reset_index(),
                     width="stretch", hide_index=True)
    with c2:
        st.markdown("按知识域")
        by_domain = counts.get("by_domain") or {}
        st.dataframe(pd.Series(by_domain, name="条数").rename_axis("知识域").reset_index().head(12),
                     width="stretch", hide_index=True)
    with c3:
        st.markdown("标签 TOP 20")
        top = extra.get("top_tags") or []
        if top:
            st.dataframe(pd.DataFrame(top, columns=["标签", "题量"]),
                         width="stretch", hide_index=True)
        st.caption(f"本人已标记练习：{extra.get('mastered', 0)} 条")
    note("命中率 = 检索到题库条目的回答数 / 助手回答数；平均延迟统计 assistant 消息的 latency_ms。"
         "指标只统计当前账号，管理员也看不到他人明细（对应权限与隐私设计）。")


# ============================================================
# 7. 账号与权限
# ============================================================
def render_account() -> None:
    page_header("账号与权限")
    user = st.session_state.get("user") or {}
    role = user.get("role", "student")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, user.get("id", "-"), "用户 ID")
        kpi(c2, user.get("username", "-"), "用户名")
        kpi(c3, ROLE_LABELS.get(role, role), "角色")
        kpi(c4, user.get("created_at", "-"), "注册时间")

    st.markdown("**角色权限矩阵（RBAC）**")
    matrix = pd.DataFrame([
        ["智能陪练 / 代码诊断 / 画像 / 推荐 / 统计", "✅", "✅", "✅"],
        ["题库浏览与练习标记", "✅", "✅", "✅"],
        ["题库录入 / 下架 / 重建索引", "—", "✅", "—"],
        ["运行状态 / 审计日志 / 检索配置", "—", "✅", "✅"],
        ["查看受限（restricted）条目", "—", "✅", "✅"],
        ["导出本人数据 / 注销本人账号", "✅", "✅", "✅"],
    ], columns=["能力", "student 学生", "admin 管理员", "auditor 审计员"])
    st.dataframe(matrix, width="stretch", hide_index=True)

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**🔐 修改口令**")
        with st.form("pwd_form"):
            old = st.text_input("原口令", type="password")
            new = st.text_input("新口令（≥8 位，含字母和数字）", type="password")
            if st.form_submit_button("确认修改"):
                try:
                    api("POST", "/api/v1/auth/change-password",
                        json={"old_password": old, "new_password": new})
                    st.success("修改成功。旧令牌已失效，请重新登录。")
                    logout()
                except ApiError as exc:
                    show_api_error(exc)
        st.caption("改密会递增 token_version，所有旧 JWT 立刻失效（安全基线要求）。")

        st.markdown("**📦 导出我的数据（PIPL 可携带权）**")
        if st.button("生成导出文件"):
            try:
                # 存进 session_state，避免下一次 rerun 把下载按钮冲掉
                st.session_state.export_payload = api("GET", "/api/v1/export/my-data")
                st.session_state.pop("export_error", None)
            except ApiError as exc:
                st.session_state.export_error = exc.message
        if st.session_state.get("export_error"):
            st.error(f"导出失败：{st.session_state.export_error}")
        if st.session_state.get("export_payload") is not None:
            payload = st.session_state.export_payload
            st.download_button(
                "下载 JSON", data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name=f"acm_coach_{user.get('username', 'user')}_data.json",
                mime="application/json",
            )
            st.caption(f"包含 {len(payload.get('sessions', []))} 个会话、"
                       f"{len(payload.get('diagnosis', []))} 条诊断、"
                       f"{len(payload.get('practice', []))} 条练习标记")
    with right:
        st.markdown("**⚠️ 注销账号（PIPL 删除权）**")
        st.caption("将按依赖顺序清除本人会话、消息、诊断记录、练习标记与检索日志，"
                   "审计快照保留。此操作不可撤销。")
        confirm_check = st.checkbox("我了解此操作不可撤销", key="del_me_check")
        confirm_name = st.text_input("请输入用户名以确认", key="del_me_name")
        if st.button("永久注销账号", disabled=not (confirm_check and confirm_name == user.get("username"))):
            try:
                api("DELETE", "/api/v1/me")
                st.success("账号已注销")
                logout()
            except ApiError as exc:
                show_api_error(exc)

        st.markdown("**🚪 退出登录**")
        if st.button("退出当前账号", width="stretch"):
            logout()


# ============================================================
# 8. 运行状态（staff only）
# ============================================================
def render_ops() -> None:
    page_header("运行状态")
    if not is_staff():
        note("该页面需要 admin / auditor 角色。学生账号不会在导航里看到它——这正是权限隔离生效。")
        return
    tabs = st.tabs(["概览", "检索参数", "抓取记录", "审计日志", "检索质量"])
    calls = [
        (tabs[0], "/api/v1/ops/overview", "系统概览"),
        (tabs[1], "/api/v1/ops/config", "检索参数白名单"),
        (tabs[2], "/api/v1/ops/crawls", "抓取批次"),
        (tabs[3], "/api/v1/ops/audit", "审计日志"),
        (tabs[4], "/api/v1/ops/retrieval", "近 24h 检索质量"),
    ]
    for tab, path, label in calls:
        with tab:
            kwargs = {"params": {"limit": 50}} if ("audit" in path or "crawl" in path) else {}
            try:
                payload = api("GET", path, **kwargs)
            except ApiError as exc:
                show_api_error(exc)
                continue
            if isinstance(payload, list):
                if payload:
                    st.dataframe(pd.DataFrame(payload), width="stretch", hide_index=True)
                else:
                    st.caption(f"{label}：暂无记录")
            elif isinstance(payload, dict):
                st.json(payload)
            else:
                st.text(str(payload))
    st.caption("密钥类配置不在此暴露：/ops/config 只返回白名单标量参数。")


# ============================================================
# 9. 答辩演示指引
# ============================================================
DEMO_STEPS = [
    ("智能陪练", "① 登录与 RAG 引导式问答",
     "用快捷提问「01 背包内层循环为什么必须倒序？」→ 观察流式生成、引用来源 [1][2]、精讲/考点提示分级。",
     "任务书：检索增强问答、防幻觉、可溯源"),
    ("智能陪练", "② 意图识别与查询改写",
     "展开「本次推理链路」，讲清 intent / level / strategy / query 四个字段来自 SSE meta 事件。",
     "任务书重点 1、2：检索质量与提示词设计"),
    ("智能陪练", "③ 多轮对话",
     "点快捷提问「那用一维数组怎么写？」→ 说明最近 3 轮历史进入上下文、查询改写补话题锚点。",
     "任务书：多轮对话"),
    ("智能陪练", "④ 拒答与反例",
     "点「直接给我完整代码」→ 演示 code_request 意图下的拒绝给代码 + 反问引导。",
     "任务书重点 2：引导式教学"),
    ("代码诊断", "⑤ 代码诊断",
     "载入演示代码（三重循环 + endl）选 TLE → 看错因、证据、静态特征与规则/混合引擎标注。",
     "任务书重点 3：静态特征 + 大模型结构化输出"),
    ("知识点画像", "⑥ 知识点画像",
     "诊断后查看掌握度、尝试次数与错因分布，讲清 0–1 滑动建模口径。",
     "任务书重点 4：掌握度建模"),
    ("薄弱点推荐", "⑦ 薄弱点推荐",
     "按最低掌握度检索题库 → 标记已练 → 在题库管理/统计页验证闭环。",
     "任务书重点 4：个性化推荐"),
    ("题库管理", "⑧ 题库管理（admin）",
     "演示筛选、详情、录入精讲条目与重建索引；说明 8,558 条语料的三类来源与版权边界。",
     "任务书：题库管理、技术要求 ≥100 条 / ≥8 类知识点"),
    ("数据统计", "⑨ 数据统计",
     "展示提问数、命中率、平均延迟、意图分布与题库构成。",
     "任务书：数据统计"),
    ("账号与权限", "⑩ 用户与权限",
     "讲 RBAC 三角色矩阵、改密使旧令牌失效、数据导出与注销（PIPL）。",
     "任务书：用户与权限、安全基线"),
]


def render_demo() -> None:
    page_header("答辩演示指引")
    st.markdown("**🎬 完整演示流程（点按钮可直接跳转）**")
    for i, (target, title, desc, mapping) in enumerate(DEMO_STEPS):
        st.markdown(
            f'<div class="acm-step"><div class="t">{esc(title)}</div>'
            f'<div class="d">{esc(desc)}</div><div class="m">{esc(mapping)}</div></div>',
            unsafe_allow_html=True,
        )
        c1, _ = st.columns([1, 6])
        if c1.button(f"前往 {PAGE_ICONS[target]} {target}", key=f"jump_{i}"):
            if target in available_pages():
                st.session_state.nav_page = target
                st.rerun()
            else:
                st.warning("当前角色无法访问该页面，请用 admin 账号登录")

    st.divider()
    st.markdown("**🩺 环境自检**（无需登录即可调用）")
    h1, h2, h3 = st.columns(3)
    try:
        health = api("GET", "/healthz")
        h1.success(f"进程存活：{health.get('status', 'ok')}")
    except ApiError as exc:
        h1.error(f"进程不可达：{exc.message}")
    try:
        ready = api("GET", "/readyz")
        h2.success(f"依赖就绪：database={ready.get('database')} · bm25={ready.get('bm25')} · vector={ready.get('vector')}")
    except ApiError as exc:
        h2.warning(f"未就绪：{exc.message}")
    try:
        ver = api("GET", "/version")
        h3.info(f"{ver.get('name', '')} v{ver.get('version', '')} · {ver.get('env', '')}")
    except ApiError:
        h3.caption("/version 需要 admin / auditor 角色，学生账号跳过即可")
    note("若 bm25=false：执行 `python -m acm_coach.cli index --max-vectors 1500` 重建索引；"
         "若 healthz 不可达：先执行 `python -m acm_coach.cli serve`。")

    st.divider()
    st.markdown("**🛠 常见报错速查（本次重点修复）**")
    faq = pd.DataFrame([
        ["红框「请确认 API 已启动」，但 /docs 打得开",
         "旧版把 403/401/404 全兜成一句话。现在按状态码分派：403 是角色不足，401 是登录过期。",
         "看新版错误提示里的技术细节 trace_id"],
        ["学生账号打开「运行状态」报红",
         "/api/v1/ops/overview 要求 admin/auditor，返回 403 是设计如此。",
         "学生导航里已隐藏该页；用 admin 登录可查看"],
        ["问一个问题后长时间不动，再问就报红",
         "JWT 120 分钟过期；旧版 SSE 不检查状态码，把 401 JSON 当事件流读。",
         "点「重新登录」即可，问答数据没有丢"],
        ["换库/删库后一进陪练页就报红",
         "浏览器缓存的 session_id 在后端已不存在，GET messages 返回 404。",
         "新版自动新建会话并提示，不再弹红框"],
        ["回答区域一片空白，也没有红框",
         "后端返回非 SSE 响应时旧版静默丢弃；也可能是未配置 API Key 走抽取式降级。",
         "看「本次推理链路」是否出现；仍未出现检查 /docs 的 stream 参数"],
        ["红框写着「生成失败：XXX」",
         "后端 event: error（流内异常转成事件），与网络无关；该分支不写日志。",
         "看启动终端堆栈，或用 trace_id 查 app.log"],
        ["代码诊断返回 422",
         "判题结果枚举不符，或代码不足 10 字符。",
         "用页面下拉框选 AC/WA/TLE/… 并贴完整代码"],
    ], columns=["现象", "真实原因", "怎么办"])
    st.dataframe(faq, width="stretch", hide_index=True)
    note("判定原则：先看 HTTP 状态码，再看后端 code/message/trace_id，最后才怀疑「服务没启动」。"
         "本页所有功能都只调用既有接口，后端零改动。")


# ============================================================
# 导航与主流程
# ============================================================
PAGES = {
    "智能陪练": render_chat,
    "代码诊断": render_diagnosis,
    "知识点画像": render_profile,
    "薄弱点推荐": render_recommend,
    "题库管理": render_knowledge,
    "数据统计": render_stats,
    "账号与权限": render_account,
    "运行状态": render_ops,
    "答辩演示指引": render_demo,
}


def available_pages() -> list[str]:
    base = ["智能陪练", "代码诊断", "知识点画像", "薄弱点推荐",
            "题库管理", "数据统计", "账号与权限"]
    if is_staff():
        base.append("运行状态")
    base.append("答辩演示指引")
    return base


def render_sidebar(pages: list[str]) -> None:
    user = st.session_state.get("user") or {}
    role = user.get("role", "student")
    st.markdown(
        '<div style="font-size:19px;font-weight:700;color:#1e3a8a;line-height:1.3">'
        '🏆 ACM 智能陪练</div>'
        '<div style="font-size:12px;color:#64748b;margin-bottom:14px">'
        'RAG 引导式陪练 · v4 </div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="acm-user"><span class="name">{esc(user.get("username", "-"))}</span>'
        f'<span class="role">{esc(ROLE_LABELS.get(role, role))}</span>'
        f'<div style="font-size:11.5px;color:#94a3b8;margin-top:6px">'
        f'API {esc(API_BASE)}</div></div>',
        unsafe_allow_html=True,
    )
    st.radio(
        "功能导航", pages, key="nav_page",
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
        label_visibility="collapsed",
    )
    st.divider()
    try:
        api("GET", "/healthz")
        st.success("● API 在线", icon="✅")
    except ApiError:
        st.warning("● API 不可达", icon="⚠️")
    st.caption("演示顺序见 🎬 答辩演示指引")
    if st.button("退出登录", width="stretch"):
        logout()


def main() -> None:
    inject_style()
    if not st.session_state.get("token"):
        render_login()
        return

    pages = available_pages()
    if st.session_state.get("nav_page") not in pages:
        st.session_state.nav_page = pages[0]
    with st.sidebar:
        render_sidebar(pages)

    page = st.session_state.nav_page
    try:
        PAGES[page]()
    except ApiError as exc:
        show_api_error(exc)
    except Exception as exc:  # 前端自身异常：明确标注，不伪装成后端故障
        # st.rerun()/st.stop() 抛的是控制流异常，必须放行，不能当成渲染错误
        if type(exc).__name__ in ("RerunException", "StopException"):
            raise
        st.error(f"前端渲染异常（{type(exc).__name__}）：{exc}")
        st.caption("这是 Streamlit 页面代码抛出的异常，与后端无关；请把这段信息记下来排查。")


main()
