# -*- coding: utf-8 -*-
"""诊断引擎：规则候选 + 大模型结构化判定 + 校验降级。

分工原则：
- **规则先行**：能用静态特征判定的（越界/溢出/递归/常数/死循环）直接出结论，
  置信度标注来源，避免把可确定的事实交给大模型"猜"；
- **模型补语义**：题意理解偏差、策略错误、DP 状态错误这类必须读代码语义的，
  由大模型在受约束的标签集合内选择；
- **三重校验**：编号必须在白名单、证据必须能在代码里找到依据、置信度必须与
  证据数量匹配，否则降级为规则结论。
"""
from __future__ import annotations

import json
import re

from acm_coach.core.logging import get_logger
from acm_coach.diagnosis.features import extract_features, normalize_language
from acm_coach.diagnosis.knowledge import ET_BY_ID, KP_BY_ID, et_name, et_prompt_list, kp_prompt_list

log = get_logger("diagnosis")

PROMPT = """你是一名 ACM 竞赛教练，正在诊断一段学生提交的代码。

判题结果：{verdict}
题目：{problem}

静态特征：
{features}

请只依据上述代码与特征，输出严格的 JSON（不要 markdown 代码块），字段：
{{
  "cause_code": "从 {et_list} 中选一个最可能的",
  "points": ["从 {kp_list} 中选 1-3 个最相关的知识点编号"],
  "evidence": ["每条必须引用代码中的具体片段或行号，最多 4 条"],
  "suggestions": ["可执行的改进动作，最多 3 条，不要给完整代码"],
  "confidence": 0.0 到 1.0 的小数
}}
约束：不得编造行号；证据不足时 confidence 必须低于 0.5；不要输出完整题解代码。

代码（语言：{language_label}）：
```{language}
{code}
```"""


def rule_candidates(features: dict) -> list[dict]:
    """由静态特征直接给出的高质量候选（置信度可解释）。"""
    out: list[dict] = []
    verdict = features["verdict"]
    lang = features.get("language", "cpp")

    if verdict == "CE":
        out.append({"cause_code": "ET01", "confidence": 0.75,
                    "evidence": ["判题结果为 CE，优先排查语法/分号/括号"], "points": ["KP01"]})
    if verdict == "TLE":
        if features["loop_nesting"] >= 3:
            out.append({"cause_code": "ET07", "confidence": 0.8,
                        "evidence": [f"循环嵌套 {features['loop_nesting']} 层，复杂度可能过高"],
                        "points": ["KP30"]})
        if lang == "cpp" and features["has_cin_cout"] and not features["sync_with_stdio"]:
            out.append({"cause_code": "ET09", "confidence": 0.7,
                        "evidence": ["cin/cout 未关闭流同步"], "points": ["KP01", "KP30"]})
        if features["uses_endl"]:
            label = "endl（每次 flush）" if lang == "cpp" else "强制 flush 输出"
            out.append({"cause_code": "ET09", "confidence": 0.55,
                        "evidence": [f"使用了 {label}"], "points": ["KP30"]})
    if verdict == "RE":
        if features["recursion"]:
            out.append({"cause_code": "ET06", "confidence": 0.65,
                        "evidence": [f"存在递归函数 {features['recursion_func']}，可能栈溢出"],
                        "points": ["KP02"]})
        if features["large_array"]:
            out.append({"cause_code": "ET03", "confidence": 0.5,
                        "evidence": [f"静态数组尺寸 {max(features['large_array'])}，注意下标与内存"],
                        "points": ["KP03"]})
    if features["mod_usage"] and not features["long_long"]:
        out.append({"cause_code": "ET14", "confidence": 0.6,
                    "evidence": ["有取模运算但未使用 long long，乘法可能溢出"],
                    "points": ["KP28"]})
    if features["sqrt_as_int"]:
        out.append({"cause_code": "ET14", "confidence": 0.5,
                    "evidence": ["sqrt 转 int 未做 ±1 校验"], "points": ["KP28"]})
    if features["abs_int"]:
        out.append({"cause_code": "ET14", "confidence": 0.45,
                    "evidence": ["对可能超 int 的值使用 abs"], "points": ["KP28"]})
    return out


def _parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def _validate(payload: dict, features: dict, code: str) -> tuple[bool, str]:
    if payload.get("cause_code") not in ET_BY_ID:
        return False, "错因编号不在白名单"
    points = payload.get("points") or []
    if not isinstance(points, list) or not all(p in KP_BY_ID for p in points):
        return False, "知识点编号不在白名单"
    evidence = payload.get("evidence") or []
    if not isinstance(evidence, list) or not evidence:
        return False, "缺少证据"
    # 证据必须能在代码里找到片段（数字或标识符）
    grounded = 0
    for item in evidence:
        tokens = re.findall(r"[A-Za-z_]\w{2,}|\b\d+\b", str(item))
        if any(tok in code for tok in tokens[:5]):
            grounded += 1
    if grounded == 0:
        return False, "证据无法在代码中定位"
    try:
        conf = float(payload.get("confidence", 0))
    except (TypeError, ValueError):
        return False, "置信度格式非法"
    if not 0.0 <= conf <= 1.0:
        return False, "置信度越界"
    payload["confidence"] = conf
    return True, ""


def diagnose(code: str, verdict: str, problem_title: str = "", llm=None,
             language: str = "cpp", mode: str = "hybrid") -> dict:
    """诊断入口。

    mode 用于开题报告承诺的「仅规则 / 仅大模型 / 融合」三组对比实验，
    便于离线复现与论文取证；生产默认 hybrid，行为与改动前完全一致：
      - rule   ：只跑规则，不调用大模型；
      - llm    ：以规则结论为安全兜底，但让通过校验的模型结论优先；
      - hybrid ：规则置信度 >=0.7 且与模型冲突时以规则为准（可确定事实优先）。
    非法 mode 一律回退 hybrid，避免评测脚本因拼写错误静默跑偏。
    """
    if mode not in ("rule", "llm", "hybrid"):
        mode = "hybrid"
    lang = normalize_language(language)
    features = extract_features(code, verdict, lang)

    # 判题结果 AC 表示代码已通过全部测试点，不存在错因。
    # 旧版仍让模型「从 ET01–ET15 里选一个最可能的错因」，会在 AC 上凭空编造错误
    # 结论（例如「输出格式错误」），与用户的判题结果直接矛盾。这里短路为「已通过」，
    # 跳过规则候选与 LLM：既不编错因，也不产生知识点扣分，只给复盘建议。
    if (verdict or "").upper() == "AC":
        return {
            "cause_code": "AC",
            "cause_label": "已通过（AC）",
            "points": [],
            "evidence": ["判题结果为 AC：代码已通过全部测试点，无需错因诊断"],
            "suggestions": [
                "把本题的关键思路与边界条件写进复盘笔记",
                "尝试同知识点的变式题或更高难度题目，巩固掌握度",
            ],
            "confidence": 1.0,
            "engine": "rule",
            "features": features,
        }

    rules = rule_candidates(features)
    best_rule = max(rules, key=lambda r: r["confidence"]) if rules else None

    result = {
        "cause_code": best_rule["cause_code"] if best_rule else "ET11",
        "cause_label": et_name(best_rule["cause_code"] if best_rule else "ET11"),
        "points": best_rule["points"] if best_rule else [],
        "evidence": best_rule["evidence"] if best_rule else ["规则未能给出确定结论"],
        "suggestions": _rule_suggestions(best_rule, features),
        "confidence": best_rule["confidence"] if best_rule else 0.3,
        "engine": "rule",
        "features": features,
    }

    # 仅规则组：到此为止，不触发任何大模型调用。
    if mode == "rule":
        return result

    if llm is None or not getattr(llm, "available", False):
        return result

    try:
        prompt = PROMPT.format(
            verdict=verdict, problem=problem_title or "未提供",
            features=json.dumps({k: v for k, v in features.items() if k != "risks"},
                                ensure_ascii=False),
            et_list=et_prompt_list(), kp_list=kp_prompt_list(), code=code[:6000],
            language=lang, language_label={"cpp": "C++", "python": "Python", "java": "Java"}[lang],
        )
        client = llm._client  # noqa: SLF001
        resp = client.chat.completions.create(
            model=llm.settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=900, timeout=45,
        )
        payload = _parse_json(resp.choices[0].message.content or "")
        if not payload:
            return result
        ok, reason = _validate(payload, features, code)
        if not ok:
            log.info("模型输出未通过校验（%s），保留规则结论", reason)
            result["evidence"].append(f"模型结论被校验拒绝：{reason}")
            return result

        # 融合组：规则置信度高且与模型冲突时，以规则为准（可确定的事实优先）。
        # 仅大模型组（llm）跳过这一步，让通过校验的模型结论直接生效，
        # 这样三组才具备可比性——否则 llm 组会被规则覆盖，退化成 hybrid。
        if (mode == "hybrid" and best_rule
                and best_rule["confidence"] >= 0.7
                and payload["cause_code"] != best_rule["cause_code"]):
            result["evidence"].append(
                f"模型倾向 {payload['cause_code']}，但静态特征更支持 {best_rule['cause_code']}，以规则为准")
            return result

        result.update({
            "cause_code": payload["cause_code"],
            "cause_label": et_name(payload["cause_code"]),
            "points": payload.get("points", []),
            "evidence": [str(e)[:160] for e in payload.get("evidence", [])][:4],
            "suggestions": [str(s)[:200] for s in payload.get("suggestions", [])][:3],
            "confidence": payload["confidence"],
            "engine": "llm" if mode == "llm" else "hybrid",
        })
    except Exception as exc:
        log.warning("大模型诊断失败，使用规则结论：%s", exc)
        result["evidence"].append(f"模型调用失败（{type(exc).__name__}），已降级为规则诊断")
    return result


def _rule_suggestions(rule: dict | None, features: dict) -> list[str]:
    if not rule:
        return ["补充题目链接与判题结果", "贴出关键代码段而不是整段代码"]
    mapping = {
        "ET06": ["把递归改成显式栈或迭代写法", "检查递归退出条件与状态重复访问"],
        "ET07": ["估算数据规模与复杂度是否匹配", "考虑用前缀和/二分/哈希降一维"],
        "ET09": ["关闭 cin/cout 流同步或改用 scanf/printf", "把 endl 换成 '\\n'"],
        "ET14": ["把相关变量改为 long long", "取模前先转 long long 再乘"],
        "ET03": ["核对数组下标上界与循环边界", "给数组多开 5-10 个冗余位"],
        "ET01": ["先本地编译一次，看第一条报错"],
    }
    return mapping.get(rule["cause_code"], ["对照静态特征逐条排查"])
