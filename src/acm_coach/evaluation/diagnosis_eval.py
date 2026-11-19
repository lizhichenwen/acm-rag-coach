# -*- coding: utf-8 -*-
"""诊断效果离线评测：仅规则 / 仅大模型 / 规则+大模型融合 三组对比。

对应开题报告「主要研究方法 4」与「关键性问题 3」承诺的对比验证。

真实性铁律（务必先读）：
- 本脚本**不生成、不推断、不编造任何标注答案**。ground truth 必须来自
  `data/labels/diagnosis_samples.json`，由人工标注；缺文件、缺字段、条数不足
  都会明确报错退出，绝不用规则输出反过来当标准答案（那会得到虚假的 100%）。
- 标注集格式：JSON 数组，每条至少含
    {"code": "...", "verdict": "TLE", "language": "cpp", "gold_cause": "ET07"}
  可选 "problem_title"、"gold_points": ["KP30"]。
- 只有配好 DeepSeek API Key 时，llm / hybrid 两组才有意义；无 Key 时
  这两组会退化为规则结果，报告里会如实标注 "llm_available": false，
  不会把降级结果冒充融合效果。

指标口径（与 evaluation/metrics.py 风格一致）：
- cause_accuracy：诊断出的错因编号 == gold_cause 的比例（开题报告「70%」对应口径）。
- top3_accuracy：规则多候选时，gold 进入前 3 候选即算命中——给规则组的宽口径，
  避免把「给了多个候选」误判为完全错误；报告同时列出严格口径，不混淆。
- point_hit_rate：gold_points 与诊断 points 有交集的比例（仅当标注集提供 gold_points）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger
from acm_coach.diagnosis.engine import diagnose, rule_candidates
from acm_coach.diagnosis.features import extract_features, normalize_language

log = get_logger("diag_eval")

MODES = ("rule", "llm", "hybrid")
MIN_SAMPLES = 10  # 低于此条数统计无意义，直接拒绝


def load_labelset(path: Path | None = None) -> list[dict]:
    """读人工标注集；任何格式/字段问题都 fail fast，不悄悄跳过样本。"""
    settings = get_settings()
    path = path or (settings.data_dir / "labels" / "diagnosis_samples.json")
    if not path.exists():
        raise FileNotFoundError(
            f"人工标注集不存在：{path}\n"
            "请按 README 说明创建 data/labels/diagnosis_samples.json，"
            "每条含 code / verdict / language / gold_cause 字段。"
            "本脚本拒绝用规则输出伪造标注答案。"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("标注集必须是 JSON 数组")
    if len(data) < MIN_SAMPLES:
        raise ValueError(f"标注集仅 {len(data)} 条，少于 {MIN_SAMPLES} 条，统计不具备参考价值")

    cleaned: list[dict] = []
    for i, item in enumerate(data, 1):
        missing = [k for k in ("code", "verdict", "gold_cause") if not item.get(k)]
        if missing:
            raise ValueError(f"第 {i} 条标注缺字段 {missing}，请补全后重跑")
        cleaned.append({
            "code": str(item["code"]),
            "verdict": str(item["verdict"]).upper(),
            "language": normalize_language(item.get("language")),
            "problem_title": str(item.get("problem_title", "")),
            "gold_cause": str(item["gold_cause"]).upper(),
            "gold_points": list(item.get("gold_points") or []),
        })
    return cleaned


def _rule_top3(code: str, verdict: str, language: str) -> list[str]:
    """规则组的多候选（按置信度降序），用于 top3 宽口径。"""
    features = extract_features(code, verdict, normalize_language(language))
    ranked = sorted(rule_candidates(features), key=lambda r: r["confidence"], reverse=True)
    return [r["cause_code"] for r in ranked]


def run_diagnosis_eval(label_path: Path | None = None, output_path: Path | None = None) -> dict:
    samples = load_labelset(label_path)
    settings = get_settings()

    # 只为评测构造一次 LLM 客户端（可能未配置 Key → available=False）
    from acm_coach.services.llm import LLMClient
    llm = LLMClient()
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": len(samples),
        "llm_available": llm.available,
        "label_source": str(label_path or (settings.data_dir / "labels" / "diagnosis_samples.json")),
        "metric_def": {
            "cause_accuracy": "诊断 cause_code == gold_cause 的严格准确率",
            "top3_accuracy": "仅规则组：gold_cause 进入规则候选前 3 的宽口径命中率",
            "point_hit_rate": "诊断知识点与 gold_points 有交集的比例（标注集提供时才统计）",
        },
        "modes": {},
    }

    for mode in MODES:
        correct = 0
        top3_hit = 0
        point_hit = 0
        point_total = 0
        per_sample = []
        for s in samples:
            result = diagnose(
                s["code"], s["verdict"], s["problem_title"],
                llm=llm, language=s["language"], mode=mode,
            )
            pred = result["cause_code"]
            hit = pred == s["gold_cause"]
            correct += int(hit)

            # top3 只对规则组有意义：其他组的「候选」概念不存在，保持口径清晰
            top3 = _rule_top3(s["code"], s["verdict"], s["language"]) if mode == "rule" else []
            t3 = s["gold_cause"] in top3[:3]
            top3_hit += int(t3) if mode == "rule" else 0

            if s["gold_points"]:
                point_total += 1
                overlap = set(result.get("points") or []) & set(s["gold_points"])
                point_hit += int(bool(overlap))

            per_sample.append({
                "gold": s["gold_cause"], "pred": pred, "hit": hit,
                "engine": result.get("engine", ""),
            })

        n = len(samples)
        report["modes"][mode] = {
            "cause_accuracy": round(correct / n, 4),
            "top3_accuracy": round(top3_hit / n, 4) if mode == "rule" else None,
            "point_hit_rate": round(point_hit / point_total, 4) if point_total else None,
            "point_samples": point_total,
            "per_sample": per_sample,
        }

    out = output_path or (settings.data_dir / "reports" / "diagnosis_eval_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("诊断三组对比完成，报告：%s", out)
    return report
