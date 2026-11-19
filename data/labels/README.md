# 诊断评测标注集说明

开题报告承诺了「仅规则 / 仅大模型 / 规则+大模型融合」三组对比验证，
`eval-diagnosis` 命令即为此实现。**准确率必须来自真实人工标注，不得编造。**

## 使用步骤

1. 复制模板：`data/labels/diagnosis_samples.template.json` → `data/labels/diagnosis_samples.json`
2. 人工标注至少 10 条（建议 40 条，与开题报告「40 份人工标注集」一致）。
   每条字段：
   - `code`：学生提交的代码（真实现场样本或从题库派生）；
   - `verdict`：`AC/WA/TLE/RE/MLE/CE/PE/OLE`；
   - `language`：`cpp` / `python` / `java`；
   - `gold_cause`：**人工判定**的错因编号 `ET01`~`ET15`（错因表见 `src/acm_coach/diagnosis/knowledge.py`）；
   - `problem_title`、`gold_points`（`KP01`~`KP30`）可选。
3. 运行：
   ```bash
   python -m acm_coach.cli eval-diagnosis
   ```
   报告写入 `data/reports/diagnosis_eval_report.json`，含三组的严格准确率、
   规则组的 top3 命中率、知识点命中率，以及逐样本明细（可复核）。

## 真实性约束（脚本已强制）

- 缺标注集 / 少于 10 条 / 任一 `gold_cause` 为空 → **直接报错退出**，不产出报告；
- 绝不用规则输出反推标注答案（否则会得到虚假的 100%）；
- 未配置 DeepSeek API Key 时，`llm`/`hybrid` 会退化为规则结果，
  报告里 `llm_available` 如实标为 `false`，不冒充融合效果。配好 Key 再跑才有对比意义。

## 指标口径

- `cause_accuracy`：`诊断 cause_code == gold_cause` 的严格准确率，对应开题报告「70%」目标；
- `top3_accuracy`：仅规则组，gold 进入规则候选前 3 即命中（宽口径，与严格口径分开列）；
- `point_hit_rate`：诊断知识点与 `gold_points` 有交集的比例（提供了 gold_points 才统计）。
