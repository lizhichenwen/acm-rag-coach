# ACM 智能陪练系统 v4 — 代码审查报告

- 审查目标：`D:/ACM_RAG/acm_rag_v4/`
- 审查依据：附件2毕业论文任务书、附件3毕业论文开题报告
- 审查范围：全量 src/ + tests/ + 参考 data/reports/
- 技术栈：Python 3.13 / FastAPI / Streamlit / SQLAlchemy / SQLite(WAL) / ChromaDB / bge-small-zh-v1.5 / DeepSeek API
- 总体评分：**8.5 / 10**（架构与工程质量在本科毕设中属上乘；扣分集中在实证数据缺失）
- 结论：⚠️ 修改后可冲优秀 —— 真实 bug 已修复，实证缺口已提供可复现工具，但**真实实验数据必须由你本人标注产出**。

---

## 一、风险摘要

代码质量本身很高：四层架构清晰、检索四段可消融、防幻觉（拒答/来源分级/证据落地）、安全（PBKDF2+JWT+token_version+审计 append-only）都落到了实处。静态脚本报的 3 个「Critical」经人工核实**全部是误报**。真正影响冲优秀的问题有三个层次：

1. **两处真实逻辑 bug**（均已修）：诊断的循环嵌套层数算错，会漏判 TLE 错因；对话链路的历史消息硬截断 + 意图误判，会让模型"看不全学生代码"而拒答（详见第四节 M-1、M-2）。两者分别压低「诊断准确率」与「多轮问答可用性」。
2. **一项开题承诺缺失**（已提供工具）：开题报告承诺的「三组诊断对比实验」和「70%准确率」在代码与数据里都没有；检索命中率实测最高约 74.7%，未达 80%。
3. **诊断模块零测试**（已补）：任务书重点研究问题之一却没有任何单测。

---

## 二、需求符合性核查（对照两个附件）

| 指标 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 题库条数 | ≥256 | 8558 | ✅ 远超 |
| 知识点覆盖 | ≥8 类 | 9 domain / 75 知识点 | ✅ |
| 功能测试用例 | ≥15 | 39（+本次新增22=61） | ✅ |
| 知识点/错因体系 | 30 KP / 15 ET | 30 / 15 | ✅ 完全吻合 |
| 用户角色 | 学生/管理员/审计员 | 三种均有且校验完整 | ✅ |
| 技术栈六项 | 见附件 | 全部属实 | ✅ |
| 三组诊断对比 | 开题报告承诺 | 原代码无 | ⚠️ 已补工具，待你标注跑数 |
| 诊断准确率70% | 开题报告预期 | 无实测报告 | ⚠️ 待你标注跑数 |
| 检索命中率80% | 开题报告预期 | 最高约74.7% | ❌ 未达标，见第五节 |

---

## 三、静态脚本 Critical 逐条核实（全部误报，未改）

- `[C-1] 硬编码密钥` @ session.py:131 — 实为迁移字典的**列名** `must_change_password`，值是 DDL 片段，不是密钥。密钥走 `config.py::resolve_secret_key()` 环境变量/持久化文件。
- `[C-2] SQL 注入` @ session.py:150 — `ALTER TABLE {table} ADD COLUMN {name}` 拼接的是**代码内硬编码的 additions 字典**，无任何用户输入，不构成注入。ORM 查询全部参数化。
- `[C-3] 不安全反序列化` @ bm25.py:130 — `pickle.load` 加载的是本地 `data/index/bm25.pkl`（系统自身生成），非用户可控输入；属可接受的内部持久化。

---

## 四、已修复的问题

### [M-1] 循环嵌套层数计算错误，漏诊 TLE — 🟠 Major（已修）

- 位置：`src/acm_coach/diagnosis/features.py::loop_nesting()`
- 问题：旧实现只在遇到 `for/while` 时入栈，却对**每一个** `}` 出栈并减层数。于是循环体里任意一个 `if {...}` 的右括号都会把循环层数抵消掉。实测：「for 里先写 if、再嵌一层 for」实际 2 层，旧版只算 1 层。
- 后果：`rule_candidates` 的 `loop_nesting >= 3 → ET07（复杂度不达标）` 判定失效。一个三层循环、中间夹着 if 的典型 TLE 代码，修复前诊断为 ET09（IO 慢，靠 endl 误判），修复后正确诊断为 ET07、置信度 0.8。直接压低诊断准确率。
- 根因：栈的语义错位——记录的"循环块"与消费的"所有块"不是同一类对象。
- 正确方案（已落地）：改为花括号栈，每个 `{` 携带「是否由 for/while 引入」的标记入栈，只有循环块出栈时才减层数；并用括号深度区分 for 头内的分号与单语句循环体。7 个场景验证全过。

### [G-1] 诊断模块零测试覆盖 — 🟠 Major（已补）

- 位置：新增 `tests/test_diagnosis.py`
- 问题：诊断是任务书「重点需要研究的问题 3」，此前 39 个测试无一覆盖特征提取、规则判定、AC 短路、白名单校验。
- 已补：22 个单测，覆盖语言归一化、7 类嵌套场景、跨语言特征键一致性、ET07 规则判定、AC 不编造错因、无 LLM 降级、递归检测等。

### [G-2] 三组对比实验能力缺失 — 🟠 Major（已补工具）

开题报告「研究方法 4」明确要「仅规则/仅大模型/融合」三组对比，原代码无法分组复现。已做：
- `diagnosis/engine.py::diagnose()` 新增 `mode` 参数（`rule`/`llm`/`hybrid`，默认 `hybrid`，**现有调用行为完全不变**）；`llm` 组跳过"规则覆盖模型"逻辑，保证三组可比。
- 新增 `evaluation/diagnosis_eval.py`：读人工标注集跑三组，输出严格准确率 + 规则组 top3 命中率 + 知识点命中率 + 逐样本明细。
- 新增 CLI 命令 `python -m acm_coach.cli eval-diagnosis`。
- **真实性铁律**：缺标注集、少于 10 条、任一 `gold_cause` 为空 → 直接报错退出；绝不用规则输出反推标注答案。已实测拒绝生效。

### [M-2] 对话链路"代码读不全" — 🟠 Major（已修，2026-11-19）

- 用户实测反馈：贴出思维题代码后，陪练系统回答"我没看完代码/看不全"。经规则链路核查，非模型能力问题，而是三处叠加：
  1. `services/chat_service.py::build_messages` 把每条历史消息从头硬截 `[:1500]`，题面一长代码整段被切掉；
  2. `services/intent.py::classify_intent` 把"你能否看懂我的代码"判为 `chitchat`，其提示词明确要求"不要扯到具体题目"；
  3. `SYSTEM_PROMPT` 原则 2「只依据参考资料、不得编造结论」未区分"标准题解"与"学生自带代码"，模型不敢分析。
- 已落地：
  - `config.py` 新增 `history_text_cap=3000 / history_code_cap=6000 / history_nl_keep=600`；
  - `chat_service.py` 新增 `keep_history_content()` 代码感知保真，替换 `[:1500]`；`SYSTEM_PROMPT` 增「代码分析例外」（优先级高于原则 2/5）；`INTENT_GUIDANCE` 的 chitchat/algorithm 分支允许代码分析；`level=none` 约束区分题解与代码；
  - `intent.py` 新增 `CODE_REVIEW_PATTERNS` + 兜底，代码复盘类追问落 `algorithm`；
  - 新增 `tests/test_chat_context.py`（11 个用例）。
- 验证：pytest 全套 **71 个通过**；端到端脚本（临时库 + 假 LLM，不联网）确认意图=`algorithm`、历史代码完整送达模型、提示词含正向禁令"不得说没看完"。**注意**：该验证证明的是"代码送达 + 提示词放行"，最终自然语言回答质量仍取决于 DeepSeek 本身，需用真实 API 重放同一对话确认（详见交付说明）。
- 对论文/答辩的影响：这是"意图识别 + 多轮上下文"两节的实证案例，可用于说明系统的防幻觉边界设计——**拒答只针对题库题解，不针对学生代码**。

补充（同日，用户追问的前端报红）：截图中的 `NotFoundError: removeChild` 全栈在 `react-dom`，
是浏览器端 Streamlit 渲染错误，**非 DeepSeek 底座问题，答案文本本身完整正确**。
根因为流式阶段逐 token 调 `markdown()` 重绘、回答里的 `<bits/stdc++.h>` 等裸尖括号被当 HTML
标签解析，节点树重绘时结构跳变。已修：流式改 `text()`（纯文本）、完成后一次性 `markdown()`，
新增 `safe_md()` 只转义代码块外尖括号；`tests/test_frontend_render.py` 锁定契约。全套 77 测试通过。

---

## 五、需要你自己完成的事（我不能代做，代做即造假）

1. **标注真实样本**：按 `data/labels/README.md`，把模板复制为 `data/labels/diagnosis_samples.json`，人工标注 40 条（与开题报告一致），然后跑：
   ```bash
   python -m acm_coach.cli eval-diagnosis
   ```
   报告写到 `data/reports/diagnosis_eval_report.json`。这几组数字才是论文和答辩能用的。

2. **检索命中率 80% 未达标**：现有 `eval_report.json` 实测 Hit@5 最高约 74.7%。不要改数字冒充达标。可选的真实改进方向：
   - 用 calibrate 重新标定 strong/weak 阈值（换库规模后绝对值阈值会失效）；
   - 增加精讲条目占比或调整 RRF 权重、curated_quota_ratio；
   - 或在论文里如实写 74.7% 并分析原因——诚实报告比虚假达标更能经得起答辩追问。

---

## 六、架构评估

- 模块边界：清晰。API 层只做 HTTP 与校验，服务层只做编排（不含 SQL/HTTP），仓储层封装 SQLAlchemy，检索层独立可消融。换前端/向量库/模型互不牵连。
- 依赖方向：单向，未见循环依赖。
- 主要风险：诊断链路缺少真实评测数据支撑，是"工程完成度 > 实证完成度"的唯一明显短板。补上标注实验后，论文的主张与代码即可闭环。

---

## 七、补充：G4 重排启用与消融隔离修复（2026-11-21）

本轮在既有代码上修了两处会**让 G4 消融失真**的问题，并重跑实验；不改诊断模块，不动标注集。

1. **全局配置串组**（真实 bug，已修）：`rag/rerank.py` 原先在 `get_reranker()` 里读 `get_settings()` 单例，且 `_MODEL_TRIED` 在 `enable_rerank` 判定前置位。评测 runner 给每个消融分组深拷贝 settings（G1/G2/G3 关重排、G4 开重排），因此模型一旦缓存：① 三组"不重排"配置会被静默重排；② 若先跑关重排分组，G4 永远拿不到模型。现改为 `rerank(query, hits, settings)` 透传调用方配置，`index.py` 同步传入；`tests/test_rerank.py` 新增 3 个用例锁定该行为。
2. **三态可观测**：新增 `rerank_backend()` 并写入检索结果 `meta.rerank_backend`（`off / cross-encoder / fusion-fallback`）。启动日志 `重排模型加载成功` 与结果 meta 双重佐证，避免"以为开了其实降级"。
3. **模型与部署**：`BAAI/bge-reranker-base` 已下载到本机 HF 缓存（约 1.1 GB），G4 现走真实 cross-encoder；`.env(.example)` 补齐 `RERANK_MODEL / RERANK_DOWNLOAD`；`docker-compose.yml` 新增宿主 HF 缓存只读挂载（Dockerfile 设了 `HF_HUB_OFFLINE=1`，不挂就必然降级）。

**证据边界**：新指标来自 `data/reports/eval_report.json`（同 150 查询、同 qrels 规则），只替换旧表，不改标注、不改指标实现；G3 与 G4 的差值现在才具备"重排贡献"的解释力。旧表中 "g4 ≈ g3 因未缓存" 的描述已作废。注意 G1–G3 数字同时上移（如 G3 Hit@1 0.607→0.653），原因是旧版它们被融合打分静默重排，修复后才是真"无重排"。

**新结果（负结果，需如实写进论文）**：G4 vs G3 的 Hit@1 / MRR@10 / NDCG@10 为 0.633 vs 0.653 / 0.667 vs 0.696 / 0.613 vs 0.627，即重排**未带来头部精度收益**，Recall@100 持平。合理解释是合成 query 含精确题名、BM25+RRF 已把目标排到第 1。延迟代价极大（评测深度 P95 8.5s）。**不得把 G4 描述为"精度提升"或声称达到 ≤800ms 目标**；答辩口径为"重排作为可选高精度模式已完成工程接入与消融验证，本数据集上未复现收益，CPU 部署默认关闭"。

**延迟优化（2026-11-21 同日追加）**：定位到真正的瓶颈不是字符数而是 token 数——语料 p99 仅 423 字符（≈300 token），`RERANK_DOC_CHARS=1200` 从不触发截断，模型一直按 512 上限跑满。落地三项：① `RERANK_MAX_LENGTH=192`（与 512 排序 Spearman 0.965，再低掉到 0.756）；② `RERANK_BATCH_SIZE=16`；③ 结果 LRU 缓存（键含 query+候选 id 序列）。实测生产 `top_n=24` **冷启动 4.1s→2.85s（-30%）、重复问 14ms**。ONNX fp32 比 PyTorch 慢、int8 量化因 shape inference 错误失败、PyTorch 动态量化不支持该激活，均已尝试并记录。优化只改推理参数与缓存，不改变打分公式，因此上面的精度结论仍然成立；81 个测试全过。

**启动延迟优化（2026-11-21 同日追加）**：用户反馈"登录后启动慢"。分段实测定位到服务冷启动 **15.7s**，其中 `import sentence_transformers` 单独占 **11.3s**（模型 load 仅 236ms、`init_db` 495ms、PBKDF2 校验 258ms）——瓶颈是**导入**不是模型推理。修复：把索引与模型加载从 `lifespan` 同步路径移到 daemon 线程（`rag/index.py::start_warmup()`），`healthz` **15.7s→1.1s**；`ensure_loaded()` 加加载锁 + `Event`，保证首个请求在预热完成前阻塞等待、不读半加载索引（`tests/test_startup.py` 用假 BM25 模拟慢加载锁定该并发行为）；`WARMUP_RERANKER=true` 让重排模型一并后台预热，消除首条对话的 3s 等待（2GB 内存机器可关）。`/readyz` 响应新增 `warmup` 进度字段。上述改动不动检索/生成算法，83 个测试全过。比赛部署口径：serve 后约 1s 可登录，等 `/readyz` 返回 200 再开始演示问答。
