# ACM 智能陪练系统 v4

> 面向 ACM 程序设计竞赛的 RAG 智能陪练系统 —— 企业级分层重构版
> 本科毕业设计《面向 ACM 程序设计竞赛的 RAG 智能陪练系统的设计与实现》

v3 是 Streamlit 单体 + 77 条题库；v4 把它重做成**四层架构的服务端 + 分层检索 + 多源语料流水线**：
FastAPI 服务层、仓储层、混合检索（BM25 + 向量 + RRF + 重排）、8,558 条语料、离线评测与阈值标定。

## 一、这套系统解决什么问题

学生做题卡住时，通用大模型会给两种坏答案：**直接甩完整代码**（学会抄）和**编造题解**（学会错）。
本项目把"陪练"约束成三件事：

1. **引导式**：不直接给可提交代码，只给关键步骤 + 反问；
2. **可溯源**：回答必须引用检索到的题库条目 `[1][2]`，且区分「精讲」与「考点提示」两种可信度；
3. **敢拒答**：检索置信度不足时明确说"题库未收录"，而不是拿勉强沾边的条目硬答。

边界说明（2026-11-19 修正）：上述"敢拒答"只约束**这道题的标准解法/官方复杂度**——
学生自己贴出的代码属于"学生提供的材料"，可以、也应当就代码本身做具体分析（做了什么、
逻辑对错、复杂度、为何能过样例），不受"题库未收录"限制。此前这条边界没写清，导致模型
把"分析学生代码"也当成"回答占题"，出现"我没看完代码"的拒答。现在 `SYSTEM_PROMPT` 里
有高优先级的「代码分析例外」明确放行。

## 二、架构

```
┌──────────────┐   REST + SSE    ┌──────────────────────────────────────────┐
│  Streamlit   │ ──────────────► │  API 层 (FastAPI)                        │
│  前端(瘦客户端)│ ◄────────────── │  auth / chat / knowledge / diagnosis ...  │
└──────────────┘                 └───────────────┬──────────────────────────┘
                                                 │
                    ┌────────────────────────────▼─────────────────────────┐
                    │ 服务层（编排，不含 SQL / 不含 HTTP）                  │
                    │ ChatService · DiagnosisService · IndexService         │
                    └───────┬──────────────────────────────┬───────────────┘
                            │                              │
              ┌─────────────▼────────────┐   ┌─────────────▼──────────────┐
              │ 检索层                   │   │ 仓储层（SQLAlchemy 2.0）    │
              │ BM25 → 向量 → RRF → 重排 │   │ users/sessions/messages/... │
              │ → 三档相关性截断          │   └─────────────┬──────────────┘
              └─────────────┬────────────┘                 │
                            │                              │
              ┌─────────────▼────────────┐   ┌─────────────▼──────────────┐
              │ ChromaDB + docstore      │   │ SQLite (WAL)                │
              └──────────────────────────┘   └────────────────────────────┘
```

设计要点（论文可直接引用）：

| 决策 | 理由 | 代价 |
|---|---|---|
| 前后端分离（Streamlit 只做 API 客户端） | 服务可独立部署/测试，接口可被小程序复用 | 多一个进程 |
| 检索分四段（召回→融合→重排→截断） | 每段可单独消融，指标可归因 | 调参维度变多 |
| 阈值由 `calibrate` 分位数标定 | 库规模从 77 涨到 8,558，绝对值阈值必然失效 | 换模型/换量级要重标 |
| 爬取语料只存元数据摘要 | 版权边界 + 不产生"伪题解" | 只能回答"考什么"，不能回答"怎么写" |
| 无 API Key 时降级为抽取式作答 | 任何环境可演示、可自动化测试 | 回答质量下降 |

## 三、目录

```
acm_rag_v4/
├─ src/acm_coach/
│  ├─ config.py            # pydantic-settings，唯一配置出口
│  ├─ core/                # logging(结构化) / security(JWT+PBKDF2) / errors / metrics
│  ├─ db/                  # models / session(WAL+迁移) / repositories
│  ├─ rag/                 # text / bm25 / vector / rerank / query / packer / index
│  ├─ services/            # chat / knowledge / index / intent / llm
│  ├─ diagnosis/           # knowledge(30 知识点+15 错因) / features / engine
│  ├─ crawler/             # client(限速重试) / sources(CF+AT) / taxonomy / builder
│  ├─ evaluation/          # metrics(唯一口径) / runner(消融) / calibrate(标定)
│  ├─ api/                 # app / deps / schemas / routers/*
│  └─ cli.py               # init-db / ingest / crawl / index / eval / calibrate / serve
├─ frontend/app.py         # Streamlit 瘦客户端（9 个功能页 + 答辩演示指引）
├─ data/                   # curated(96) + chroma + index + taxonomy + reports
├─ scripts/                # 冒烟、报告、清理
├─ tests/                  # 单元 + 接口测试（含 test_chat_context 对话链路回归）
└─ docs/                   # 01 定位路线 / 02 资产盘点 / 03 加工 / 04 检索调优 / 05 安全合规
```

### 前端与画像口径（2026-11-12 与代码对齐）

`frontend/app.py` 现为 **9 个功能页 + 答辩演示指引**，覆盖任务书的题库管理、检索增强问答、意图识别、多轮对话、代码诊断、知识点画像、薄弱点推荐、用户与权限、数据统计；仍是瘦客户端，只发 HTTP、不访问数据库。逐页映射与演示步骤见该文件头注释及「答辩演示指引」页。

- **掌握度口径**（`db/repositories.py::update_skills`）：初始 `0.3`，钳制在 `[0, 1]`；判题结果 `AC` 视为已掌握 → `+0.12`，其余结果 → `-0.06`；每次诊断命中的知识点 `attempts + 1`。画像按掌握度升序，最低的 N 个即薄弱点。2026-11-12 修正了旧版「AC 反而扣分、做错反而加分」的方向错误。
- **薄弱点推荐没有独立后端接口**：前端取 `GET /api/v1/diagnosis/skills` 的最低掌握度知识点，逐点调 `GET /api/v1/knowledge` 检索题目，再用 `POST /api/v1/practice/{item_id}` 标记已练。这是前端编排的闭环，答辩需如实说明，不要说成后端推荐服务。
- **前端错误处理**：统一 `ApiError`（`code / message / trace_id / details`）按状态码分派（401 重新登录、403 角色不足、404 自动重建会话、5xx 看 trace_id）；SSE 请求先判 HTTP 状态码再解析事件，不再把 4xx 的 JSON 错误体当事件流读，也不再把权限错误笼统提示成「API 未启动」。

### 对话链路口径（2026-11-19 与代码对齐）

- **意图识别**（`services/intent.py`，规则版）：五类 `greeting/chitchat/code_request/algorithm/strategy`。
  "你能否看懂我的代码""这段代码对不对""刚才那份的复杂度"等**代码复盘类追问**归入 `algorithm`；
  纯学习方法闲聊才归 `chitchat`。意图只影响提示词策略，不影响是否检索。
- **多轮上下文保真**（`services/chat_service.py::keep_history_content`）：历史消息不再从头硬截 1500 字。
  短消息原样保留；长消息按是否含代码分流——纯文本留尾部，代码消息拆成自然语言与代码两段，
  代码按 `history_code_cap` 完整保留，避免"题面太长把代码切没、模型答没看完"。三个上限
  （`history_text_cap / history_code_cap / history_nl_keep`）在 `config.py` 可调。
- 上述行为由 `tests/test_chat_context.py` 锁定（意图判定 + 代码保真 + 提示词契约，11 个用例）。

> `docs/` 下 5 篇是**规划/设计基线**（量化数值多为设计目标），接口路径与数量一律以运行中的 `/openapi.json` 为准：当前 **30 个路径 / 34 个端点**。

## 四、快速开始

```bash
# 1) 依赖（建议虚拟环境）
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2) 配置
copy .env.example .env      # 留空 DEEPSEEK_API_KEY 也能跑（降级模式）

# 3) 建库 + 导入语料 + 建索引
python -m acm_coach.cli init-db
python -m acm_coach.cli ingest
python -m acm_coach.cli index --max-vectors 1500   # 分批建向量，可反复执行续跑

# 4) 启动
1.进入D:\ACM_RAG\acm_rag_v4：cmd后输入venv\Scripts\activate
2.python -m acm_coach.cli serve          # http://127.0.0.1:8000/docs
3.新窗口D:\ACM_RAG\acm_rag_v4：streamlit run frontend/app.py          # http://localhost:8501
```

> 首次登录：管理员口令见 `.env` 的 `ADMIN_PASSWORD`；未配置时启动日志会打印**一次性随机口令**。
> 不提供任何默认口令（`admin123` 之类）。

## 五、语料与数据

| 来源 | 条数 | 内容 | 可信度 |
|---|---|---|---|
| curated 精讲 | 96 | 66 题解 + 11 竞赛元知识 + 19 知识点卡 | 可作题解依据 |
| Codeforces | 5,962 | 元数据（标题/rating/tags/通过数）加工的考点摘要 | 仅考点提示 |
| AtCoder | 2,500 | 同上（1,771 条有难度，729 条标"未知"） | 仅考点提示 |
| **合计** | **8,558** | | |

- CF 全量 11,503 → 分层采样 6,000（易/中/难 45%/35%/20%）→ 门禁丢弃 38 → 5,962
- 难度分布：简单 3,835 / 中等 3,000 / 困难 1,723（20.1%）
- 入库报告：`data/reports/ingest_report.json`；标签分布：`data/taxonomy/cf_tag_stats.json`

**版权边界**：只存平台 API 返回的元数据 + 自撰考点摘要，**不抓题面、不存样例**；
入库门禁按形态限制正文长度（考点提示 ≤500 字），并从结构上排除"整篇题面"特征。

**溯源契约（`knowledge_items.meta_json`，docs/05 §6.7）**：8 键
`entry_type/content_origin/processed_by/owner_type/fetched_at/crawl_batch/license/mirror`，
取值严格沿用文档枚举（`self_written|metadata`、`solution|concept|strategy|hint`、
`user|platform`、`internal-only|Codeforces-Terms|AtCoder-Terms`、
`official-api|kenkoooo|n/a(self_written)`）。`mirror` 是通道维度：AtCoder 数据经
kenkoooo 镜像取得，权利人主张删除时可据此回答"该停哪个通道"；两条写入路径
（`knowledge_service` / `crawler.builder`）与全部 8,558 条历史数据已统一——
历史数据由 `python scripts/migrate_meta_json.py` 幂等回填，缺失键实测为 0；
`tests/test_meta_contract.py` 对全库做可执行断言。历史数据无抓取时间戳，
按契约如实标 `unknown`/`legacy`，**不用 `created_at` 冒充**。
`source` 列维持 `curated/cf/atcoder`（检索 type_boost 承重契约，见 docs/05 §6.7 仲裁说明）。

## 六、检索与评测

```bash
python -m acm_coach.cli eval --configs g1,g2,g3,g4 --max-queries 150
python -m acm_coach.cli calibrate --sample 120
```

- `g1` BM25 / `g2` 向量 / `g3` 混合 / `g4` 混合+重排，四组消融
- 指标：Hit@1/3/5/10、MRR@10、NDCG@10、Recall@100、P50/P95 延迟、空结果率
- 相关性标注规则写在 `evaluation/runner.py`（目标条目 rel=2，标签 Jaccard≥0.5 的 rel=1，上限 8 条）
- 阈值标定输出 `data/reports/calibration.json`，含"库规模+模型+度量"指纹
- `--verify` 护栏：type_boost 开启但候选池 curated 条目为 0 时直接报错退出，防加权静默失效

**实测结果**（数据源：`data/reports/eval_report.json`，生成于 2026-11-08 01:43，150 查询，语料 8,558，向量覆盖率 100%。项目文档、答辩 PPT 与演示口径均以此表为准）：

| 配置 | Hit@1 | Hit@10 | MRR@10 | NDCG@10 | Recall@100 | P50 | P95 |
|---|---|---|---|---|---|---|---|
| g1 BM25 | 0.667 | 0.747 | 0.702 | 0.627 | 0.607 | 23.1ms | 27.7ms |
| g2 向量 | 0.280 | 0.333 | 0.294 | 0.233 | 0.206 | 39.5ms | 45.3ms |
| g3 混合 | 0.607 | 0.733 | 0.658 | 0.609 | **0.622** | 65.0ms | 73.4ms |
| g4 混合+重排 | 0.607 | 0.733 | 0.658 | 0.609 | 0.622 | 65.3ms | 75.9ms |

读法（论文请如实呈现，不要只挑好看的）：
- 合成查询含精确题名，词法通道天然占优，故 g1 头部精度最高；混合的 **Recall@100 最好**，
  说明向量路确实召回了 BM25 漏掉的相关条目，只是 RRF 把它们排在中段；
- g2 远低于 g1 是"短文本标题 + bge-small"的固有上限，恰是引入混合架构的动机；
- g4 ≈ g3 因 `bge-reranker-base` 未缓存，走融合打分降级路径（见已知限制 2）。

**三档门控口径**：`similarity = max(向量余弦, BM25 归一化)`，取两路置信度较强者——
只取向量分会把"BM25 第 1 但余弦一般"的正确条目挡在截断线外（实测误杀案例：「01背包内层循环
为什么必须倒序」的 01背包条目）。阈值由 `calibrate` 在**门关闭**状态测量裸分后回填
（2026-11-08：strong=0.9382 / weak=0.572，域内 p10=0.837 vs 域外 p90=0.552）。

## 七、安全基线

| 项 | 实现 |
|---|---|
| 口令 | PBKDF2-HMAC-SHA256，600k 次迭代，16 字节随机盐；哈希串自带迭代次数，校验时按该次数复算以兼容旧口令（**本版未做登录时 rehash 升级**） |
| 会话 | JWT HS256，claims 仅 `sub/role/ver/iat/nbf/exp/jti/iss`；`token_version` 支持改密即失效 |
| 越权 | 所有 `/{id}` 查询的 SQL 同时带 `user_id`；越权返回 404 而非 403（防枚举） |
| 爆破 | 同账号连续 5 次失败锁定 5 分钟 |
| 审计 | `audit_logs` append-only（SQLite 触发器拒绝 UPDATE/DELETE）；越权访问统一落 `access_denied` 审计 |
| 合规 | `knowledge_items` 具备 status/visibility/sensitivity/retention_days/meta_json；检索入口 fail-closed |
| 密钥 | 无硬编码；`.env` 不入库；`data/.secret` 自动生成 JWT 密钥 |
| 行为日志脱敏 | `retrieval_logs.query` 只存 `sha256 前 16 位 + 长度` 指纹（`core/security.fingerprint_query`），不落查询原文 |
| PIPL | 导出：`GET /api/v1/export/my-data`；删除权：`DELETE /api/v1/me` 按依赖顺序清除本人全部数据（含检索日志），注销动作留审计快照 |

## 八、已知限制（答辩请主动说明）

1. ~~向量索引未全量建完~~（已解决，2026-11-08 补齐至 8,558/8,558，覆盖率 100%）。若新增语料，执行 `python -m acm_coach.cli index --max-vectors 1500` 增量补建即可。
2. **重排模型未启用**：`bge-reranker-base` 未缓存，当前走"0.55 语义 + 0.30 词面 + 0.15 词法"融合打分（`rag/rerank.py` 的降级路径），启用方式见 `.env` 的 `RERANK_MODEL`。
3. **删除不是即时生效**：向量是快照，删除后需重建索引；`scripts/purge_source.py` 提供按来源清除。
4. **未做真实判题**：诊断基于静态特征 + 大模型，不执行代码、不接 OJ 沙箱。
5. **单机部署形态**：登录限流与缓存是进程内实现，多实例需换 Redis。
6. **知识点画像是规则式滑动更新**：不是 DKT/BKT 一类知识追踪模型，掌握度只由代码诊断命中的知识点与判题结果驱动，无跨会话长期记忆；推荐也不做难度自适应（属规划中的 P2 加分项）。
7. **前端流式渲染报红**（2026-11-19 已修）：此前逐 token 用 `markdown()` 重绘，回答里的 `<...>` 会被当 HTML 标签解析，偶发 `removeChild` 节点错乱（浏览器端报错，**与大模型底座无关，答案本身正确**）。现改为流式阶段用纯文本 `text()`、完成后一次性 `markdown()`，并新增 `safe_md()` 只转义代码块外的裸尖括号。由 `tests/test_frontend_render.py` 锁定。

## 兜底排查：如果你仍看到红框
- 该红框是 Streamlit 前端的 DOM 报错，不是后端 500、也不是 DeepSeek 故障：答案已正常显示即可忽略。
- 多为浏览器扩展（截图/翻译/油猴脚本）注入 DOM 引起，可用无痕窗口复测；Streamlit 版本相关问题升级 `streamlit` 亦可。

## 九、v3 → v4 变化

| 维度 | v3 | v4 |
|---|---|---|       
| 架构 | Streamlit 单体（页面函数直连 DB/RAG） | API / service / repository / infra 四层 |
| 语料 | 77 条 | 8,558 条（三类来源，含合规字段） |
| 检索 | 单路向量 + 同 query 多路 + 自实现 RRF | BM25 + 向量双路 + 加权 RRF + 重排 + 按题聚合 |
| 评测 | 25 问，指标散落在脚本里 | 150 问，指标单模块，四组消融，阈值标定 |
| 安全 | PBKDF2 200k，无令牌，无审计 | JWT + RBAC + 审计触发器 + 限流 + 合规字段 |
| 运维 | 有日志 | 结构化日志 + trace_id + Prometheus 指标 + 健康检查 |
| 工程 | 38 单测 | 单测 + 接口测试 + CI + Docker |

## 十、常用命令

```bash
python -m acm_coach.cli stats                 # 语料与运行态统计
python -m acm_coach.cli crawl --dry-run       # 只看抓取统计，不写库
python -m acm_coach.cli crawl --cf-limit 6000 --at-limit 2500
python scripts/build_ingest_report.py         # 重新生成入库报告
python scripts/purge_source.py --source cf    # 按来源清除（需 --yes）
pytest -q                                     # 跑测试
```
