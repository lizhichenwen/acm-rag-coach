# -*- coding: utf-8 -*-
"""命令行入口：`python -m acm_coach.cli <command>`。

子命令：
  init-db     建表 + 初始化管理员
  ingest      加载 data/ 下的语料并入结构化表
  crawl       抓取 Codeforces / AtCoder 元数据并入库（可 --dry-run 只看统计）
  index       重建 BM25 + 向量索引
  eval        离线检索评测（消融对比）
  eval-diagnosis  诊断三组对比评测（仅规则/仅大模型/融合，需人工标注集）
  calibrate   用真实语料标定相关性阈值
  serve       启动 API 服务
  stats       打印知识库与运行态统计
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger, setup_logging

log = get_logger("cli")


def _load_all(db) -> list:
    from acm_coach.db.repositories import KnowledgeRepository

    return KnowledgeRepository(db).all_for_index()


def cmd_init_db(args) -> None:
    from acm_coach.db import init_db

    init_db()
    print("数据库初始化完成：", get_settings().sqlalchemy_url)


def cmd_ingest(args) -> None:
    from acm_coach.db import init_db, session_scope
    from acm_coach.db.repositories import KnowledgeRepository
    from acm_coach.services import knowledge_service

    settings = get_settings()
    init_db(seed_admin=False)
    items = knowledge_service.load_directory(settings.knowledge_paths)
    records = knowledge_service.to_records(items)
    with session_scope() as db:
        inserted, updated = KnowledgeRepository(db).upsert_many(records)
    print(f"语料 {len(items)} 条 → 入库 新增 {inserted} / 更新 {updated}")


def cmd_crawl(args) -> None:
    from acm_coach.crawler.builder import build_records, crawl_summary
    from acm_coach.crawler.sources import fetch_atcoder, fetch_codeforces, stratified_sample
    from acm_coach.crawler.taxonomy import Taxonomy
    from acm_coach.db import init_db, session_scope
    from acm_coach.db.repositories import KnowledgeRepository, OpsRepository

    settings = get_settings()
    init_db(seed_admin=False)
    taxonomy = Taxonomy(settings.taxonomy_dir)

    problems = []
    if args.source in ("cf", "all"):
        cf = fetch_codeforces()
        cf_pick = stratified_sample(cf, args.cf_limit, seed=args.seed)
        print(f"Codeforces：全量 {len(cf)} → 分层采样 {len(cf_pick)}")
        problems.extend(cf_pick)
    if args.source in ("at", "all"):
        at = fetch_atcoder()
        at_pick = stratified_sample(at, args.at_limit, seed=args.seed)
        print(f"AtCoder：全量 {len(at)} → 分层采样 {len(at_pick)}")
        problems.extend(at_pick)

    crawl_batch = f"{args.source}-{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    records = build_records(problems, taxonomy, crawl_batch=crawl_batch)
    summary = crawl_summary(records)
    print("合成条目：", json.dumps(summary, ensure_ascii=False))

    report_dir = settings.data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "crawl_report.json").write_text(
        json.dumps(
            {"summary": summary,
             "args": {k: v for k, v in vars(args).items() if not callable(v)}},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        print("dry-run：未写库")
        return

    with session_scope() as db:
        ops = OpsRepository(db)
        run = ops.start_crawl(args.source)
        repo = KnowledgeRepository(db)
        inserted, updated = repo.upsert_many(records)
        ops.finish_crawl(run.id, status="success", total=len(problems),
                         inserted=inserted, updated=updated,
                         skipped=len(problems) - len(records))
    print(f"入库完成：新增 {inserted} / 更新 {updated} / 门禁丢弃 {len(problems) - len(records)}")


def cmd_index(args) -> None:
    from acm_coach.db import init_db, session_scope
    from acm_coach.services.index_service import build_index

    init_db(seed_admin=False)  # 确保新增列已迁移，否则 ORM 查询会报 no such column
    with session_scope() as db:
        rows = _load_all(db)
    stats = build_index(rows, with_vector=not args.no_vector, reset_vector=args.reset,
                        max_vectors=args.max_vectors)
    print("索引构建完成：", json.dumps(stats, ensure_ascii=False))


def cmd_eval(args) -> None:
    from acm_coach.evaluation.runner import run_evaluation

    report = run_evaluation(
        configs=args.configs.split(","), max_queries=args.max_queries,
        rebuild=args.rebuild, verify=args.verify,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_eval_diagnosis(args) -> None:
    from acm_coach.evaluation.diagnosis_eval import run_diagnosis_eval

    label_path = Path(args.labels) if args.labels else None
    report = run_diagnosis_eval(label_path=label_path)
    # 控制台只打摘要，明细（逐样本）写入报告文件，避免刷屏
    summary = {
        "sample_count": report["sample_count"],
        "llm_available": report["llm_available"],
        "modes": {
            mode: {k: v for k, v in m.items() if k != "per_sample"}
            for mode, m in report["modes"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_calibrate(args) -> None:
    from acm_coach.evaluation.calibrate import calibrate

    result = calibrate(sample_size=args.sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_stats(args) -> None:
    from acm_coach.db import session_scope
    from acm_coach.db.repositories import KnowledgeRepository, OpsRepository

    with session_scope() as db:
        counts = KnowledgeRepository(db).count_by()
        tags = KnowledgeRepository(db).tag_distribution(10)
        retrieval = OpsRepository(db).retrieval_stats()
        crawls = [
            {"source": c.source, "status": c.status, "total": c.total,
             "inserted": c.inserted, "updated": c.updated}
            for c in OpsRepository(db).list_crawls(5)
        ]
    print(json.dumps({"knowledge": counts, "top_tags": tags,
                      "retrieval": retrieval, "recent_crawls": crawls},
                     ensure_ascii=False, indent=2))


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("acm_coach.api.app:app", host=args.host, port=args.port,
                reload=args.reload, log_config=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acm_coach", description="ACM 智能陪练系统 CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    sub.add_parser("ingest").set_defaults(func=cmd_ingest)

    c = sub.add_parser("crawl")
    c.add_argument("--source", default="all", choices=["cf", "at", "all"])
    c.add_argument("--cf-limit", type=int, default=6000)
    c.add_argument("--at-limit", type=int, default=2500)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_crawl)

    i = sub.add_parser("index")
    i.add_argument("--no-vector", action="store_true", help="只建 BM25（快速验证用）")
    i.add_argument("--reset", action="store_true", help="重建向量集合")
    i.add_argument("--max-vectors", type=int, default=None,
                   help="本次最多补多少条向量（分批构建，中断可续）")
    i.set_defaults(func=cmd_index)

    e = sub.add_parser("eval")
    e.add_argument("--configs", default="g1,g2,g3,g4")
    e.add_argument("--max-queries", type=int, default=150)
    e.add_argument("--rebuild", action="store_true")
    e.add_argument("--verify", action="store_true")
    e.set_defaults(func=cmd_eval)

    ed = sub.add_parser("eval-diagnosis",
                        help="诊断三组对比（需 data/labels/diagnosis_samples.json 人工标注集）")
    ed.add_argument("--labels", default=None, help="标注集路径，缺省用 data/labels/diagnosis_samples.json")
    ed.set_defaults(func=cmd_eval_diagnosis)

    cal = sub.add_parser("calibrate")
    cal.add_argument("--sample", type=int, default=120)
    cal.set_defaults(func=cmd_calibrate)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)
    return p


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
