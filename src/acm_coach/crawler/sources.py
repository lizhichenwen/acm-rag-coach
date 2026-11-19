# -*- coding: utf-8 -*-
"""数据源适配器：Codeforces / AtCoder。

两个源都只走官方或社区公开的 JSON API，不解析 HTML、不抓题面：
- Codeforces: `GET /api/problemset.problems` 一次返回全量题目元数据（rating/tags/solvedCount）
- AtCoder: `problems.json`（题号/标题/比赛）+ `problem-models.json`（难度，缺失记为未知）
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger
from acm_coach.crawler.client import HttpClient

log = get_logger("crawler.sources")


@dataclass
class RawProblem:
    platform: str
    problem_id: str          # 平台内唯一 id，如 1845C / abc300_a
    title: str
    url: str
    rating: int | None
    tags: list[str]
    solved: int | None = None
    contest: str = ""
    index: str = ""


def fetch_codeforces(client: HttpClient | None = None) -> list[RawProblem]:
    settings = get_settings()
    client = client or HttpClient()
    payload = client.get_json(f"{settings.cf_api_base}/problemset.problems")
    if payload.get("status") != "OK":
        raise RuntimeError(f"Codeforces API 返回异常：{payload.get('comment')}")

    stats = {
        (p["contestId"], p["index"]): p.get("solvedCount", 0)
        for p in payload["result"].get("problemStatistics", [])
    }
    problems: list[RawProblem] = []
    for item in payload["result"]["problems"]:
        contest_id = item.get("contestId")
        index = item.get("index", "")
        if contest_id is None:
            continue
        problems.append(RawProblem(
            platform="codeforces",
            problem_id=f"{contest_id}{index}",
            title=item.get("name", "").strip(),
            url=f"https://codeforces.com/problemset/problem/{contest_id}/{index}",
            rating=item.get("rating"),
            tags=[t.strip().lower() for t in item.get("tags", [])],
            solved=stats.get((contest_id, index), 0),
            contest=str(item.get("contestId", "")),
            index=index,
        ))
    log.info("Codeforces 全量题目：%d 条", len(problems))
    return problems


def fetch_atcoder(client: HttpClient | None = None) -> list[RawProblem]:
    settings = get_settings()
    client = client or HttpClient()
    problems = client.get_json(settings.atcoder_problems_url)
    models = client.get_json(settings.atcoder_models_url)
    difficulty_map = {
        pid: (meta or {}).get("difficulty") for pid, meta in (models or {}).items()
    }
    out: list[RawProblem] = []
    for item in problems or []:
        pid = item.get("id", "")
        if not pid:
            continue
        difficulty = difficulty_map.get(pid)
        out.append(RawProblem(
            platform="atcoder",
            problem_id=pid,
            title=item.get("title", "").strip(),
            url=f"https://atcoder.jp/contests/{item.get('contest_id', '')}/tasks/{pid}",
            rating=int(difficulty) if isinstance(difficulty, (int, float)) else None,
            tags=[],  # AtCoder 无官方标签体系，域由难度与题号推断
            contest=item.get("contest_id", ""),
            index=pid.split("_")[-1] if "_" in pid else "",
        ))
    log.info("AtCoder 全量题目：%d 条（有难度 %d 条）",
             len(out), sum(1 for p in out if p.rating is not None))
    return out


# ---------------- 采样策略 ----------------
def stratified_sample(problems: list[RawProblem], total: int, *, seed: int = 42,
                      easy_ratio: float = 0.45, mid_ratio: float = 0.35) -> list[RawProblem]:
    """按难度分层采样（易/中/难 = 45%/35%/20%）。

    直接取"rating 从低到高前 N 条"会让语料难度整体上移或下移，
    分层采样保证三个难度档在语料中的占比稳定，也让评测的困难档查询有题可召。
    """
    if total >= len(problems):
        return list(problems)

    from acm_coach.crawler.taxonomy import atcoder_tier, cf_tier

    buckets: dict[str, list[RawProblem]] = {"易": [], "中": [], "难": []}
    for p in problems:
        tier, _ = cf_tier(p.rating) if p.platform == "codeforces" else atcoder_tier(p.rating)
        if tier in ("入门", "简单"):
            buckets["易"].append(p)
        elif tier in ("中等", "较难"):
            buckets["中"].append(p)
        elif tier == "困难":
            buckets["难"].append(p)
        else:
            buckets["中"].append(p)  # 未知难度归入中档，保持可召回

    rng = random.Random(seed)
    quotas = {
        "易": int(total * easy_ratio),
        "中": int(total * mid_ratio),
    }
    quotas["难"] = total - quotas["易"] - quotas["中"]

    picked: list[RawProblem] = []
    for key, quota in quotas.items():
        pool = buckets[key]
        rng.shuffle(pool)
        # 优先保留被更多人做过的题（solved 高 → 质量与可讨论性更可靠）
        pool.sort(key=lambda p: (p.solved or 0), reverse=True)
        head = pool[: int(quota * 0.7)]
        rest = pool[int(quota * 0.7):]
        rng.shuffle(rest)
        picked.extend(head)
        picked.extend(rest[: max(0, quota - len(head))])
    return picked
