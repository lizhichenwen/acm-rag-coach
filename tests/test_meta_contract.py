# -*- coding: utf-8 -*-
"""溯源契约测试（docs/05 §6.7）：meta_json 7 键、枚举取值、新旧写入路径一致。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from acm_coach.crawler.builder import build_records
from acm_coach.crawler.sources import RawProblem
from acm_coach.crawler.taxonomy import Taxonomy
from acm_coach.services.knowledge_service import to_records

CONTRACT_KEYS = {
    "entry_type", "content_origin", "processed_by", "owner_type",
    "fetched_at", "crawl_batch", "license", "mirror",
}
MIRROR_VALUES = {"official-api", "kenkoooo", "n/a(self_written)", "unknown"}


class TestToRecordsContract:
    def _item(self, **over):
        item = {
            "title": "测试题", "difficulty": "中等", "tags": "动态规划",
            "description": "描述" * 10, "solution": "思路" * 30, "pitfalls": "边界",
            "source": "curated",
        }
        item.update(over)
        return item

    def test_curated_meta_has_contract(self):
        meta = json.loads(to_records([self._item()])[0]["meta_json"])
        assert CONTRACT_KEYS <= set(meta)
        assert meta["entry_type"] == "solution"
        assert meta["content_origin"] == "self_written"
        assert meta["owner_type"] == "user"
        assert meta["license"] == "internal-only"
        assert meta["mirror"] == "n/a(self_written)"
        assert "license_note" not in meta  # 契约键名为 license

    def test_entry_type_enum_mapping(self):
        meta = json.loads(to_records([self._item(entry_type="竞赛元知识")])[0]["meta_json"])
        assert meta["entry_type"] == "strategy"
        assert meta["entry_type_label"] == "竞赛元知识"

    def test_crawled_side_defaults(self):
        meta = json.loads(to_records([self._item(source="cf")])[0]["meta_json"])
        assert meta["content_origin"] == "metadata"
        assert meta["owner_type"] == "platform"
        assert meta["entry_type"] == "hint"


class TestBuilderContract:
    def test_build_records_meta(self):
        p = RawProblem(platform="codeforces", problem_id="1845C",
                       title="Test Problem", url="https://codeforces.com/1845C",
                       rating=1400, tags=["dp", "greedy"], solved=1234)
        records = build_records([p], Taxonomy(None), crawl_batch="cf-20261108")
        assert len(records) == 1
        meta = json.loads(records[0]["meta_json"])
        assert CONTRACT_KEYS <= set(meta)
        assert meta["entry_type"] == "hint"
        assert meta["crawl_batch"] == "cf-20261108"
        assert meta["content_origin"] == "metadata"
        assert meta["license"] == "Codeforces-Terms"
        assert meta["mirror"] == "official-api"

    def test_atcoder_mirror(self):
        p = RawProblem(platform="atcoder", problem_id="abc300_a",
                       title="Test Problem", url="https://atcoder.jp/abc300_a",
                       rating=None, tags=["implementation"])
        records = build_records([p], Taxonomy(None), crawl_batch="at-20261108")
        assert len(records) == 1
        meta = json.loads(records[0]["meta_json"])
        assert meta["mirror"] == "kenkoooo"
        assert meta["license"] == "AtCoder-Terms"
        assert records[0]["visibility"] == "public"
        assert records[0]["sensitivity"] == "L1"


class TestLiveDatabaseContract:
    """真实库全量校验：答辩演示时随便抽一条都能对上契约。"""

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "data" / "acm_coach.db").exists(),
        reason="未初始化语料库")
    def test_all_rows_have_contract(self):
        db = Path(__file__).resolve().parents[1] / "data" / "acm_coach.db"
        con = sqlite3.connect(db)
        rows = con.execute("select meta_json, visibility from knowledge_items").fetchall()
        con.close()
        assert len(rows) == 8558
        for meta_json, visibility in rows:
            meta = json.loads(meta_json or "{}")
            assert CONTRACT_KEYS <= set(meta)
            assert meta["entry_type"] in {"solution", "concept", "strategy", "hint"}
            assert meta["content_origin"] in {"self_written", "metadata"}
            assert meta["owner_type"] in {"user", "platform"}
            assert meta["license"] in {"internal-only", "Codeforces-Terms", "AtCoder-Terms",
                                       "CC-BY-NC-SA-4.0"}
            assert meta["mirror"] in MIRROR_VALUES
            assert meta["fetched_at"] not in {"", None}
            assert meta["crawl_batch"] not in {"", None}
            assert visibility in {"internal", "public", "restricted"}

    def test_atcoder_rows_record_mirror_channel(self):
        """守库必做项：历史 AtCoder 行必须能回答"该停哪个通道"。"""
        db = Path(__file__).resolve().parents[1] / "data" / "acm_coach.db"
        if not db.exists():
            pytest.skip("未初始化语料库")
        con = sqlite3.connect(db)
        rows = con.execute(
            "select meta_json from knowledge_items where source = 'atcoder'").fetchall()
        con.close()
        assert len(rows) == 2500
        for (meta_json,) in rows:
            assert json.loads(meta_json)["mirror"] == "kenkoooo"
