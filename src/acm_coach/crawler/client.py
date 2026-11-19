# -*- coding: utf-8 -*-
"""抓取客户端：限速、重试退避、磁盘缓存、UA 标识。

合规约束（见 docs/05-安全合规方案.md）：
- 只访问平台公开 API，不抓取题面 HTML、不绕过任何反爬；
- 单域名 QPS 限速（默认 1 req/s），失败指数退避，最多 3 次；
- 响应按 URL 哈希落盘缓存，默认 24h 内不重复请求（增量抓取）；
- UA 显式标识用途与联系方式，便于对方封禁而非误伤。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

from acm_coach.config import get_settings
from acm_coach.core.logging import get_logger

log = get_logger("crawler.http")


class CrawlError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, cache_dir: Path | None = None, qps: float | None = None,
                 timeout: int | None = None, retries: int | None = None):
        s = get_settings()
        self.cache_dir = cache_dir or s.raw_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max(qps or s.crawler_qps, 0.01)
        self.timeout = timeout or s.crawler_timeout
        self.retries = retries or s.crawler_retries
        self.ttl = s.crawler_cache_ttl
        self._last_call = 0.0

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha1(url.encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def _wait(self) -> None:
        gap = time.time() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.time()

    def get_json(self, url: str, *, use_cache: bool = True) -> dict:
        path = self._cache_path(url)
        if use_cache and path.exists() and (time.time() - path.stat().st_mtime) < self.ttl:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._wait()
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                    resp = client.get(url, headers={
                        "User-Agent": get_settings().crawler_user_agent,
                        "Accept": "application/json",
                    })
                if resp.status_code == 403:
                    # 平台反爬返回 403：视为"拒绝服务"，立即停止而不是重试打爆对方
                    raise CrawlError(f"403 Forbidden：{url}（对方拒绝抓取，请降低频率或改用官方 API）")
                resp.raise_for_status()
                payload = resp.json()
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return payload
            except CrawlError:
                raise
            except Exception as exc:  # 网络/JSON 异常
                last_error = exc
                backoff = min(2 ** attempt, 8)
                log.warning("抓取失败（第 %d/%d 次），%ds 后重试：%s", attempt, self.retries, backoff, exc)
                time.sleep(backoff)
        raise CrawlError(f"抓取失败：{url}（{last_error}）")
