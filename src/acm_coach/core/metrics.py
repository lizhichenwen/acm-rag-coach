# -*- coding: utf-8 -*-
"""轻量指标采集（不引入 prometheus_client 依赖）。

对外暴露 Prometheus 文本格式，`/metrics` 直接给 Grafana / VictoriaMetrics 抓取。
只做三件事：计数器、直方图（固定桶）、运行态快照。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

_LOCK = threading.Lock()
_COUNTERS: dict[tuple[str, tuple], float] = defaultdict(float)
_OBSERVATIONS: dict[tuple[str, tuple], list[float]] = defaultdict(list)
_MAX_SAMPLES = 2048

LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


def inc(name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
    key = (name, tuple(sorted((labels or {}).items())))
    with _LOCK:
        _COUNTERS[key] += value


def observe(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    key = (name, tuple(sorted((labels or {}).items())))
    with _LOCK:
        bucket = _OBSERVATIONS[key]
        bucket.append(value)
        if len(bucket) > _MAX_SAMPLES:
            del bucket[: len(bucket) - _MAX_SAMPLES]


class timer:
    """with timer("retrieval", {"layer": "hybrid"}): ..."""

    def __init__(self, name: str, labels: dict[str, str] | None = None):
        self.name, self.labels = name, labels
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        observe(self.name + "_seconds", time.perf_counter() - self.start, self.labels)
        if exc and exc[0]:
            inc(self.name + "_errors_total", self.labels)
        return False


def _fmt_labels(labels: tuple) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + inner + "}"


def render_prometheus() -> str:
    lines: list[str] = ["# HELP acm_coach_info build info", "# TYPE acm_coach_info gauge",
                        'acm_coach_info{version="4.0.0"} 1']
    with _LOCK:
        counters = dict(_COUNTERS)
        observations = {k: list(v) for k, v in _OBSERVATIONS.items()}

    for (name, labels), value in sorted(counters.items()):
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name}{_fmt_labels(labels)} {value:g}")

    for (name, labels), samples in sorted(observations.items()):
        if not samples:
            continue
        lines.append(f"# TYPE {name} histogram")
        total = sum(samples)
        for b in LATENCY_BUCKETS:
            count = sum(1 for s in samples if s <= b)
            lines.append(f'{name}_bucket{_fmt_labels(labels + (("le", str(b)),))} {count}')
        lines.append(f'{name}_bucket{_fmt_labels(labels + (("le", "+Inf"),))} {len(samples)}')
        lines.append(f"{name}_sum{_fmt_labels(labels)} {total:.6f}")
        lines.append(f"{name}_count{_fmt_labels(labels)} {len(samples)}")
    return "\n".join(lines) + "\n"


def snapshot() -> dict[str, float]:
    """给 /ops 概览页用的聚合快照。"""
    out: dict[str, float] = {}
    with _LOCK:
        for (name, _labels), value in _COUNTERS.items():
            out[name] = out.get(name, 0.0) + value
        for (name, _labels), samples in _OBSERVATIONS.items():
            if samples:
                ordered = sorted(samples)
                out[name + "_p50"] = round(ordered[len(ordered) // 2], 4)
                out[name + "_p95"] = round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 4)
    return out


def reset() -> None:
    with _LOCK:
        _COUNTERS.clear()
        _OBSERVATIONS.clear()
