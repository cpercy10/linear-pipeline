"""Lightweight thread-safe metrics collector for the end-of-run summary.

`stage_timer` feeds per-stage elapsed_ms here; the runner reads percentiles +
lane counts at the end. Threads (preprocess + diffusion pools) write concurrently,
so all access is lock-guarded.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage_ms: Dict[str, List[float]] = defaultdict(list)
        self._lanes: Counter = Counter()
        self._errors: List[Tuple[str, str]] = []

    def record_stage(self, stage: str, ms: float) -> None:
        with self._lock:
            self._stage_ms[stage].append(ms)

    def record_lane(self, lane: str) -> None:
        with self._lock:
            self._lanes[lane] += 1

    def record_error(self, name: str, msg: str) -> None:
        with self._lock:
            self._errors.append((name, msg))

    def stage_summary(self) -> Dict[str, dict]:
        with self._lock:
            out: Dict[str, dict] = {}
            for stage, vals in self._stage_ms.items():
                vs = sorted(vals)
                out[stage] = {
                    "count": len(vs),
                    "p50_ms": round(_percentile(vs, 50), 1),
                    "p95_ms": round(_percentile(vs, 95), 1),
                    "mean_ms": round(sum(vs) / len(vs), 1),
                    "max_ms": round(vs[-1], 1),
                }
            return out

    def lane_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._lanes)

    def errors(self) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self._errors)


_metrics: Metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics
