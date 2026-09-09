"""Request-level and stage-wise efficiency metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def efficiency_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute mean/P95 latency, tokens per request, and mean stage latency."""

    rows = list(records)
    latencies = np.asarray([float(row["latency_s"]) for row in rows], dtype=float)
    stage_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for stage, value in row.get("stage_latency_s", {}).items():
            stage_values[str(stage)].append(float(value))
    return {
        "mean_latency_s": float(latencies.mean()) if len(latencies) else 0.0,
        "p95_latency_s": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        "tokens_per_request": (
            sum(float(row.get("tokens", 0)) for row in rows) / len(rows) if rows else 0.0
        ),
        "stage_wise_latency_s": {
            stage: sum(values) / len(values) for stage, values in sorted(stage_values.items())
        },
        "n_requests": len(rows),
    }
