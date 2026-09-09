"""Scene-grounding metrics used by the MSQA-derived evaluation track."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def _mean(flags: Iterable[bool]) -> float:
    values = list(flags)
    return sum(values) / len(values) if values else 0.0


def scene_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute overall/category accuracy, support rate, and hallucination rate.

    Each record accepts ``prediction``, ``reference``, ``category``,
    ``scene_supported``, and ``hallucinated`` fields.
    """

    rows = list(records)
    correct = [row.get("prediction") == row.get("reference") for row in rows]
    by_category: dict[str, list[bool]] = defaultdict(list)
    for row, flag in zip(rows, correct):
        by_category[str(row.get("category", "uncategorized"))].append(flag)
    return {
        "overall_accuracy": _mean(correct),
        "category_accuracy": {category: _mean(flags) for category, flags in sorted(by_category.items())},
        "scene_support_rate": _mean(bool(row.get("scene_supported")) for row in rows),
        "hallucination_rate": _mean(bool(row.get("hallucinated")) for row in rows),
        "n_requests": len(rows),
    }
