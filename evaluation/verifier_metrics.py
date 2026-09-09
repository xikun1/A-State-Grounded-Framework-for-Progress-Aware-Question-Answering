"""Metrics for clean and controlled-violation verifier evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def verifier_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute detection, residual, false rejection, retry, recovery, and fallback rates.

    Expected fields are ``injected_label`` (null for clean), ``initial_labels``,
    ``final_labels``, ``initial_accepted``, ``final_accepted``, ``retried``, and
    ``used_fallback``.
    """

    rows = list(records)
    perturbed = [row for row in rows if row.get("injected_label")]
    clean = [row for row in rows if not row.get("injected_label")]
    detections: dict[str, list[bool]] = defaultdict(list)
    residuals: dict[str, list[bool]] = defaultdict(list)
    for row in perturbed:
        label = str(row["injected_label"])
        detections[label].append(label in set(row.get("initial_labels", [])))
        residuals[label].append(label in set(row.get("final_labels", [])))
    retried = [row for row in rows if row.get("retried")]
    return {
        "detection_recall": _rate(
            sum(str(row["injected_label"]) in set(row.get("initial_labels", [])) for row in perturbed),
            len(perturbed),
        ),
        "detection_recall_by_type": {
            label: _rate(sum(values), len(values)) for label, values in sorted(detections.items())
        },
        "residual_violation_rate": _rate(
            sum(bool(set(row.get("final_labels", []))) for row in perturbed), len(perturbed)
        ),
        "residual_violation_rate_by_type": {
            label: _rate(sum(values), len(values)) for label, values in sorted(residuals.items())
        },
        "false_rejection_rate": _rate(
            sum(not bool(row.get("initial_accepted")) for row in clean), len(clean)
        ),
        "retry_rate": _rate(len(retried), len(rows)),
        "retry_recovery_rate": _rate(sum(bool(row.get("final_accepted")) for row in retried), len(retried)),
        "fallback_rate": _rate(sum(bool(row.get("used_fallback")) for row in rows), len(rows)),
        "n_candidates": len(rows),
    }
