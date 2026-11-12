"""Metrics for clean and controlled-violation verifier evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


_LEGACY_TRACK_ALIASES = {
    "table11_full_request": "table12_full_request",
    "natural_errors": "table11_natural_errors",
}


def _normalized_track(value: Any) -> Any:
    return _LEGACY_TRACK_ALIASES.get(value, value)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _independent_valid(row: Mapping[str, Any]) -> bool | None:
    value = row.get(
        "independent_valid",
        row.get("reference_valid", row.get("initial_reference_valid")),
    )
    return value if isinstance(value, bool) else None


def _final_reference(row: Mapping[str, Any]) -> tuple[set[str], bool]:
    """Read an independent final reference result and its evaluability flag."""

    evaluable = row.get("final_reference_evaluable")
    if evaluable is False:
        return set(), False
    valid = row.get("final_reference_valid", row.get("reference_final_valid"))
    if isinstance(valid, bool):
        labels = row.get("final_reference_labels", row.get("reference_final_labels", []))
        return (set(labels) if isinstance(labels, (list, tuple, set)) else set()), True
    if "final_reference_labels" in row or "reference_final_labels" in row:
        labels = row.get("final_reference_labels", row.get("reference_final_labels", []))
        if isinstance(labels, (list, tuple, set)):
            return set(labels), True
    return set(), False


def _macro(values: Mapping[str, float]) -> float | None:
    return sum(values.values()) / len(values) if values else None


def _final_reference_valid(row: Mapping[str, Any]) -> bool:
    labels, evaluable = _final_reference(row)
    if not evaluable:
        return False
    explicit = row.get("final_reference_valid", row.get("reference_final_valid"))
    return explicit if isinstance(explicit, bool) else not labels


def _natural_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate request-level metrics for naturally generated candidates."""

    n_requests = len(rows)
    initial_evaluable = [
        row
        for row in rows
        if row.get("initial_reference_evaluable") is not False
        and isinstance(row.get("initial_reference_valid"), bool)
    ]
    initial_invalid = [row for row in initial_evaluable if not row["initial_reference_valid"]]
    initial_valid = [row for row in initial_evaluable if row["initial_reference_valid"]]
    initial_rejected = [
        row
        for row in rows
        if not bool(row.get("initial_verifier_accepted", row.get("initial_accepted")))
    ]
    detection_numerator = sum(
        not bool(row.get("initial_verifier_accepted", row.get("initial_accepted")))
        for row in initial_invalid
    )
    false_rejection_numerator = sum(
        not bool(row.get("initial_verifier_accepted", row.get("initial_accepted")))
        for row in initial_valid
    )
    retried = [row for row in rows if bool(row.get("retried"))]
    retry_verifier_accepted = sum(
        bool(
            row.get(
                "retry_verifier_accepted",
                bool(row.get("retry_recovered")) and not bool(row.get("used_fallback")),
            )
        )
        for row in retried
    )
    retry_reference_evaluable = [
        row
        for row in retried
        if bool(
            row.get(
                "retry_verifier_accepted",
                bool(row.get("retry_recovered")) and not bool(row.get("used_fallback")),
            )
        )
        and isinstance(row.get("retry_reference_valid"), bool)
    ]
    retry_reference_valid = sum(
        bool(row["retry_reference_valid"]) for row in retry_reference_evaluable
    )
    final_evaluable = [
        row
        for row in rows
        if row.get("final_reference_evaluable") is not False
        and isinstance(row.get("final_reference_valid"), bool)
    ]
    residual_numerator = sum(not row["final_reference_valid"] for row in final_evaluable)
    valid_non_fallback_numerator = sum(
        row["final_reference_valid"] and not bool(row.get("used_fallback"))
        for row in final_evaluable
    )

    def ratio(numerator: int, denominator: int) -> float | None:
        return _optional_rate(numerator, denominator)

    initial_complete = n_requests > 0 and len(initial_evaluable) == n_requests
    final_complete = n_requests > 0 and len(final_evaluable) == n_requests
    metrics = {
        "natural_violation_rate_before_verify": (
            ratio(len(initial_invalid), n_requests) if initial_complete else None
        ),
        "natural_detection_recall": (
            ratio(detection_numerator, len(initial_invalid)) if initial_complete else None
        ),
        "natural_false_rejection_rate": (
            ratio(false_rejection_numerator, len(initial_valid)) if initial_complete else None
        ),
        "natural_retry_rate": ratio(len(retried), n_requests),
        "natural_retry_recovery_rate": _rate(retry_verifier_accepted, len(retried)),
        "natural_fallback_rate": ratio(
            sum(bool(row.get("used_fallback")) for row in rows), n_requests
        ),
        # These are Table 11 full-sample metrics.  A partial reference track is
        # not reported as a deceptively low rate: the diagnostic numerator and
        # counts remain available below, while the formal rate is undefined.
        "natural_residual_violation_rate": (
            ratio(residual_numerator, n_requests) if final_complete else None
        ),
        "natural_valid_non_fallback_coverage": (
            ratio(valid_non_fallback_numerator, n_requests) if final_complete else None
        ),
        "n_initial_reference_evaluable": len(initial_evaluable),
        "n_initial_reference_unevaluable": n_requests - len(initial_evaluable),
        "n_initial_reference_invalid": len(initial_invalid),
        "n_initial_reference_valid": len(initial_valid),
        "n_initial_invalid_rejected": detection_numerator,
        "n_initial_valid_rejected": false_rejection_numerator,
        "n_initial_verifier_rejected": len(initial_rejected),
        "n_final_reference_evaluable": len(final_evaluable),
        "n_final_reference_unevaluable": n_requests - len(final_evaluable),
        "n_final_reference_invalid": residual_numerator,
        "n_final_reference_valid": len(final_evaluable) - residual_numerator,
        "n_retried": len(retried),
        "n_retry_recovered": retry_verifier_accepted,
        "n_retry_verifier_accepted": retry_verifier_accepted,
        "n_retry_reference_evaluable": len(retry_reference_evaluable),
        "n_retry_reference_valid": retry_reference_valid,
        "n_retry_reference_invalid": len(retry_reference_evaluable) - retry_reference_valid,
        "n_fallback": sum(bool(row.get("used_fallback")) for row in rows),
        "n_valid_non_fallback": valid_non_fallback_numerator,
        "n_requests": n_requests,
        "reference_evaluation_complete": final_complete,
        "initial_reference_evaluation_complete": initial_complete,
        "final_reference_evaluation_complete": final_complete,
        "natural_residual_violation_rate_evaluable_subset": ratio(
            residual_numerator, len(final_evaluable)
        ),
        "natural_valid_non_fallback_coverage_evaluable_subset": ratio(
            valid_non_fallback_numerator, len(final_evaluable)
        ),
        "evaluation_track": "table11_natural_errors",
    }
    suffix_pairs = (
        ("natural_violation_rate_before_verify", len(initial_invalid), n_requests),
        ("natural_detection_recall", detection_numerator, len(initial_invalid)),
        ("natural_false_rejection_rate", false_rejection_numerator, len(initial_valid)),
        ("natural_retry_rate", len(retried), n_requests),
        ("natural_retry_recovery_rate", retry_verifier_accepted, len(retried)),
        (
            "natural_fallback_rate",
            metrics["n_fallback"],
            n_requests,
        ),
        ("natural_residual_violation_rate", residual_numerator, n_requests),
        ("natural_valid_non_fallback_coverage", valid_non_fallback_numerator, n_requests),
    )
    for name, numerator, denominator in suffix_pairs:
        metrics[f"{name}_numerator"] = numerator
        metrics[f"{name}_denominator"] = denominator
        value = metrics[name]
        metrics[f"{name}_percent"] = round(value * 100.0, 1) if value is not None else None

    # Keep the established generic names available for callers that consume a
    # single evaluation track, while exposing the natural-prefixed names above.
    metrics.update(
        {
            "detection_recall": metrics["natural_detection_recall"],
            "detection_recall_micro": metrics["natural_detection_recall"],
            "detection_recall_by_type": {},
            "residual_violation_rate": metrics["natural_residual_violation_rate"],
            "residual_violation_rate_macro": metrics["natural_residual_violation_rate"],
            "residual_violation_rate_by_type": {},
            "false_rejection_rate": metrics["natural_false_rejection_rate"],
            "retry_rate": metrics["natural_retry_rate"],
            "retry_recovery_rate": metrics["natural_retry_recovery_rate"],
            "fallback_rate": metrics["natural_fallback_rate"],
            "valid_non_fallback_coverage": metrics["natural_valid_non_fallback_coverage"],
            "n_candidates": n_requests,
        }
    )
    return metrics


def verifier_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute detection, residual, false rejection, retry, recovery, and fallback rates.

    ``initial_labels`` come from the actual verifier.  Residual metrics require
    ``final_reference_labels``/``final_reference_valid`` from an independent
    offline reference check and exclude rows marked as not evaluable.
    ``independent_valid``/``reference_valid`` is the initial-candidate oracle
    used only for False Rejection Rate.
    """

    rows = [
        {
            **row,
            "evaluation_track": _normalized_track(row.get("evaluation_track")),
        }
        for row in records
    ]
    tracks = {str(row["evaluation_track"]) for row in rows if row.get("evaluation_track")}
    if len(tracks) > 1:
        raise ValueError("Different verifier evaluation tracks must be aggregated separately.")
    evaluation_track = next(iter(tracks), None)
    if evaluation_track in {"natural_errors", "table11_natural_errors"}:
        return _natural_metrics(rows)
    perturbed = [row for row in rows if row.get("injected_label")]
    clean = [row for row in rows if not row.get("injected_label")]
    detections: dict[str, list[bool]] = defaultdict(list)
    residuals: dict[str, list[bool]] = defaultdict(list)
    for row in perturbed:
        label = str(row["injected_label"])
        detections[label].append(label in set(row.get("initial_labels", [])))
        final_labels, evaluable = _final_reference(row)
        if evaluable:
            residuals[label].append(bool(final_labels))
    detection_by_type = {
        label: _optional_rate(sum(values), len(values)) for label, values in sorted(detections.items())
    }
    detection_by_type = {label: value for label, value in detection_by_type.items() if value is not None}
    residual_by_type = {
        label: _optional_rate(sum(values), len(values)) for label, values in sorted(residuals.items())
    }
    residual_by_type = {label: value for label, value in residual_by_type.items() if value is not None}
    retried = [row for row in rows if row.get("retried")]
    recovered = [
        bool(row.get("retry_recovered", row.get("final_accepted") and not row.get("used_fallback")))
        for row in retried
    ]
    independently_valid_initial = [row for row in rows if _independent_valid(row) is True]
    final_reference_rows = []
    for row in rows:
        final_labels, evaluable = _final_reference(row)
        if evaluable:
            final_reference_rows.append((row, final_labels))
    residual_reference_rows = [
        (row, labels)
        for row, labels in final_reference_rows
        if row.get("injected_label")
    ]
    residual_overall = _optional_rate(
        sum(not _final_reference_valid(row) for row, _ in residual_reference_rows),
        len(residual_reference_rows),
    )
    valid_non_fallback = sum(
        _final_reference_valid(row)
        for row, _ in final_reference_rows
        if bool(row.get("final_accepted")) and not bool(row.get("used_fallback"))
    )
    return {
        "detection_recall": _macro(detection_by_type),
        "detection_recall_micro": _optional_rate(
            sum(str(row["injected_label"]) in set(row.get("initial_labels", [])) for row in perturbed),
            len(perturbed),
        ),
        "detection_recall_by_type": detection_by_type,
        "residual_violation_rate": residual_overall,
        "residual_violation_rate_macro": _macro(residual_by_type),
        "residual_violation_rate_by_type": residual_by_type,
        "false_rejection_rate": _optional_rate(
            sum(not bool(row.get("initial_accepted")) for row in independently_valid_initial),
            len(independently_valid_initial),
        ),
        "retry_rate": _rate(len(retried), len(rows)),
        "retry_recovery_rate": _rate(sum(recovered), len(retried)),
        "fallback_rate": _rate(sum(bool(row.get("used_fallback")) for row in rows), len(rows)),
        "valid_non_fallback_coverage": _optional_rate(valid_non_fallback, len(rows)),
        "n_independently_valid_initial": len(independently_valid_initial),
        "n_independently_valid_clean": len([row for row in clean if _independent_valid(row) is True]),
        "n_final_reference_evaluable": len(final_reference_rows),
        "n_candidates": len(rows),
        "evaluation_track": evaluation_track,
    }
