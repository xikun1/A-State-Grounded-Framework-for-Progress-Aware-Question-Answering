"""State-construction and progress/action metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


STATUSES = ("not_started", "in_progress", "completed")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def set_f1(predicted: Iterable[str], reference: Iterable[str]) -> float:
    """Compute set F1 with exact empty-set handling."""

    predicted_set, reference_set = set(predicted), set(reference)
    if not predicted_set and not reference_set:
        return 1.0
    if not predicted_set or not reference_set:
        return 0.0
    intersection = len(predicted_set & reference_set)
    precision = intersection / len(predicted_set)
    recall = intersection / len(reference_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def macro_f1(predicted: Sequence[str], reference: Sequence[str], labels: Sequence[str] = STATUSES) -> float:
    """Compute unweighted one-vs-rest Macro-F1 over fixed labels."""

    scores = []
    for label in labels:
        tp = sum(p == label and r == label for p, r in zip(predicted, reference))
        fp = sum(p == label and r != label for p, r in zip(predicted, reference))
        fn = sum(p != label and r == label for p, r in zip(predicted, reference))
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return _mean(scores)


def state_construction_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Evaluate current step, status, completed/remaining steps, and legal actions."""

    rows = list(records)
    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    return {
        "current_step_accuracy": _mean(
            [float(p.get("current_step") == r.get("current_step")) for p, r in zip(predictions, references)]
        ),
        "progress_status_macro_f1": macro_f1(
            [str(p.get("progress_status")) for p in predictions],
            [str(r.get("progress_status")) for r in references],
        ),
        "completed_step_f1": _mean(
            [set_f1(p.get("completed_steps", []), r.get("completed_steps", [])) for p, r in zip(predictions, references)]
        ),
        "remaining_step_f1": _mean(
            [set_f1(p.get("remaining_steps", []), r.get("remaining_steps", [])) for p, r in zip(predictions, references)]
        ),
        "legal_action_set_f1": _mean(
            [set_f1(p.get("legal_actions", []), r.get("legal_actions", [])) for p, r in zip(predictions, references)]
        ),
    }


def progress_action_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Compute progress alignment and next-action consistency metrics."""

    rows = list(records)
    base = state_construction_metrics(rows)
    next_correct = []
    valid = []
    precondition_order = []
    duplicate = []
    for row in rows:
        prediction, reference = row["prediction"], row["reference"]
        acceptable = set(reference.get("acceptable_next_actions", []))
        next_action = prediction.get("next_action_id")
        next_correct.append(next_action in acceptable if next_action is not None else not acceptable)
        valid.append(next_action is None or next_action in set(reference.get("legal_actions", [])))
        precondition_order.append(bool(prediction.get("precondition_or_order_violation", False)))
        steps = prediction.get("remaining_steps", [])
        duplicate.append(len(steps) != len(set(steps)))
    base.update(
        {
            "next_action_accuracy": _mean([float(value) for value in next_correct]),
            "action_validity_rate": _mean([float(value) for value in valid]),
            "illegal_action_rate": 1.0 - _mean([float(value) for value in valid]),
            "precondition_order_violation_rate": _mean([float(value) for value in precondition_order]),
            "duplicate_step_rate": _mean([float(value) for value in duplicate]),
        }
    )
    return base
