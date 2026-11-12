"""State-construction and progress/action metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Literal, Mapping, Sequence


STATUSES = ("not_started", "in_progress", "completed")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


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


EvaluationStage = Literal["prediction", "state", "initial_candidate", "final_verified"]


def _normalize_progress_prediction(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize runtime-state and structured-response progress field names.

    Optional fields are deliberately not synthesized.  In particular,
    structured responses do not normally contain ``completed_steps`` or
    ``legal_actions``; treating those missing fields as empty lists would turn
    an unavailable F1 into a falsely perfect score.
    """

    normalized = dict(prediction)
    if "current_step" not in normalized and "current_step_id" in normalized:
        normalized["current_step"] = normalized["current_step_id"]
    if "remaining_steps" not in normalized and "remaining_step_ids" in normalized:
        normalized["remaining_steps"] = normalized["remaining_step_ids"]
    task = normalized.get("task")
    if isinstance(task, Mapping):
        for field in ("current_step", "progress_status", "completed_steps", "remaining_steps"):
            if field not in normalized and field in task:
                normalized[field] = task[field]
    action = normalized.get("action")
    if isinstance(action, Mapping):
        if "legal_actions" not in normalized and "available_actions" in action:
            normalized["legal_actions"] = action["available_actions"]
        if "acceptable_next_actions" not in normalized and "acceptable_next_actions" in action:
            normalized["acceptable_next_actions"] = action["acceptable_next_actions"]
    return normalized


def _predictions_for_stage(
    rows: Sequence[Mapping[str, Any]], stage: EvaluationStage | None
) -> list[Mapping[str, Any] | None]:
    """Select the explicitly named prediction stage without changing old inputs."""

    if stage is None or stage == "prediction":
        return [
            _normalize_progress_prediction(row["prediction"])
            if isinstance(row.get("prediction"), Mapping)
            else None
            for row in rows
        ]
    selected: list[Mapping[str, Any] | None] = []
    for row in rows:
        value = row.get(stage)
        if value is None and isinstance(row.get("predictions"), Mapping):
            value = row["predictions"].get(stage)
        selected.append(_normalize_progress_prediction(value) if isinstance(value, Mapping) else None)
    return selected


def _references_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("reference", row.get("reference_state"))
        if not isinstance(value, Mapping) and isinstance(row.get("reference_labels"), Mapping):
            labels = row["reference_labels"]
            value = {
                "current_step": labels.get("reference_current_step"),
                "progress_status": labels.get("reference_progress_status"),
                "completed_steps": labels.get("reference_completed_steps"),
                "remaining_steps": labels.get("reference_remaining_steps"),
                "legal_actions": labels.get("reference_legal_actions"),
                "acceptable_next_actions": labels.get("acceptable_next_actions"),
            }
        references.append(_normalize_progress_prediction(value) if isinstance(value, Mapping) else {})
    return references


def _evaluation_status(n_requests: int, n_evaluable_predictions: int) -> str:
    if n_evaluable_predictions == n_requests:
        return "complete"
    if n_evaluable_predictions:
        return "partial"
    return "unavailable"


def _optional_set_f1(
    prediction: Mapping[str, Any], reference: Mapping[str, Any], field: str
) -> float | None:
    """Return F1 only when the prediction explicitly supplies ``field``."""

    if (
        field not in prediction
        or prediction[field] is None
        or field not in reference
        or reference[field] is None
    ):
        return None
    return set_f1(prediction[field], reference[field])


def _mean_optional(values: Sequence[float | None]) -> tuple[float | None, int]:
    available = [value for value in values if value is not None]
    return (sum(available) / len(available), len(available)) if available else (None, 0)


def state_construction_metrics(
    records: Iterable[Mapping[str, Any]], stage: EvaluationStage | None = None
) -> dict[str, Any]:
    """Evaluate current step, status, completed/remaining steps, and legal actions."""

    rows = list(records)
    predictions = _predictions_for_stage(rows, stage)
    references = _references_for_rows(rows)
    pairs = [
        (prediction, reference)
        for prediction, reference in zip(predictions, references)
        if prediction is not None
    ]
    current_values = [
        float(prediction["current_step"] == reference["current_step"])
        for prediction, reference in pairs
        if "current_step" in prediction and "current_step" in reference
    ]
    status_pairs = [
        (prediction["progress_status"], reference["progress_status"])
        for prediction, reference in pairs
        if prediction.get("progress_status") is not None and reference.get("progress_status") is not None
    ]
    remaining_values = [
        set_f1(prediction["remaining_steps"], reference["remaining_steps"])
        for prediction, reference in pairs
        if prediction.get("remaining_steps") is not None and reference.get("remaining_steps") is not None
    ]
    completed_f1, completed_f1_n = _mean_optional(
        [_optional_set_f1(p, r, "completed_steps") for p, r in pairs]
    )
    legal_f1, legal_f1_n = _mean_optional(
        [_optional_set_f1(p, r, "legal_actions") for p, r in pairs]
    )
    stage_name = stage or "prediction"
    return {
        "evaluation_stage": stage_name,
        "evaluation_status": _evaluation_status(len(rows), len(pairs)),
        "n_requests": len(rows),
        "n_evaluable_predictions": len(pairs),
        "n_missing_stage_predictions": len(rows) - len(pairs),
        "n_current_step_evaluable": len(current_values),
        "current_step_accuracy": _mean_or_none(current_values),
        "n_progress_status_evaluable": len(status_pairs),
        "progress_status_macro_f1": (
            macro_f1([str(p) for p, _ in status_pairs], [str(r) for _, r in status_pairs])
            if status_pairs
            else None
        ),
        "completed_step_f1": completed_f1,
        "completed_step_f1_n": completed_f1_n,
        "n_remaining_step_evaluable": len(remaining_values),
        "remaining_step_f1": _mean_or_none(remaining_values),
        "legal_action_set_f1": legal_f1,
        "legal_action_set_f1_n": legal_f1_n,
    }


def _reference_precondition_order_violation(
    row: Mapping[str, Any], reference: Mapping[str, Any]
) -> bool | None:
    """Read an independently computed precondition/order flag if available."""

    candidates: list[Mapping[str, Any]] = [reference, row]
    for container_key in (
        "reference_constraints",
        "reference_task_constraints",
        "reference_action_evaluation",
        "reference_evaluation",
    ):
        container = row.get(container_key)
        if isinstance(container, Mapping):
            candidates.append(container)
    for container_key in ("action", "constraints"):
        container = reference.get(container_key)
        if isinstance(container, Mapping):
            candidates.append(container)

    flag_keys = (
        "precondition_order_violation",
        "precondition_or_order_violation",
        "reference_precondition_order_violation",
        "precondition_violation",
        "order_violation",
    )
    label_keys = ("reference_violation_labels", "violation_labels", "labels")
    violation_labels = {
        "precondition_violation",
        "order_violation",
        "precondition_order_violation",
        "precondition_or_order_violation",
    }
    for candidate in candidates:
        for key in flag_keys:
            value = candidate.get(key)
            if isinstance(value, bool):
                return value
        for key in label_keys:
            labels = candidate.get(key)
            if isinstance(labels, (list, tuple, set)):
                return bool(set(labels) & violation_labels)
    return None


def progress_action_metrics(
    records: Iterable[Mapping[str, Any]], stage: EvaluationStage | None = None
) -> dict[str, Any]:
    """Compute progress/action metrics for an explicit evaluation stage."""

    rows = list(records)
    predictions = _predictions_for_stage(rows, stage)
    references = _references_for_rows(rows)
    base = state_construction_metrics(rows, stage=stage)
    next_correct: list[bool] = []
    valid_actions: list[bool] = []
    non_null_actions = 0
    precondition_order: list[bool] = []
    duplicate: list[bool] = []
    for prediction, reference, row in zip(predictions, references, rows):
        if prediction is None:
            continue
        acceptable = set(reference["acceptable_next_actions"]) if reference.get("acceptable_next_actions") is not None else None
        next_action = prediction.get("next_action_id")
        if acceptable is not None:
            next_correct.append(next_action is not None and next_action in acceptable)
        if next_action is not None:
            non_null_actions += 1
            if reference.get("legal_actions") is not None:
                valid_actions.append(next_action in set(reference["legal_actions"]))
            flag = _reference_precondition_order_violation(row, reference)
            if flag is not None:
                precondition_order.append(flag)
        steps = prediction.get("remaining_steps")
        if steps is not None:
            duplicate.append(len(steps) != len(set(steps)))
    n_requests = len(rows)
    n_valid_actions = sum(valid_actions)
    n_illegal_actions = len(valid_actions) - n_valid_actions
    action_rate = (
        n_valid_actions / non_null_actions
        if non_null_actions and len(valid_actions) == non_null_actions
        else None
    )
    illegal_rate = (
        n_illegal_actions / non_null_actions
        if non_null_actions and len(valid_actions) == non_null_actions
        else None
    )
    non_null_rate = (
        non_null_actions / n_requests
        if n_requests and len([prediction for prediction in predictions if prediction is not None]) == n_requests
        else None
    )
    next_accuracy = sum(next_correct) / n_requests if n_requests and len(next_correct) == n_requests else None
    precondition_rate = (
        sum(precondition_order) / non_null_actions
        if non_null_actions and len(precondition_order) == non_null_actions
        else None
    )
    base.update(
        {
            "next_action_accuracy": next_accuracy,
            "action_validity_rate": action_rate,
            "illegal_action_rate": illegal_rate,
            "non_null_action_recommendation_rate": non_null_rate,
            "n_requests": n_requests,
            "n_non_null_actions": non_null_actions,
            "n_valid_actions": n_valid_actions,
            "n_illegal_actions": n_illegal_actions,
            "n_next_action_evaluable": len(next_correct),
            "n_action_validity_evaluable": len(valid_actions),
            "n_action_validity_unevaluable": non_null_actions - len(valid_actions),
            "n_precondition_order_evaluable": len(precondition_order),
            "n_precondition_order_violation_evaluable": len(precondition_order),
            "precondition_order_violation_rate_n": len(precondition_order),
            "n_precondition_order_unevaluable": non_null_actions - len(precondition_order),
            "precondition_order_violation_rate": precondition_rate,
            "duplicate_step_rate": _mean_or_none([float(value) for value in duplicate]),
        }
    )
    return base
