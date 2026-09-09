"""Controlled transformations from an accepted response to seven violation classes."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Mapping


def _copy(response: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(response))


def inject_schema_violation(response: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    value = _copy(response)
    value.pop("answer", None)
    return value


def inject_scene_mismatch(response: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    value = _copy(response)
    value["scene_refs"].setdefault("entity_ids", []).append("out_of_state_entity")
    return value


def inject_evidence_mismatch(response: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    value = _copy(response)
    value["evidence_ids"].append("out_of_set_evidence")
    return value


def inject_progress_mismatch(response: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    value = _copy(response)
    value["progress_status"] = "completed" if value.get("progress_status") != "completed" else "in_progress"
    return value


def inject_order_violation(response: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    value = _copy(response)
    remaining = list(value.get("remaining_step_ids", []))
    value["remaining_step_ids"] = list(reversed(remaining)) if len(remaining) > 1 else ["out_of_order_step"]
    return value


def inject_illegal_action(response: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    value = _copy(response)
    value["next_action_id"] = "out_of_state_action"
    return value


def inject_duplicate_step(response: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    value = _copy(response)
    steps = list(value.get("remaining_step_ids", []))
    duplicate = steps[0] if steps else value.get("current_step_id") or "repeated_step"
    value["remaining_step_ids"] = steps + [duplicate, duplicate]
    return value


INJECTORS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]] = {
    "schema_violation": inject_schema_violation,
    "scene_mismatch": inject_scene_mismatch,
    "evidence_mismatch": inject_evidence_mismatch,
    "progress_mismatch": inject_progress_mismatch,
    "order_violation": inject_order_violation,
    "illegal_action": inject_illegal_action,
    "duplicate_step": inject_duplicate_step,
}


def inject_violation(
    label: str, valid_response: Mapping[str, Any], context: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Generate one labeled perturbation from a valid structured response."""

    if label not in INJECTORS:
        raise ValueError(f"Unknown violation label: {label}")
    return INJECTORS[label](valid_response, context or {})
