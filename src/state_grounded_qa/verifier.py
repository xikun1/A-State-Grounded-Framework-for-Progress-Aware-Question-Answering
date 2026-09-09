"""Deterministic response predicates for schema, evidence, scene, progress, and action."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .progress import TaskGraph
from .schemas import EvidenceRecord, RuntimeState, StructuredResponse


FAILURE_LABELS = {
    "schema_violation",
    "scene_mismatch",
    "evidence_mismatch",
    "progress_mismatch",
    "order_violation",
    "illegal_action",
    "duplicate_step",
}


@dataclass(slots=True)
class VerificationResult:
    """Top-level verifier decision and stable failure labels."""

    accepted: bool
    failure_labels: list[str]


def _as_mapping(candidate: Any) -> Mapping[str, Any] | None:
    if isinstance(candidate, StructuredResponse):
        return candidate.to_dict()
    if isinstance(candidate, Mapping):
        return candidate
    if isinstance(candidate, str):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, Mapping) else None
    return None


def chi_schema(candidate: Any) -> tuple[bool, list[str]]:
    """Check required JSON fields, field types, and allowed status values."""

    value = _as_mapping(candidate)
    if value is None:
        return False, ["schema_violation"]
    required = {
        "answer",
        "evidence_ids",
        "scene_refs",
        "current_step_id",
        "progress_status",
        "remaining_step_ids",
        "next_action_id",
    }
    if set(value) != required:
        return False, ["schema_violation"]
    valid = (
        isinstance(value["answer"], str)
        and isinstance(value["evidence_ids"], list)
        and all(isinstance(item, str) for item in value["evidence_ids"])
        and isinstance(value["scene_refs"], Mapping)
        and (value["current_step_id"] is None or isinstance(value["current_step_id"], str))
        and value["progress_status"] in {None, "not_started", "in_progress", "completed"}
        and isinstance(value["remaining_step_ids"], list)
        and all(isinstance(item, str) for item in value["remaining_step_ids"])
        and (value["next_action_id"] is None or isinstance(value["next_action_id"], str))
    )
    return (True, []) if valid else (False, ["schema_violation"])


def chi_evidence(candidate: Mapping[str, Any], evidence: Sequence[EvidenceRecord]) -> tuple[bool, list[str]]:
    """Require every cited evidence identifier to belong to the fixed E_t."""

    allowed = {item.evidence_id for item in evidence}
    valid = all(identifier in allowed for identifier in candidate["evidence_ids"])
    return (True, []) if valid else (False, ["evidence_mismatch"])


def _relations(state: RuntimeState) -> set[tuple[str, str, str]]:
    return {
        (str(item.get("subject")), str(item.get("relation")), str(item.get("object")))
        for item in state.scene.object_relations
    }


def _entities(state: RuntimeState) -> set[str]:
    result = {state.scene.active_object} if state.scene.active_object else set()
    for item in state.scene.objects:
        if isinstance(item, Mapping) and item.get("object_id") is not None:
            result.add(str(item["object_id"]))
    for subject, _, object_id in _relations(state):
        result.update({subject, object_id})
    return result


def chi_scene(candidate: Mapping[str, Any], state: RuntimeState) -> tuple[bool, list[str]]:
    """Check entity, location, and relation references against current scene facts."""

    refs = candidate["scene_refs"]
    if not isinstance(refs, Mapping):
        return False, ["scene_mismatch"]
    entity_ids = refs.get("entity_ids", [])
    locations = refs.get("locations", [])
    relation_triples = refs.get("relation_triples", [])
    if not all(isinstance(items, list) for items in (entity_ids, locations, relation_triples)):
        return False, ["scene_mismatch"]
    if any(str(identifier) not in _entities(state) for identifier in entity_ids):
        return False, ["scene_mismatch"]
    allowed_locations = {item for item in (state.scene.scene_id, state.scene.location) if item}
    if any(str(location) not in allowed_locations for location in locations):
        return False, ["scene_mismatch"]
    parsed_relations: list[tuple[str, str, str]] = []
    for triple in relation_triples:
        if isinstance(triple, Mapping):
            parsed_relations.append(
                (str(triple.get("subject")), str(triple.get("relation")), str(triple.get("object")))
            )
        elif isinstance(triple, list) and len(triple) == 3:
            parsed_relations.append(tuple(map(str, triple)))
        else:
            return False, ["scene_mismatch"]
    valid = all(triple in _relations(state) for triple in parsed_relations)
    return (True, []) if valid else (False, ["scene_mismatch"])


def chi_progress(
    candidate: Mapping[str, Any], state: RuntimeState, task_graph: TaskGraph | None = None
) -> tuple[bool, list[str]]:
    """Check current step, status, remaining order, prerequisites, and duplicates."""

    labels: list[str] = []
    remaining = list(candidate["remaining_step_ids"])
    combined = state.task.completed_steps + remaining
    if len(remaining) != len(set(remaining)) or len(combined) != len(set(combined)):
        labels.append("duplicate_step")
    if candidate["current_step_id"] in remaining or candidate["current_step_id"] in state.task.completed_steps:
        labels.append("duplicate_step")
    if (
        candidate["current_step_id"] != state.task.current_step
        or candidate["progress_status"] != state.task.progress_status
        or remaining != state.task.remaining_steps
    ):
        labels.append("progress_mismatch")
    if candidate["progress_status"] == "completed" and (
        candidate["current_step_id"] is not None or remaining
    ):
        labels.append("progress_mismatch")
    if task_graph is not None:
        expected_order = [
            node_id
            for node_id in task_graph.topological_order
            if node_id in remaining
        ]
        if remaining != expected_order:
            labels.append("order_violation")
        prior = set(state.task.completed_steps)
        if candidate["current_step_id"] is not None:
            node = task_graph.nodes.get(candidate["current_step_id"])
            active_nodes = prior | set(remaining) | {candidate["current_step_id"]}
            if node is None or any(
                pred in active_nodes and pred not in prior for pred in node.predecessors
            ):
                labels.append("order_violation")
    unique_labels = list(dict.fromkeys(labels))
    return not unique_labels, unique_labels


def chi_action(candidate: Mapping[str, Any], state: RuntimeState) -> tuple[bool, list[str]]:
    """Accept null or require the suggestion to belong to A_t_legal."""

    action_id = candidate["next_action_id"]
    valid = action_id is None or action_id in state.action.available_actions
    return (True, []) if valid else (False, ["illegal_action"])


def verify_response(
    candidate: Any,
    state: RuntimeState,
    evidence: Sequence[EvidenceRecord],
    task_graph: TaskGraph | None = None,
) -> VerificationResult:
    """Apply the five top-level predicates and return their union of labels."""

    schema_ok, schema_labels = chi_schema(candidate)
    if not schema_ok:
        return VerificationResult(False, schema_labels)
    value = _as_mapping(candidate)
    assert value is not None
    checks = [
        chi_evidence(value, evidence),
        chi_scene(value, state),
        chi_progress(value, state, task_graph),
        chi_action(value, state),
    ]
    labels = list(dict.fromkeys(label for _, check_labels in checks for label in check_labels))
    return VerificationResult(accepted=not labels, failure_labels=labels)
