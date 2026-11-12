"""Independent offline reference checks for final verifier evaluation.

This module intentionally does not call ``state_grounded_qa.verifier``.  It
implements the reference contract from the supplied state, evidence, and task
graph so that final residual violations are not obtained by asking the system
under evaluation to judge its own output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from state_grounded_qa.progress import TaskGraph
from state_grounded_qa.schemas import EvidenceRecord, RuntimeState, StructuredResponse


REQUIRED_FIELDS = {
    "answer",
    "evidence_ids",
    "scene_refs",
    "current_step_id",
    "progress_status",
    "remaining_step_ids",
    "next_action_id",
}

_MISSING = object()


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


def _schema_labels(candidate: Any) -> tuple[Mapping[str, Any] | None, list[str]]:
    value = _as_mapping(candidate)
    if value is None or set(value) != REQUIRED_FIELDS:
        return None, ["schema_violation"]
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
    return (value, []) if valid else (None, ["schema_violation"])


def _relations(state: RuntimeState | None) -> set[tuple[str, str, str]]:
    if state is None:
        return set()
    return {
        (str(item.get("subject")), str(item.get("relation")), str(item.get("object")))
        for item in state.scene.object_relations
    }


def _entities(state: RuntimeState | None) -> set[str]:
    if state is None:
        return set()
    result = {state.scene.active_object} if state.scene.active_object else set()
    result.update(
        str(item["object_id"])
        for item in state.scene.objects
        if isinstance(item, Mapping) and item.get("object_id") is not None
    )
    for subject, _, object_id in _relations(state):
        result.update({subject, object_id})
    return result


def _active_route(
    state: RuntimeState | None,
    graph: TaskGraph | None,
    reference_fields: Mapping[str, Any] | None = None,
) -> list[str]:
    reference_fields = reference_fields or {}
    has_explicit_task_fields = any(
        key in reference_fields
        for key in ("current_step", "progress_status", "completed_steps", "remaining_steps")
    )
    if state is None or has_explicit_task_fields:
        completed = list(reference_fields.get("completed_steps", []))
        current = reference_fields.get("current_step")
        remaining = list(reference_fields.get("remaining_steps", []))
        route = completed + ([current] if current is not None else []) + remaining
        if graph is not None:
            graph_route = graph.active_order({"task": dict(reference_fields)})
            return list(dict.fromkeys(graph_route + route))
        return list(dict.fromkeys(route))
    state_route = list(state.task.completed_steps)
    if state.task.current_step is not None:
        state_route.append(state.task.current_step)
    state_route.extend(state.task.remaining_steps)
    if graph is None:
        return list(dict.fromkeys(state_route))
    graph_route = graph.active_order(state.to_dict())
    graph_route = [
        node_id
        for node_id in graph_route
        if graph.nodes[node_id].active_if is None or node_id in state_route
    ]
    return list(dict.fromkeys(graph_route + state_route))


def _progress_labels(
    value: Mapping[str, Any],
    state: RuntimeState | None,
    graph: TaskGraph | None,
    reference_fields: Mapping[str, Any] | None = None,
) -> list[str]:
    current = value["current_step_id"]
    remaining = list(value["remaining_step_ids"])
    reference_fields = reference_fields or {}
    if state is not None and not any(
        key in reference_fields
        for key in ("current_step", "progress_status", "completed_steps", "remaining_steps")
    ):
        runtime_current = state.task.current_step
        runtime_status = state.task.progress_status
        completed = set(state.task.completed_steps)
        runtime_remaining = list(state.task.remaining_steps)
    else:
        runtime_current = reference_fields.get("current_step")
        runtime_status = reference_fields.get("progress_status")
        completed = set(reference_fields.get("completed_steps", []))
        runtime_remaining = list(reference_fields.get("remaining_steps", []))
    route = _active_route(state, graph, reference_fields)
    route_set = set(route)
    positions = {step_id: index for index, step_id in enumerate(route)}
    labels: list[str] = []

    if len(remaining) != len(set(remaining)) or len(completed | set(remaining)) != len(completed) + len(remaining):
        labels.append("duplicate_step")
    if current in remaining or current in completed:
        labels.append("duplicate_step")
    if current is not None and current not in route_set:
        labels.extend(["progress_mismatch", "order_violation"])
    if any(step_id not in route_set for step_id in remaining):
        labels.extend(["progress_mismatch", "order_violation"])

    runtime_completed = runtime_status == "completed" or (
        bool(route) and route_set.issubset(completed)
    )
    if not runtime_completed and runtime_current is not None and current is None:
        labels.append("progress_mismatch")
    if not runtime_completed and current is not None and current in positions:
        required = {
            step_id
            for step_id in route[positions[current] + 1 :]
            if step_id not in completed
        }
        if not required.issubset(set(remaining)):
            labels.append("progress_mismatch")
    if value["progress_status"] == "completed" and (
        current is not None or remaining or not runtime_completed
    ):
        labels.append("progress_mismatch")
    if runtime_completed and (current is not None or remaining):
        labels.append("progress_mismatch")
    if runtime_completed and value["progress_status"] not in {None, "completed"}:
        labels.append("progress_mismatch")
    if value["progress_status"] == "not_started" and completed:
        labels.append("progress_mismatch")

    if graph is not None:
        if any(
            positions[left] >= positions[right]
            for left, right in zip(remaining, remaining[1:])
            if left in positions and right in positions
        ):
            labels.append("order_violation")
        prior = set(completed)
        if current is not None:
            node = graph.nodes.get(current)
            if node is not None and any(
                predecessor in route_set and predecessor not in prior
                for predecessor in node.predecessors
            ):
                labels.append("order_violation")
            prior.add(current)
        for step_id in remaining:
            node = graph.nodes.get(step_id)
            if node is None:
                continue
            if any(
                predecessor in route_set and predecessor not in prior
                for predecessor in node.predecessors
            ):
                labels.append("order_violation")
            prior.add(step_id)
    elif any(
        positions[left] >= positions[right]
        for left, right in zip(remaining, remaining[1:])
        if left in positions and right in positions
    ):
        labels.append("order_violation")
    return list(dict.fromkeys(labels))


def resolve_reference_legal_actions(
    row: Mapping[str, Any],
) -> tuple[list[str] | None, bool]:
    """Resolve the independent reference action set without runtime fallback.

    The presence of a higher-priority key is authoritative, including an empty
    list.  A missing or malformed value is therefore unevaluable rather than a
    reason to silently consult the online runtime state.
    """

    sources = (
        (row, "reference_legal_actions"),
        (row.get("reference_labels"), "reference_legal_actions"),
        (row.get("reference"), "legal_actions"),
    )
    reference_state = row.get("reference_runtime_state")
    if isinstance(reference_state, Mapping):
        action = reference_state.get("action")
        sources += ((action, "available_actions"),)
    for source, key in sources:
        if isinstance(source, Mapping) and key in source:
            value = source[key]
            if isinstance(value, (list, tuple, set)):
                return [str(item) for item in value], True
            return None, False
    return None, False


def reference_fields_from_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Collect explicitly supplied, non-runtime reference task/scene fields."""

    fields: dict[str, Any] = {}
    labels = row.get("reference_labels")
    if isinstance(labels, Mapping):
        for key, value in labels.items():
            fields[key[len("reference_") :] if key.startswith("reference_") else key] = value
    reference = row.get("reference")
    if isinstance(reference, Mapping):
        fields.update(reference)
    for key, value in row.items():
        if key.startswith("reference_") and key not in {
            "reference_runtime_state",
            "reference_evidence",
            "reference_task_graph",
            "reference_task_constraints",
            "reference_legal_actions",
        }:
            fields.setdefault(key[len("reference_") :], value)
    return fields


def reference_violation_labels(
    candidate: Any,
    state: RuntimeState | None,
    evidence: Sequence[EvidenceRecord] | None,
    task_graph: TaskGraph | None = None,
    *,
    reference_legal_actions: Sequence[str] | None | object = _MISSING,
    reference_fields: Mapping[str, Any] | None = None,
) -> tuple[list[str], bool]:
    """Return independent reference labels and whether the check was evaluable."""

    value, labels = _schema_labels(candidate)
    if value is None:
        return labels, True

    evaluable = True
    reference_fields = reference_fields or {}
    admissible_evidence = reference_fields.get("admissible_evidence_ids")
    if evidence is None and not isinstance(admissible_evidence, (list, tuple, set)):
        evaluable = False
    else:
        evidence_ids = (
            {str(item) for item in admissible_evidence}
            if isinstance(admissible_evidence, (list, tuple, set))
            else (
                {str(item.evidence_id) for item in evidence}
                if evidence is not None
                else set()
            )
        )
        if any(identifier not in evidence_ids for identifier in value["evidence_ids"]):
            labels.append("evidence_mismatch")

    refs = value["scene_refs"]
    if not isinstance(refs, Mapping):
        labels.append("scene_mismatch")
    else:
        entity_ids = refs.get("entity_ids", [])
        locations = refs.get("locations", [])
        triples = refs.get("relation_triples", [])
        if not all(isinstance(items, list) for items in (entity_ids, locations, triples)):
            labels.append("scene_mismatch")
        else:
            entity_source = reference_fields.get("scene_entities")
            relation_source = reference_fields.get("scene_relations")
            entities = (
                {str(item) for item in entity_source}
                if isinstance(entity_source, (list, tuple, set))
                else _entities(state)
            )
            relations = (
                {
                    (str(item.get("subject")), str(item.get("relation")), str(item.get("object")))
                    for item in relation_source
                    if isinstance(item, Mapping)
                }
                if isinstance(relation_source, (list, tuple, set))
                else _relations(state)
            )
            allowed_locations = {
                item
                for item in (
                    reference_fields.get("scene_id"),
                    reference_fields.get("location"),
                    state.scene.scene_id if state is not None else None,
                    state.scene.location if state is not None else None,
                )
                if item
            }
            if state is None and entity_source is None:
                evaluable = False
            if any(str(item) not in entities for item in entity_ids):
                labels.append("scene_mismatch")
            if state is None and not any(
                key in reference_fields for key in ("scene_id", "location")
            ):
                # No independent location fact was supplied.  An empty
                # candidate location list remains structurally checkable.
                if locations:
                    evaluable = False
            if any(str(item) not in allowed_locations for item in locations):
                labels.append("scene_mismatch")
            parsed: list[tuple[str, str, str]] = []
            for triple in triples:
                if isinstance(triple, Mapping):
                    parsed.append((str(triple.get("subject")), str(triple.get("relation")), str(triple.get("object"))))
                elif isinstance(triple, list) and len(triple) == 3:
                    parsed.append(tuple(map(str, triple)))
                else:
                    labels.append("scene_mismatch")
                    continue
            if state is None and relation_source is None:
                evaluable = False
            if not relations and state is None and parsed:
                evaluable = False
            if any(triple not in relations for triple in parsed):
                labels.append("scene_mismatch")

    reference_fields = reference_fields or {}
    progress_keys = {"current_step", "progress_status", "completed_steps", "remaining_steps"}
    if state is None and not progress_keys.issubset(reference_fields):
        evaluable = False
    labels.extend(_progress_labels(value, state, task_graph, reference_fields))

    if reference_legal_actions is _MISSING:
        # The independent checker never infers an oracle action set from the
        # online state.  Callers must pass the reference set explicitly.
        legal_actions = None
        action_evaluable = False
    else:
        legal_actions = (
            list(reference_legal_actions)
            if isinstance(reference_legal_actions, (list, tuple, set))
            else None
        )
        action_evaluable = legal_actions is not None
    if not action_evaluable:
        evaluable = False
    if value["next_action_id"] is not None and action_evaluable and value["next_action_id"] not in legal_actions:
        labels.append("illegal_action")
    return list(dict.fromkeys(labels)), evaluable
