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
    """Check structural progress consistency against the authoritative state.

    The response does not have to reproduce every progress field in ``S_t``.
    It must, however, describe a possible point on the active task route.  The
    runtime completed steps are treated as confirmed facts; graph membership,
    prerequisite order, completion semantics, and duplicate-step checks remain
    deterministic.
    """

    labels: list[str] = []
    remaining = list(candidate["remaining_step_ids"])
    completed = list(state.task.completed_steps)
    confirmed_completed = set(completed)

    # Without a graph, the normalized runtime state is the only available
    # description of the active route.  With a graph, retain the runtime route
    # as well as the graph-derived route because active_if predicates may refer
    # to source fields that are intentionally not copied into RuntimeState.
    state_route = list(completed)
    if state.task.current_step is not None:
        state_route.append(state.task.current_step)
    state_route.extend(state.task.remaining_steps)
    if task_graph is None:
        active_route = list(dict.fromkeys(state_route))
    else:
        graph_route = task_graph.active_order(state.to_dict())
        # RuntimeState already records the selected branch's current and
        # remaining nodes.  Do not force graph nodes guarded by an unresolved
        # active_if predicate into that route merely because they are present
        # in the graph.
        graph_route = [
            node_id
            for node_id in graph_route
            if task_graph.nodes[node_id].active_if is None or node_id in state_route
        ]
        active_route = list(dict.fromkeys(graph_route + state_route))
    active_steps = set(active_route)
    route_position = {step_id: index for index, step_id in enumerate(active_route)}

    runtime_completed = state.task.progress_status == "completed"
    if active_route and active_steps.issubset(confirmed_completed):
        runtime_completed = True

    combined = completed + remaining
    if len(remaining) != len(set(remaining)) or len(combined) != len(set(combined)):
        labels.append("duplicate_step")
    current = candidate["current_step_id"]
    if current in remaining or current in confirmed_completed:
        labels.append("duplicate_step")
    if current is not None and current not in active_steps:
        labels.append("progress_mismatch")
        labels.append("order_violation")
    unknown_remaining = any(step_id not in active_steps for step_id in remaining)
    if unknown_remaining:
        labels.append("progress_mismatch")
        labels.append("order_violation")

    # An unfinished runtime task with a known active step cannot be represented
    # by an empty current step.  This closes the otherwise ambiguous
    # ``current=None, remaining=[]`` case without restoring equality checks for
    # otherwise non-authoritative fields.
    if not runtime_completed and state.task.current_step is not None and current is None:
        labels.append("progress_mismatch")

    # Every active node after the candidate current step is a necessary future
    # step for the selected route.  The candidate may contain additional
    # structurally valid information, but may not silently omit this suffix.
    if not runtime_completed and current is not None and current in route_position:
        required_remaining = {
            step_id
            for step_id in active_route[route_position[current] + 1 :]
            if step_id not in confirmed_completed
        }
        if not required_remaining.issubset(set(remaining)):
            labels.append("progress_mismatch")

    # A completed response is only valid when the environment has confirmed
    # completion.  Conversely, a non-completed status cannot contradict a
    # confirmed completed runtime task.  ``None`` remains an unspecified value
    # and is therefore not forced to equal the runtime status.
    if candidate["progress_status"] == "completed" and (
        current is not None or remaining or not runtime_completed
    ):
        labels.append("progress_mismatch")
    if runtime_completed and (current is not None or remaining):
        labels.append("progress_mismatch")
    if runtime_completed and candidate["progress_status"] not in {None, "completed"}:
        labels.append("progress_mismatch")
    if candidate["progress_status"] == "not_started" and confirmed_completed:
        labels.append("progress_mismatch")

    if task_graph is not None:
        if any(
            route_position[left] >= route_position[right]
            for left, right in zip(remaining, remaining[1:])
            if left in route_position and right in route_position
        ):
            labels.append("order_violation")
        prior = set(confirmed_completed)
        if current is not None:
            node = task_graph.nodes.get(current)
            if node is not None:
                active_predecessors = [pred for pred in node.predecessors if pred in active_steps]
                if any(pred not in prior for pred in active_predecessors):
                    labels.append("order_violation")
            prior.add(current)
        for step_id in remaining:
            node = task_graph.nodes.get(step_id)
            if node is None:
                continue
            active_predecessors = [pred for pred in node.predecessors if pred in active_steps]
            if any(pred not in prior for pred in active_predecessors):
                labels.append("order_violation")
            prior.add(step_id)
    elif any(
        route_position[left] >= route_position[right]
        for left, right in zip(remaining, remaining[1:])
        if left in route_position and right in route_position
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
