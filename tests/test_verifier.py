"""Deterministic evidence and action checks."""

from __future__ import annotations

from state_grounded_qa.schemas import (
    ActionState,
    DialogueState,
    EvidenceRecord,
    RuntimeState,
    SceneState,
    StateMetadata,
    StructuredResponse,
    TaskState,
)
from state_grounded_qa.progress import TaskGraph, TaskNode
from state_grounded_qa.verifier import verify_response


def state() -> RuntimeState:
    return RuntimeState(
        scene=SceneState(
            scene_id="room",
            location="north",
            active_object="lever",
            objects=[{"object_id": "lever"}],
        ),
        task=TaskState(
            goal="finish",
            current_step="step_1",
            progress_status="in_progress",
            completed_steps=[],
            remaining_steps=["step_2"],
            milestone_progress=0.0,
        ),
        action=ActionState(available_actions=["inspect"]),
        dialogue=DialogueState(),
        metadata=StateMetadata(snapshot_id="s1", source_adapter="test"),
    )


def response() -> StructuredResponse:
    return StructuredResponse(
        answer="Inspect the lever.",
        evidence_ids=["e1"],
        scene_refs={"entity_ids": ["lever"], "locations": ["north"], "relation_triples": []},
        current_step_id="step_1",
        progress_status="in_progress",
        remaining_step_ids=["step_2"],
        next_action_id="inspect",
    )


def test_illegal_action_is_rejected() -> None:
    candidate = response().to_dict()
    candidate["next_action_id"] = "skip_ahead"
    result = verify_response(candidate, state(), [EvidenceRecord("e1", "Inspect first.", "fixture")])
    assert not result.accepted
    assert "illegal_action" in result.failure_labels


def test_nonexistent_evidence_is_rejected() -> None:
    candidate = response().to_dict()
    candidate["evidence_ids"] = ["missing"]
    result = verify_response(candidate, state(), [EvidenceRecord("e1", "Inspect first.", "fixture")])
    assert not result.accepted
    assert "evidence_mismatch" in result.failure_labels


def progress_graph() -> TaskGraph:
    return TaskGraph(
        [
            TaskNode("step_1"),
            TaskNode("step_2", predecessors=["step_1"]),
        ],
        goal="finish",
    )


def test_progress_allows_structurally_consistent_non_authoritative_fields() -> None:
    candidate = response().to_dict()
    candidate["progress_status"] = "not_started"
    result = verify_response(
        candidate,
        state(),
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        progress_graph(),
    )
    assert result.accepted


def test_progress_rejects_unmet_prerequisite_for_current_step() -> None:
    candidate = response().to_dict()
    candidate["current_step_id"] = "step_2"
    candidate["remaining_step_ids"] = []
    result = verify_response(
        candidate,
        state(),
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        progress_graph(),
    )
    assert not result.accepted
    assert "order_violation" in result.failure_labels


def test_progress_rejects_unconfirmed_completion() -> None:
    candidate = response().to_dict()
    candidate["current_step_id"] = None
    candidate["progress_status"] = "completed"
    candidate["remaining_step_ids"] = []
    result = verify_response(
        candidate,
        state(),
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        progress_graph(),
    )
    assert not result.accepted
    assert "progress_mismatch" in result.failure_labels


def linear_graph() -> TaskGraph:
    return TaskGraph(
        [
            TaskNode("step_1"),
            TaskNode("step_2", predecessors=["step_1"]),
            TaskNode("step_3", predecessors=["step_2"]),
        ],
        goal="finish",
    )


def branch_graph() -> TaskGraph:
    return TaskGraph(
        [
            TaskNode("step_1"),
            TaskNode("primary", predecessors=["step_1"], active_if={"path": "branch", "equals": "primary"}),
            TaskNode("alternate", predecessors=["step_1"], active_if={"path": "branch", "equals": "alternate"}),
        ],
        goal="finish",
    )


def progress_candidate(current: str | None, remaining: list[str], status: str = "in_progress") -> dict:
    candidate = response().to_dict()
    candidate["current_step_id"] = current
    candidate["remaining_step_ids"] = remaining
    candidate["progress_status"] = status
    candidate["next_action_id"] = None
    return candidate


def test_complete_current_and_remaining_steps_are_accepted() -> None:
    runtime = state()
    runtime.task.current_step = "step_2"
    runtime.task.completed_steps = ["step_1"]
    runtime.task.remaining_steps = ["step_3"]
    result = verify_response(
        progress_candidate("step_2", ["step_3"]),
        runtime,
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        linear_graph(),
    )
    assert result.accepted


def test_unfinished_task_cannot_omit_current_step() -> None:
    result = verify_response(
        progress_candidate(None, []),
        state(),
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        progress_graph(),
    )
    assert not result.accepted
    assert "progress_mismatch" in result.failure_labels


def test_required_remaining_suffix_cannot_be_omitted() -> None:
    runtime = state()
    runtime.task.current_step = "step_2"
    runtime.task.completed_steps = ["step_1"]
    runtime.task.remaining_steps = ["step_3"]
    result = verify_response(
        progress_candidate("step_2", []),
        runtime,
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        linear_graph(),
    )
    assert not result.accepted
    assert "progress_mismatch" in result.failure_labels


def test_inactive_branch_step_is_rejected() -> None:
    runtime = state()
    runtime.task.current_step = "primary"
    runtime.task.completed_steps = ["step_1"]
    runtime.task.remaining_steps = []
    result = verify_response(
        progress_candidate("alternate", []),
        runtime,
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        branch_graph(),
    )
    assert not result.accepted
    assert "progress_mismatch" in result.failure_labels


def test_current_step_cannot_skip_prerequisite() -> None:
    runtime = state()
    runtime.task.current_step = "step_3"
    runtime.task.completed_steps = ["step_1"]
    runtime.task.remaining_steps = []
    result = verify_response(
        progress_candidate("step_3", []),
        runtime,
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        linear_graph(),
    )
    assert not result.accepted
    assert "order_violation" in result.failure_labels


def test_duplicate_remaining_step_is_rejected() -> None:
    result = verify_response(
        progress_candidate("step_1", ["step_2", "step_2"]),
        state(),
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        progress_graph(),
    )
    assert not result.accepted
    assert "duplicate_step" in result.failure_labels


def test_completed_task_allows_empty_progress_fields() -> None:
    runtime = state()
    runtime.task.current_step = None
    runtime.task.progress_status = "completed"
    runtime.task.completed_steps = ["step_1", "step_2"]
    runtime.task.remaining_steps = []
    result = verify_response(
        progress_candidate(None, [], "completed"),
        runtime,
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        progress_graph(),
    )
    assert result.accepted


def test_selected_branch_does_not_require_inactive_branch() -> None:
    runtime = state()
    runtime.task.current_step = "alternate"
    runtime.task.completed_steps = ["step_1"]
    runtime.task.remaining_steps = []
    result = verify_response(
        progress_candidate("alternate", []),
        runtime,
        [EvidenceRecord("e1", "Inspect first.", "fixture")],
        branch_graph(),
    )
    assert result.accepted
