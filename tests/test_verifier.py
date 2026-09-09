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
