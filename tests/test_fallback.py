"""Fallback and pipeline state-immutability tests."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np

from state_grounded_qa.fallback import ConservativeFallback
from state_grounded_qa.pipeline import StateGroundedPipeline
from state_grounded_qa.schemas import (
    ActionState,
    DialogueState,
    EvidenceRecord,
    RuntimeState,
    SceneState,
    StateMetadata,
    TaskState,
)
from state_grounded_qa.state_adapter import StateAdapter


def runtime_state() -> RuntimeState:
    return RuntimeState(
        scene=SceneState(scene_id="room", location="north", active_object="lever", objects=[{"object_id": "lever"}]),
        task=TaskState(
            goal="finish",
            current_step="step_1",
            progress_status="in_progress",
            remaining_steps=["step_2"],
            milestone_progress=0.0,
        ),
        action=ActionState(available_actions=["inspect"]),
        dialogue=DialogueState(),
        metadata=StateMetadata(snapshot_id="s1", source_adapter="test"),
    )


class FixedAdapter(StateAdapter):
    def __init__(self, state: RuntimeState) -> None:
        super().__init__()
        self.state = state

    def build_state(self, raw_record: Mapping[str, Any]) -> RuntimeState:
        return self.state


class FixedEmbedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, float(index + 1)] for index, _ in enumerate(texts)])


class RejectingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, serialized_input: str, verifier_feedback: Sequence[str] | None = None) -> dict[str, Any]:
        self.calls += 1
        return {"answer": "unstructured"}


def test_fallback_is_non_executing_and_does_not_modify_state() -> None:
    state = runtime_state()
    before = copy.deepcopy(state.to_dict())
    response = ConservativeFallback(state)
    assert response.next_action_id is None
    assert state.to_dict() == before


def test_retry_and_fallback_reuse_state_without_modification() -> None:
    state = runtime_state()
    before = copy.deepcopy(state.to_dict())
    generator = RejectingGenerator()
    pipeline = StateGroundedPipeline(
        adapter=FixedAdapter(state),
        task_graph=None,
        knowledge_base=[EvidenceRecord("e1", "Inspect the lever.", "fixture")],
        embedder=FixedEmbedder(),
        generator=generator,
    )
    result = pipeline.run({}, "What next?")
    assert generator.calls == 2
    assert result.used_fallback
    assert result.response.next_action_id is None
    assert state.to_dict() == before
