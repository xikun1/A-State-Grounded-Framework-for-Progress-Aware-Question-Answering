"""Adapter normalization and temporal-separation tests."""

from __future__ import annotations

import json
from pathlib import Path

from state_grounded_qa.adapters import EAIVirtualHomeAdapter, MSQAAdapter, UE5Adapter


ROOT = Path(__file__).resolve().parents[1]


def test_ue5_adapter_maps_exported_record_without_reference_labels() -> None:
    record = json.loads(
        (ROOT / "tests" / "fixtures" / "ue5" / "raw_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    adapter = UE5Adapter.from_yaml(ROOT / "configs" / "ue5_adapter.yaml")
    state = adapter.build_state(record)
    assert state.scene.active_object == "lever_01"
    assert state.task.current_step == "step_inspect"
    assert state.metadata.event_index == 0
    assert "reference_labels" not in json.dumps(state.to_dict())


def test_msqa_scene_only_fields_remain_non_applicable() -> None:
    state = MSQAAdapter().build_state(
        {
            "scene_id": "Room 1",
            "objects": [{"id": "Chair 7"}],
            "question": "What is visible?",
            "answer": "A chair",
        }
    )
    assert state.scene.scene_id == "room_1"
    assert state.task.progress_status is None
    assert state.task.current_step is None
    assert state.action.available_actions == []


def test_eai_adapter_does_not_consume_future_trajectory_items() -> None:
    record = {
        "request_id": "eai_fixture_2",
        "request_index": 1,
        "task_specification": {
            "task_id": "fixture",
            "goal": "Complete two steps",
            "nodes": [
                {
                    "node_id": "step_1",
                    "predecessors": [],
                    "allowed_actions": ["inspect"],
                    "acceptable_next_actions": ["inspect"],
                    "preconditions": {},
                    "completion_rule": None,
                    "evidence_ids": [],
                },
                {
                    "node_id": "step_2",
                    "predecessors": ["step_1"],
                    "allowed_actions": ["attach"],
                    "acceptable_next_actions": ["attach"],
                    "preconditions": {},
                    "completion_rule": None,
                    "evidence_ids": [],
                },
            ],
        },
        "trajectory": [
            {"action_id": "inspect", "completed_step_id": "step_1"},
            {"action_id": "attach", "completed_step_id": "step_2", "event": {"type": "final_outcome"}},
        ],
    }
    state = EAIVirtualHomeAdapter().build_state(record)
    assert state.task.completed_steps == ["step_1"]
    assert state.task.current_step == "step_2"
    assert state.task.progress_status == "in_progress"
