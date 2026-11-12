"""Deterministic environment-record anomaly input tests."""

from __future__ import annotations

import copy

from evaluation.record_anomalies import apply_record_anomaly


def _records() -> list[dict]:
    return [
        {
            "sequence_id": "sequence_1",
            "request_id": "request_1",
            "request_index": 1,
            "runtime_state": {"task": {"current_step": "step_1"}},
            "reference_labels": {"reference_current_step": "step_1"},
        },
        {
            "sequence_id": "sequence_1",
            "request_id": "request_2",
            "request_index": 2,
            "runtime_state": {"task": {"current_step": "step_2"}},
            "reference_labels": {"reference_current_step": "step_2"},
        },
    ]


def test_record_anomaly_conditions_preserve_request_and_reference_fields() -> None:
    records = _records()
    clean = apply_record_anomaly(records, "clean")
    incomplete = apply_record_anomaly(
        records, "incomplete", field_paths=["runtime_state.task.current_step"], missing_rule="remove"
    )
    contradictory = apply_record_anomaly(
        records,
        "contradictory",
        field_paths=["runtime_state.task.current_step"],
        conflict_value="conflicting_step",
    )

    assert [row["runtime_state"] for row in clean] == [row["runtime_state"] for row in records]
    assert [row["reference_labels"] for row in clean] == [row["reference_labels"] for row in records]
    assert "current_step" not in incomplete[0]["runtime_state"]["task"]
    assert contradictory[0]["runtime_state"]["task"]["current_step"] == "conflicting_step"
    for output in (incomplete, contradictory):
        assert [row["request_id"] for row in output] == ["request_1", "request_2"]
        assert [row["reference_labels"] for row in output] == [row["reference_labels"] for row in records]


def test_delayed_anomaly_uses_earlier_state_within_sequence() -> None:
    records = _records()
    delayed = apply_record_anomaly(records, "delayed", delay_steps=1)

    assert delayed[0]["runtime_state"]["task"]["current_step"] == "step_1"
    assert delayed[1]["runtime_state"]["task"]["current_step"] == "step_1"
    assert delayed[1]["delayed_from_request_id"] == "request_1"
    assert delayed[1]["reference_labels"] == records[1]["reference_labels"]


def test_record_anomaly_field_selection_is_reproducible_with_seed() -> None:
    records = _records()
    first = apply_record_anomaly(
        records,
        "incomplete",
        field_paths=["runtime_state.task.current_step", "runtime_state.task.missing"],
        seed=7,
    )
    second = apply_record_anomaly(
        copy.deepcopy(records),
        "incomplete",
        field_paths=["runtime_state.task.current_step", "runtime_state.task.missing"],
        seed=7,
    )

    assert first == second
    assert first[0]["record_anomaly"]["seed"] == 7
