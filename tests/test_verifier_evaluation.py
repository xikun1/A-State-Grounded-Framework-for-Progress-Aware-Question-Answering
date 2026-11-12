"""Executable full verifier procedure tests without model or environment data."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.verifier_metrics import verifier_metrics  # noqa: E402
from evaluation.reference_constraints import reference_violation_labels  # noqa: E402
from evaluation.violation_injection import INJECTORS  # noqa: E402
from run_verifier_evaluation import evaluate  # noqa: E402
from state_grounded_qa.schemas import EvidenceRecord, RuntimeState  # noqa: E402
from state_grounded_qa.progress import TaskGraph  # noqa: E402
from state_grounded_qa.verifier import verify_response  # noqa: E402
from audit_table11 import (  # noqa: E402
    DuplicateRecordIdentityError,
    _compare_main_and_natural_tracks,
    _index_records,
)
from run_verifier_evaluation import _run_pipeline, _run_with_retry_if_needed  # noqa: E402


def fixture_row() -> dict:
    path = ROOT / "tests" / "fixtures" / "verifier" / "accepted_candidates.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    # The production evaluator must never infer independent truth from the
    # online state.  Supply the independent side explicitly for unit tests.
    row["reference_runtime_state"] = copy.deepcopy(row["runtime_state"])
    row["reference_evidence"] = copy.deepcopy(row["evidence"])
    row["reference_task_graph"] = copy.deepcopy(row["task_graph"])
    return row


def test_controlled_full_verifier_evaluation_runs_retry_and_final_recheck() -> None:
    row = fixture_row()
    row["retry_responses"] = {
        label: copy.deepcopy(row["response"])
        for label in INJECTORS
    }
    outcomes = evaluate([row], procedure="controlled-full")
    assert len(outcomes) == len(INJECTORS)
    assert sum(bool(item["retried"]) for item in outcomes) == len(INJECTORS)
    assert all(item["retry_recovered"] for item in outcomes)
    assert all(not item["used_fallback"] for item in outcomes)
    metric = verifier_metrics(outcomes)
    assert metric["detection_recall"] == 1.0
    assert metric["residual_violation_rate"] == 0.0
    assert metric["retry_rate"] == 1.0


def test_full_verifier_evaluation_does_not_inject_and_accepts_without_retry() -> None:
    row = fixture_row()
    outcomes = evaluate([row], procedure="full")
    assert len(outcomes) == 1
    assert outcomes[0]["injected_label"] is None
    assert outcomes[0]["retried"] is False
    assert outcomes[0]["used_fallback"] is False


def test_natural_evaluation_accepts_clean_candidate_without_retry() -> None:
    row = fixture_row()

    outcomes = evaluate([row], procedure="natural")
    metric = verifier_metrics(outcomes)

    assert outcomes[0]["evaluation_track"] == "table11_natural_errors"
    assert outcomes[0]["initial_reference_valid"] is True
    assert outcomes[0]["initial_verifier_accepted"] is True
    assert outcomes[0]["retried"] is False
    assert metric["natural_violation_rate_before_verify"] == 0.0
    assert metric["natural_retry_rate"] == 0.0
    assert metric["natural_residual_violation_rate"] == 0.0
    assert metric["natural_valid_non_fallback_coverage"] == 1.0


def test_natural_evaluation_detects_reference_violation_and_recovers_on_retry() -> None:
    row = fixture_row()
    row["response"] = copy.deepcopy(row["response"])
    row["response"]["next_action_id"] = "illegal"
    row["retry_response"] = copy.deepcopy(fixture_row()["response"])

    outcomes = evaluate([row], procedure="natural")
    metric = verifier_metrics(outcomes)

    assert outcomes[0]["initial_reference_valid"] is False
    assert "illegal_action" in outcomes[0]["initial_reference_labels"]
    assert outcomes[0]["initial_verifier_accepted"] is False
    assert outcomes[0]["retried"] is True
    assert outcomes[0]["retry_recovered"] is True
    assert metric["natural_detection_recall"] == 1.0
    assert metric["natural_retry_recovery_rate"] == 1.0
    assert metric["natural_residual_violation_rate"] == 0.0


def test_natural_evaluation_keeps_undetected_reference_violation() -> None:
    row = fixture_row()
    row["reference_runtime_state"] = copy.deepcopy(row["runtime_state"])
    row["reference_runtime_state"]["action"]["available_actions"] = []

    outcomes = evaluate([row], procedure="natural")
    metric = verifier_metrics(outcomes)

    assert outcomes[0]["initial_reference_valid"] is False
    assert outcomes[0]["initial_verifier_accepted"] is True
    assert outcomes[0]["retried"] is False
    assert metric["natural_detection_recall"] == 0.0
    assert metric["natural_residual_violation_rate"] == 1.0


def test_natural_missing_reference_is_marked_unevaluable() -> None:
    row = fixture_row()
    row["reference_runtime_state"] = "not supplied"

    outcomes = evaluate([row], procedure="natural")
    metric = verifier_metrics(outcomes)

    assert outcomes[0]["initial_reference_evaluable"] is False
    assert outcomes[0]["initial_reference_valid"] is None
    assert outcomes[0]["final_reference_evaluable"] is False
    assert outcomes[0]["final_reference_valid"] is None
    assert metric["n_initial_reference_unevaluable"] == 1
    assert metric["n_final_reference_unevaluable"] == 1
    assert metric["natural_valid_non_fallback_coverage"] is None


def test_natural_evaluation_counts_false_rejection_independently() -> None:
    row = fixture_row()
    row["runtime_state"]["scene"]["active_object"] = None
    row["runtime_state"]["scene"]["objects"] = []
    row["retry_response"] = copy.deepcopy(row["response"])
    row["retry_response"]["scene_refs"] = {
        "entity_ids": [],
        "locations": [],
        "relation_triples": [],
    }

    outcomes = evaluate([row], procedure="natural")
    metric = verifier_metrics(outcomes)

    assert outcomes[0]["initial_reference_valid"] is True
    assert outcomes[0]["initial_verifier_accepted"] is False
    assert metric["natural_false_rejection_rate"] == 1.0
    assert metric["natural_retry_recovery_rate"] == 1.0
    assert metric["natural_valid_non_fallback_coverage"] == 1.0


def test_natural_evaluation_uses_fallback_after_retry_failure() -> None:
    row = fixture_row()
    row["response"] = {"answer": "malformed"}
    row["retry_response"] = {"answer": "still malformed"}

    outcomes = evaluate([row], procedure="natural")
    metric = verifier_metrics(outcomes)

    assert outcomes[0]["retried"] is True
    assert outcomes[0]["used_fallback"] is True
    assert outcomes[0]["final_reference_valid"] is True
    assert outcomes[0]["final_reference_labels"] == []
    assert metric["natural_retry_rate"] == 1.0
    assert metric["natural_fallback_rate"] == 1.0


def test_natural_retry_acceptance_with_reference_invalid_is_residual_not_coverage() -> None:
    row = fixture_row()
    row["response"] = {"answer": "malformed"}
    row["retry_response"] = copy.deepcopy(fixture_row()["response"])
    row["reference_legal_actions"] = []

    outcomes = evaluate([row], procedure="natural")
    metric = verifier_metrics(outcomes)

    assert outcomes[0]["retry_verifier_accepted"] is True
    assert outcomes[0]["retry_reference_valid"] is False
    assert outcomes[0]["final_reference_valid"] is False
    assert metric["natural_retry_recovery_rate"] == 1.0
    assert metric["natural_residual_violation_rate"] == 1.0
    assert metric["natural_valid_non_fallback_coverage"] == 0.0


def test_natural_and_controlled_tracks_are_aggregated_separately() -> None:
    row = fixture_row()
    natural = evaluate([row], procedure="natural")
    controlled = evaluate([row], procedure="controlled")

    assert natural[0]["evaluation_track"] == "table11_natural_errors"
    with pytest.raises(ValueError):
        verifier_metrics(natural + controlled)


def _synthetic_natural_row(
    *,
    initial_valid: bool,
    initial_accepted: bool,
    retried: bool,
    retry_accepted: bool,
    fallback: bool,
    final_valid: bool,
) -> dict:
    return {
        "evaluation_track": "table11_natural_errors",
        "initial_reference_evaluable": True,
        "initial_reference_valid": initial_valid,
        "initial_verifier_accepted": initial_accepted,
        "initial_accepted": initial_accepted,
        "retried": retried,
        "retry_verifier_accepted": retry_accepted if retried else None,
        "retry_recovered": retry_accepted if retried else False,
        "used_fallback": fallback,
        "final_reference_evaluable": True,
        "final_reference_valid": final_valid,
        "final_reference_labels": [] if final_valid else ["illegal_action"],
        "final_accepted": not fallback,
    }


def test_natural_simulated_eai_counts_match_table11_formulas() -> None:
    rows = []
    rows += [_synthetic_natural_row(initial_valid=True, initial_accepted=True, retried=False, retry_accepted=False, fallback=False, final_valid=True) for _ in range(1351)]
    rows += [_synthetic_natural_row(initial_valid=False, initial_accepted=False, retried=True, retry_accepted=True, fallback=False, final_valid=True) for _ in range(451)]
    rows += [_synthetic_natural_row(initial_valid=False, initial_accepted=False, retried=True, retry_accepted=True, fallback=False, final_valid=False) for _ in range(22)]
    rows += [_synthetic_natural_row(initial_valid=False, initial_accepted=False, retried=True, retry_accepted=False, fallback=True, final_valid=True) for _ in range(53)]
    rows += [_synthetic_natural_row(initial_valid=True, initial_accepted=False, retried=True, retry_accepted=False, fallback=True, final_valid=True) for _ in range(31)]
    rows += [_synthetic_natural_row(initial_valid=False, initial_accepted=True, retried=False, retry_accepted=False, fallback=False, final_valid=False) for _ in range(92)]

    metrics = verifier_metrics(rows)

    assert metrics["n_requests"] == 2000
    assert metrics["n_initial_reference_invalid"] == 618
    assert metrics["n_initial_invalid_rejected"] == 526
    assert metrics["n_initial_valid_rejected"] == 31
    assert metrics["n_retried"] == 557
    assert metrics["n_retry_verifier_accepted"] == 473
    assert metrics["n_fallback"] == 84
    assert metrics["n_final_reference_invalid"] == 114
    assert metrics["n_valid_non_fallback"] == 1802
    assert metrics["natural_violation_rate_before_verify_percent"] == 30.9
    assert metrics["natural_detection_recall_percent"] == 85.1
    assert metrics["natural_false_rejection_rate_percent"] == 2.2
    assert metrics["natural_retry_rate_percent"] == 27.9
    assert metrics["natural_retry_recovery_rate_percent"] == 84.9
    assert metrics["natural_fallback_rate_percent"] == 4.2
    assert metrics["natural_residual_violation_rate_percent"] == 5.7
    assert metrics["natural_valid_non_fallback_coverage_percent"] == 90.1


def test_natural_simulated_ue5_counts_match_table11_formulas() -> None:
    rows = []
    rows += [_synthetic_natural_row(initial_valid=True, initial_accepted=True, retried=False, retry_accepted=False, fallback=False, final_valid=True) for _ in range(311)]
    rows += [_synthetic_natural_row(initial_valid=False, initial_accepted=False, retried=True, retry_accepted=True, fallback=False, final_valid=True) for _ in range(115)]
    rows += [_synthetic_natural_row(initial_valid=False, initial_accepted=False, retried=True, retry_accepted=True, fallback=False, final_valid=False) for _ in range(10)]
    rows += [_synthetic_natural_row(initial_valid=False, initial_accepted=False, retried=True, retry_accepted=False, fallback=True, final_valid=True) for _ in range(11)]
    rows += [_synthetic_natural_row(initial_valid=True, initial_accepted=False, retried=True, retry_accepted=False, fallback=True, final_valid=True) for _ in range(8)]
    rows += [_synthetic_natural_row(initial_valid=False, initial_accepted=True, retried=False, retry_accepted=False, fallback=False, final_valid=False) for _ in range(25)]

    metrics = verifier_metrics(rows)

    assert metrics["n_requests"] == 480
    assert metrics["n_initial_reference_invalid"] == 161
    assert metrics["n_initial_invalid_rejected"] == 136
    assert metrics["n_initial_valid_rejected"] == 8
    assert metrics["n_retried"] == 144
    assert metrics["n_retry_verifier_accepted"] == 125
    assert metrics["n_fallback"] == 19
    assert metrics["n_final_reference_invalid"] == 35
    assert metrics["n_valid_non_fallback"] == 426
    assert metrics["natural_violation_rate_before_verify_percent"] == 33.5
    assert metrics["natural_detection_recall_percent"] == 84.5
    assert metrics["natural_false_rejection_rate_percent"] == 2.5
    assert metrics["natural_retry_rate_percent"] == 30.0
    assert metrics["natural_retry_recovery_rate_percent"] == 86.8
    assert metrics["natural_fallback_rate_percent"] == 4.0
    assert metrics["natural_residual_violation_rate_percent"] == 7.3
    assert metrics["natural_valid_non_fallback_coverage_percent"] == 88.8


def test_full_evaluation_counts_independent_false_rejection_and_retry_recovery() -> None:
    row = fixture_row()
    row["independent_valid"] = True
    row["response"] = {"answer": "malformed"}
    row["retry_response"] = fixture_row()["response"]
    outcomes = evaluate([row], procedure="full")
    assert len(outcomes) == 1
    assert outcomes[0]["initial_accepted"] is False
    assert outcomes[0]["retried"] is True
    assert outcomes[0]["retry_recovered"] is True
    metric = verifier_metrics(outcomes)
    assert metric["false_rejection_rate"] == 1.0
    assert metric["retry_recovery_rate"] == 1.0
    assert metric["valid_non_fallback_coverage"] == 1.0


def test_full_evaluation_retries_once_then_uses_fallback() -> None:
    row = fixture_row()
    row["response"] = {"answer": "malformed"}
    row["retry_response"] = {"answer": "still malformed"}
    outcomes = evaluate([row], procedure="full")
    assert outcomes[0]["retried"] is True
    assert outcomes[0]["retry_recovered"] is False
    assert outcomes[0]["used_fallback"] is True
    metric = verifier_metrics(outcomes)
    assert metric["retry_rate"] == 1.0
    assert metric["retry_recovery_rate"] == 0.0
    assert metric["fallback_rate"] == 1.0
    assert metric["valid_non_fallback_coverage"] == 0.0


def test_controlled_full_residual_uses_independent_reference_state() -> None:
    row = fixture_row()
    row["reference_runtime_state"] = copy.deepcopy(row["runtime_state"])
    row["reference_runtime_state"]["action"]["available_actions"] = []
    row["retry_responses"] = {
        label: copy.deepcopy(row["response"])
        for label in INJECTORS
    }
    outcomes = evaluate([row], procedure="controlled-full")
    metric = verifier_metrics(outcomes)
    assert metric["residual_violation_rate"] == 1.0
    assert metric["residual_violation_rate_by_type"]["illegal_action"] == 1.0


def test_table10_and_table11_tracks_are_not_mixed() -> None:
    row = fixture_row()
    controlled = evaluate([row], procedure="controlled")
    full = evaluate([row], procedure="full")
    assert all(item["evaluation_track"] == "table10_detection" for item in controlled)
    assert all(item["evaluation_track"] == "table12_full_request" for item in full)
    assert all(item["injected_label"] is not None for item in controlled[1:])
    assert len(full) == 1 and full[0]["injected_label"] is None
    with pytest.raises(ValueError, match="Different verifier evaluation tracks"):
        verifier_metrics(controlled + full)


def test_legacy_full_track_is_normalized_to_table12() -> None:
    metric = verifier_metrics(
        [
            {
                "evaluation_track": "table11_full_request",
                "final_reference_evaluable": False,
                "used_fallback": False,
            }
        ]
    )

    assert metric["evaluation_track"] == "table12_full_request"


def test_independent_reference_actions_do_not_fall_back_to_runtime_actions() -> None:
    row = fixture_row()
    state = RuntimeState.from_dict(row["runtime_state"])
    evidence = [EvidenceRecord(**item) for item in row["evidence"]]
    graph = TaskGraph.from_dict(row["task_graph"])
    candidate = row["response"]

    online = verify_response(candidate, state, evidence, graph)
    labels, evaluable = reference_violation_labels(
        candidate,
        state,
        evidence,
        graph,
        reference_legal_actions=[],
    )
    assert online.accepted is True
    assert evaluable is True
    assert "illegal_action" in labels

    labels, evaluable = reference_violation_labels(
        candidate,
        state,
        evidence,
        graph,
        reference_legal_actions=["inspect"],
    )
    assert evaluable is True
    assert "illegal_action" not in labels

    labels, evaluable = reference_violation_labels(
        candidate,
        state,
        evidence,
        graph,
        reference_legal_actions=None,
    )
    assert evaluable is False
    assert "illegal_action" not in labels


def test_final_reference_labels_are_recomputed_for_actual_response() -> None:
    row = fixture_row()
    row["runtime_state"]["action"]["available_actions"] = ["inspect", "rotate"]
    row["final_reference_valid"] = True
    row["final_reference_labels"] = []
    row["response"] = copy.deepcopy(row["response"])
    row["response"]["next_action_id"] = "rotate"
    outcomes = evaluate([row], procedure="natural")

    assert outcomes[0]["initial_verifier_accepted"] is True
    assert outcomes[0]["final_reference_valid"] is False
    assert outcomes[0]["final_reference_labels"] == ["illegal_action"]
    assert outcomes[0]["supplied_final_reference_valid"] is True
    assert outcomes[0]["final_reference_agreement"] is False


def test_partial_natural_reference_evaluation_does_not_report_zero_rates() -> None:
    complete = {
        "evaluation_track": "table11_natural_errors",
        "initial_reference_evaluable": True,
        "initial_reference_valid": True,
        "initial_verifier_accepted": True,
        "final_reference_evaluable": True,
        "final_reference_valid": True,
        "used_fallback": False,
    }
    missing = {
        "evaluation_track": "table11_natural_errors",
        "initial_reference_evaluable": False,
        "initial_reference_valid": None,
        "initial_verifier_accepted": True,
        "final_reference_evaluable": False,
        "final_reference_valid": None,
        "used_fallback": False,
    }
    metrics = verifier_metrics([complete] * 80 + [missing] * 20)
    assert metrics["n_final_reference_evaluable"] == 80
    assert metrics["n_final_reference_unevaluable"] == 20
    assert metrics["natural_residual_violation_rate"] is None
    assert metrics["natural_valid_non_fallback_coverage"] is None
    assert metrics["reference_evaluation_complete"] is False


def test_natural_reference_completeness_controls_formal_full_sample_rates() -> None:
    valid = {
        "evaluation_track": "table11_natural_errors",
        "initial_reference_evaluable": True,
        "initial_reference_valid": True,
        "initial_verifier_accepted": True,
        "final_reference_evaluable": True,
        "final_reference_valid": True,
        "used_fallback": False,
    }
    residual = {**valid, "final_reference_valid": False}
    fallback = {**valid, "used_fallback": True}
    complete = verifier_metrics([residual] * 10 + [valid] * 80 + [fallback] * 10)
    assert complete["natural_residual_violation_rate"] == 0.1
    assert complete["natural_valid_non_fallback_coverage"] == 0.8
    assert complete["reference_evaluation_complete"] is True

    unevaluable = {
        "evaluation_track": "table11_natural_errors",
        "initial_reference_evaluable": False,
        "initial_reference_valid": None,
        "final_reference_evaluable": False,
        "final_reference_valid": None,
        "used_fallback": False,
    }
    missing = verifier_metrics([unevaluable] * 100)
    assert missing["n_final_reference_unevaluable"] == 100
    assert missing["natural_residual_violation_rate"] is None
    assert missing["natural_valid_non_fallback_coverage"] is None


def test_table11_main_track_comparison_reports_request_level_differences() -> None:
    table11 = [
        {
            "request_id": "r1",
            "dataset": "UE5",
            "run_id": "run-a",
            "evaluation_stage": "final_verified",
            "final_next_action_id": "inspect",
            "reference_legal_actions": ["inspect"],
            "reference_legal_actions_evaluable": True,
            "final_reference_valid": True,
        },
        {
            "request_id": "r2",
            "dataset": "UE5",
            "run_id": "run-a",
            "evaluation_stage": "final_verified",
            "final_next_action_id": "bad",
            "reference_legal_actions": ["inspect"],
            "reference_legal_actions_evaluable": True,
            "final_reference_valid": False,
        },
    ]
    main = [
        {
            "request_id": "r1",
            "dataset": "UE5",
            "run_id": "run-a",
            "evaluation_stage": "final_verified",
            "final_next_action_id": "rotate",
            "reference_legal_actions": ["inspect"],
        },
        {
            "request_id": "r2",
            "dataset": "UE5",
            "run_id": "run-a",
            "evaluation_stage": "final_verified",
            "final_next_action_id": "bad",
            "reference_legal_actions": ["inspect"],
        },
    ]
    audit = _compare_main_and_natural_tracks(table11, main, "UE5", "table14")
    assert audit["n_matched_requests"] == 2
    assert audit["different_final_action_request_ids"] == ["r1"]
    assert audit["n_table11_illegal_actions"] == 1
    assert audit["n_illegal_actions_outside_residual"] == 0


def _audit_record(
    request_id: str = "r1",
    run_id: str | None = "run-a",
    stage: str | None = "final_verified",
    action: str | None = "inspect",
    legal: list[str] | None = None,
    final_valid: bool = True,
) -> dict:
    row = {
        "dataset": "UE5",
        "request_id": request_id,
        "final_next_action_id": action,
        "reference_legal_actions": ["inspect"] if legal is None else legal,
        "reference_legal_actions_evaluable": True,
        "final_reference_valid": final_valid,
    }
    if run_id is not None:
        row["run_id"] = run_id
    if stage is not None:
        row["evaluation_stage"] = stage
    return row


def test_fixed_evidence_over_five_items_is_not_reranked_or_truncated() -> None:
    row = fixture_row()
    evidence = [
        {
            "evidence_id": f"e{index}",
            "text": f"Evidence {index}",
            "source": "test",
            "scene_constraints": {},
            "task_step_constraints": {},
            "applicability_metadata": {},
        }
        for index in range(1, 7)
    ]
    candidate = copy.deepcopy(row["response"])
    candidate["evidence_ids"] = ["e6"]
    row["retrieved_evidence"] = evidence
    row["evidence"] = evidence[:1]
    result, _, selected, _ = _run_pipeline(row, candidate)
    assert [item.evidence_id for item in selected] == [f"e{index}" for index in range(1, 7)]
    assert [item.evidence_id for item in result.retrieved_evidence] == [f"e{index}" for index in range(1, 7)]
    assert result.verification.accepted is True


def test_fixed_evidence_order_and_legacy_field_compatibility() -> None:
    row = fixture_row()
    evidence = [
        {
            "evidence_id": item,
            "text": item,
            "source": "test",
            "scene_constraints": {},
            "task_step_constraints": {},
            "applicability_metadata": {},
        }
        for item in ("e3", "e1", "e5", "e2", "e4")
    ]
    row["evidence"] = evidence
    result, _, selected, _ = _run_pipeline(row, row["response"])
    assert [item.evidence_id for item in selected] == ["e3", "e1", "e5", "e2", "e4"]
    assert [item.evidence_id for item in result.retrieved_evidence] == [
        "e3", "e1", "e5", "e2", "e4"
    ]


def test_retry_uses_the_same_fixed_evidence_once() -> None:
    row = fixture_row()
    row["retrieved_evidence"] = copy.deepcopy(row["evidence"])
    row["response"] = {"answer": "malformed"}
    row["retry_response"] = copy.deepcopy(fixture_row()["response"])
    result, _, selected, _ = _run_with_retry_if_needed(row, row["response"], "full")
    assert result.retry_count == 1
    assert result.used_fallback is False
    assert result.verification.accepted is True
    assert [item.evidence_id for item in result.retrieved_evidence] == [
        item.evidence_id for item in selected
    ]


def test_audit_requires_complete_identity_for_strict_matching() -> None:
    table11 = [_audit_record()]
    assert _compare_main_and_natural_tracks(table11, [_audit_record(run_id=None)], "UE5")[
        "n_matched_requests"
    ] == 0
    missing = _compare_main_and_natural_tracks(table11, [_audit_record(run_id=None)], "UE5")
    assert missing["n_metadata_unconfirmed"] == 1
    assert missing["audit_status"] == "metadata_incomplete"

    different_run = _compare_main_and_natural_tracks(
        table11, [_audit_record(run_id="run-b")], "UE5"
    )
    assert different_run["n_different_run"] == 1
    assert different_run["n_matched_requests"] == 0
    assert different_run["audit_status"] == "different_execution"

    different_stage = _compare_main_and_natural_tracks(
        table11, [_audit_record(stage="initial_candidate")], "UE5"
    )
    assert different_stage["n_different_stage"] == 1
    assert different_stage["audit_status"] == "different_execution"


def test_audit_preserves_distinct_runs_and_rejects_duplicate_identity() -> None:
    rows = [_audit_record(run_id="run-a"), _audit_record(run_id="run-b")]
    assert len(_index_records(rows, "UE5", "Table 11")) == 2
    with pytest.raises(DuplicateRecordIdentityError):
        _index_records([_audit_record(), _audit_record()], "UE5", "Table 11")


def test_audit_marks_matched_action_difference_failed() -> None:
    result = _compare_main_and_natural_tracks(
        [_audit_record(action="inspect")],
        [_audit_record(action="rotate")],
        "UE5",
    )
    assert result["n_matched_requests"] == 1
    assert result["audit_status"] == "failed"
    assert result["different_final_action_request_ids"] == ["r1"]


def test_audit_detects_illegal_action_outside_residual_set() -> None:
    result = _compare_main_and_natural_tracks(
        [_audit_record(action="illegal", final_valid=True)],
        [_audit_record(action="illegal", final_valid=True)],
        "UE5",
    )
    assert result["audit_status"] == "failed"
    assert result["illegal_action_outside_residual_request_ids"] == ["r1"]
