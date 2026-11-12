"""Reviewer 4 metric denominator and stage-selection tests."""

from __future__ import annotations

import pytest

from evaluation.progress_metrics import progress_action_metrics, state_construction_metrics
from evaluation.verifier_metrics import verifier_metrics
from evaluation.efficiency_metrics import compare_efficiency_protocols, efficiency_metrics


def _row(next_action: str | None) -> dict:
    prediction = {
        "current_step": "step_1",
        "progress_status": "in_progress",
        "completed_steps": [],
        "remaining_steps": ["step_2"],
        "legal_actions": ["inspect"],
        "next_action_id": next_action,
    }
    return {
        "prediction": prediction,
        "reference": {
            "current_step": "step_1",
            "progress_status": "in_progress",
            "completed_steps": [],
            "remaining_steps": ["step_2"],
            "legal_actions": ["inspect"],
            "acceptable_next_actions": ["inspect"],
        },
    }


def test_action_rates_use_non_null_action_denominator() -> None:
    metric = progress_action_metrics([_row("inspect"), _row(None)])
    assert metric["action_validity_rate"] == 1.0
    assert metric["illegal_action_rate"] == 0.0
    assert metric["non_null_action_recommendation_rate"] == 0.5
    assert metric["n_non_null_actions"] == 1
    assert metric["n_valid_actions"] == 1
    assert metric["n_illegal_actions"] == 0


def test_action_rates_are_not_defined_when_all_actions_are_null() -> None:
    metric = progress_action_metrics([_row(None)])
    assert metric["action_validity_rate"] is None
    assert metric["illegal_action_rate"] is None
    assert metric["non_null_action_recommendation_rate"] == 0.0
    assert metric["precondition_order_violation_rate"] is None


def test_null_action_is_incorrect_when_reference_accepts_no_action() -> None:
    row = _row(None)
    row["reference"]["acceptable_next_actions"] = []

    metric = progress_action_metrics([row])

    assert metric["next_action_accuracy"] == 0.0
    assert metric["n_next_action_evaluable"] == 1


def test_state_and_final_verified_stages_are_scored_independently() -> None:
    row = _row("inspect")
    row["state"] = {
        "task": {
            "current_step": "step_1",
            "progress_status": "in_progress",
            "completed_steps": [],
            "remaining_steps": ["step_2"],
        },
        "action": {"available_actions": ["inspect"]},
    }
    row["final_verified"] = {
        "current_step_id": "step_2",
        "progress_status": "in_progress",
        "remaining_step_ids": [],
        "next_action_id": "inspect",
    }

    state_metric = state_construction_metrics([row], stage="state")
    final_metric = state_construction_metrics([row], stage="final_verified")

    assert state_metric["evaluation_stage"] == "state"
    assert state_metric["n_requests"] == 1
    assert state_metric["current_step_accuracy"] == 1.0
    assert state_metric["completed_step_f1"] == 1.0
    assert state_metric["legal_action_set_f1"] == 1.0
    assert final_metric["evaluation_stage"] == "final_verified"
    assert final_metric["current_step_accuracy"] == 0.0
    assert final_metric["remaining_step_f1"] == 0.0


def test_precondition_order_rate_uses_reference_constraint_and_non_null_denominator() -> None:
    valid = _row("inspect")
    valid["reference"]["precondition_order_violation"] = False
    valid["prediction"]["precondition_or_order_violation"] = True
    invalid = _row("illegal")
    invalid["reference"]["precondition_order_violation"] = True
    invalid["prediction"]["precondition_or_order_violation"] = False

    metric = progress_action_metrics([valid, invalid])

    assert metric["action_validity_rate"] == 0.5
    assert metric["illegal_action_rate"] == 0.5
    assert metric["precondition_order_violation_rate"] == 0.5
    assert metric["n_precondition_order_evaluable"] == 2


def test_missing_reference_constraints_are_not_scored_as_no_violation() -> None:
    row = _row("inspect")
    del row["reference"]["legal_actions"]

    metric = progress_action_metrics([row])

    assert metric["action_validity_rate"] is None
    assert metric["illegal_action_rate"] is None
    assert metric["precondition_order_violation_rate"] is None
    assert metric["n_action_validity_unevaluable"] == 1
    assert metric["n_precondition_order_unevaluable"] == 1


def test_missing_stage_is_reported_without_fake_valid_metrics() -> None:
    metric = progress_action_metrics([_row("inspect")], stage="final_verified")

    assert metric["evaluation_status"] == "unavailable"
    assert metric["n_requests"] == 1
    assert metric["n_missing_stage_predictions"] == 1
    assert metric["current_step_accuracy"] is None
    assert metric["next_action_accuracy"] is None
    assert metric["non_null_action_recommendation_rate"] is None


def test_progress_metrics_can_select_initial_candidate_stage() -> None:
    row = _row("inspect")
    row["initial_candidate"] = dict(row["prediction"])
    row["initial_candidate"]["next_action_id"] = "illegal"
    metric = progress_action_metrics([row], stage="initial_candidate")
    assert metric["evaluation_stage"] == "initial_candidate"
    assert metric["action_validity_rate"] == 0.0
    assert metric["illegal_action_rate"] == 1.0


def test_structured_progress_fields_are_normalized_for_all_stages() -> None:
    row = _row("inspect")
    structured = {
        "current_step_id": "step_1",
        "progress_status": "in_progress",
        "remaining_step_ids": ["step_2"],
        "next_action_id": "inspect",
    }
    row["state"] = dict(row["prediction"])
    row["initial_candidate"] = dict(structured)
    row["final_verified"] = dict(structured)
    final_metric = progress_action_metrics([row], stage="final_verified")
    initial_metric = progress_action_metrics([row], stage="initial_candidate")
    state_metric = progress_action_metrics([row], stage="state")
    prediction_metric = progress_action_metrics([row])
    explicit_prediction_metric = progress_action_metrics([row], stage="prediction")
    assert final_metric["current_step_accuracy"] == 1.0
    assert final_metric["remaining_step_f1"] == 1.0
    assert initial_metric["current_step_accuracy"] == 1.0
    assert initial_metric["remaining_step_f1"] == 1.0
    assert state_metric["current_step_accuracy"] == 1.0
    assert prediction_metric["remaining_step_f1"] == 1.0
    assert explicit_prediction_metric["current_step_accuracy"] == 1.0
    assert final_metric["completed_step_f1"] is None
    assert final_metric["legal_action_set_f1"] is None
    assert final_metric["action_validity_rate"] == 1.0


def test_missing_structured_optional_fields_are_not_scored_as_empty_sets() -> None:
    row = _row("inspect")
    row["final_verified"] = {
        "current_step_id": "step_1",
        "progress_status": "in_progress",
        "remaining_step_ids": ["step_2"],
        "next_action_id": "inspect",
    }
    metric = progress_action_metrics([row], stage="final_verified")
    assert metric["completed_step_f1"] is None
    assert metric["legal_action_set_f1"] is None
    assert metric["completed_step_f1_n"] == 0
    assert metric["legal_action_set_f1_n"] == 0


def test_verifier_metrics_require_independent_validity_for_false_rejection() -> None:
    rows = [
        {
            "injected_label": None,
            "initial_accepted": False,
            "final_accepted": False,
            "retried": False,
            "used_fallback": False,
            "independent_valid": True,
        },
        {
            "injected_label": "illegal_action",
            "initial_labels": ["illegal_action"],
            "final_labels": [],
            "final_reference_labels": [],
            "initial_accepted": False,
            "final_accepted": True,
            "retried": True,
            "retry_recovered": True,
            "used_fallback": False,
            "independent_valid": False,
        },
    ]
    metric = verifier_metrics(rows)
    assert metric["false_rejection_rate"] == 1.0
    assert metric["retry_rate"] == 0.5
    assert metric["retry_recovery_rate"] == 1.0
    assert metric["residual_violation_rate"] == 0.0
    assert metric["valid_non_fallback_coverage"] == 0.5


def test_valid_non_fallback_coverage_uses_all_requests_as_denominator() -> None:
    rows = [
        {
            "initial_accepted": True,
            "final_accepted": True,
            "used_fallback": False,
            "final_reference_valid": True,
            "final_reference_evaluable": True,
        },
        {
            "initial_accepted": False,
            "final_accepted": True,
            "used_fallback": True,
            "final_reference_valid": True,
            "final_reference_evaluable": True,
        },
    ]
    assert verifier_metrics(rows)["valid_non_fallback_coverage"] == 0.5
    rows.append(
        {
            "initial_accepted": True,
            "final_accepted": True,
            "used_fallback": False,
            "final_reference_valid": True,
            "final_reference_evaluable": True,
        }
    )
    assert verifier_metrics(rows)["valid_non_fallback_coverage"] == 2 / 3


def test_residual_uses_final_reference_labels_not_initial_detection() -> None:
    row = {
        "injected_label": "illegal_action",
        "initial_labels": ["illegal_action"],
        "final_labels": [],
        "final_reference_labels": ["illegal_action"],
        "final_reference_valid": False,
        "final_reference_evaluable": True,
        "initial_accepted": False,
        "final_accepted": True,
        "retried": True,
        "retry_recovered": True,
        "used_fallback": False,
    }
    metric = verifier_metrics([row])
    assert metric["residual_violation_rate"] == 1.0
    assert metric["residual_violation_rate_by_type"]["illegal_action"] == 1.0


def test_residual_violation_counts_cross_type_remaining_error() -> None:
    row = {
        "injected_label": "schema_violation",
        "initial_labels": ["schema_violation"],
        "final_reference_labels": ["illegal_action"],
        "final_reference_valid": False,
        "final_reference_evaluable": True,
    }

    metric = verifier_metrics([row])

    assert metric["residual_violation_rate_by_type"]["schema_violation"] == 1.0
    assert metric["residual_violation_rate"] == 1.0


def test_efficiency_metrics_report_explicit_protocol_metadata() -> None:
    metric = efficiency_metrics(
        [
            {
                "latency_s": 1.0,
                "stage_latency_s": {"generation": 0.8},
                "evaluation_protocol": "ue5_offline_replay_evaluation",
                "evaluation_subset": "ue5_full",
                "timing_scope": "request_end_to_end",
                "execution_config": "batch1",
            },
            {
                "latency_s": 3.0,
                "evaluation_protocol": "ue5_offline_replay_evaluation",
                "evaluation_subset": "ue5_full",
                "timing_scope": "request_end_to_end",
                "execution_config": "batch1",
            },
        ],
        protocol="ue5_offline_replay_evaluation",
    )
    assert metric["mean_latency_s"] == 2.0
    assert metric["timing_protocol"]["evaluation_protocol"] == "ue5_offline_replay_evaluation"
    assert metric["timing_protocol"]["metadata_available"] is True


def test_efficiency_metrics_report_p95_stage_sum_and_configuration_consistency() -> None:
    metric = efficiency_metrics(
        [
            {
                "request_id": "r1",
                "latency_s": 1.0,
                "stage_latency_s": {"retrieval": 0.2, "generation": 0.5},
                "tokens": 10,
                "evaluation_protocol": "protocol_a",
                "evaluation_subset": "subset_a",
                "timing_scope": "end_to_end",
                "execution_config": {"batch_size": 1},
            },
            {
                "request_id": "r2",
                "latency_s": 3.0,
                "stage_latency_s": {"retrieval": 0.4, "generation": 1.0},
                "tokens": 20,
                "evaluation_protocol": "protocol_a",
                "evaluation_subset": "subset_a",
                "timing_scope": "end_to_end",
                "execution_config": {"batch_size": 2},
            },
        ],
        protocol="protocol_a",
    )

    assert metric["n_requests"] == 2
    assert metric["mean_latency_s"] == 2.0
    assert metric["p95_latency_s"] == 2.9
    assert metric["tokens_per_request"] == 15.0
    assert metric["stage_wise_latency_s"]["generation"] == 0.75
    assert metric["mean_stage_latency_sum_s"] == pytest.approx(1.05)
    assert metric["mean_latency_minus_stage_sum_s"] == pytest.approx(0.95)
    assert metric["timing_protocol"]["protocol_consistency"]["consistent"] is False
    assert "execution_config" in metric["timing_protocol"]["protocol_consistency"]["inconsistent_fields"]


def test_efficiency_metrics_compare_protocols_without_mixing_records() -> None:
    comparison = compare_efficiency_protocols(
        [
            {
                "request_id": "offline-1",
                "latency_s": 1.0,
                "evaluation_protocol": "ue5_offline_replay_evaluation",
                "evaluation_subset": "ue5_full",
                "timing_scope": "end_to_end",
                "execution_config": {"batch_size": 1},
            },
            {
                "request_id": "online-1",
                "latency_s": 2.0,
                "evaluation_protocol": "end_to_end_computational_cost_evaluation",
                "evaluation_subset": "ue5_full",
                "timing_scope": "end_to_end",
                "execution_config": {"batch_size": 1},
            },
        ]
    )

    assert comparison["n_protocols"] == 2
    assert comparison["protocols"]["ue5_offline_replay_evaluation"]["n_requests"] == 1
    assert comparison["protocols"]["end_to_end_computational_cost_evaluation"]["mean_latency_s"] == 2.0
