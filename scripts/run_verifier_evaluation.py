"""Evaluate controlled violations or the complete verify/retry/fallback flow."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from _common import read_jsonl, write_json, write_jsonl
from evaluation.reference_constraints import (
    reference_fields_from_record,
    reference_violation_labels,
    resolve_reference_legal_actions,
)
from evaluation.verifier_metrics import verifier_metrics
from evaluation.violation_injection import INJECTORS, inject_violation
from state_grounded_qa.fallback import ConservativeFallback
from state_grounded_qa.pipeline import PipelineResult
from state_grounded_qa.progress import TaskGraph
from state_grounded_qa.schemas import EvidenceRecord, RuntimeState
from state_grounded_qa.serialization import parse_structured_output, serialize_input
from state_grounded_qa.verifier import verify_response


class _SequenceGenerator:
    """Return supplied candidates in order, allowing Pipeline to own control flow."""

    def __init__(
        self,
        candidates: Sequence[Any],
        retry_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._candidates = [copy.deepcopy(candidate) for candidate in candidates]
        self._retry_provider = retry_provider
        self.feedback: list[Sequence[str] | None] = []

    def generate(self, serialized_input: str, verifier_feedback: Sequence[str] | None = None) -> Any:
        self.feedback.append(verifier_feedback)
        if self._candidates:
            return self._candidates.pop(0)
        if self._retry_provider is not None:
            provider = self._retry_provider
            self._retry_provider = None
            return provider()
        raise RuntimeError("The complete evaluation needs a candidate for every pipeline attempt")


def _independent_valid(row: Mapping[str, Any]) -> bool | None:
    value = row.get("independent_valid", row.get("reference_valid"))
    return value if isinstance(value, bool) else None


def _response_field(response: Any, field: str) -> Any:
    """Read a field from a JSON response or a schema object."""

    if isinstance(response, Mapping):
        return response.get(field)
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value.get(field) if isinstance(value, Mapping) else None
    return getattr(response, field, None)


def _reference_legal_actions(row: Mapping[str, Any]) -> list[str]:
    """Read legal actions from the independent reference side of a record."""

    value, _ = resolve_reference_legal_actions(row)
    return value or []


def _reference_legal_actions_evaluable(row: Mapping[str, Any]) -> bool:
    _, evaluable = resolve_reference_legal_actions(row)
    return evaluable


def _outcome(
    row: Mapping[str, Any],
    injected_label: str | None,
    result: Any,
    state: RuntimeState,
    evidence: Sequence[EvidenceRecord],
    graph: TaskGraph | None,
    independent_valid: bool | None,
    final_reference_labels: Sequence[str],
    final_reference_evaluable: bool,
    evaluation_track: str,
    initial_reference_labels: Sequence[str] = (),
    initial_reference_valid: bool | None = None,
    initial_reference_evaluable: bool = False,
    initial_response: Any | None = None,
    dataset: str | None = None,
    retry_verifier_accepted: bool | None = None,
    retry_reference_valid: bool | None = None,
    final_reference_metadata: Mapping[str, Any] | None = None,
    initial_reference_metadata: Mapping[str, Any] | None = None,
    retry_reference_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    initial = result.initial_verification or result.verification
    retried = result.retry_count > 0
    final_reference_valid = (
        None
        if not final_reference_evaluable
        else not bool(final_reference_labels)
    )
    retry_verifier_accepted = (
        retried and not result.used_fallback
        if retry_verifier_accepted is None
        else retry_verifier_accepted
    )
    table_name = {
        "table10_detection": "Table 10",
        "table10_controlled_full": "Table 10",
        "natural_errors": "Table 11",
        "table11_natural_errors": "Table 11",
        "table12_full_request": "Table 12",
    }.get(evaluation_track)
    dataset_name = dataset or row.get("dataset") or row.get("dataset_name") or row.get("source_dataset")
    if dataset_name is None:
        dataset_name = "unknown"
    initial_response = row.get("response") if initial_response is None else initial_response
    valid_non_fallback = (
        bool(final_reference_valid and not result.used_fallback)
        if final_reference_evaluable
        else None
    )
    return {
        "request_id": row.get("request_id"),
        "dataset": dataset_name,
        "injected_label": injected_label,
        "initial_labels": initial.failure_labels,
        "final_labels": result.verification.failure_labels,
        "initial_reference_labels": list(initial_reference_labels),
        "initial_reference_valid": initial_reference_valid,
        "initial_reference_evaluable": initial_reference_evaluable,
        "final_reference_labels": list(final_reference_labels),
        "final_reference_valid": final_reference_valid,
        "final_reference_evaluable": final_reference_evaluable,
        "supplied_final_reference_valid": (final_reference_metadata or {}).get("valid"),
        "supplied_final_reference_labels": (final_reference_metadata or {}).get("labels"),
        "final_reference_agreement": (final_reference_metadata or {}).get("agreement"),
        "supplied_initial_reference_valid": (initial_reference_metadata or {}).get("valid"),
        "supplied_initial_reference_labels": (initial_reference_metadata or {}).get("labels"),
        "initial_reference_agreement": (initial_reference_metadata or {}).get("agreement"),
        "supplied_retry_reference_valid": (retry_reference_metadata or {}).get("valid"),
        "supplied_retry_reference_labels": (retry_reference_metadata or {}).get("labels"),
        "retry_reference_agreement": (retry_reference_metadata or {}).get("agreement"),
        "initial_accepted": initial.accepted,
        "initial_verifier_accepted": initial.accepted,
        "initial_verifier_labels": list(initial.failure_labels),
        "final_accepted": result.verification.accepted,
        "final_verifier_accepted": result.verification.accepted,
        "initial_next_action_id": _response_field(initial_response, "next_action_id"),
        "final_next_action_id": _response_field(result.response, "next_action_id"),
        "reference_legal_actions": _reference_legal_actions(row),
        "reference_legal_actions_evaluable": _reference_legal_actions_evaluable(row),
        "retried": retried,
        "retry_verifier_accepted": retry_verifier_accepted if retried else None,
        "retry_reference_valid": retry_reference_valid if retried and retry_verifier_accepted else None,
        "retry_recovered": retried and not result.used_fallback,
        "used_fallback": result.used_fallback,
        "valid_non_fallback": valid_non_fallback,
        "independent_valid": independent_valid,
        "evaluation_track": evaluation_track,
        "evaluation_table": table_name,
        "run_id": row.get("run_id", row.get("run_identifier")),
        "evaluation_stage": "final_verified",
    }


def _reference_context(
    row: Mapping[str, Any],
) -> tuple[
    RuntimeState | None,
    list[EvidenceRecord] | None,
    TaskGraph | None,
    list[str] | None,
    bool,
    dict[str, Any],
]:
    """Build only the independent reference context.

    In particular, an online ``runtime_state`` is never substituted for a
    missing reference state.  Missing reference dimensions remain
    unevaluable and are reported as such by the offline checker.
    """

    state_value = row.get("reference_runtime_state")
    state = RuntimeState.from_dict(state_value) if isinstance(state_value, Mapping) else None
    evidence_value = row.get("reference_evidence")
    evidence: list[EvidenceRecord] | None = None
    if isinstance(evidence_value, Sequence) and not isinstance(evidence_value, (str, bytes)):
        evidence = [
            item if isinstance(item, EvidenceRecord) else EvidenceRecord(**item)
            for item in evidence_value
            if isinstance(item, (EvidenceRecord, Mapping))
        ]
    graph_value = row.get("reference_task_graph", row.get("reference_task_constraints"))
    if isinstance(graph_value, Mapping) and isinstance(graph_value.get("task_graph"), Mapping):
        graph_value = graph_value["task_graph"]
    graph = TaskGraph.from_dict(graph_value) if graph_value else None
    legal_actions, legal_actions_evaluable = resolve_reference_legal_actions(row)
    return (
        state,
        evidence,
        graph,
        legal_actions,
        legal_actions_evaluable,
        reference_fields_from_record(row),
    )


def _supplied_reference_result(
    row: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    aliases = {
        "initial": ("initial_reference_valid", "reference_initial_valid"),
        "retry": ("retry_reference_valid", "reference_retry_valid"),
        "final": ("final_reference_valid", "reference_final_valid"),
    }[stage]
    label_aliases = {
        "initial": ("initial_reference_labels", "reference_initial_labels"),
        "retry": ("retry_reference_labels", "reference_retry_labels"),
        "final": ("final_reference_labels", "reference_final_labels"),
    }[stage]
    valid = next((row[key] for key in aliases if isinstance(row.get(key), bool)), None)
    labels_value = next(
        (row[key] for key in label_aliases if isinstance(row.get(key), (list, tuple, set))),
        None,
    )
    labels = list(labels_value) if labels_value is not None else None
    return {
        "present": valid is not None or labels is not None,
        "valid": valid,
        "labels": labels,
    }


def _reference_agreement(
    supplied: Mapping[str, Any], actual_labels: Sequence[str], actual_evaluable: bool
) -> bool | None:
    if not supplied.get("present") or not actual_evaluable:
        return None
    actual_valid = not bool(actual_labels)
    if isinstance(supplied.get("valid"), bool) and supplied["valid"] != actual_valid:
        return False
    if supplied.get("labels") is not None and set(supplied["labels"]) != set(actual_labels):
        return False
    return True


def _run_reference_check(
    row: Mapping[str, Any], response: Any, stage: str
) -> tuple[list[str], bool, dict[str, Any]]:
    supplied = _supplied_reference_result(row, stage)
    try:
        state, evidence, graph, legal_actions, legal_evaluable, fields = _reference_context(row)
        labels, evaluable = reference_violation_labels(
            response,
            state,
            evidence,
            graph,
            reference_legal_actions=legal_actions,
            reference_fields=fields,
        )
        # A missing/malformed action set is an unevaluable reference dimension;
        # keep any labels found in the dimensions that were available.
        evaluable = evaluable and legal_evaluable
    except (KeyError, TypeError, ValueError):
        labels, evaluable = [], False
    supplied["agreement"] = _reference_agreement(supplied, labels, evaluable)
    return labels, evaluable, supplied


def _initial_reference_check_with_metadata(
    row: Mapping[str, Any], response: Any
) -> tuple[list[str], bool | None, bool, dict[str, Any]]:
    """Return independent initial labels, validity, and evaluability."""
    labels, evaluable, metadata = _run_reference_check(row, response, "initial")
    return labels, (not labels if evaluable else None), evaluable, metadata


def _initial_reference_check(
    row: Mapping[str, Any], response: Any
) -> tuple[list[str], bool | None, bool]:
    """Compatibility wrapper for the independent initial reference check."""

    labels, valid, evaluable, _ = _initial_reference_check_with_metadata(row, response)
    return labels, valid, evaluable


def _final_reference_check_with_metadata(
    row: Mapping[str, Any], response: Any
) -> tuple[list[str], bool, dict[str, Any]]:
    """Re-check the actual final response against independent constraints."""

    return _run_reference_check(row, response, "final")


def _final_reference_check(
    row: Mapping[str, Any], response: Any
) -> tuple[list[str], bool]:
    """Compatibility wrapper returning labels and evaluability."""

    labels, evaluable, _ = _final_reference_check_with_metadata(row, response)
    return labels, evaluable


def _retry_reference_check_with_metadata(
    row: Mapping[str, Any], response: Any
) -> tuple[bool | None, bool, dict[str, Any]]:
    """Check a retry independently, without reusing final-response labels."""
    labels, evaluable, metadata = _run_reference_check(row, response, "retry")
    return (not labels if evaluable else None), evaluable, metadata


def _retry_reference_check(
    row: Mapping[str, Any], response: Any
) -> tuple[bool | None, bool]:
    """Compatibility wrapper for the independent retry reference check."""

    valid, evaluable, _ = _retry_reference_check_with_metadata(row, response)
    return valid, evaluable


def _run_pipeline(
    row: Mapping[str, Any],
    initial_candidate: Any,
    retry_candidate: Any | None = None,
    retry_provider: Callable[[], Any] | None = None,
) -> tuple[Any, RuntimeState, list[EvidenceRecord], TaskGraph | None]:
    """Run the existing verification/retry/fallback control flow with fixed S_t/E_t.

    The normal online pipeline owns retrieval.  Offline evaluation already
    receives the selected evidence from the generation run, so it serializes
    and verifies that exact ordered collection without filtering, embedding,
    ranking, truncating, or re-selecting it.
    """

    state = RuntimeState.from_dict(row["runtime_state"])
    evidence_value = (
        row["retrieved_evidence"]
        if "retrieved_evidence" in row
        else row.get("evidence", [])
    )
    if evidence_value is None:
        evidence_value = []
    if not isinstance(evidence_value, Sequence) or isinstance(evidence_value, (str, bytes)):
        raise TypeError("retrieved_evidence/evidence must be an ordered sequence")
    evidence = [
        item if isinstance(item, EvidenceRecord) else EvidenceRecord(**dict(item))
        for item in evidence_value
    ]
    evidence_ids = [item.evidence_id for item in evidence]
    graph = TaskGraph.from_dict(row["task_graph"]) if row.get("task_graph") else None
    candidates = [initial_candidate]
    if retry_candidate is not None:
        candidates.append(retry_candidate)
    generator = _SequenceGenerator(candidates, retry_provider=retry_provider)
    request = row.get("request")
    request_question = request.get("question", "") if isinstance(request, Mapping) else ""
    question = str(row.get("question", request_question))
    state_before = copy.deepcopy(state.to_dict())
    serialized = serialize_input(question, state, evidence)
    first = generator.generate(serialized, verifier_feedback=None)
    first_verification = verify_response(first, state, evidence, graph)
    if first_verification.accepted:
        response = parse_structured_output(first)
        result = PipelineResult(
            response, first_verification, evidence, 0, False, first_verification
        )
    else:
        second = generator.generate(serialized, verifier_feedback=first_verification.failure_labels)
        second_verification = verify_response(second, state, evidence, graph)
        if second_verification.accepted:
            response = parse_structured_output(second)
            result = PipelineResult(
                response, second_verification, evidence, 1, False, first_verification
            )
        else:
            fallback = ConservativeFallback(state)
            fallback_verification = verify_response(fallback, state, evidence, graph)
            result = PipelineResult(
                fallback, fallback_verification, evidence, 1, True, first_verification
            )
    if state.to_dict() != state_before:
        raise RuntimeError("Offline evaluation modified authoritative runtime state")
    if [item.evidence_id for item in result.retrieved_evidence] != evidence_ids:
        raise RuntimeError("Offline evaluation changed the selected evidence order or IDs")
    return result, state, evidence, graph


def _retry_candidate(row: Mapping[str, Any], label: str) -> Any:
    candidates = row.get("retry_responses")
    if isinstance(candidates, Mapping) and label in candidates:
        return candidates[label]
    if label == "full" and isinstance(candidates, Mapping):
        for alias in ("clean", "normal", "default"):
            if alias in candidates:
                return candidates[alias]
    if "retry_response" in row:
        return row["retry_response"]
    raise ValueError(
        f"Request {row.get('request_id')!r} requires a retry candidate for {label!r}: "
        "use row['retry_responses'][label] or row['retry_response']"
    )


def _run_with_retry_if_needed(
    row: Mapping[str, Any], initial_candidate: Any, retry_key: str
) -> tuple[Any, RuntimeState, list[EvidenceRecord], TaskGraph | None]:
    """Only request a retry candidate when the initial verifier rejects."""

    return _run_pipeline(
        row,
        initial_candidate,
        retry_provider=lambda: _retry_candidate(row, retry_key),
    )


def _evaluate_controlled(rows: list[dict]) -> list[dict]:
    """Table 10 detection: one direct verifier pass per injected candidate."""

    results = []
    for row in rows:
        state = RuntimeState.from_dict(row["runtime_state"])
        evidence = [EvidenceRecord(**item) for item in row.get("evidence", [])]
        graph = TaskGraph.from_dict(row["task_graph"]) if row.get("task_graph") else None
        clean = row["response"]
        clean_result = verify_response(clean, state, evidence, graph)
        if not clean_result.accepted:
            raise ValueError("Every source response must pass verification before perturbation")
        results.append(
            {
                "request_id": row.get("request_id"),
                "injected_label": None,
                "initial_labels": [],
                "final_labels": [],
                "initial_accepted": True,
                "final_accepted": True,
                "retried": False,
                "retry_recovered": False,
                "used_fallback": False,
                "independent_valid": _independent_valid(row),
                "final_reference_evaluable": False,
                "evaluation_track": "table10_detection",
            }
        )
        for label in INJECTORS:
            perturbed = inject_violation(label, clean, {"state": row["runtime_state"]})
            verification = verify_response(perturbed, state, evidence, graph)
            results.append(
                {
                    "request_id": row.get("request_id"),
                    "injected_label": label,
                    "initial_labels": verification.failure_labels,
                    "final_labels": verification.failure_labels,
                    "initial_accepted": verification.accepted,
                    "final_accepted": False,
                    "retried": False,
                    "retry_recovered": False,
                    "used_fallback": False,
                    "independent_valid": False,
                    "final_reference_evaluable": False,
                    "evaluation_track": "table10_detection",
                }
            )
    return results


def _evaluate_controlled_full(rows: list[dict]) -> list[dict]:
    """Table 10 residuals: run the complete flow for injected candidates."""

    results = []
    for row in rows:
        state = RuntimeState.from_dict(row["runtime_state"])
        evidence = [EvidenceRecord(**item) for item in row.get("evidence", [])]
        graph = TaskGraph.from_dict(row["task_graph"]) if row.get("task_graph") else None
        clean = row["response"]
        clean_result = verify_response(clean, state, evidence, graph)
        if not clean_result.accepted:
            raise ValueError("Every source response must pass verification before controlled perturbation")
        for label in INJECTORS:
            perturbed = inject_violation(label, clean, {"state": row["runtime_state"]})
            result, _, _, _ = _run_with_retry_if_needed(row, perturbed, label)
            reference_labels, reference_evaluable, reference_metadata = _final_reference_check_with_metadata(
                row, result.response
            )
            results.append(
                _outcome(
                    row,
                    label,
                    result,
                    state,
                    evidence,
                    graph,
                    False,
                    reference_labels,
                    reference_evaluable,
                    "table10_controlled_full",
                    final_reference_metadata=reference_metadata,
                )
            )
    return results


def _evaluate_full(rows: list[dict], dataset: str | None = None) -> list[dict]:
    """Table 12: full-request verification, retry, and fallback evaluation."""

    results = []
    for row in rows:
        state = RuntimeState.from_dict(row["runtime_state"])
        evidence = [EvidenceRecord(**item) for item in row.get("evidence", [])]
        graph = TaskGraph.from_dict(row["task_graph"]) if row.get("task_graph") else None
        result, _, _, _ = _run_with_retry_if_needed(row, row["response"], "full")
        reference_labels, reference_evaluable, reference_metadata = _final_reference_check_with_metadata(
            row, result.response
        )
        results.append(
            _outcome(
                row,
                None,
                result,
                state,
                evidence,
                graph,
                _independent_valid(row),
                reference_labels,
                reference_evaluable,
                "table12_full_request",
                initial_response=row["response"],
                dataset=dataset,
                final_reference_metadata=reference_metadata,
            )
        )
    return results


def _evaluate_natural(rows: list[dict], dataset: str | None = None) -> list[dict]:
    """Table 11: verification of naturally generated candidates."""

    results = []
    for row in rows:
        candidate = row.get("response", row.get("initial_candidate"))
        if candidate is None:
            raise ValueError("Natural evaluation requires row['response'] or row['initial_candidate']")
        initial_labels, initial_valid, initial_evaluable, initial_metadata = _initial_reference_check_with_metadata(
            row, candidate
        )
        result, state, evidence, graph = _run_with_retry_if_needed(row, candidate, "full")
        final_labels, final_evaluable, final_metadata = _final_reference_check_with_metadata(
            row, result.response
        )
        retry_verifier_accepted = result.retry_count > 0 and not result.used_fallback
        retry_reference_valid = None
        retry_reference_metadata: dict[str, Any] | None = None
        if retry_verifier_accepted:
            retry_candidate = _retry_candidate(row, "full")
            retry_reference_valid, _, retry_reference_metadata = _retry_reference_check_with_metadata(
                row, retry_candidate
            )
        results.append(
            _outcome(
                row,
                None,
                result,
                state,
                evidence,
                graph,
                initial_valid,
                final_labels,
                final_evaluable,
                "table11_natural_errors",
                initial_reference_labels=initial_labels,
                initial_reference_valid=initial_valid if initial_evaluable else None,
                initial_reference_evaluable=initial_evaluable,
                initial_response=candidate,
                dataset=dataset,
                retry_verifier_accepted=retry_verifier_accepted,
                retry_reference_valid=retry_reference_valid,
                final_reference_metadata=final_metadata,
                initial_reference_metadata=initial_metadata,
                retry_reference_metadata=retry_reference_metadata,
            )
        )
    return results


def evaluate(
    rows: list[dict], procedure: str = "controlled", dataset: str | None = None
) -> list[dict]:
    """Evaluate one independent Table 10, Table 11, or Table 12 procedure."""

    if procedure == "controlled":
        return _evaluate_controlled(rows)
    if procedure == "controlled-full":
        return _evaluate_controlled_full(rows)
    if procedure == "full":
        return _evaluate_full(rows, dataset=dataset)
    if procedure == "natural":
        return _evaluate_natural(rows, dataset=dataset)
    raise ValueError(f"Unknown verifier evaluation procedure: {procedure}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL of candidate requests with state and E_t")
    parser.add_argument("--output", required=True, help="Metric JSON")
    parser.add_argument("--evaluation-set", help="Optional per-request outcome JSONL")
    parser.add_argument(
        "--dataset",
        help="Optional dataset label copied to request-level outcomes (e.g. EAI-VirtualHome or UE5)",
    )
    parser.add_argument(
        "--procedure",
        choices=("controlled", "controlled-full", "full", "natural"),
        default="controlled",
        help="Table 10 controlled flow, Table 11 natural candidates, or Table 12 full-request recovery",
    )
    parser.add_argument(
        "--input-mode",
        choices=("accepted", "outcomes"),
        default="accepted",
        help="Inject violations from accepted candidates, or aggregate supplied verifier outcomes",
    )
    args = parser.parse_args()
    source_rows = read_jsonl(args.input)
    if args.input_mode == "outcomes" and args.procedure != "controlled":
        raise ValueError("--input-mode outcomes cannot execute a pipeline procedure")
    results = (
        evaluate(source_rows, procedure=args.procedure, dataset=args.dataset)
        if args.input_mode == "accepted"
        else source_rows
    )
    if args.evaluation_set:
        write_jsonl(args.evaluation_set, results)
    write_json(args.output, verifier_metrics(results))


if __name__ == "__main__":
    main()
