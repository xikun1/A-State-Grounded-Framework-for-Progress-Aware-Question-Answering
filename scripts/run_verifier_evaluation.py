"""Create controlled violations from valid candidates and evaluate the verifier."""

from __future__ import annotations

import argparse

from _common import read_jsonl, write_json, write_jsonl
from evaluation.verifier_metrics import verifier_metrics
from evaluation.violation_injection import INJECTORS, inject_violation
from state_grounded_qa.progress import TaskGraph
from state_grounded_qa.schemas import EvidenceRecord, RuntimeState
from state_grounded_qa.verifier import verify_response


def evaluate(rows: list[dict]) -> list[dict]:
    """Evaluate clean inputs and one transformation per configured violation type."""

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
                "used_fallback": False,
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
                    "used_fallback": False,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL of accepted responses with state and E_t")
    parser.add_argument("--output", required=True, help="Metric JSON")
    parser.add_argument("--evaluation-set", help="Optional controlled-candidate JSONL")
    parser.add_argument(
        "--input-mode",
        choices=("accepted", "outcomes"),
        default="accepted",
        help="Inject violations from accepted candidates, or aggregate supplied verifier outcomes",
    )
    args = parser.parse_args()
    source_rows = read_jsonl(args.input)
    results = evaluate(source_rows) if args.input_mode == "accepted" else source_rows
    if args.evaluation_set:
        write_jsonl(args.evaluation_set, results)
    write_json(args.output, verifier_metrics(results))


if __name__ == "__main__":
    main()
