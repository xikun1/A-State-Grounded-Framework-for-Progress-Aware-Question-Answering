"""Run the Table 11 natural-error evaluation for EAI-VirtualHome and UE5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from _common import read_jsonl, write_json
from evaluation.verifier_metrics import verifier_metrics
from run_verifier_evaluation import evaluate


SUMMARY_FIELDS = (
    "natural_violation_rate_before_verify",
    "natural_detection_recall",
    "natural_false_rejection_rate",
    "natural_retry_rate",
    "natural_retry_recovery_rate",
    "natural_fallback_rate",
    "natural_residual_violation_rate",
    "natural_valid_non_fallback_coverage",
)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_records_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    rows = list(records)
    preferred = (
        "request_id",
        "dataset",
        "evaluation_track",
        "run_id",
        "evaluation_stage",
        "initial_reference_valid",
        "initial_reference_labels",
        "initial_reference_evaluable",
        "initial_verifier_accepted",
        "initial_verifier_labels",
        "retried",
        "retry_verifier_accepted",
        "retry_reference_valid",
        "retry_reference_agreement",
        "used_fallback",
        "final_verifier_accepted",
        "final_reference_valid",
        "final_reference_labels",
        "final_reference_evaluable",
        "supplied_final_reference_valid",
        "supplied_final_reference_labels",
        "final_reference_agreement",
        "initial_next_action_id",
        "final_next_action_id",
        "reference_legal_actions",
        "reference_legal_actions_evaluable",
        "valid_non_fallback",
    )
    keys = list(preferred)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _summary_row(dataset: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "evaluation_track": metrics.get("evaluation_track"),
        "n_requests": metrics.get("n_requests", metrics.get("n_candidates")),
    }
    for name in SUMMARY_FIELDS:
        row[name] = metrics.get(name)
        row[f"{name}_numerator"] = metrics.get(f"{name}_numerator")
        row[f"{name}_denominator"] = metrics.get(f"{name}_denominator")
        row[f"{name}_percent"] = metrics.get(f"{name}_percent")
    row.update(
        {
            "n_initial_reference_invalid": metrics.get("n_initial_reference_invalid"),
            "n_initial_reference_valid": metrics.get("n_initial_reference_valid"),
            "n_initial_reference_evaluable": metrics.get("n_initial_reference_evaluable"),
            "n_initial_reference_unevaluable": metrics.get("n_initial_reference_unevaluable"),
            "n_initial_invalid_rejected": metrics.get("n_initial_invalid_rejected"),
            "n_initial_valid_rejected": metrics.get("n_initial_valid_rejected"),
            "n_retried": metrics.get("n_retried"),
            "n_retry_verifier_accepted": metrics.get("n_retry_verifier_accepted"),
            "n_fallback": metrics.get("n_fallback"),
            "n_final_reference_invalid": metrics.get("n_final_reference_invalid"),
            "n_final_reference_valid": metrics.get("n_final_reference_valid"),
            "n_final_reference_evaluable": metrics.get("n_final_reference_evaluable"),
            "n_final_reference_unevaluable": metrics.get("n_final_reference_unevaluable"),
            "n_valid_non_fallback": metrics.get("n_valid_non_fallback"),
            "reference_evaluation_complete": metrics.get("reference_evaluation_complete"),
        }
    )
    return row


def run(eai_input: str | Path, ue5_input: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Evaluate both independent request tracks and write the four Table 11 files."""

    eai_rows = evaluate(read_jsonl(eai_input), procedure="natural", dataset="EAI-VirtualHome")
    ue5_rows = evaluate(read_jsonl(ue5_input), procedure="natural", dataset="UE5")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_records_csv(output / "table11_eai_records.csv", eai_rows)
    _write_records_csv(output / "table11_ue5_records.csv", ue5_rows)

    eai_metrics = verifier_metrics(eai_rows)
    ue5_metrics = verifier_metrics(ue5_rows)
    all_metrics = verifier_metrics(eai_rows + ue5_rows)
    payload = {
        "evaluation_track": "table11_natural_errors",
        "datasets": {
            "EAI-VirtualHome": eai_metrics,
            "UE5": ue5_metrics,
        },
        "overall": all_metrics,
        "n_requests": len(eai_rows) + len(ue5_rows),
    }
    write_json(output / "table11_metrics.json", payload)

    summary_rows = [
        _summary_row("EAI-VirtualHome", eai_metrics),
        _summary_row("UE5", ue5_metrics),
        _summary_row("overall", all_metrics),
    ]
    summary_keys = list(summary_rows[0])
    with (output / "table11_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_keys)
        writer.writeheader()
        writer.writerows(summary_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eai-input", required=True, help="EAI-VirtualHome natural candidates JSONL")
    parser.add_argument("--ue5-input", required=True, help="UE5 natural candidates JSONL")
    parser.add_argument("--output-dir", required=True, help="External directory for Table 11 outputs")
    args = parser.parse_args()
    run(args.eai_input, args.ue5_input, args.output_dir)


if __name__ == "__main__":
    main()
