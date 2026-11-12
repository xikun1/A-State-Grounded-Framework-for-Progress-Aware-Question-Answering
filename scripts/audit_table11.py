"""Audit Table 11 records against independent action references and main-track outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from _common import read_jsonl, write_json
from evaluation.reference_constraints import resolve_reference_legal_actions


RecordIdentity = tuple[str, str | None, str | None, str | None]


class DuplicateRecordIdentityError(ValueError):
    """Raised when an audit input contains an exact duplicate record identity."""

    def __init__(self, source: str, details: list[dict[str, Any]]) -> None:
        self.source = source
        self.details = details
        super().__init__(
            f"Duplicate record identities in {source}: "
            + ", ".join(str(item["identity"]) for item in details)
        )


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "true":
            return True
        if stripped.lower() == "false":
            return False
        if stripped.lower() in {"", "none", "null"}:
            return None
        if stripped.startswith(("[", "{", '"')):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as stream:
            return [
                {key: _decode(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_jsonl(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("records", "outcomes", "rows"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
    return []


def _actions(value: Any) -> set[str]:
    value = _decode(value)
    return set(value) if isinstance(value, (list, tuple, set)) else set()


def _extract_action(row: Mapping[str, Any], stage: str = "final_verified") -> Any:
    """Extract a stage-specific action without changing source field names."""

    candidates: list[tuple[Any, str]] = []
    if stage == "initial_candidate":
        candidates.append((row, "initial_next_action_id"))
        for key in ("initial_candidate", "initial"):
            if isinstance(row.get(key), Mapping):
                candidates.append((row[key], "next_action_id"))
    if stage == "final_verified":
        candidates.append((row, "final_next_action_id"))
        if isinstance(row.get("final_verified"), Mapping):
            candidates.append((row["final_verified"], "next_action_id"))
    candidates.extend((row, key) for key in ("next_action_id", "final_next_action_id"))
    for source, key in candidates:
        if isinstance(source, Mapping) and key in source:
            value = _decode(source[key])
            return None if value in (None, "", "null") else value
    for key in ("prediction", "response"):
        source = row.get(key)
        if isinstance(source, Mapping) and "next_action_id" in source:
            value = _decode(source["next_action_id"])
            return None if value in (None, "", "null") else value
    return None


def _extract_reference_actions(row: Mapping[str, Any]) -> tuple[set[str], bool]:
    value, evaluable = resolve_reference_legal_actions(row)
    return set(value or []), evaluable


def _row_stage(row: Mapping[str, Any], default: str = "final_verified") -> str:
    return str(row.get("evaluation_stage") or default)


def _record_stage(row: Mapping[str, Any]) -> str | None:
    value = row.get("evaluation_stage")
    return str(value) if value not in (None, "") else None


def _row_run_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("run_id", row.get("run_identifier"))
    return str(value) if value not in (None, "") else None


def _row_dataset(row: Mapping[str, Any], default: str) -> str:
    return str(row.get("dataset") or row.get("dataset_name") or default)


def _record_identity(
    row: Mapping[str, Any], default_dataset: str
) -> RecordIdentity:
    """Return the complete identity, retaining missing metadata as ``None``."""

    request_id = row.get("request_id")
    return (
        _row_dataset(row, default_dataset),
        str(request_id) if request_id is not None else None,
        _row_run_id(row),
        _record_stage(row),
    )


def _identity_dict(identity: RecordIdentity) -> dict[str, Any]:
    return {
        "dataset": identity[0],
        "request_id": identity[1],
        "run_id": identity[2],
        "evaluation_stage": identity[3],
    }


def _index_records(
    rows: list[Mapping[str, Any]],
    dataset: str,
    source: str = "records",
) -> dict[RecordIdentity, Mapping[str, Any]]:
    """Index all records by complete identity and reject exact duplicates."""

    index: dict[RecordIdentity, Mapping[str, Any]] = {}
    positions: dict[RecordIdentity, list[int]] = {}
    for position, row in enumerate(rows):
        if _row_dataset(row, dataset) != dataset:
            continue
        identity = _record_identity(row, dataset)
        positions.setdefault(identity, []).append(position)
        index[identity] = row
    details = [
        {
            "source": source,
            "identity": _identity_dict(identity),
            "count": len(record_positions),
            "positions": record_positions,
        }
        for identity, record_positions in positions.items()
        if len(record_positions) > 1
    ]
    if details:
        raise DuplicateRecordIdentityError(source, details)
    return index


def _action_summary(
    rows: list[Mapping[str, Any]],
    action_field: str | None = None,
    stage: str = "final_verified",
) -> dict[str, Any]:
    non_null = 0
    valid = 0
    illegal_ids: list[str] = []
    illegal_details: list[dict[str, Any]] = []
    reference_unevaluable = 0
    for row in rows:
        row_stage = _row_stage(row, stage)
        action = row.get(action_field) if action_field else _extract_action(row, row_stage)
        legal, legal_evaluable = _extract_reference_actions(row)
        if row.get("reference_legal_actions_evaluable") is False or not legal_evaluable:
            reference_unevaluable += 1
            continue
        if action is None or action == "" or action == "null":
            continue
        non_null += 1
        if action in legal:
            valid += 1
        else:
            request_id = str(row.get("request_id"))
            illegal_ids.append(request_id)
            illegal_details.append(
                {
                    "request_id": request_id,
                    "suggested_action": action,
                    "reference_legal_actions": sorted(legal),
                    "final_reference_valid": row.get("final_reference_valid"),
                    "final_verifier_accepted": row.get("final_verifier_accepted"),
                    "initial_verifier_accepted": row.get("initial_verifier_accepted"),
                }
            )
    return {
        "action_field": action_field or f"{stage} (normalized)",
        "n_reference_action_unevaluable": reference_unevaluable,
        "n_non_null_actions": non_null,
        "n_valid_actions": valid,
        "n_illegal_actions": non_null - valid,
        "action_validity_rate": valid / non_null if non_null else None,
        "illegal_action_rate": (non_null - valid) / non_null if non_null else None,
        "illegal_request_ids": illegal_ids,
        "illegal_action_details": illegal_details,
    }


def _same_request_candidates(
    index: Mapping[RecordIdentity, Mapping[str, Any]], identity: RecordIdentity
) -> list[tuple[RecordIdentity, Mapping[str, Any]]]:
    return [
        (candidate_identity, row)
        for candidate_identity, row in index.items()
        if candidate_identity[0] == identity[0]
        and candidate_identity[1] == identity[1]
    ]


def _pair_status(left: RecordIdentity, right: RecordIdentity) -> str:
    if left[2] is None or right[2] is None or left[3] is None or right[3] is None:
        return "metadata_unconfirmed"
    if left[2] != right[2]:
        return "different_run"
    if left[3] != right[3]:
        return "different_stage"
    return "matched"


def _best_comparison_status(
    identity: RecordIdentity,
    candidates: list[tuple[RecordIdentity, Mapping[str, Any]]],
) -> str:
    statuses = [_pair_status(identity, candidate_identity) for candidate_identity, _ in candidates]
    if "matched" in statuses:
        return "matched"
    if "different_run" in statuses:
        return "different_run"
    if "different_stage" in statuses:
        return "different_stage"
    return "metadata_unconfirmed"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _compare_main_and_natural_tracks(
    table11_rows: list[Mapping[str, Any]],
    main_rows: list[Mapping[str, Any]],
    dataset: str,
    main_track: str | None = None,
) -> dict[str, Any]:
    """Compare only records with a complete, identical execution identity."""

    table11_index = _index_records(table11_rows, dataset, "Table 11")
    main_index = _index_records(main_rows, dataset, main_track or "main experiment")
    matched_identities = sorted(
        identity
        for identity in table11_index.keys() & main_index.keys()
        if _pair_status(identity, identity) == "matched"
    )
    matched = [(identity, table11_index[identity], main_index[identity]) for identity in matched_identities]

    status_details: list[dict[str, Any]] = []
    for identity in table11_index:
        if identity in matched_identities:
            continue
        candidates = _same_request_candidates(main_index, identity)
        if candidates:
            status = _best_comparison_status(identity, candidates)
            candidate_identity = candidates[0][0]
            status_details.append(
                {
                    **_identity_dict(identity),
                    "table11_run_id": identity[2],
                    "main_run_id": candidate_identity[2],
                    "table11_evaluation_stage": identity[3],
                    "main_evaluation_stage": candidate_identity[3],
                    "comparison_status": status,
                }
            )
    for identity in main_index:
        if identity in matched_identities:
            continue
        candidates = _same_request_candidates(table11_index, identity)
        if candidates:
            candidate_identity = candidates[0][0]
            if not any(
                item["request_id"] == identity[1]
                and item["main_run_id"] == identity[2]
                and item["main_evaluation_stage"] == identity[3]
                for item in status_details
            ):
                status_details.append(
                    {
                        **_identity_dict(identity),
                        "table11_run_id": candidate_identity[2],
                        "main_run_id": identity[2],
                        "table11_evaluation_stage": candidate_identity[3],
                        "main_evaluation_stage": identity[3],
                        "comparison_status": _best_comparison_status(identity, candidates),
                    }
                )

    table11_request_ids = {identity[1] for identity in table11_index}
    main_request_ids = {identity[1] for identity in main_index}
    table11_only = [identity for identity in table11_index if identity[1] not in main_request_ids]
    main_only = [identity for identity in main_index if identity[1] not in table11_request_ids]
    metadata_unconfirmed = [item for item in status_details if item["comparison_status"] == "metadata_unconfirmed"]
    different_run = [item for item in status_details if item["comparison_status"] == "different_run"]
    different_stage = [item for item in status_details if item["comparison_status"] == "different_stage"]

    different_actions: list[dict[str, Any]] = []
    reference_mismatches: list[dict[str, Any]] = []
    final_valid_mismatches: list[dict[str, Any]] = []
    same_actions = 0
    matched_table11_illegal_ids: set[str] = set()
    matched_residual_ids: set[str] = set()
    for identity, table11, main in matched:
        request_id = str(identity[1])
        table11_action = _extract_action(table11, identity[3] or "final_verified")
        main_action = _extract_action(main, identity[3] or "final_verified")
        if table11_action == main_action:
            same_actions += 1
        else:
            different_actions.append(
                {
                    "dataset": identity[0],
                    "request_id": request_id,
                    "table11_run_id": identity[2],
                    "main_run_id": identity[2],
                    "table11_evaluation_stage": identity[3],
                    "main_evaluation_stage": identity[3],
                    "comparison_status": "matched",
                    "table11_action": table11_action,
                    "main_action": main_action,
                }
            )
        table11_legal, table11_legal_eval = _extract_reference_actions(table11)
        main_legal, main_legal_eval = _extract_reference_actions(main)
        if table11_legal_eval and main_legal_eval and table11_legal != main_legal:
            reference_mismatches.append(
                {
                    "dataset": identity[0],
                    "request_id": request_id,
                    "table11_run_id": identity[2],
                    "main_run_id": identity[2],
                    "table11_evaluation_stage": identity[3],
                    "main_evaluation_stage": identity[3],
                    "comparison_status": "matched",
                    "table11_reference_legal_actions": sorted(table11_legal),
                    "main_reference_legal_actions": sorted(main_legal),
                }
            )
        if (
            "final_reference_valid" in table11
            and "final_reference_valid" in main
            and table11.get("final_reference_valid") != main.get("final_reference_valid")
        ):
            final_valid_mismatches.append(
                {
                    "dataset": identity[0],
                    "request_id": request_id,
                    "table11_final_reference_valid": table11.get("final_reference_valid"),
                    "main_final_reference_valid": main.get("final_reference_valid"),
                    "comparison_status": "matched",
                }
            )
        if table11_legal_eval and table11_action is not None and table11_action not in table11_legal:
            matched_table11_illegal_ids.add(request_id)
        if table11.get("final_reference_valid") is False:
            matched_residual_ids.add(request_id)

    table11_summary = _action_summary(list(table11_index.values()), stage="final_verified")
    main_summary = _action_summary(list(main_index.values()), stage="final_verified")
    outside_residual = sorted(matched_table11_illegal_ids - matched_residual_ids)
    failures = bool(different_actions or reference_mismatches or final_valid_mismatches or outside_residual)
    if failures:
        audit_status = "failed"
    elif different_run or different_stage:
        audit_status = "different_execution"
    elif metadata_unconfirmed or table11_only or main_only or not matched:
        audit_status = "metadata_incomplete"
    else:
        audit_status = "passed"

    return {
        "dataset": dataset,
        "main_track": main_track,
        "audit_status": audit_status,
        "n_table11_requests": len(table11_index),
        "n_main_requests": len(main_index),
        "n_matched_requests": len(matched),
        "n_table11_only_requests": len(table11_only),
        "n_main_only_requests": len(main_only),
        "n_incomparable_run_or_stage_requests": len(status_details),
        "n_metadata_unconfirmed": len(metadata_unconfirmed),
        "n_different_run": len(different_run),
        "n_different_stage": len(different_stage),
        "n_duplicate_records": 0,
        "n_same_final_actions": same_actions,
        "n_different_final_actions": len(different_actions),
        "n_reference_action_set_mismatches": len(reference_mismatches),
        "n_final_reference_valid_mismatches": len(final_valid_mismatches),
        "n_table11_non_null_actions": table11_summary["n_non_null_actions"],
        "n_main_non_null_actions": main_summary["n_non_null_actions"],
        "n_table11_illegal_actions": table11_summary["n_illegal_actions"],
        "n_main_illegal_actions": main_summary["n_illegal_actions"],
        "n_illegal_actions_outside_residual": len(outside_residual),
        "unmatched_request_ids": sorted(
            {str(identity[1]) for identity in table11_only + main_only}
        ),
        "metadata_unconfirmed_request_ids": _unique(
            [str(item["request_id"]) for item in metadata_unconfirmed]
        ),
        "different_run_request_ids": _unique(
            [str(item["request_id"]) for item in different_run]
        ),
        "different_stage_request_ids": _unique(
            [str(item["request_id"]) for item in different_stage]
        ),
        "different_final_action_request_ids": [item["request_id"] for item in different_actions],
        "reference_action_mismatch_request_ids": [item["request_id"] for item in reference_mismatches],
        "illegal_action_outside_residual_request_ids": outside_residual,
        "different_final_actions": different_actions,
        "reference_action_mismatches": reference_mismatches,
        "final_reference_valid_mismatches": final_valid_mismatches,
        "matched_record_identities": [_identity_dict(identity) for identity in matched_identities],
        "comparison_status_details": status_details,
        "table11_reference_source": "independent reference constraints",
        "main_reference_source": "record reference fields",
        "table11_action_validity_rate": table11_summary["action_validity_rate"],
        "main_action_validity_rate": main_summary["action_validity_rate"],
        "table11_illegal_action_rate": table11_summary["illegal_action_rate"],
        "main_illegal_action_rate": main_summary["illegal_action_rate"],
    }


def _audit_records(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    final_invalid_ids = {
        str(row.get("request_id"))
        for row in rows
        if row.get("final_reference_valid") is False
    }
    final_actions = _action_summary(rows, stage="final_verified")
    initial_actions = _action_summary(rows, stage="initial_candidate")
    illegal_ids = set(final_actions["illegal_request_ids"])
    residual_check = illegal_ids <= final_invalid_ids
    return {
        "dataset": dataset,
        "n_records": len(rows),
        "request_ids_unique": len({row.get("request_id") for row in rows}) == len(rows),
        "n_final_reference_invalid": len(final_invalid_ids),
        "n_final_reference_unevaluable": sum(
            row.get("final_reference_evaluable") is False for row in rows
        ),
        "initial_action_metrics": initial_actions,
        "final_action_metrics": final_actions,
        "final_illegal_action_le_final_invalid": residual_check,
        "final_illegal_action_ids_are_residual": residual_check,
        "final_illegal_action_ids_not_residual": sorted(illegal_ids - final_invalid_ids),
        "reference_action_source": "independent reference constraints",
        "audit_status": "passed" if residual_check else "failed",
    }


def _audit_ue5_reference_alignment(
    replay_path: str | Path, labels_path: str | Path
) -> dict[str, Any]:
    replay = {str(row.get("request_id")): row for row in _read_records(replay_path)}
    labels = {str(row.get("request_id")): row for row in _read_records(labels_path)}
    common = sorted(set(replay) & set(labels))
    mismatches = []
    for request_id in common:
        runtime_state = replay[request_id].get("runtime_state", replay[request_id])
        action = runtime_state.get("action", {}) if isinstance(runtime_state, Mapping) else {}
        runtime_actions = _actions(action.get("available_actions")) if isinstance(action, Mapping) else set()
        reference_actions, reference_evaluable = resolve_reference_legal_actions(labels[request_id])
        reference_actions = set(reference_actions or [])
        if not reference_evaluable or runtime_actions != reference_actions:
            mismatches.append(
                {
                    "request_id": request_id,
                    "runtime_available_actions": sorted(runtime_actions),
                    "reference_legal_actions": sorted(reference_actions),
                    "reference_legal_actions_evaluable": reference_evaluable,
                }
            )
    return {
        "replay_records": len(replay),
        "reference_records": len(labels),
        "aligned_records": len(common),
        "action_set_mismatch_count": len(mismatches),
        "action_set_mismatches": mismatches,
    }


def _optional_main_track(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"available": False, "reason": "no main-experiment output supplied"}
    source = Path(path)
    if not source.exists():
        return {"available": False, "path": str(source), "reason": "file not found"}
    rows = _read_records(source)
    return {
        "available": True,
        "path": str(source),
        "n_records": len(rows),
        "request_ids": len({row.get("request_id") for row in rows}),
        "run_ids": sorted({_row_run_id(row) for row in rows if _row_run_id(row)}),
        "evaluation_stages": sorted({_record_stage(row) for row in rows if _record_stage(row)}),
        "action_metrics": _action_summary(rows, stage="final_verified"),
        "records": rows,
    }


def _overall_audit_status(comparisons: Mapping[str, Any]) -> str:
    statuses = [
        result.get("audit_status")
        for dataset in comparisons.values()
        for result in dataset.values()
        if isinstance(result, Mapping) and result.get("audit_status")
    ]
    if "failed" in statuses:
        return "failed"
    if "metadata_incomplete" in statuses:
        return "metadata_incomplete"
    if "different_execution" in statuses:
        return "different_execution"
    return "passed" if statuses else "metadata_incomplete"


def audit(
    eai_records: str | Path,
    ue5_records: str | Path,
    output: str | Path,
    ue5_replay: str | Path | None = None,
    ue5_labels: str | Path | None = None,
    table6_output: str | Path | None = None,
    table14_output: str | Path | None = None,
) -> dict[str, Any]:
    eai_rows = _read_records(eai_records)
    ue5_rows = _read_records(ue5_records)
    table6 = _optional_main_track(table6_output)
    table14 = _optional_main_track(table14_output)
    comparisons: dict[str, dict[str, Any]] = {}
    try:
        for dataset, natural_rows in (("EAI-VirtualHome", eai_rows), ("UE5", ue5_rows)):
            comparisons[dataset] = {
                "table6": (
                    _compare_main_and_natural_tracks(
                        natural_rows, table6["records"], dataset, "table6"
                    )
                    if table6.get("available")
                    else {"available": False, "reason": table6.get("reason", "not supplied")}
                ),
                "table14": (
                    _compare_main_and_natural_tracks(
                        natural_rows, table14["records"], dataset, "table14"
                    )
                    if table14.get("available")
                    else {"available": False, "reason": table14.get("reason", "not supplied")}
                ),
            }
    except DuplicateRecordIdentityError as exc:
        payload = {
            "evaluation_track": "table11_consistency_audit",
            "audit_status": "failed",
            "error": str(exc),
            "duplicate_record_details": exc.details,
            "n_duplicate_records": sum(
                max(0, int(item.get("count", 0)) - 1) for item in exc.details
            ),
        }
        write_json(output, payload)
        raise

    payload: dict[str, Any] = {
        "evaluation_track": "table11_consistency_audit",
        "audit_status": _overall_audit_status(comparisons),
        "table11": {
            "EAI-VirtualHome": _audit_records(eai_rows, "EAI-VirtualHome"),
            "UE5": _audit_records(ue5_rows, "UE5"),
        },
        "main_experiment": {
            "table6": {key: value for key, value in table6.items() if key != "records"},
            "table14": {key: value for key, value in table14.items() if key != "records"},
        },
        "comparisons": comparisons,
    }
    if ue5_replay and ue5_labels:
        payload["ue5_reference_alignment"] = _audit_ue5_reference_alignment(ue5_replay, ue5_labels)
    else:
        payload["ue5_reference_alignment"] = {
            "available": False,
            "reason": "supply --ue5-replay and --ue5-labels for the 480-request alignment check",
        }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eai-records", required=True)
    parser.add_argument("--ue5-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ue5-replay")
    parser.add_argument("--ue5-labels")
    parser.add_argument("--table6-output")
    parser.add_argument("--table14-output")
    args = parser.parse_args()
    audit(
        args.eai_records,
        args.ue5_records,
        args.output,
        ue5_replay=args.ue5_replay,
        ue5_labels=args.ue5_labels,
        table6_output=args.table6_output,
        table14_output=args.table14_output,
    )


if __name__ == "__main__":
    main()
