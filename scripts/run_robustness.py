"""Build stale-state or distractor-evidence evaluation inputs from supplied records."""

from __future__ import annotations

import argparse
import json

from _common import read_jsonl, write_jsonl
from evaluation.distractor_evidence import LEVELS, contaminate_evidence
from evaluation.record_anomalies import CONDITIONS, apply_record_anomaly
from evaluation.stale_state import apply_stale_state
from state_grounded_qa.retrieval import SentenceTransformerBackend
from state_grounded_qa.schemas import EvidenceRecord, RuntimeState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stale", "distractor", "record-anomaly"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--level", type=int, choices=LEVELS)
    parser.add_argument("--evidence-pool")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--field-path", action="append")
    parser.add_argument("--missing-rule", choices=("null", "remove"), default="null")
    parser.add_argument("--conflict-value")
    parser.add_argument("--delay-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    if args.mode == "record-anomaly":
        if args.condition is None:
            raise SystemExit("--condition is required for record-anomaly mode")
        conflict_value = None
        if args.conflict_value is not None:
            try:
                conflict_value = json.loads(args.conflict_value)
            except json.JSONDecodeError:
                conflict_value = args.conflict_value
        write_jsonl(
            args.output,
            apply_record_anomaly(
                rows,
                args.condition,
                field_paths=args.field_path,
                missing_rule=args.missing_rule,
                conflict_value=conflict_value,
                delay_steps=args.delay_steps,
                seed=args.seed,
            ),
        )
        return
    if args.mode == "stale":
        if args.delta is None:
            raise SystemExit("--delta is required for stale mode")
        write_jsonl(args.output, apply_stale_state(rows, args.delta))
        return
    if args.level is None or not args.evidence_pool:
        raise SystemExit("--level and --evidence-pool are required for distractor mode")
    pool = [EvidenceRecord(**item) for item in read_jsonl(args.evidence_pool)]
    backend = SentenceTransformerBackend()
    output = []
    for row in rows:
        base = [EvidenceRecord(**item) for item in row["evidence"]]
        contaminated = contaminate_evidence(
            base,
            pool,
            RuntimeState.from_dict(row["runtime_state"]),
            row["question"],
            backend,
            args.level,
            row.get("supporting_evidence_ids", []),
            row.get("acceptable_next_actions", []),
        )
        converted = dict(row)
        converted["evidence"] = [item.to_dict() for item in contaminated]
        converted["distractor_level_percent"] = args.level
        output.append(converted)
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
