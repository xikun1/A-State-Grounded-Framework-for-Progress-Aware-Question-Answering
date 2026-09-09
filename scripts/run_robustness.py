"""Build stale-state or distractor-evidence evaluation inputs from supplied records."""

from __future__ import annotations

import argparse

from _common import read_jsonl, write_jsonl
from evaluation.distractor_evidence import LEVELS, contaminate_evidence
from evaluation.stale_state import apply_stale_state
from state_grounded_qa.retrieval import SentenceTransformerBackend
from state_grounded_qa.schemas import EvidenceRecord, RuntimeState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stale", "distractor"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--level", type=int, choices=LEVELS)
    parser.add_argument("--evidence-pool")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
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
