"""Aggregate request-level timing and token records."""

from __future__ import annotations

import argparse

from _common import read_jsonl, write_json
from evaluation.efficiency_metrics import compare_efficiency_protocols, efficiency_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol",
        help="Optional semantic evaluation_protocol value, such as ue5_offline_replay_evaluation or end_to_end_computational_cost_evaluation",
    )
    parser.add_argument(
        "--compare-protocols",
        action="store_true",
        help="Aggregate each evaluation_protocol separately instead of mixing records",
    )
    args = parser.parse_args()
    if args.compare_protocols and args.protocol is not None:
        raise SystemExit("--compare-protocols cannot be combined with --protocol")
    rows = read_jsonl(args.input)
    metric = (
        compare_efficiency_protocols(rows)
        if args.compare_protocols
        else efficiency_metrics(rows, protocol=args.protocol)
    )
    write_json(args.output, metric)


if __name__ == "__main__":
    main()
