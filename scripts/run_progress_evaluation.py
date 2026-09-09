"""Compute state-construction or progress/action metrics."""

from __future__ import annotations

import argparse

from _common import read_jsonl, write_json
from evaluation.progress_metrics import progress_action_metrics, state_construction_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--track", choices=("state", "progress_action"), default="progress_action")
    args = parser.parse_args()
    metric = state_construction_metrics if args.track == "state" else progress_action_metrics
    write_json(args.output, metric(read_jsonl(args.input)))


if __name__ == "__main__":
    main()
