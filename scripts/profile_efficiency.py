"""Aggregate request-level timing and token records."""

from __future__ import annotations

import argparse

from _common import read_jsonl, write_json
from evaluation.efficiency_metrics import efficiency_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, efficiency_metrics(read_jsonl(args.input)))


if __name__ == "__main__":
    main()
