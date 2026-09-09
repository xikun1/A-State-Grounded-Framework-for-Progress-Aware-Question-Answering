"""Normalize user-supplied MSQA-style JSONL without redistributing source data."""

from __future__ import annotations

import argparse

from _common import read_jsonl, write_jsonl
from state_grounded_qa.adapters import MSQAAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MSQA-style JSONL obtained under the source terms")
    parser.add_argument("--output", required=True, help="Normalized JSONL destination")
    args = parser.parse_args()
    adapter = MSQAAdapter()
    write_jsonl(args.output, (adapter.adapt_record(record) for record in read_jsonl(args.input)))


if __name__ == "__main__":
    main()
