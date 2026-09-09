"""Normalize user-supplied EAI-VirtualHome request prefixes."""

from __future__ import annotations

import argparse

from _common import read_jsonl, write_jsonl
from state_grounded_qa.adapters import EAIVirtualHomeAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Request-level JSONL with task specification and trajectory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    adapter = EAIVirtualHomeAdapter()
    write_jsonl(args.output, (adapter.adapt_record(record) for record in read_jsonl(args.input)))


if __name__ == "__main__":
    main()
