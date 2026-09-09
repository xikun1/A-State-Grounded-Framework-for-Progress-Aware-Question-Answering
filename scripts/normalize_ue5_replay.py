"""Convert exported UE5 JSON, JSONL, or CSV logs to normalized request JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from _common import REPOSITORY_ROOT, read_jsonl, write_jsonl
from jsonschema import Draft202012Validator
from state_grounded_qa.adapters import UE5Adapter


def _decode_cell(value: str) -> Any:
    stripped = value.strip()
    if stripped[:1] in {"{", "["}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load supported exported-log formats."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value if isinstance(value, list) else value.get("records", [value])
        if not all(isinstance(item, dict) for item in records):
            raise ValueError("JSON input must contain request objects")
        return records
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [{key: _decode_cell(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    raise ValueError("Input must be .json, .jsonl, or .csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(REPOSITORY_ROOT / "configs" / "ue5_adapter.yaml"))
    parser.add_argument(
        "--schema", default=str(REPOSITORY_ROOT / "data" / "ue5" / "request_record.schema.json")
    )
    parser.add_argument(
        "--runtime-schema",
        default=str(REPOSITORY_ROOT / "data" / "schemas" / "runtime_state.schema.json"),
    )
    args = parser.parse_args()
    adapter = UE5Adapter.from_yaml(args.config)
    normalized = [adapter.normalize_request(record) for record in load_records(Path(args.input))]
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    runtime_schema = json.loads(Path(args.runtime_schema).read_text(encoding="utf-8"))
    runtime_validator = Draft202012Validator(runtime_schema)
    for index, record in enumerate(normalized, start=1):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"Normalized record {index} failed schema validation: {errors[0].message}")
        runtime_errors = sorted(
            runtime_validator.iter_errors(record["runtime_state"]), key=lambda error: list(error.path)
        )
        if runtime_errors:
            raise ValueError(
                f"Runtime state {index} failed schema validation: {runtime_errors[0].message}"
            )
    write_jsonl(args.output, normalized)


if __name__ == "__main__":
    main()
