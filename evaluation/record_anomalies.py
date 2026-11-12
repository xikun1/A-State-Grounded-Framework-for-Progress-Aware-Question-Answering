"""Deterministic environment-record anomaly generation for robustness inputs."""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


CONDITIONS = ("clean", "incomplete", "contradictory", "delayed")


def _parts(path: str) -> list[str]:
    value = path.strip().strip(".")
    if value.startswith("$"):
        value = value[1:].lstrip(".")
    parts = [part for part in value.split(".") if part]
    if not parts:
        raise ValueError("field_path must contain at least one field")
    return parts


def _get_path(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in _parts(path):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(record: dict[str, Any], path: str, value: Any, remove: bool = False) -> None:
    parts = _parts(path)
    current: dict[str, Any] = record
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    if remove:
        current.pop(parts[-1], None)
    else:
        current[parts[-1]] = copy.deepcopy(value)


def _default_conflict(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value + 1
    if isinstance(value, list):
        return list(value) + ["__contradictory_value__"]
    if value is None:
        return "__contradictory_value__"
    return f"{value}__contradictory__"


def _selected_path(field_paths: Sequence[str] | None, rng: random.Random) -> str | None:
    if not field_paths:
        return None
    return field_paths[rng.randrange(len(field_paths))]


def apply_record_anomaly(
    records: Iterable[Mapping[str, Any]],
    condition: str,
    *,
    field_paths: Sequence[str] | None = None,
    missing_rule: str = "null",
    conflict_value: Any = None,
    delay_steps: int = 1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Create one deterministic anomaly condition while preserving references.

    Only the configured environment-record path is changed.  Request identifiers
    and reference labels remain copied from the source record.  ``delayed``
    replaces the configured field with the value from an earlier request in the
    same sequence; the default field is ``runtime_state``.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if missing_rule not in {"null", "remove"}:
        raise ValueError("missing_rule must be 'null' or 'remove'")
    if delay_steps < 1:
        raise ValueError("delay_steps must be at least 1")

    source = [copy.deepcopy(dict(record)) for record in records]
    rng = random.Random(seed)
    selected_path = _selected_path(field_paths, rng)
    if condition in {"incomplete", "contradictory"} and selected_path is None:
        raise ValueError(f"{condition} requires at least one --field-path")
    if condition == "delayed" and selected_path is None:
        selected_path = "runtime_state"

    output = [copy.deepcopy(record) for record in source]
    if condition == "incomplete":
        for row in output:
            _set_path(row, selected_path, None, remove=missing_rule == "remove")
    elif condition == "contradictory":
        for row in output:
            original = _get_path(row, selected_path)
            value = _default_conflict(original) if conflict_value is None else conflict_value
            _set_path(row, selected_path, value)
    elif condition == "delayed":
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, row in enumerate(source):
            sequence_id = str(row.get("sequence_id", "__default_sequence__"))
            grouped[sequence_id].append((index, row))
        for sequence_rows in grouped.values():
            sequence_rows.sort(
                key=lambda pair: (
                    int(pair[1].get("event_index", pair[1].get("request_index", pair[0]))),
                    pair[0],
                )
            )
            for position, (index, row) in enumerate(sequence_rows):
                source_position = max(0, position - delay_steps)
                delayed_value = _get_path(sequence_rows[source_position][1], selected_path)
                _set_path(output[index], selected_path, delayed_value)
                output[index]["delayed_from_request_id"] = sequence_rows[source_position][1].get(
                    "request_id"
                )

    for row in output:
        row["record_anomaly"] = {
            "condition": condition,
            "field_path": selected_path,
            "missing_rule": missing_rule if condition == "incomplete" else None,
            "delay_steps": delay_steps if condition == "delayed" else None,
            "seed": seed,
        }
    return output
