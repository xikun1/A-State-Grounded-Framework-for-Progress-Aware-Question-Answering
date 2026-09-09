"""State-staleness perturbations that preserve current reference labels."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterable, Mapping


def apply_stale_state(
    records: Iterable[Mapping[str, Any]], delta: int
) -> list[dict[str, Any]]:
    """Replace S_t with S_(t-delta), retaining labels associated with current t."""

    if delta not in {0, 1, 2, 3}:
        raise ValueError("delta must be one of 0, 1, 2, or 3")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sequence_id"])].append(record)
    output: list[dict[str, Any]] = []
    for sequence in grouped.values():
        sequence.sort(key=lambda item: int(item.get("event_index", item["runtime_state"]["metadata"]["event_index"])))
        for index, current in enumerate(sequence):
            source = sequence[max(0, index - delta)]
            row = copy.deepcopy(dict(current))
            row["runtime_state"] = copy.deepcopy(source["runtime_state"])
            row["reference_current_state"] = copy.deepcopy(current["runtime_state"])
            row["staleness_delta"] = delta
            output.append(row)
    return output
