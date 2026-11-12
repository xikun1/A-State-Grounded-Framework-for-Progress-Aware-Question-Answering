"""Request-level and stage-wise efficiency metrics."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Iterable, Mapping

import numpy as np


UE5_OFFLINE_REPLAY_PROTOCOL = "ue5_offline_replay_evaluation"
END_TO_END_COMPUTATIONAL_COST_PROTOCOL = "end_to_end_computational_cost_evaluation"


def _metadata_values(rows: list[Mapping[str, Any]], key: str) -> list[Any]:
    """Return stable distinct protocol metadata values supplied by the caller."""

    values: dict[str, Any] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        values[marker] = value
    return [values[marker] for marker in sorted(values)]


def _protocol_consistency(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Report metadata fields that disagree inside one timing aggregate."""

    issues: list[str] = []
    values_by_key: dict[str, list[Any]] = {}
    for key in ("evaluation_protocol", "evaluation_subset", "timing_scope", "execution_config"):
        values = _metadata_values(rows, key)
        values_by_key[key] = values
        if len(values) > 1:
            issues.append(key)
    return {
        "consistent": not issues,
        "inconsistent_fields": issues,
        "values": values_by_key,
    }


def efficiency_metrics(
    records: Iterable[Mapping[str, Any]], protocol: str | None = None
) -> dict[str, Any]:
    """Compute timing metrics and expose the supplied measurement protocol.

    ``latency_s`` is treated as the request-level endpoint-to-endpoint value
    supplied in each input record.  This function does not add stages or infer
    a UE5 execution protocol.  When ``protocol`` is requested, records must
    explicitly carry the matching ``evaluation_protocol`` field.
    """

    source_rows = list(records)
    if protocol is not None:
        rows = [row for row in source_rows if row.get("evaluation_protocol") == protocol]
        if not rows:
            raise ValueError(
                f"No timing records carry evaluation_protocol={protocol!r}; "
                "The measurement protocol cannot be inferred from the supplied timing records."
            )
    else:
        rows = source_rows
        protocols = _metadata_values(rows, "evaluation_protocol")
        if len(protocols) > 1:
            raise ValueError(
                "Timing records contain multiple evaluation_protocol values; "
                "use --protocol or --compare-protocols instead of mixing protocols"
            )
    latencies = np.asarray([float(row["latency_s"]) for row in rows], dtype=float)
    stage_values: dict[str, list[float]] = defaultdict(list)
    stage_sums: list[float] = []
    for row in rows:
        stages = row.get("stage_latency_s", {})
        stage_sums.append(sum(float(value) for value in stages.values()))
        for stage, value in stages.items():
            stage_values[str(stage)].append(float(value))
    protocol_values = _metadata_values(rows, "evaluation_protocol")
    subset_values = _metadata_values(rows, "evaluation_subset")
    scope_values = _metadata_values(rows, "timing_scope")
    config_values = _metadata_values(rows, "execution_config")
    mean_stage_sum = sum(stage_sums) / len(stage_sums) if stage_sums else 0.0
    mean_latency = float(latencies.mean()) if len(latencies) else 0.0
    tokens_per_request = (
        sum(float(row.get("tokens", row.get("total_tokens", 0))) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    return {
        "mean_latency_s": mean_latency,
        "p95_latency_s": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        "tokens_per_request": tokens_per_request,
        "mean_total_tokens_per_request": tokens_per_request,
        "stage_wise_latency_s": {
            stage: sum(values) / len(values) for stage, values in sorted(stage_values.items())
        },
        "mean_stage_latency_sum_s": mean_stage_sum,
        "mean_latency_minus_stage_sum_s": mean_latency - mean_stage_sum,
        "latency_components_are_additive": False,
        "n_requests": len(rows),
        "timing_protocol": {
            "evaluation_protocol": protocol_values[0] if len(protocol_values) == 1 else None,
            "evaluation_protocols": protocol_values,
            "evaluation_subsets": subset_values,
            "timing_scopes": scope_values,
            "execution_configs": config_values,
            "metadata_available": bool(protocol_values or subset_values or scope_values or config_values),
            "latency_field": "latency_s",
            "latency_definition": "request-level actual endpoint-to-endpoint value supplied by the timing input",
            "stage_latency_definition": "reported per-stage components; their sum is reported separately and is not substituted for latency_s",
            "protocol_consistency": _protocol_consistency(rows),
        },
    }


def compare_efficiency_protocols(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate each evaluation protocol independently for comparison."""

    rows = list(records)
    protocols = [str(value) for value in _metadata_values(rows, "evaluation_protocol")]
    if not protocols:
        raise ValueError("--compare-protocols requires timing records with evaluation_protocol")
    by_protocol = {
        protocol: efficiency_metrics(rows, protocol=protocol) for protocol in protocols
    }
    return {
        "comparison": True,
        "n_protocols": len(by_protocol),
        "protocols": by_protocol,
        "configuration_values_by_protocol": {
            protocol: metrics["timing_protocol"]
            for protocol, metrics in by_protocol.items()
        },
    }
