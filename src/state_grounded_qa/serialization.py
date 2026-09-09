"""Prompt serialization and structured-output parsing."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .schemas import EvidenceRecord, RuntimeState, StructuredResponse


def serialize_input(question: str, state: RuntimeState, evidence: Sequence[EvidenceRecord]) -> str:
    """Serialize input in the exact method order described in the paper."""

    sections = [
        ("Task goal", {"goal": state.task.goal}),
        ("Scene state", state.to_dict()["scene"]),
        (
            "Progress state",
            {
                "current_step": state.task.current_step,
                "progress_status": state.task.progress_status,
                "completed_steps": state.task.completed_steps,
                "remaining_steps": state.task.remaining_steps,
                "milestone_progress": state.task.milestone_progress,
            },
        ),
        (
            "Action constraints",
            {
                "legal_actions": state.action.available_actions,
                "preconditions": state.action.preconditions,
                "last_action": state.action.last_action,
            },
        ),
        ("Retrieved evidence", [item.to_dict() for item in evidence]),
        ("User question", question),
    ]
    return "\n\n".join(
        f"## {title}\n{json.dumps(value, ensure_ascii=False, sort_keys=True)}"
        for title, value in sections
    )


def parse_structured_output(output: str | Mapping[str, Any] | StructuredResponse) -> StructuredResponse:
    """Parse a JSON object into the shared structured-response dataclass."""

    if isinstance(output, StructuredResponse):
        return output
    value = json.loads(output) if isinstance(output, str) else dict(output)
    return StructuredResponse.from_dict(value)
