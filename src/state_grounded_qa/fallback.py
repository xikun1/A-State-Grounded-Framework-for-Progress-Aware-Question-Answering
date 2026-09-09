"""Conservative response used after one unsuccessful regeneration."""

from __future__ import annotations

from .schemas import RuntimeState, StructuredResponse


def ConservativeFallback(state: RuntimeState) -> StructuredResponse:
    """Return a valid non-executing response without modifying authoritative state."""

    if state.task.progress_status is None:
        answer = "I cannot provide a state-supported action recommendation for this request."
    elif state.task.progress_status == "completed":
        answer = "The available state marks the task as completed; no next action is recommended."
    else:
        answer = "I cannot verify a safe next action from the available evidence and current state."
    return StructuredResponse(
        answer=answer,
        evidence_ids=[],
        scene_refs={"entity_ids": [], "locations": [], "relation_triples": []},
        current_step_id=state.task.current_step,
        progress_status=state.task.progress_status,
        remaining_step_ids=list(state.task.remaining_steps),
        next_action_id=None,
    )
