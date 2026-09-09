"""Metadata-admissible semantic hard-distractor contamination."""

from __future__ import annotations

from math import ceil
from typing import Sequence

from state_grounded_qa.retrieval import EmbeddingBackend, evidence_is_admissible, rank_evidence
from state_grounded_qa.schemas import EvidenceRecord, RuntimeState


LEVELS = (0, 50, 100, 200)


def contaminate_evidence(
    base_candidates: Sequence[EvidenceRecord],
    evidence_pool: Sequence[EvidenceRecord],
    state: RuntimeState,
    question: str,
    backend: EmbeddingBackend,
    level_percent: int,
    supporting_ids: Sequence[str],
    acceptable_action_ids: Sequence[str],
) -> list[EvidenceRecord]:
    """Add semantically hard items that pass metadata filters but support neither target."""

    if level_percent not in LEVELS:
        raise ValueError(f"level_percent must be one of {LEVELS}")
    supporting = set(supporting_ids)
    acceptable = set(acceptable_action_ids)
    base_ids = {item.evidence_id for item in base_candidates}
    eligible = [
        item
        for item in evidence_pool
        if item.evidence_id not in base_ids
        and item.evidence_id not in supporting
        and evidence_is_admissible(item, state, state.task.goal)
        and not acceptable.intersection(item.applicability_metadata.get("supported_action_ids", []))
        and not item.applicability_metadata.get("supports_current_answer", False)
    ]
    count = ceil(len(base_candidates) * level_percent / 100)
    ranked = rank_evidence(question, eligible, backend)
    return list(base_candidates) + [item.record for item in ranked[:count]]
