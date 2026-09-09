"""State-conditioned evidence filtering, query composition, ranking, and Top-k."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .schemas import EvidenceRecord, RuntimeState


class EmbeddingBackend(Protocol):
    """Embedding interface that permits offline tests without model weights."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a two-dimensional float array, one row per text."""


class SentenceTransformerBackend:
    """Lazy BAAI/bge-m3 backend; no download occurs during module import."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode and normalize text using the configured sentence transformer."""

        model = self._load()
        return np.asarray(
            model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float32,
        )

    def count_tokens(self, text: str) -> int:
        """Count model tokens for enforcing the configured evidence chunk limit."""

        model = self._load()
        return len(model.tokenizer.encode(text, add_special_tokens=True))


@dataclass(slots=True)
class RankedEvidence:
    """Evidence and its cosine-similarity score."""

    record: EvidenceRecord
    score: float


def _constraint_matches(expected: Any, actual: Any) -> bool:
    if expected in (None, [], {}):
        return True
    expected_values = set(expected if isinstance(expected, list) else [expected])
    actual_values = set(actual if isinstance(actual, list) else [actual])
    return bool(expected_values & actual_values)


def _scene_entity_ids(state: RuntimeState) -> list[str]:
    identifiers = [state.scene.active_object] if state.scene.active_object else []
    identifiers.extend(
        str(item.get("object_id"))
        for item in state.scene.objects
        if isinstance(item, Mapping) and item.get("object_id") is not None
    )
    for relation in state.scene.object_relations:
        identifiers.extend(
            str(relation.get(key))
            for key in ("subject", "object")
            if relation.get(key) is not None
        )
    return list(dict.fromkeys(identifiers))


def evidence_is_admissible(record: EvidenceRecord, state: RuntimeState, goal: str | None = None) -> bool:
    """Return whether evidence metadata is compatible with S_t and goal G."""

    scene = record.scene_constraints
    if not _constraint_matches(scene.get("scene_ids"), state.scene.scene_id):
        return False
    if not _constraint_matches(scene.get("locations"), state.scene.location):
        return False
    if not _constraint_matches(scene.get("entity_ids"), _scene_entity_ids(state)):
        return False
    task = record.task_step_constraints
    if not _constraint_matches(task.get("step_ids"), state.task.current_step):
        return False
    statuses = task.get("progress_statuses")
    if not _constraint_matches(statuses, state.task.progress_status):
        return False
    goal_terms = record.applicability_metadata.get("goal_terms", [])
    if goal_terms and goal:
        lowered = goal.lower()
        if not any(str(term).lower() in lowered for term in goal_terms):
            return False
    return True


def filter_evidence(
    knowledge_base: Sequence[EvidenceRecord], state: RuntimeState, goal: str | None = None
) -> list[EvidenceRecord]:
    """Implement C_t = Filter(K; S_t, G)."""

    return [record for record in knowledge_base if evidence_is_admissible(record, state, goal)]


def summarize_state(state: RuntimeState) -> str:
    """Create a compact state summary for state-enhanced retrieval."""

    task = state.task
    return (
        f"scene={state.scene.scene_id or 'n/a'}; location={state.scene.location or 'n/a'}; "
        f"active_object={state.scene.active_object or 'n/a'}; goal={task.goal or 'n/a'}; "
        f"current_step={task.current_step or 'none'}; progress={task.progress_status or 'n/a'}; "
        f"completed={','.join(task.completed_steps) or 'none'}; "
        f"remaining={','.join(task.remaining_steps) or 'none'}; "
        f"legal_actions={','.join(state.action.available_actions) or 'none'}"
    )


def compose_query(question: str, state: RuntimeState) -> str:
    """Compose the state-enhanced query used for ranking."""

    return f"State: {summarize_state(state)}\nQuestion: {question.strip()}"


def rank_evidence(
    query: str, candidates: Sequence[EvidenceRecord], backend: EmbeddingBackend
) -> list[RankedEvidence]:
    """Rank candidates using cosine similarity / normalized inner product."""

    if not candidates:
        return []
    embeddings = np.asarray(backend.encode([query] + [item.text for item in candidates]), dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(candidates) + 1:
        raise ValueError("Embedding backend returned an unexpected shape")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-12, None)
    scores = normalized[1:] @ normalized[0]
    ranked = [RankedEvidence(record=record, score=float(score)) for record, score in zip(candidates, scores)]
    return sorted(ranked, key=lambda item: (-item.score, item.record.evidence_id))


def validate_chunk_lengths(
    records: Sequence[EvidenceRecord], token_counter: Any, max_chunk_tokens: int = 256
) -> None:
    """Reject evidence chunks exceeding the configured model-token limit."""

    if max_chunk_tokens < 1:
        raise ValueError("max_chunk_tokens must be positive")
    for record in records:
        length = int(token_counter.count_tokens(record.text))
        if length > max_chunk_tokens:
            raise ValueError(
                f"Evidence {record.evidence_id!r} has {length} tokens; limit is {max_chunk_tokens}"
            )


def select_top_k(ranked: Sequence[RankedEvidence], top_k: int = 5) -> list[EvidenceRecord]:
    """Select E_t with the configured Top-k (paper default: 5)."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    return [item.record for item in ranked[:top_k]]


# Paper notation aliases.
Filter = filter_evidence
Summarize = summarize_state
ComposeQuery = compose_query
