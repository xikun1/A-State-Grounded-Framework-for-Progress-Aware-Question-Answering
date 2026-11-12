"""FAISS retrieval and independent Qwen-tokenizer counting tests."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from state_grounded_qa import retrieval
from state_grounded_qa.retrieval import (
    QwenTokenizerTokenCounter,
    filter_evidence,
    rank_evidence,
    validate_chunk_lengths,
)
from state_grounded_qa.schemas import (
    ActionState,
    DialogueState,
    EvidenceRecord,
    RuntimeState,
    SceneState,
    StateMetadata,
    TaskState,
)


class VectorBackend:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float64)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        del texts
        return self.vectors


def record(evidence_id: str, text: str | None = None, **metadata: object) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        text=text or evidence_id,
        source="fixture",
        applicability_metadata=dict(metadata),
    )


def test_faiss_exact_cosine_top_five_and_metadata_are_preserved() -> None:
    records = [record(f"e{i}", source_index=i) for i in range(1, 7)]
    backend = VectorBackend(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
            [0.0, 1.0],
            [0.5, 0.8660254],
            [0.4, 0.9165151],
        ]
    )

    ranked = rank_evidence("query", records, backend)

    assert [item.record.evidence_id for item in ranked[:5]] == ["e1", "e2", "e3", "e5", "e6"]
    assert ranked[0].record.applicability_metadata == {"source_index": 1}


def test_filtering_happens_before_ranking_and_empty_or_single_inputs_work() -> None:
    state = RuntimeState(
        scene=SceneState(scene_id="room"),
        task=TaskState(current_step="step_1"),
        action=ActionState(),
        dialogue=DialogueState(),
        metadata=StateMetadata(snapshot_id="s1"),
    )
    candidates = [
        EvidenceRecord("allowed", "allowed", "fixture", scene_constraints={"scene_ids": ["room"]}),
        EvidenceRecord("blocked", "blocked", "fixture", scene_constraints={"scene_ids": ["other"]}),
    ]
    filtered = filter_evidence(candidates, state)

    assert [item.record.evidence_id for item in rank_evidence("query", filtered, VectorBackend([[1, 0], [1, 0]]))] == [
        "allowed"
    ]
    assert rank_evidence("query", [], VectorBackend([])) == []


def test_equal_faiss_scores_are_sorted_by_evidence_id() -> None:
    candidates = [record("z"), record("a"), record("m")]
    ranked = rank_evidence("query", candidates, VectorBackend([[1, 0], [1, 0], [1, 0], [1, 0]]))

    assert [item.record.evidence_id for item in ranked] == ["a", "m", "z"]


def test_qwen_token_counter_is_lazy_and_counts_without_special_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool) -> list[str]:
            calls.append((text, add_special_tokens))
            return text.split()

    monkeypatch.setattr(retrieval, "_load_tokenizer", lambda model_name: FakeTokenizer())
    counter = QwenTokenizerTokenCounter()

    assert counter.count_tokens("alpha beta") == 2
    assert calls == [("alpha beta", False)]


def test_chunk_length_boundary_uses_injected_counter() -> None:
    item = record("boundary", "fixture")

    validate_chunk_lengths([item], lambda text: 256)
    with pytest.raises(ValueError, match="limit is 256"):
        validate_chunk_lengths([item], lambda text: 257)
