"""Algorithm 1 orchestration with a single retry and conservative fallback."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .fallback import ConservativeFallback
from .progress import TaskGraph
from .retrieval import (
    EmbeddingBackend,
    compose_query,
    filter_evidence,
    rank_evidence,
    select_top_k,
    validate_chunk_lengths,
)
from .schemas import EvidenceRecord, RuntimeState, StructuredResponse
from .serialization import parse_structured_output, serialize_input
from .state_adapter import StateAdapter
from .verifier import VerificationResult, verify_response


class GeneratorBackend(Protocol):
    """Text/structured generation interface."""

    def generate(self, serialized_input: str, verifier_feedback: Sequence[str] | None = None) -> Any:
        """Generate one structured-response candidate."""


@dataclass(slots=True)
class PipelineResult:
    """Final response and request-level control-flow metadata."""

    response: StructuredResponse
    verification: VerificationResult
    retrieved_evidence: list[EvidenceRecord]
    retry_count: int
    used_fallback: bool


class StateGroundedPipeline:
    """Executable implementation of the paper's request-level pipeline."""

    def __init__(
        self,
        adapter: StateAdapter,
        task_graph: TaskGraph | None,
        knowledge_base: Sequence[EvidenceRecord],
        embedder: EmbeddingBackend,
        generator: GeneratorBackend,
        top_k: int = 5,
        max_chunk_tokens: int = 256,
        max_retry: int = 1,
    ) -> None:
        if max_retry != 1:
            raise ValueError("The method permits exactly one maximum retry")
        self.adapter = adapter
        self.task_graph = task_graph
        self.knowledge_base = list(knowledge_base)
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k
        self.max_chunk_tokens = max_chunk_tokens
        self.max_retry = max_retry

    def run(self, raw_record: Mapping[str, Any], question: str) -> PipelineResult:
        """Build S_t, retrieve E_t once, then generate, verify, retry, or fall back."""

        state = self.adapter.build_state(raw_record)
        state_before = copy.deepcopy(state.to_dict())
        candidates = filter_evidence(self.knowledge_base, state, state.task.goal)
        if hasattr(self.embedder, "count_tokens"):
            validate_chunk_lengths(candidates, self.embedder, self.max_chunk_tokens)
        query = compose_query(question, state)
        evidence = select_top_k(rank_evidence(query, candidates, self.embedder), self.top_k)
        serialized = serialize_input(question, state, evidence)

        first = self.generator.generate(serialized, verifier_feedback=None)
        first_verification = verify_response(first, state, evidence, self.task_graph)
        if first_verification.accepted:
            response = parse_structured_output(first)
            result = PipelineResult(response, first_verification, evidence, 0, False)
        else:
            second = self.generator.generate(serialized, verifier_feedback=first_verification.failure_labels)
            second_verification = verify_response(second, state, evidence, self.task_graph)
            if second_verification.accepted:
                response = parse_structured_output(second)
                result = PipelineResult(response, second_verification, evidence, 1, False)
            else:
                fallback = ConservativeFallback(state)
                fallback_verification = verify_response(fallback, state, evidence, self.task_graph)
                result = PipelineResult(fallback, fallback_verification, evidence, 1, True)

        if state.to_dict() != state_before:
            raise RuntimeError("Pipeline components modified authoritative runtime state")
        return result
