"""Dependency-light dataclasses shared by adapters, retrieval, and verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


ProgressStatus = Literal["not_started", "in_progress", "completed"]


@dataclass(slots=True)
class SceneState:
    """Current scene facts exposed by an environment adapter."""

    scene_id: str | None = None
    location: str | None = None
    active_object: str | None = None
    object_relations: list[dict[str, str]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TaskState:
    """Request-level task state; fields may be non-applicable for scene-only data."""

    goal: str | None = None
    current_step: str | None = None
    progress_status: ProgressStatus | None = None
    completed_steps: list[str] = field(default_factory=list)
    remaining_steps: list[str] = field(default_factory=list)
    milestone_progress: float | None = None


@dataclass(slots=True)
class ActionState:
    """Executable actions and their current preconditions.

    ``available_actions`` is the legal action set. ``acceptable_next_actions`` is
    an optional, narrower reference set used by evaluation, not legality checks.
    """

    available_actions: list[str] = field(default_factory=list)
    preconditions: dict[str, Any] = field(default_factory=dict)
    last_action: str | None = None
    acceptable_next_actions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DialogueState:
    """Dialogue context available before the current request."""

    dialogue_history: list[dict[str, Any]] = field(default_factory=list)
    unresolved_issue: str | None = None


@dataclass(slots=True)
class StateMetadata:
    """Snapshot provenance and temporal identity."""

    snapshot_id: str
    timestamp: str | float | int | None = None
    event_index: int | None = None
    source_adapter: str = "unknown"


@dataclass(slots=True)
class RuntimeState:
    """Authoritative state S_t constructed from records available at request t."""

    scene: SceneState
    task: TaskState
    action: ActionState
    dialogue: DialogueState
    metadata: StateMetadata

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeState":
        """Construct a runtime state from normalized nested mappings."""

        return cls(
            scene=SceneState(**dict(value.get("scene", {}))),
            task=TaskState(**dict(value.get("task", {}))),
            action=ActionState(**dict(value.get("action", {}))),
            dialogue=DialogueState(**dict(value.get("dialogue", {}))),
            metadata=StateMetadata(**dict(value.get("metadata", {}))),
        )


@dataclass(slots=True)
class EvidenceRecord:
    """Knowledge-base evidence with scene and task applicability constraints."""

    evidence_id: str
    text: str
    source: str
    scene_constraints: dict[str, Any] = field(default_factory=dict)
    task_step_constraints: dict[str, Any] = field(default_factory=dict)
    applicability_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(slots=True)
class StructuredResponse:
    """Structured candidate or final response checked by the verifier."""

    answer: str
    evidence_ids: list[str]
    scene_refs: dict[str, Any]
    current_step_id: str | None
    progress_status: ProgressStatus | None
    remaining_step_ids: list[str]
    next_action_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StructuredResponse":
        """Construct a response after schema-level validation."""

        return cls(
            answer=value["answer"],
            evidence_ids=list(value["evidence_ids"]),
            scene_refs=dict(value["scene_refs"]),
            current_step_id=value["current_step_id"],
            progress_status=value["progress_status"],
            remaining_step_ids=list(value["remaining_step_ids"]),
            next_action_id=value["next_action_id"],
        )
