"""Configurable adapter for exported UE5 JSON, JSONL, or CSV records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ..schemas import ActionState, DialogueState, RuntimeState, SceneState, StateMetadata, TaskState
from ..state_adapter import StateAdapter


class UE5Adapter(StateAdapter):
    """Normalize exported scene, task, milestone, action, event, and dialogue fields."""

    source_name = "ue5"

    def __init__(self, config: Mapping[str, Any]) -> None:
        policy = config.get("missing_value_policy", {})
        super().__init__(missing_value=policy.get("canonical", None))
        self.config = dict(config)
        self._missing_values = list(policy.get("accepted_source_values", [None, ""]))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UE5Adapter":
        """Load an adapter mapping without requiring an Unreal Engine installation."""

        with Path(path).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        return cls(config)

    def _mapped(self, record: Mapping[str, Any], section: str, target: str, default: Any = None) -> Any:
        path = self.config.get(section, {}).get(target)
        return self.get_path(record, path, default) if path else default

    def is_missing(self, value: Any) -> bool:
        """Apply configured source missing values without conflating absent fields with references."""

        return any(value == missing for missing in self._missing_values)

    def _objects(self, raw: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in raw or []:
            if not isinstance(item, Mapping):
                continue
            converted = dict(item)
            object_id = self.normalize_identifier(item.get("object_id", item.get("id", item.get("actor_id"))))
            if object_id is not None:
                converted["object_id"] = object_id
            result.append(converted)
        return result

    def _relations(self, raw: Any) -> list[dict[str, str]]:
        result = []
        for item in raw or []:
            if not isinstance(item, Mapping):
                continue
            result.append(
                {
                    "subject": self.normalize_identifier(item.get("subject", item.get("source"))) or "",
                    "relation": self.normalize_identifier(item.get("relation", item.get("type"))) or "",
                    "object": self.normalize_identifier(item.get("object", item.get("target"))) or "",
                }
            )
        return result

    def build_state(self, raw_record: Mapping[str, Any]) -> RuntimeState:
        """Map one exported request record to S_t, excluding offline references."""

        completed = self.normalize_identifier_list(self._mapped(raw_record, "task_mapping", "completed_steps", []))
        remaining = self.normalize_identifier_list(self._mapped(raw_record, "task_mapping", "remaining_steps", []))
        current = self.normalize_identifier(self._mapped(raw_record, "task_mapping", "current_step"))
        remaining = [step for step in remaining if step != current and step not in completed]
        progress_status = self._mapped(raw_record, "task_mapping", "progress_status")
        if progress_status not in {None, "not_started", "in_progress", "completed"}:
            raise ValueError(f"Unsupported progress status: {progress_status}")
        if progress_status == "completed":
            current, remaining = None, []
        snapshot = self._mapped(raw_record, "field_mapping", "snapshot_id", raw_record.get("request_id"))
        if self.is_missing(snapshot):
            raise ValueError("UE5 request record requires a snapshot or request identifier")
        dialogue = self._mapped(raw_record, "field_mapping", "dialogue_history", []) or []
        events = self._mapped(raw_record, "event_mapping", "environment_events", []) or []
        last_action = self._mapped(raw_record, "action_mapping", "last_action")
        if last_action is None and events:
            last_event = events[-1] if isinstance(events, list) else None
            if isinstance(last_event, Mapping):
                last_action = last_event.get("action_id")
        return RuntimeState(
            scene=SceneState(
                scene_id=self.normalize_identifier(self._mapped(raw_record, "scene_mapping", "scene_id")),
                location=self.normalize_identifier(self._mapped(raw_record, "scene_mapping", "location")),
                active_object=self.normalize_identifier(self._mapped(raw_record, "scene_mapping", "active_object")),
                object_relations=self._relations(
                    self._mapped(raw_record, "scene_mapping", "object_relations", [])
                ),
                objects=self._objects(self._mapped(raw_record, "scene_mapping", "objects", [])),
            ),
            task=TaskState(
                goal=self._mapped(raw_record, "task_mapping", "goal"),
                current_step=current,
                progress_status=progress_status,
                completed_steps=completed,
                remaining_steps=remaining,
                milestone_progress=self._mapped(raw_record, "task_mapping", "milestone_progress"),
            ),
            action=ActionState(
                available_actions=self.normalize_identifier_list(
                    self._mapped(raw_record, "action_mapping", "available_actions", [])
                ),
                preconditions=dict(self._mapped(raw_record, "action_mapping", "preconditions", {}) or {}),
                last_action=self.normalize_identifier(last_action),
                acceptable_next_actions=self.normalize_identifier_list(
                    self._mapped(raw_record, "action_mapping", "acceptable_next_actions", [])
                ),
            ),
            dialogue=DialogueState(
                dialogue_history=list(dialogue),
                unresolved_issue=self._mapped(raw_record, "field_mapping", "unresolved_issue"),
            ),
            metadata=StateMetadata(
                snapshot_id=str(snapshot),
                timestamp=self._mapped(raw_record, "field_mapping", "timestamp"),
                event_index=self._mapped(raw_record, "event_mapping", "event_index"),
                source_adapter=self.source_name,
            ),
        )

    def normalize_request(self, raw_record: Mapping[str, Any]) -> dict[str, Any]:
        """Return a normalized request-level record suitable for JSONL output."""

        state = self.build_state(raw_record)
        return {
            "sequence_id": self._mapped(raw_record, "field_mapping", "sequence_id"),
            "request_id": self._mapped(raw_record, "field_mapping", "request_id"),
            "timestamp": state.metadata.timestamp,
            "scene": state.to_dict()["scene"],
            "task": state.to_dict()["task"],
            "action": state.to_dict()["action"],
            "event": self._mapped(raw_record, "event_mapping", "current_event"),
            "dialogue": state.to_dict()["dialogue"],
            "runtime_state": state.to_dict(),
            "question": self._mapped(raw_record, "field_mapping", "question"),
        }
