"""Adapter for MSQA-style scene-grounding records from MSR3D."""

from __future__ import annotations

from typing import Any, Mapping

from ..schemas import ActionState, DialogueState, RuntimeState, SceneState, StateMetadata, TaskState
from ..state_adapter import StateAdapter


class MSQAAdapter(StateAdapter):
    """Map scene, object, relation, and QA fields without adding task labels."""

    source_name = "msqa"

    def build_state(self, raw_record: Mapping[str, Any]) -> RuntimeState:
        """Construct scene-only S_t; question and references remain outside state."""

        scene_source = raw_record.get("scene", raw_record)
        objects = scene_source.get("objects", raw_record.get("objects", []))
        normalized_objects: list[dict[str, Any]] = []
        for item in objects or []:
            if not isinstance(item, Mapping):
                continue
            normalized = dict(item)
            identifier = self.normalize_identifier(item.get("object_id", item.get("id")))
            if identifier is not None:
                normalized["object_id"] = identifier
            normalized_objects.append(normalized)
        relations = scene_source.get("object_relations", scene_source.get("relations", [])) or []
        normalized_relations = []
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            normalized_relations.append(
                {
                    "subject": self.normalize_identifier(relation.get("subject")) or "",
                    "relation": self.normalize_identifier(relation.get("relation", relation.get("predicate"))) or "",
                    "object": self.normalize_identifier(relation.get("object")) or "",
                }
            )
        metadata = self.metadata_without_references(raw_record)
        snapshot_id = str(
            metadata.get("snapshot_id")
            or raw_record.get("question_id")
            or raw_record.get("request_id")
            or "msqa_request"
        )
        return RuntimeState(
            scene=SceneState(
                scene_id=self.normalize_identifier(scene_source.get("scene_id", raw_record.get("scene_id"))),
                location=self.normalize_identifier(scene_source.get("location")),
                active_object=self.normalize_identifier(scene_source.get("active_object")),
                object_relations=normalized_relations,
                objects=normalized_objects,
            ),
            task=TaskState(),
            action=ActionState(),
            dialogue=DialogueState(dialogue_history=list(raw_record.get("dialogue_history", []))),
            metadata=StateMetadata(
                snapshot_id=snapshot_id,
                timestamp=metadata.get("timestamp"),
                event_index=metadata.get("event_index"),
                source_adapter=self.source_name,
            ),
        )

    def adapt_record(self, raw_record: Mapping[str, Any]) -> dict[str, Any]:
        """Keep evaluation references separate from the runtime state."""

        references = {
            key: raw_record[key]
            for key in ("answer", "answers", "reference", "reference_information", "category")
            if key in raw_record
        }
        return {
            "runtime_state": self.build_state(raw_record).to_dict(),
            "question": raw_record.get("question"),
            "references": references,
        }
