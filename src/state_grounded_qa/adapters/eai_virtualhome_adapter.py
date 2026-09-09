"""Past-only adapter for EAI-VirtualHome trajectories and task specifications."""

from __future__ import annotations

from typing import Any, Mapping

from ..progress import TaskGraph
from ..schemas import ActionState, DialogueState, RuntimeState, SceneState, StateMetadata
from ..state_adapter import StateAdapter


class EAIVirtualHomeAdapter(StateAdapter):
    """Build task progress from a trajectory prefix ending strictly before t."""

    source_name = "eai_virtualhome"

    def _past_prefix(self, raw_record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        trajectory = list(raw_record.get("trajectory", []))
        request_index = int(raw_record.get("request_index", raw_record.get("prefix_length", 0)))
        if request_index < 0 or request_index > len(trajectory):
            raise ValueError("request_index must identify a valid trajectory prefix")
        return [item for item in trajectory[:request_index] if isinstance(item, Mapping)]

    def build_state(self, raw_record: Mapping[str, Any]) -> RuntimeState:
        """Map only task specification, action rules, and the prefix before t."""

        prefix = self._past_prefix(raw_record)
        specification = raw_record.get("task_specification", {})
        graph = TaskGraph.from_dict(specification)
        context: dict[str, Any] = {"events": {}, "state": dict(raw_record.get("initial_state", {}))}
        completed_hint: list[str] = []
        for item in prefix:
            action = item.get("action_id", item.get("action"))
            if item.get("completed_step_id") is not None:
                completed_hint.append(str(item["completed_step_id"]))
            event = item.get("event")
            if isinstance(event, Mapping):
                event_type = str(event.get("event_type", event.get("type", "event")))
                context["events"][event_type] = event.get("value", True)
            update = item.get("state_update")
            if isinstance(update, Mapping):
                context["state"].update(update)
        progress = graph.derive(context, completed_hint=completed_hint, started=bool(prefix))
        scene = raw_record.get("scene", {})
        last_action = prefix[-1].get("action_id", prefix[-1].get("action")) if prefix else None
        request_index = int(raw_record.get("request_index", raw_record.get("prefix_length", 0)))
        return RuntimeState(
            scene=SceneState(
                scene_id=self.normalize_identifier(scene.get("scene_id")),
                location=self.normalize_identifier(scene.get("location")),
                active_object=self.normalize_identifier(scene.get("active_object")),
                object_relations=list(scene.get("object_relations", [])),
                objects=list(scene.get("objects", [])),
            ),
            task=progress.task,
            action=ActionState(
                available_actions=progress.legal_actions,
                preconditions={
                    action: graph.nodes[progress.task.current_step].preconditions.get(action, {})
                    for action in progress.legal_actions
                    if progress.task.current_step is not None
                },
                last_action=self.normalize_identifier(last_action),
                acceptable_next_actions=progress.acceptable_next_actions,
            ),
            dialogue=DialogueState(dialogue_history=list(raw_record.get("dialogue_history", []))),
            metadata=StateMetadata(
                snapshot_id=str(raw_record.get("request_id", f"eai_request_{request_index}")),
                timestamp=raw_record.get("timestamp"),
                event_index=request_index,
                source_adapter=self.source_name,
            ),
        )

    def adapt_record(self, raw_record: Mapping[str, Any]) -> dict[str, Any]:
        """Return runtime state and current labels while excluding future trajectory items."""

        state = self.build_state(raw_record)
        return {
            "runtime_state": state.to_dict(),
            "question": raw_record.get("question"),
            "reference_labels": dict(raw_record.get("reference_labels", {})),
        }
