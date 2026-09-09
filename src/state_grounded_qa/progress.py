"""Task progress, milestone, and legal-action construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .schemas import ProgressStatus, TaskState


Predicate = Callable[[Mapping[str, Any]], bool]


def _path_value(context: Mapping[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def evaluate_predicate(rule: Mapping[str, Any] | Predicate | None, context: Mapping[str, Any]) -> bool:
    """Evaluate a small, auditable completion/precondition rule."""

    if rule is None:
        return False
    if callable(rule):
        return bool(rule(context))
    path = str(rule.get("path", ""))
    actual = _path_value(context, path)
    if "equals" in rule:
        return actual == rule["equals"]
    if "in" in rule:
        return actual in rule["in"]
    if rule.get("exists") is True:
        return actual is not None
    if rule.get("truthy") is True:
        return bool(actual)
    return False


@dataclass(slots=True)
class TaskNode:
    """A task-graph node and its progress/action constraints."""

    node_id: str
    predecessors: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    acceptable_next_actions: list[str] = field(default_factory=list)
    preconditions: dict[str, Mapping[str, Any] | Predicate] = field(default_factory=dict)
    completion_rule: Mapping[str, Any] | Predicate | None = None
    evidence_ids: list[str] = field(default_factory=list)
    active_if: Mapping[str, Any] | Predicate | None = None

    def is_active(self, context: Mapping[str, Any]) -> bool:
        """Return whether this node belongs to the active task branch."""

        if self.active_if is None:
            return True
        return evaluate_predicate(self.active_if, context)


@dataclass(slots=True)
class ProgressResult:
    """Derived task state plus legal and acceptable action sets."""

    task: TaskState
    legal_actions: list[str]
    acceptable_next_actions: list[str]


class TaskGraph:
    """Directed acyclic task graph with deterministic request-level progress."""

    def __init__(self, nodes: Iterable[TaskNode], goal: str | None = None) -> None:
        node_list = list(nodes)
        self.nodes = {node.node_id: node for node in node_list}
        self.goal = goal
        if len(self.nodes) != len(node_list):
            raise ValueError("Task node identifiers must be unique")
        self._order = self._topological_order()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskGraph":
        """Load a graph from the repository task-graph schema."""

        nodes = [TaskNode(**dict(node)) for node in value.get("nodes", [])]
        return cls(nodes, goal=value.get("goal"))

    def _topological_order(self) -> list[str]:
        missing = {
            predecessor
            for node in self.nodes.values()
            for predecessor in node.predecessors
            if predecessor not in self.nodes
        }
        if missing:
            raise ValueError(f"Unknown predecessor nodes: {sorted(missing)}")
        indegree = {node_id: 0 for node_id in self.nodes}
        successors: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for node in self.nodes.values():
            for predecessor in node.predecessors:
                indegree[node.node_id] += 1
                successors[predecessor].append(node.node_id)
        queue = [node_id for node_id in self.nodes if indegree[node_id] == 0]
        ordered: list[str] = []
        while queue:
            node_id = queue.pop(0)
            ordered.append(node_id)
            for successor in successors[node_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        if len(ordered) != len(self.nodes):
            raise ValueError("Task graph must be acyclic")
        return ordered

    def active_order(self, context: Mapping[str, Any]) -> list[str]:
        """Return topological order restricted to the active branch."""

        return [node_id for node_id in self._order if self.nodes[node_id].is_active(context)]

    def completed_steps(
        self, context: Mapping[str, Any], completed_hint: Iterable[str] = ()
    ) -> list[str]:
        """Derive completed nodes from authoritative facts and validated hints."""

        active = self.active_order(context)
        hinted = {node_id for node_id in completed_hint if node_id in active}
        completed: set[str] = set()
        changed = True
        while changed:
            changed = False
            for node_id in active:
                node = self.nodes[node_id]
                if node_id in completed:
                    continue
                if all(pred in completed or pred not in active for pred in node.predecessors):
                    if node_id in hinted or evaluate_predicate(node.completion_rule, context):
                        completed.add(node_id)
                        changed = True
        return [node_id for node_id in active if node_id in completed]

    @property
    def topological_order(self) -> list[str]:
        """Return a copy of the graph's deterministic topological order."""

        return list(self._order)

    def current_step(self, active: list[str], completed: list[str]) -> str | None:
        """Return CurrentStep() under predecessor constraints."""

        completed_set = set(completed)
        for node_id in active:
            if node_id in completed_set:
                continue
            predecessors = [pred for pred in self.nodes[node_id].predecessors if pred in active]
            if all(pred in completed_set for pred in predecessors):
                return node_id
        return None

    def derive(
        self, context: Mapping[str, Any], completed_hint: Iterable[str] = (), started: bool | None = None
    ) -> ProgressResult:
        """Derive Section 3.4 progress and A_t_legal from the current state."""

        active = self.active_order(context)
        completed = self.completed_steps(context, completed_hint)
        current = self.current_step(active, completed)
        if active and len(completed) == len(active):
            status: ProgressStatus = "completed"
            current = None
            remaining: list[str] = []
        else:
            inferred_started = bool(completed) if started is None else started
            status = "in_progress" if inferred_started else "not_started"
            remaining = [node_id for node_id in active if node_id not in completed and node_id != current]
        denominator = len(completed) + (1 if current is not None else 0) + len(remaining)
        milestone = len(completed) / denominator if denominator else 1.0
        task = TaskState(
            goal=self.goal,
            current_step=current,
            progress_status=status,
            completed_steps=completed,
            remaining_steps=remaining,
            milestone_progress=milestone,
        )
        legal = self.legal_actions(current, context)
        acceptable = self.acceptable_next_actions(current, legal)
        return ProgressResult(task=task, legal_actions=legal, acceptable_next_actions=acceptable)

    def legal_actions(self, current_step: str | None, context: Mapping[str, Any]) -> list[str]:
        """Construct A_t_legal using node permissions and action preconditions."""

        if current_step is None:
            return []
        node = self.nodes[current_step]
        legal: list[str] = []
        for action_id in node.allowed_actions:
            rule = node.preconditions.get(action_id)
            if rule is None or evaluate_predicate(rule, context):
                legal.append(action_id)
        return legal

    def acceptable_next_actions(self, current_step: str | None, legal: list[str]) -> list[str]:
        """Return the narrower evaluation set of acceptable next actions."""

        if current_step is None:
            return []
        preferred = self.nodes[current_step].acceptable_next_actions
        return [action_id for action_id in preferred if action_id in legal]


def CurrentStep(graph: TaskGraph, context: Mapping[str, Any], completed: Iterable[str] = ()) -> str | None:
    """Paper-style functional entry point for current-step construction."""

    active = graph.active_order(context)
    completed_steps = graph.completed_steps(context, completed)
    return graph.current_step(active, completed_steps)


def construct_legal_actions(
    graph: TaskGraph, current_step: str | None, context: Mapping[str, Any]
) -> list[str]:
    """Paper-style functional entry point for A_t_legal."""

    return graph.legal_actions(current_step, context)
