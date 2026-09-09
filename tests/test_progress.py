"""Task graph, active-branch, and completion semantics."""

from __future__ import annotations

from state_grounded_qa.progress import TaskGraph, TaskNode


def make_graph() -> TaskGraph:
    return TaskGraph(
        [
            TaskNode("step_1", allowed_actions=["inspect"], completion_rule={"path": "done.step_1", "truthy": True}),
            TaskNode(
                "step_2",
                predecessors=["step_1"],
                allowed_actions=["attach", "inspect"],
                acceptable_next_actions=["attach"],
                preconditions={"attach": {"path": "ready", "equals": True}},
                completion_rule={"path": "done.step_2", "truthy": True},
                active_if={"path": "branch", "equals": "primary"},
            ),
            TaskNode(
                "step_alternate",
                predecessors=["step_1"],
                allowed_actions=["alternate"],
                active_if={"path": "branch", "equals": "alternate"},
            ),
        ],
        goal="Finish the task",
    )


def test_completed_task_has_no_current_or_remaining_steps() -> None:
    result = make_graph().derive(
        {"branch": "primary", "ready": True, "done": {"step_1": True, "step_2": True}}
    )
    assert result.task.progress_status == "completed"
    assert result.task.current_step is None
    assert result.task.remaining_steps == []


def test_remaining_excludes_current_and_inactive_branch() -> None:
    result = make_graph().derive(
        {"branch": "primary", "ready": True, "done": {"step_1": True, "step_2": False}}
    )
    assert result.task.current_step == "step_2"
    assert "step_2" not in result.task.remaining_steps
    assert "step_alternate" not in result.task.remaining_steps
    assert result.legal_actions == ["attach", "inspect"]
    assert result.acceptable_next_actions == ["attach"]
