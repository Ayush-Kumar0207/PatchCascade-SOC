"""
PatchCascade SOC - Task Definitions
=====================================

This package defines the task registry for the PatchCascade SOC environment.
Each task maps to a difficulty level with an associated grader.

Tasks:
    - easy: 3-5 nodes, no dependencies, 1 vulnerability
    - medium: 5-8 nodes, linear dependency chain, 2 vulnerabilities
    - hard: 10-15 nodes, complex dependency graph, multiple critical vulns

Required by OpenEnv hackathon validation: at least 3 tasks with graders.

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

from tasks.easy import EASY_TASK
from tasks.medium import MEDIUM_TASK
from tasks.hard import HARD_TASK


# =============================================================================
# TASK REGISTRY — All tasks with their graders
# =============================================================================

TASK_REGISTRY: dict[str, dict] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}


def get_task(task_id: str) -> dict:
    """Get a task configuration by ID."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_id}' not found. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> list[dict]:
    """List all available tasks."""
    return list(TASK_REGISTRY.values())


def list_tasks_with_graders() -> list[dict]:
    """List all tasks that have graders attached."""
    return [t for t in TASK_REGISTRY.values() if t.get("has_grader", False)]


__all__ = [
    "TASK_REGISTRY",
    "EASY_TASK",
    "MEDIUM_TASK",
    "HARD_TASK",
    "get_task",
    "list_tasks",
    "list_tasks_with_graders",
]
