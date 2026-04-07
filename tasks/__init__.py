"""
PatchCascade SOC - Task Definitions
=====================================

This package defines the task registry for the PatchCascade SOC environment.
Each task maps to a difficulty level with an associated multi-dimensional grader.

Task Curriculum (5 levels, progressive difficulty):

    1. Easy (difficulty 1):
       3-5 nodes, no dependencies, 1 vulnerability.
       Teaches basic patch mechanics.

    2. Medium (difficulty 2):
       5-8 nodes, linear dependency chain, 2 vulnerabilities.
       Introduces dependency awareness and suspend-patch-resume workflow.

    3. Hard (difficulty 3):
       10-15 nodes, complex dependency graph, multiple critical vulns.
       Tests multi-objective optimization with exploit prioritization.

    4. Incident Response (difficulty 4):
       8 nodes (2 pre-crashed), active breach, exploit spreading.
       Tests triage, recovery-under-pressure, and threat containment.

    5. Zero-Day Cascade (difficulty 5):
       10 nodes, dynamic CVE injection at turns 5 and 15.
       Tests adaptive planning and strategy revision under uncertainty.

Required by OpenEnv hackathon validation: at least 3 tasks with graders.
We provide 5 graders for comprehensive evaluation.

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

from tasks.easy import EASY_TASK
from tasks.medium import MEDIUM_TASK
from tasks.hard import HARD_TASK
from tasks.incident_response import INCIDENT_RESPONSE_TASK
from tasks.zero_day import ZERO_DAY_TASK


# =============================================================================
# TASK REGISTRY — All tasks with their graders (5 total)
# =============================================================================

TASK_REGISTRY: dict[str, dict] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
    "incident_response": INCIDENT_RESPONSE_TASK,
    "zero_day": ZERO_DAY_TASK,
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
    "INCIDENT_RESPONSE_TASK",
    "ZERO_DAY_TASK",
    "get_task",
    "list_tasks",
    "list_tasks_with_graders",
]
