"""
PatchCascade SOC - Programmatic Graders
=========================================

This module implements programmatic graders for each task (easy, medium, hard).
Each grader is a callable class that evaluates agent performance and returns
a normalized score (0.0 - 1.0).

Required by OpenEnv hackathon validation: at least 3 tasks with graders.

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# GRADER RESULT
# =============================================================================

@dataclass
class GraderResult:
    """Result returned by a grader after evaluating an episode."""
    task_id: str
    score: float  # Normalized 0.0 - 1.0
    passed: bool
    success_threshold: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "score": round(self.score, 4),
            "passed": self.passed,
            "success_threshold": self.success_threshold,
            "details": self.details,
        }


# =============================================================================
# BASE GRADER
# =============================================================================

class TaskGrader:
    """
    Base class for programmatic graders.

    Each grader evaluates an episode by consuming the episode's reward history,
    final state, and action log, then computes a normalized score in [0.0, 1.0].

    Subclasses MUST override:
        - task_id: str
        - task_name: str
        - success_threshold: float
        - grade(episode_data: dict) -> GraderResult
    """
    task_id: str = ""
    task_name: str = ""
    description: str = ""
    success_threshold: float = 0.5
    min_reward: float = -300.0
    max_reward: float = 50.0

    def grade(self, episode_data: dict[str, Any]) -> GraderResult:
        """
        Grade an episode.

        Args:
            episode_data: Dict containing at minimum:
                - total_reward (float): Cumulative reward over the episode
                - rewards (list[float]): Per-step rewards
                - success (bool): Whether the agent achieved the goal
                - steps (int): Number of steps taken
                - state (dict, optional): Final environment state

        Returns:
            GraderResult with normalized score and pass/fail.
        """
        total_reward = episode_data.get("total_reward", 0.0)
        rewards = episode_data.get("rewards", [])
        success = episode_data.get("success", False)
        steps = episode_data.get("steps", 0)

        # If rewards list is provided but total_reward is not, compute it
        if rewards and total_reward == 0.0:
            total_reward = sum(rewards)

        # Compute normalized score
        score = self._normalize_reward(total_reward)

        # Check pass criteria
        passed = score >= self.success_threshold and self._check_success_criteria(episode_data)

        return GraderResult(
            task_id=self.task_id,
            score=score,
            passed=passed,
            success_threshold=self.success_threshold,
            details={
                "raw_total_reward": total_reward,
                "normalized_score": round(score, 4),
                "steps_taken": steps,
                "episode_success": success,
                "grader_type": "programmatic",
                "min_reward": self.min_reward,
                "max_reward": self.max_reward,
                "formula": "(total_reward - min_reward) / (max_reward - min_reward)",
            },
        )

    def _normalize_reward(self, total_reward: float) -> float:
        """Normalize reward to [0.0, 1.0] range."""
        if self.max_reward == self.min_reward:
            return 0.5
        score = (total_reward - self.min_reward) / (self.max_reward - self.min_reward)
        return max(0.0, min(1.0, score))

    def _check_success_criteria(self, episode_data: dict[str, Any]) -> bool:
        """Check task-specific success criteria. Override in subclasses."""
        return episode_data.get("success", False)

    def to_dict(self) -> dict:
        """Serialize grader configuration for API responses."""
        return {
            "type": "programmatic",
            "module": "grader",
            "function": f"{self.__class__.__name__}.grade",
            "description": f"Programmatic grader for {self.task_name}",
            "success_threshold": self.success_threshold,
            "scoring": {
                "method": "normalized_reward",
                "min_reward": self.min_reward,
                "max_reward": self.max_reward,
                "formula": "(total_reward - min_reward) / (max_reward - min_reward)",
            },
        }


# =============================================================================
# TASK-SPECIFIC GRADERS
# =============================================================================

class EasyGrader(TaskGrader):
    """
    Grader for Easy task: 3-5 nodes, no dependencies, 1 vulnerability.

    Success Criteria:
        - All vulnerabilities must be patched
        - No catastrophic failures (all nodes crashed)
        - Normalized score >= 0.5
    """
    task_id = "easy"
    task_name = "Easy Mode"
    description = "Grades easy-mode episodes based on normalized cumulative reward"
    success_threshold = 0.5
    min_reward = -300.0
    max_reward = 50.0

    def _check_success_criteria(self, episode_data: dict[str, Any]) -> bool:
        state = episode_data.get("state", {})
        # Check all vulnerabilities patched
        vulns = state.get("vulnerabilities", episode_data.get("vulnerabilities", []))
        all_patched = len(vulns) == 0 or episode_data.get("success", False)
        # Check no catastrophic failure
        nodes = state.get("nodes", [])
        if nodes:
            all_crashed = all(n.get("state") == "crashed" for n in nodes)
        else:
            all_crashed = False
        no_catastrophe = not all_crashed
        return all_patched and no_catastrophe


class MediumGrader(TaskGrader):
    """
    Grader for Medium task: 5-8 nodes, linear dependency chain, 2 vulnerabilities.

    Success Criteria:
        - All vulnerabilities must be patched
        - No catastrophic failures
        - Dependencies must be respected (no avoidable cascade failures)
        - Normalized score >= 0.6
    """
    task_id = "medium"
    task_name = "Medium Mode"
    description = "Grades medium-mode episodes based on normalized cumulative reward"
    success_threshold = 0.6
    min_reward = -300.0
    max_reward = 50.0

    def _check_success_criteria(self, episode_data: dict[str, Any]) -> bool:
        state = episode_data.get("state", {})
        vulns = state.get("vulnerabilities", episode_data.get("vulnerabilities", []))
        all_patched = len(vulns) == 0 or episode_data.get("success", False)
        nodes = state.get("nodes", [])
        if nodes:
            all_crashed = all(n.get("state") == "crashed" for n in nodes)
        else:
            all_crashed = False
        no_catastrophe = not all_crashed
        return all_patched and no_catastrophe


class HardGrader(TaskGrader):
    """
    Grader for Hard task: 10-15 nodes, complex dependency graph, multiple critical vulns.

    Success Criteria:
        - All vulnerabilities must be patched
        - No catastrophic failures
        - Dependencies must be respected
        - Minimize downtime (fewer turns with nodes offline)
        - Normalized score >= 0.7
    """
    task_id = "hard"
    task_name = "Hard Mode"
    description = "Grades hard-mode episodes based on normalized cumulative reward"
    success_threshold = 0.7
    min_reward = -300.0
    max_reward = 50.0

    def _check_success_criteria(self, episode_data: dict[str, Any]) -> bool:
        state = episode_data.get("state", {})
        vulns = state.get("vulnerabilities", episode_data.get("vulnerabilities", []))
        all_patched = len(vulns) == 0 or episode_data.get("success", False)
        nodes = state.get("nodes", [])
        if nodes:
            all_crashed = all(n.get("state") == "crashed" for n in nodes)
        else:
            all_crashed = False
        no_catastrophe = not all_crashed
        return all_patched and no_catastrophe


# =============================================================================
# GRADER REGISTRY — Maps task IDs to grader instances
# =============================================================================

GRADERS: dict[str, TaskGrader] = {
    "easy": EasyGrader(),
    "medium": MediumGrader(),
    "hard": HardGrader(),
}


def get_grader(task_id: str) -> TaskGrader:
    """Get a grader instance by task ID."""
    if task_id not in GRADERS:
        raise ValueError(f"No grader found for task_id='{task_id}'. Available: {list(GRADERS.keys())}")
    return GRADERS[task_id]


def grade_episode(task_id: str, episode_data: dict[str, Any]) -> GraderResult:
    """
    Grade an episode for a specific task.

    This is the main entry point for programmatic grading.

    Args:
        task_id: One of 'easy', 'medium', 'hard'
        episode_data: Episode results dict

    Returns:
        GraderResult with normalized score and pass/fail determination.
    """
    grader = get_grader(task_id)
    return grader.grade(episode_data)


def list_graders() -> list[dict]:
    """List all available graders with their configurations."""
    return [
        {
            "task_id": grader.task_id,
            "task_name": grader.task_name,
            "grader_type": "programmatic",
            "module": "grader",
            "class": grader.__class__.__name__,
            "success_threshold": grader.success_threshold,
            "has_grader": True,
        }
        for grader in GRADERS.values()
    ]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TaskGrader",
    "EasyGrader",
    "MediumGrader",
    "HardGrader",
    "GraderResult",
    "GRADERS",
    "get_grader",
    "grade_episode",
    "list_graders",
]
