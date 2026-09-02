"""
PatchCascade SOC - Multi-Dimensional Programmatic Graders
==========================================================

This module implements programmatic graders for each task (easy, medium, hard,
incident_response, zero_day). Each grader evaluates agent performance across
FOUR scoring dimensions, producing a normalized composite score in [0.0, 1.0].

Scoring Dimensions:
    1. Completion (40%): Were all vulnerabilities patched?
    2. Efficiency (20%): Steps taken vs. theoretical optimum
    3. Safety (20%): Cascade failures and catastrophic events avoided
    4. Strategy (20%): Correct dependency ordering, exploit prioritization

This design enables nuanced evaluation:
- An agent that patches everything but slowly scores ~0.7
- An agent that patches everything optimally scores ~1.0
- An agent that causes cascades but recovers scores ~0.4
- A random agent scores ~0.1-0.2

OpenEnv protocol validation requires at least 3 tasks with graders.
We provide 5 graders for comprehensive evaluation.

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# SCORING WEIGHTS — Configurable per-task dimension weights
# =============================================================================

@dataclass(frozen=True)
class ScoringWeights:
    """
    Dimension weights for multi-dimensional scoring.

    All weights must sum to 1.0. Each weight controls how much
    that dimension contributes to the final composite score.
    """
    completion: float = 0.40
    efficiency: float = 0.20
    safety: float = 0.20
    strategy: float = 0.20

    def __post_init__(self):
        total = self.completion + self.efficiency + self.safety + self.strategy
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# Preset weight configurations for different task types
STANDARD_WEIGHTS = ScoringWeights(0.40, 0.20, 0.20, 0.20)
SAFETY_FOCUSED_WEIGHTS = ScoringWeights(0.30, 0.15, 0.35, 0.20)
EFFICIENCY_FOCUSED_WEIGHTS = ScoringWeights(0.35, 0.30, 0.15, 0.20)


# =============================================================================
# GRADER RESULT
# =============================================================================

@dataclass
class GraderResult:
    """
    Result returned by a grader after evaluating an episode.

    Contains both the composite score and detailed per-dimension breakdowns,
    enabling transparent evaluation and agent debugging.
    """
    task_id: str
    score: float  # Normalized composite score in (0.0, 1.0)
    passed: bool
    success_threshold: float
    dimensions: dict = field(default_factory=dict)  # Per-dimension scores
    details: dict = field(default_factory=dict)      # Additional metadata

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "score": round(self.score, 4),
            "passed": self.passed,
            "success_threshold": self.success_threshold,
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "details": self.details,
        }


# =============================================================================
# BASE GRADER — Multi-Dimensional Scoring Engine
# =============================================================================

class TaskGrader:
    """
    Multi-dimensional programmatic grader for PatchCascade SOC tasks.

    Evaluates agent performance across four orthogonal dimensions:

    1. COMPLETION (default 40%):
       Measures how many vulnerabilities were successfully patched.
       Full credit requires patching ALL vulnerabilities.
       Partial credit proportional to fraction patched.

    2. EFFICIENCY (default 20%):
       Measures how quickly the agent completed the task relative to
       the theoretical minimum steps. Rewards concise action sequences.
       Score = max(0, 1 - (steps_taken - optimal) / max_turns)

    3. SAFETY (default 20%):
       Penalizes cascade failures and catastrophic events.
       Full credit if no nodes crashed due to dependency violations.
       Partial credit reduced by (cascade_count / total_nodes).

    4. STRATEGY (default 20%):
       Measures intelligent decision-making:
       - Did the agent prioritize exploit_in_wild CVEs?
       - Did the agent suspend dependents before dependencies?
       - Did the agent avoid wasting turns on invalid actions?

    Composite: score = w1*completion + w2*efficiency + w3*safety + w4*strategy
    """
    task_id: str = ""
    task_name: str = ""
    description: str = ""
    success_threshold: float = 0.5
    min_reward: float = -300.0
    max_reward: float = 50.0
    optimal_steps: int = 5  # Theoretical minimum steps for this task
    weights: ScoringWeights = STANDARD_WEIGHTS

    def grade(self, episode_data: dict[str, Any]) -> GraderResult:
        """
        Grade an episode using multi-dimensional scoring.

        Args:
            episode_data: Dict containing:
                - total_reward (float): Cumulative reward
                - rewards (list[float]): Per-step rewards
                - success (bool): Whether agent achieved the goal
                - steps (int): Number of steps taken
                - state (dict): Final environment state
                - cascade_failures (int): Total cascade crashes
                - invalid_actions (int): Count of invalid actions
                - exploited_patched_first (bool): Whether exploited CVEs were prioritized
                - correct_suspend_order (bool): Whether dependencies were respected

        Returns:
            GraderResult with composite score and per-dimension breakdowns.
        """
        # Extract episode data with safe defaults
        total_reward = episode_data.get("total_reward", 0.0)
        rewards = episode_data.get("rewards", [])
        success = episode_data.get("success", False)
        steps = episode_data.get("steps", 0)
        state = episode_data.get("state", {})
        cascade_failures = episode_data.get("cascade_failures", 0)
        invalid_actions = episode_data.get("invalid_actions", 0)

        # Compute total reward from list if not provided
        if rewards and total_reward == 0.0:
            total_reward = sum(rewards)

        # ---- Dimension 1: Completion Score ----
        completion = self._score_completion(episode_data, state, success)

        # ---- Dimension 2: Efficiency Score ----
        efficiency = self._score_efficiency(steps, success)

        # ---- Dimension 3: Safety Score ----
        safety = self._score_safety(state, cascade_failures)

        # ---- Dimension 4: Strategy Score ----
        strategy = self._score_strategy(episode_data, invalid_actions)

        # ---- Composite Score ----
        w = self.weights
        composite = (
            w.completion * completion
            + w.efficiency * efficiency
            + w.safety * safety
            + w.strategy * strategy
        )

        # Clamp to open interval (0, 1) — OpenEnv result-contract requirement
        composite = max(0.001, min(0.999, composite))

        # Determine pass/fail
        passed = composite >= self.success_threshold and success

        # Build dimension breakdown
        dimensions = {
            "completion": completion,
            "efficiency": efficiency,
            "safety": safety,
            "strategy": strategy,
        }

        # Build detailed metadata
        details = {
            "raw_total_reward": total_reward,
            "normalized_reward": round(self._normalize_reward(total_reward), 4),
            "composite_score": round(composite, 4),
            "steps_taken": steps,
            "optimal_steps": self.optimal_steps,
            "cascade_failures": cascade_failures,
            "invalid_actions": invalid_actions,
            "episode_success": success,
            "grader_type": "programmatic_multidimensional",
            "weights": {
                "completion": w.completion,
                "efficiency": w.efficiency,
                "safety": w.safety,
                "strategy": w.strategy,
            },
            "scoring_formula": (
                f"score = {w.completion}*completion + {w.efficiency}*efficiency "
                f"+ {w.safety}*safety + {w.strategy}*strategy"
            ),
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
        }

        return GraderResult(
            task_id=self.task_id,
            score=composite,
            passed=passed,
            success_threshold=self.success_threshold,
            dimensions=dimensions,
            details=details,
        )

    # -------------------------------------------------------------------------
    # Scoring Dimension Implementations
    # -------------------------------------------------------------------------

    def _score_completion(
        self, episode_data: dict, state: dict, success: bool
    ) -> float:
        """
        Score based on vulnerability completion rate.

        Returns 1.0 if all vulnerabilities patched, otherwise
        proportional to fraction completed.
        """
        if success:
            return 1.0

        # Check remaining vulnerabilities in final state
        vulns = state.get("vulnerabilities", episode_data.get("vulnerabilities", []))
        initial_vulns = episode_data.get("initial_vulnerability_count", None)

        if initial_vulns and initial_vulns > 0:
            remaining = len(vulns)
            return max(0.0, 1.0 - remaining / initial_vulns)

        # Fallback: use reward-based estimation
        total_reward = episode_data.get("total_reward", sum(episode_data.get("rewards", [0])))
        return max(0.0, min(1.0, self._normalize_reward(total_reward)))

    def _score_efficiency(self, steps: int, success: bool) -> float:
        """
        Score based on step efficiency relative to theoretical optimum.

        An agent that completes in optimal_steps gets 1.0.
        Score degrades linearly as steps increase beyond optimal.
        Non-completing agents get a small base score.
        """
        if not success or steps == 0:
            return 0.1  # Small base score for attempting

        if steps <= self.optimal_steps:
            return 1.0

        # Linear decay from optimal to max allowed
        max_steps = self.optimal_steps * 5  # Beyond 5x optimal is very poor
        overshoot = steps - self.optimal_steps
        max_overshoot = max_steps - self.optimal_steps

        score = max(0.1, 1.0 - (overshoot / max_overshoot))
        return min(1.0, score)

    def _score_safety(self, state: dict, cascade_failures: int) -> float:
        """
        Score based on avoiding cascade failures and catastrophic events.

        Full credit (1.0) for zero cascades.
        Penalty proportional to cascades relative to total nodes.
        """
        nodes = state.get("nodes", [])
        total_nodes = len(nodes) if nodes else 6  # Default estimate

        # Check for catastrophic failure (all crashed)
        if nodes and all(n.get("state") == "crashed" for n in nodes):
            return 0.0

        if cascade_failures == 0:
            return 1.0

        # Proportional penalty: each cascade reduces score
        penalty = cascade_failures / max(total_nodes, 1)
        return max(0.0, 1.0 - penalty)

    def _score_strategy(
        self, episode_data: dict, invalid_actions: int
    ) -> float:
        """
        Score based on strategic decision-making quality.

        Factors:
        - Exploit prioritization: Did agent patch exploit_in_wild CVEs first?
        - Dependency ordering: Did agent suspend dependents before dependencies?
        - Action validity: Percentage of valid actions (no wasted turns)
        """
        score = 0.5  # Base score

        # Exploit prioritization bonus
        if episode_data.get("exploited_patched_first", False):
            score += 0.25

        # Correct suspend ordering bonus
        if episode_data.get("correct_suspend_order", True):
            score += 0.15

        # Invalid action penalty
        steps = episode_data.get("steps", 1)
        if steps > 0:
            valid_ratio = max(0, 1.0 - invalid_actions / steps)
            score += 0.10 * valid_ratio

        return max(0.0, min(1.0, score))

    # -------------------------------------------------------------------------
    # Reward Normalization (backward compatibility)
    # -------------------------------------------------------------------------

    def _normalize_reward(self, total_reward: float) -> float:
        """Normalize reward to strict (0, 1) range — never exactly 0.0 or 1.0."""
        if self.max_reward == self.min_reward:
            return 0.5
        score = (total_reward - self.min_reward) / (self.max_reward - self.min_reward)
        return max(0.001, min(0.999, score))

    def _check_success_criteria(self, episode_data: dict[str, Any]) -> bool:
        """Check task-specific success criteria. Override in subclasses."""
        return episode_data.get("success", False)

    def to_dict(self) -> dict:
        """Serialize grader configuration for API responses."""
        return {
            "type": "programmatic",
            "module": "grader",
            "function": f"{self.__class__.__name__}.grade",
            "description": f"Multi-dimensional programmatic grader for {self.task_name}",
            "success_threshold": self.success_threshold,
            "scoring": {
                "method": "multi_dimensional",
                "dimensions": ["completion", "efficiency", "safety", "strategy"],
                "weights": {
                    "completion": self.weights.completion,
                    "efficiency": self.weights.efficiency,
                    "safety": self.weights.safety,
                    "strategy": self.weights.strategy,
                },
                "formula": (
                    "score = w_completion * completion + w_efficiency * efficiency "
                    "+ w_safety * safety + w_strategy * strategy"
                ),
                "min_reward": self.min_reward,
                "max_reward": self.max_reward,
            },
        }


# =============================================================================
# TASK-SPECIFIC GRADERS
# =============================================================================


class EasyGrader(TaskGrader):
    """
    Grader for Easy task: 3-5 nodes, no dependencies, 1 vulnerability.

    Evaluation Focus:
        - Completion (40%): Patch the single vulnerability
        - Efficiency (20%): Should complete in ~3 steps (scan + patch + done)
        - Safety (20%): No cascades expected (no dependencies)
        - Strategy (20%): Basic action validity

    Success Criteria:
        - All vulnerabilities patched
        - No catastrophic failures
        - Composite score >= 0.5
    """
    task_id = "easy"
    task_name = "Easy Mode"
    description = "Grades easy-mode episodes: basic patching without dependencies"
    success_threshold = 0.5
    optimal_steps = 3
    weights = STANDARD_WEIGHTS

    def _score_completion(self, episode_data, state, success):
        """Easy mode: binary completion (1 vuln to patch)."""
        if success:
            return 1.0

        vulns = state.get("vulnerabilities", episode_data.get("vulnerabilities", []))
        if len(vulns) == 0:
            return 1.0
        return 0.2  # Partial credit for attempting


class MediumGrader(TaskGrader):
    """
    Grader for Medium task: 5-8 nodes, linear dependency chain, 2 vulnerabilities.

    Evaluation Focus:
        - Completion (40%): Patch both vulnerabilities
        - Efficiency (20%): ~8 optimal steps (suspend chain + patch + resume)
        - Safety (20%): Avoid cascade from DB dependency
        - Strategy (20%): Correct suspend-before-patch on Tier 1

    Success Criteria:
        - All vulnerabilities patched
        - No catastrophic failures
        - Dependencies respected
        - Composite score >= 0.6
    """
    task_id = "medium"
    task_name = "Medium Mode"
    description = "Grades medium-mode episodes: dependency-aware patching"
    success_threshold = 0.6
    optimal_steps = 8
    weights = STANDARD_WEIGHTS

    def _score_strategy(self, episode_data, invalid_actions):
        """Medium mode: extra weight on dependency awareness."""
        base = super()._score_strategy(episode_data, invalid_actions)

        # Bonus for zero cascades AND completion
        if (episode_data.get("cascade_failures", 0) == 0 and
                episode_data.get("success", False)):
            base = min(1.0, base + 0.15)

        return base


class HardGrader(TaskGrader):
    """
    Grader for Hard task: 10-15 nodes, complex graph, multiple critical vulns.

    Evaluation Focus:
        - Completion (40%): Patch all 5 vulnerabilities across 13+ nodes
        - Efficiency (20%): ~18 optimal steps in complex graph
        - Safety (20%): Avoid multi-level cascade failures
        - Strategy (20%): Prioritize exploited CVEs, correct dependency ordering

    Success Criteria:
        - All vulnerabilities patched
        - No catastrophic failures
        - Dependencies respected
        - Minimal downtime
        - Composite score >= 0.7
    """
    task_id = "hard"
    task_name = "Hard Mode"
    description = "Grades hard-mode episodes: complex graph with exploited vulnerabilities"
    success_threshold = 0.7
    optimal_steps = 18
    weights = STANDARD_WEIGHTS

    def _score_strategy(self, episode_data, invalid_actions):
        """Hard mode: strict exploit prioritization scoring."""
        base = super()._score_strategy(episode_data, invalid_actions)

        # Penalize if exploited CVEs were NOT prioritized
        if not episode_data.get("exploited_patched_first", True):
            base = max(0.0, base - 0.2)

        return base


class IncidentResponseGrader(TaskGrader):
    """
    Grader for Incident Response task: active breach with pre-crashed nodes.

    Evaluation Focus:
        - Completion (30%): Patch vulns AND recover crashed nodes
        - Efficiency (15%): Speed is critical during active breach
        - Safety (35%): Prevent further cascade spread (safety-focused)
        - Strategy (20%): Triage correctly — recover before patching, isolate threats

    This task uses SAFETY_FOCUSED_WEIGHTS because the scenario involves
    an active breach where preventing further damage is paramount.

    Success Criteria:
        - All vulnerabilities patched
        - All crashed nodes recovered
        - No further cascade failures beyond initial breach
        - Composite score >= 0.5
    """
    task_id = "incident_response"
    task_name = "Incident Response"
    description = "Grades incident-response episodes: active breach triage and recovery"
    success_threshold = 0.5
    optimal_steps = 12
    weights = SAFETY_FOCUSED_WEIGHTS

    def _score_completion(self, episode_data, state, success):
        """IR mode: completion includes node recovery."""
        if success:
            return 1.0

        # Partial credit: 50% for patching, 50% for recovery
        vulns = state.get("vulnerabilities", episode_data.get("vulnerabilities", []))
        nodes = state.get("nodes", [])

        patch_score = 1.0 if len(vulns) == 0 else 0.3
        recovery_score = 1.0
        if nodes:
            crashed = sum(1 for n in nodes if n.get("state") == "crashed")
            if crashed > 0:
                recovery_score = max(0.0, 1.0 - crashed / len(nodes))

        return 0.5 * patch_score + 0.5 * recovery_score

    def _score_safety(self, state, cascade_failures):
        """IR mode: extra penalty for additional cascades beyond initial breach."""
        # Initial breach has 2 crashed nodes — only penalize NEW cascades
        initial_crashed = 2
        new_cascades = max(0, cascade_failures - initial_crashed)

        if new_cascades == 0:
            return 1.0

        nodes = state.get("nodes", [])
        total = len(nodes) if nodes else 8
        return max(0.0, 1.0 - (new_cascades * 1.5 / total))


class ZeroDayGrader(TaskGrader):
    """
    Grader for Zero-Day Cascade task: dynamic CVE injection mid-episode.

    Evaluation Focus:
        - Completion (35%): Patch both initial AND dynamically injected CVEs
        - Efficiency (30%): Must adapt quickly when new threats appear
        - Safety (15%): Manage cascades during strategy adaptation
        - Strategy (20%): Ability to reprioritize when new CVEs arrive

    This task uses EFFICIENCY_FOCUSED_WEIGHTS because the dynamic
    threat injection tests the agent's ability to adapt quickly.

    Success Criteria:
        - All vulnerabilities patched (including dynamically injected ones)
        - Strategy adapted when new CVEs appeared
        - Composite score >= 0.6
    """
    task_id = "zero_day"
    task_name = "Zero-Day Cascade"
    description = "Grades zero-day episodes: adaptive response to dynamic threats"
    success_threshold = 0.6
    optimal_steps = 15
    weights = EFFICIENCY_FOCUSED_WEIGHTS

    def _score_strategy(self, episode_data, invalid_actions):
        """Zero-day mode: reward fast adaptation to new threats."""
        base = super()._score_strategy(episode_data, invalid_actions)

        # Bonus if agent patched dynamic CVEs quickly after injection
        if episode_data.get("dynamic_cves_patched_promptly", False):
            base = min(1.0, base + 0.2)

        return base


# =============================================================================
# GRADER REGISTRY — Maps task IDs to grader instances
# =============================================================================

GRADERS: dict[str, TaskGrader] = {
    "easy": EasyGrader(),
    "medium": MediumGrader(),
    "hard": HardGrader(),
    "incident_response": IncidentResponseGrader(),
    "zero_day": ZeroDayGrader(),
}


def get_grader(task_id: str) -> TaskGrader:
    """Get a grader instance by task ID."""
    if task_id not in GRADERS:
        raise ValueError(f"No grader found for task_id='{task_id}'. Available: {list(GRADERS.keys())}")
    return GRADERS[task_id]


def grade_episode(task_id: str, episode_data: dict[str, Any]) -> GraderResult:
    """
    Grade an episode for a specific task using multi-dimensional scoring.

    This is the main entry point for programmatic grading. Each episode is
    evaluated across four dimensions (completion, efficiency, safety, strategy)
    and produces a weighted composite score.

    Args:
        task_id: One of 'easy', 'medium', 'hard', 'incident_response', 'zero_day'
        episode_data: Episode results dict

    Returns:
        GraderResult with composite score, per-dimension breakdowns, and metadata.
    """
    grader = get_grader(task_id)
    return grader.grade(episode_data)


def list_graders() -> list[dict]:
    """List all available graders with their configurations."""
    return [
        {
            "task_id": grader.task_id,
            "task_name": grader.task_name,
            "grader_type": "programmatic_multidimensional",
            "module": "grader",
            "class": grader.__class__.__name__,
            "success_threshold": grader.success_threshold,
            "has_grader": True,
            "dimensions": ["completion", "efficiency", "safety", "strategy"],
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
    "IncidentResponseGrader",
    "ZeroDayGrader",
    "GraderResult",
    "ScoringWeights",
    "GRADERS",
    "get_grader",
    "grade_episode",
    "list_graders",
    "STANDARD_WEIGHTS",
    "SAFETY_FOCUSED_WEIGHTS",
    "EFFICIENCY_FOCUSED_WEIGHTS",
]
