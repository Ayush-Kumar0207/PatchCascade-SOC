"""
PatchCascade SOC - Grading Logic Tests
========================================

Tests for the multi-dimensional grading system, verifying that each
grader produces correct scores across all dimensions.

Tests verify:
- Composite score calculation
- Per-dimension scoring (completion, efficiency, safety, strategy)
- Weight profiles per task type
- Edge cases (zero rewards, max rewards, no steps)
- Grader registry and lookup

Author: PatchCascade SOC Team
License: Apache 2.0
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grader import (
    TaskGrader,
    EasyGrader,
    MediumGrader,
    HardGrader,
    IncidentResponseGrader,
    ZeroDayGrader,
    GraderResult,
    ScoringWeights,
    GRADERS,
    get_grader,
    grade_episode,
    list_graders,
    STANDARD_WEIGHTS,
    SAFETY_FOCUSED_WEIGHTS,
    EFFICIENCY_FOCUSED_WEIGHTS,
)


# =============================================================================
# SCORING WEIGHTS TESTS
# =============================================================================


class TestScoringWeights:
    """Test weight configuration validation."""

    def test_standard_weights_sum_to_one(self):
        """Standard weights should sum to 1.0."""
        w = STANDARD_WEIGHTS
        assert abs(w.completion + w.efficiency + w.safety + w.strategy - 1.0) < 0.01

    def test_safety_weights_sum_to_one(self):
        """Safety-focused weights should sum to 1.0."""
        w = SAFETY_FOCUSED_WEIGHTS
        assert abs(w.completion + w.efficiency + w.safety + w.strategy - 1.0) < 0.01

    def test_efficiency_weights_sum_to_one(self):
        """Efficiency-focused weights should sum to 1.0."""
        w = EFFICIENCY_FOCUSED_WEIGHTS
        assert abs(w.completion + w.efficiency + w.safety + w.strategy - 1.0) < 0.01

    def test_invalid_weights_raise_error(self):
        """Weights that don't sum to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringWeights(0.5, 0.5, 0.5, 0.5)

    def test_safety_weights_prioritize_safety(self):
        """Safety-focused weights should have safety as highest weight."""
        w = SAFETY_FOCUSED_WEIGHTS
        assert w.safety > w.efficiency
        assert w.safety > w.strategy

    def test_efficiency_weights_prioritize_efficiency(self):
        """Efficiency-focused weights should have efficiency elevated."""
        w = EFFICIENCY_FOCUSED_WEIGHTS
        assert w.efficiency > w.safety


# =============================================================================
# GRADER RESULT TESTS
# =============================================================================


class TestGraderResult:
    """Test GraderResult serialization."""

    def test_to_dict(self):
        """GraderResult.to_dict() should produce valid JSON-serializable dict."""
        result = GraderResult(
            task_id="easy",
            score=0.85,
            passed=True,
            success_threshold=0.5,
            dimensions={"completion": 1.0, "efficiency": 0.8, "safety": 1.0, "strategy": 0.7},
            details={"steps_taken": 5},
        )
        d = result.to_dict()
        assert d["task_id"] == "easy"
        assert d["score"] == 0.85
        assert d["passed"] is True
        assert "completion" in d["dimensions"]

    def test_score_rounding(self):
        """Scores should be rounded to 4 decimal places."""
        result = GraderResult(
            task_id="test",
            score=0.123456789,
            passed=True,
            success_threshold=0.1,
            dimensions={"completion": 0.999999},
        )
        d = result.to_dict()
        assert d["score"] == 0.1235
        assert d["dimensions"]["completion"] == 1.0


# =============================================================================
# GRADING LOGIC TESTS
# =============================================================================


class TestGradingLogic:
    """Test the core grading calculations."""

    def _make_perfect_episode(self) -> dict:
        """Create episode data for a perfect run."""
        return {
            "total_reward": 50.0,
            "rewards": [10.0, 15.0, 25.0],
            "success": True,
            "steps": 3,
            "state": {"vulnerabilities": [], "nodes": []},
            "cascade_failures": 0,
            "invalid_actions": 0,
            "exploited_patched_first": True,
            "correct_suspend_order": True,
        }

    def _make_poor_episode(self) -> dict:
        """Create episode data for a poor run."""
        return {
            "total_reward": -50.0,
            "rewards": [-10.0, -15.0, -25.0],
            "success": False,
            "steps": 50,
            "state": {
                "vulnerabilities": [
                    {"cve_id": "CVE-2024-1001", "affected_hosts": ["web-01"]},
                ],
                "nodes": [
                    {"hostname": "web-01", "state": "crashed"},
                    {"hostname": "db-01", "state": "crashed"},
                ],
            },
            "cascade_failures": 5,
            "invalid_actions": 10,
            "exploited_patched_first": False,
            "correct_suspend_order": False,
        }

    def test_perfect_run_scores_high(self):
        """A perfect episode should score > 0.9."""
        grader = EasyGrader()
        result = grader.grade(self._make_perfect_episode())
        assert result.score > 0.9
        assert result.passed is True

    def test_poor_run_scores_low(self):
        """A poor episode should score < 0.4."""
        grader = EasyGrader()
        result = grader.grade(self._make_poor_episode())
        assert result.score < 0.4
        assert result.passed is False

    def test_score_always_in_open_interval(self):
        """Score should always be in (0, 1), never exactly 0.0 or 1.0."""
        grader = EasyGrader()

        # Perfect run
        result1 = grader.grade(self._make_perfect_episode())
        assert 0.0 < result1.score < 1.0

        # Terrible run
        result2 = grader.grade(self._make_poor_episode())
        assert 0.0 < result2.score < 1.0

    def test_all_dimensions_present(self):
        """Grade result should include all 4 dimensions."""
        grader = MediumGrader()
        result = grader.grade(self._make_perfect_episode())
        assert "completion" in result.dimensions
        assert "efficiency" in result.dimensions
        assert "safety" in result.dimensions
        assert "strategy" in result.dimensions

    def test_completion_score_binary_for_success(self):
        """Completion should be 1.0 if episode was successful."""
        grader = EasyGrader()
        episode = self._make_perfect_episode()
        result = grader.grade(episode)
        assert result.dimensions["completion"] == 1.0

    def test_safety_score_penalizes_cascades(self):
        """Safety dimension should decrease with cascade failures."""
        grader = HardGrader()

        # No cascades
        ep_safe = self._make_perfect_episode()
        ep_safe["cascade_failures"] = 0
        ep_safe["state"]["nodes"] = [{"hostname": f"n{i}", "state": "online"} for i in range(10)]
        result_safe = grader.grade(ep_safe)

        # Many cascades
        ep_unsafe = self._make_perfect_episode()
        ep_unsafe["cascade_failures"] = 5
        ep_unsafe["state"]["nodes"] = [{"hostname": f"n{i}", "state": "online"} for i in range(10)]
        result_unsafe = grader.grade(ep_unsafe)

        assert result_safe.dimensions["safety"] > result_unsafe.dimensions["safety"]

    def test_efficiency_rewards_fewer_steps(self):
        """Efficiency should be higher for fewer steps."""
        grader = EasyGrader()  # optimal_steps = 3

        fast = self._make_perfect_episode()
        fast["steps"] = 3
        result_fast = grader.grade(fast)

        slow = self._make_perfect_episode()
        slow["steps"] = 20
        result_slow = grader.grade(slow)

        assert result_fast.dimensions["efficiency"] > result_slow.dimensions["efficiency"]


# =============================================================================
# TASK-SPECIFIC GRADER TESTS
# =============================================================================


class TestTaskGraders:
    """Test that each task grader uses correct weight profiles."""

    def test_easy_uses_standard_weights(self):
        """Easy grader should use standard weights."""
        grader = EasyGrader()
        assert grader.weights == STANDARD_WEIGHTS

    def test_ir_uses_safety_weights(self):
        """IR grader should use safety-focused weights."""
        grader = IncidentResponseGrader()
        assert grader.weights == SAFETY_FOCUSED_WEIGHTS

    def test_zd_uses_efficiency_weights(self):
        """Zero-day grader should use efficiency-focused weights."""
        grader = ZeroDayGrader()
        assert grader.weights == EFFICIENCY_FOCUSED_WEIGHTS

    def test_each_grader_has_unique_task_id(self):
        """All graders should have unique task IDs."""
        ids = [g.task_id for g in GRADERS.values()]
        assert len(ids) == len(set(ids))

    def test_grader_to_dict(self):
        """Grader serialization should produce valid dict."""
        grader = MediumGrader()
        d = grader.to_dict()
        assert d["type"] == "programmatic"
        assert d["module"] == "grader"
        assert d["success_threshold"] == 0.6
        assert "weights" in d["scoring"]


# =============================================================================
# REGISTRY TESTS
# =============================================================================


class TestGraderRegistry:
    """Test grader registry and lookup."""

    def test_five_graders_registered(self):
        """Registry should contain exactly 5 graders."""
        assert len(GRADERS) == 5

    def test_all_task_ids_present(self):
        """All expected task IDs should be in registry."""
        expected = {"easy", "medium", "hard", "incident_response", "zero_day"}
        assert set(GRADERS.keys()) == expected

    def test_get_grader_valid(self):
        """get_grader should return correct grader for valid ID."""
        grader = get_grader("medium")
        assert isinstance(grader, MediumGrader)

    def test_get_grader_invalid_raises(self):
        """get_grader should raise ValueError for invalid ID."""
        with pytest.raises(ValueError, match="No grader found"):
            get_grader("nonexistent")

    def test_grade_episode_function(self):
        """grade_episode should delegate to correct grader."""
        episode = {
            "total_reward": 30.0,
            "success": True,
            "steps": 5,
            "state": {"vulnerabilities": [], "nodes": []},
            "cascade_failures": 0,
            "invalid_actions": 0,
        }
        result = grade_episode("easy", episode)
        assert isinstance(result, GraderResult)
        assert result.task_id == "easy"

    def test_list_graders(self):
        """list_graders should return metadata for all graders."""
        graders = list_graders()
        assert len(graders) == 5
        for g in graders:
            assert g["has_grader"] is True
            assert "completion" in g["dimensions"]
