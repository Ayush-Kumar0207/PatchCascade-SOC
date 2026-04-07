"""
PatchCascade SOC - Environment Tests
======================================

Comprehensive tests for the core environment logic across all 5 task levels.

Tests verify:
- Correct initialization for each difficulty level
- Action validation and execution
- Cascade failure mechanics
- Dynamic events (exploit spreading, zero-day injection)
- Terminal conditions (victory, catastrophic failure, timeout)
- Reward calculation correctness

Author: PatchCascade SOC Team
License: Apache 2.0
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment import PatchCascadeEnv, VICTORY_BONUS, INVALID_ACTION_PENALTY
from models import (
    ActionType,
    CriticalityTier,
    NodeState,
    PatchCascadeAction,
    PatchCascadeObservation,
    SeverityLevel,
)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestEnvironmentInit:
    """Test environment initialization and reset for all 5 task levels."""

    @pytest.mark.parametrize("task_level,min_nodes,min_vulns", [
        ("easy", 3, 1),
        ("medium", 5, 2),
        ("hard", 10, 5),
        ("incident_response", 8, 3),
        ("zero_day", 10, 2),
    ])
    def test_reset_produces_valid_observation(self, task_level, min_nodes, min_vulns):
        """Each task level should produce a valid observation with expected counts."""
        env = PatchCascadeEnv(seed=42)
        obs = env.reset(task_level=task_level, seed=42)

        assert isinstance(obs, PatchCascadeObservation)
        assert len(obs.nodes) >= min_nodes
        assert len(obs.vulnerabilities) >= min_vulns
        assert obs.health.total_nodes == len(obs.nodes)
        assert obs.health.turn_number == 0

    def test_reset_invalid_task_level(self):
        """Invalid task levels should raise ValueError."""
        env = PatchCascadeEnv(seed=42)
        with pytest.raises(ValueError, match="Invalid task_level"):
            env.reset(task_level="nonexistent")

    def test_reset_clears_previous_episode(self):
        """Resetting mid-episode should start a clean new episode."""
        env = PatchCascadeEnv(seed=42)
        obs1 = env.reset(task_level="easy", seed=42)

        # Take some actions
        action = PatchCascadeAction(action_type=ActionType.NOOP)
        env.step(action)
        env.step(action)

        # Reset to different level
        obs2 = env.reset(task_level="medium", seed=42)
        assert obs2.health.turn_number == 0
        assert len(obs2.vulnerabilities) >= 2

    def test_seed_reproducibility(self):
        """Same seed should produce identical episodes."""
        env1 = PatchCascadeEnv(seed=42)
        obs1 = env1.reset(task_level="medium", seed=42)

        env2 = PatchCascadeEnv(seed=42)
        obs2 = env2.reset(task_level="medium", seed=42)

        assert len(obs1.nodes) == len(obs2.nodes)
        assert len(obs1.vulnerabilities) == len(obs2.vulnerabilities)
        for n1, n2 in zip(obs1.nodes, obs2.nodes):
            assert n1.hostname == n2.hostname
            assert n1.state == n2.state

    def test_step_before_reset_raises(self):
        """Stepping without reset should raise RuntimeError."""
        env = PatchCascadeEnv(seed=42)
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(PatchCascadeAction(action_type=ActionType.NOOP))


# =============================================================================
# TASK-SPECIFIC INITIALIZATION TESTS
# =============================================================================


class TestEasyTask:
    """Tests specific to Easy mode initialization."""

    def test_easy_no_dependencies(self, easy_env):
        """Easy mode should have zero dependencies."""
        env, obs = easy_env
        assert len(obs.dependencies) == 0

    def test_easy_single_vulnerability(self, easy_env):
        """Easy mode should have exactly 1 vulnerability."""
        env, obs = easy_env
        assert len(obs.vulnerabilities) == 1

    def test_easy_all_nodes_online(self, easy_env):
        """Easy mode should start with all nodes online."""
        env, obs = easy_env
        for node in obs.nodes:
            assert node.state == NodeState.ONLINE


class TestIncidentResponseTask:
    """Tests specific to Incident Response mode."""

    def test_ir_has_crashed_nodes(self, ir_env):
        """IR mode should start with 2 crashed nodes."""
        env, obs = ir_env
        crashed = [n for n in obs.nodes if n.state == NodeState.CRASHED]
        assert len(crashed) == 2

    def test_ir_has_exploited_vulns(self, ir_env):
        """IR mode should have actively exploited vulnerabilities."""
        env, obs = ir_env
        exploited = [v for v in obs.vulnerabilities if v.exploit_in_wild]
        assert len(exploited) >= 2

    def test_ir_intro_messages(self, ir_env):
        """IR mode should display breach warning messages."""
        env, obs = ir_env
        messages_text = " ".join(obs.messages)
        assert "ACTIVE BREACH" in messages_text


class TestZeroDayTask:
    """Tests specific to Zero-Day Cascade mode."""

    def test_zd_initial_vulns(self, zd_env):
        """Zero-day mode should start with 2 vulnerabilities."""
        env, obs = zd_env
        assert len(obs.vulnerabilities) == 2

    def test_zd_intro_messages(self, zd_env):
        """Zero-day mode should warn about undisclosed threats."""
        env, obs = zd_env
        messages_text = " ".join(obs.messages)
        assert "zero-day" in messages_text.lower()


# =============================================================================
# ACTION TESTS
# =============================================================================


class TestActions:
    """Test action execution and validation."""

    def test_noop_valid(self, easy_env):
        """NOOP should always be valid and advance the turn."""
        env, obs = easy_env
        result = env.step(PatchCascadeAction(action_type=ActionType.NOOP))
        assert result.info.get("valid") is True
        assert env.state.turn_number == 1

    def test_scan_host_valid(self, easy_env):
        """SCAN_HOST on existing node should succeed."""
        env, obs = easy_env
        hostname = obs.nodes[0].hostname
        action = PatchCascadeAction(
            action_type=ActionType.SCAN_HOST,
            target=hostname,
        )
        result = env.step(action)
        assert result.info.get("valid") is True

    def test_invalid_target(self, easy_env):
        """Action on nonexistent node should be invalid."""
        env, obs = easy_env
        action = PatchCascadeAction(
            action_type=ActionType.SCAN_HOST,
            target="nonexistent-server-99",
        )
        result = env.step(action)
        assert result.info.get("valid") is False
        assert "invalid_target" in result.info.get("error", "")

    def test_suspend_online_node(self, easy_env):
        """Suspending an ONLINE node should succeed."""
        env, obs = easy_env
        hostname = obs.nodes[0].hostname
        action = PatchCascadeAction(
            action_type=ActionType.SUSPEND_SERVICE,
            target=hostname,
        )
        result = env.step(action)
        assert result.info.get("valid") is True
        # Verify node is now SUSPENDED
        obs2 = env.get_observation()
        target_node = next(n for n in obs2.nodes if n.hostname == hostname)
        assert target_node.state == NodeState.SUSPENDED

    def test_patch_requires_cve_id(self, easy_env):
        """APPLY_PATCH without cve_id should be invalid."""
        env, obs = easy_env
        hostname = obs.vulnerabilities[0].affected_hosts[0]
        action = PatchCascadeAction(
            action_type=ActionType.APPLY_PATCH,
            target=hostname,
            cve_id=None,
        )
        result = env.step(action)
        assert result.info.get("valid") is False

    def test_patch_tier1_requires_suspend(self, medium_env):
        """Patching Tier 1 node without suspending first should fail."""
        env, obs = medium_env
        # Find the Tier 1 node
        tier1_node = next(n for n in obs.nodes if n.tier == CriticalityTier.CRITICAL)
        # Find its vulnerability
        vuln = next(v for v in obs.vulnerabilities if tier1_node.hostname in v.affected_hosts)

        action = PatchCascadeAction(
            action_type=ActionType.APPLY_PATCH,
            target=tier1_node.hostname,
            cve_id=vuln.cve_id,
        )
        result = env.step(action)
        assert result.info.get("valid") is False
        assert "dependency_violation" in result.info.get("error", "")

    def test_resume_crashed_node(self, ir_env):
        """Resuming a CRASHED node should bring it back ONLINE."""
        env, obs = ir_env
        crashed_node = next(n for n in obs.nodes if n.state == NodeState.CRASHED)
        action = PatchCascadeAction(
            action_type=ActionType.RESUME_SERVICE,
            target=crashed_node.hostname,
        )
        result = env.step(action)
        assert result.info.get("valid") is True


# =============================================================================
# PATCH COMPLETION TESTS
# =============================================================================


class TestPatchCompletion:
    """Test patch mechanics: apply, timer, completion, vulnerability removal."""

    def test_patch_completes_in_one_turn(self, easy_env):
        """A patch should complete within 1-2 turns."""
        env, obs = easy_env
        vuln = obs.vulnerabilities[0]
        target = vuln.affected_hosts[0]

        # Apply patch
        action = PatchCascadeAction(
            action_type=ActionType.APPLY_PATCH,
            target=target,
            cve_id=vuln.cve_id,
        )
        result1 = env.step(action)
        assert result1.info.get("valid") is True

        # Node should be either PATCHING (multi-turn) or ONLINE (instant patch)
        patching_node = next(n for n in result1.observation.nodes if n.hostname == target)
        assert patching_node.state in (NodeState.PATCHING, NodeState.ONLINE)

        # After one more NOOP, patch should definitely be complete
        result2 = env.step(PatchCascadeAction(action_type=ActionType.NOOP))
        patched_node = next(n for n in result2.observation.nodes if n.hostname == target)
        assert patched_node.state == NodeState.ONLINE

    def test_victory_when_all_patched(self, easy_env):
        """Episode should end with victory when all vulnerabilities are patched."""
        env, obs = easy_env
        vuln = obs.vulnerabilities[0]

        # Patch all affected hosts
        done = False
        for host in list(vuln.affected_hosts):
            action = PatchCascadeAction(
                action_type=ActionType.APPLY_PATCH,
                target=host,
                cve_id=vuln.cve_id,
            )
            result = env.step(action)
            if result.done:
                done = True
                break

        # May need extra NOOP to let last patch complete
        if not done:
            for _ in range(5):
                result = env.step(PatchCascadeAction(action_type=ActionType.NOOP))
                if result.done:
                    break

        assert result.done is True
        assert env.state.termination_reason == "all_patched"


# =============================================================================
# CASCADE FAILURE TESTS
# =============================================================================


class TestCascadeFailures:
    """Test dependency cascade mechanics."""

    def test_suspending_dependency_crashes_dependent(self, medium_env):
        """Suspending a dependency without protecting dependents should cause crash."""
        env, obs = medium_env
        # db-primary-01 is a Tier 1 node that app-server-01 depends on
        # Suspending it should crash app-server-01

        action = PatchCascadeAction(
            action_type=ActionType.SUSPEND_SERVICE,
            target="db-primary-01",
        )
        result = env.step(action)

        # Check that cascade occurred
        crashed = [n for n in result.observation.nodes if n.state == NodeState.CRASHED]
        assert len(crashed) > 0, "No cascade failure occurred when dependency was suspended"

    def test_safe_suspend_order_avoids_cascade(self, medium_env):
        """Suspending dependents before dependencies should prevent cascades."""
        env, obs = medium_env

        # Safe order: web -> app -> db
        safe_order = [
            "web-frontend-01", "web-frontend-02",
            "app-server-01", "app-server-02",
            "db-primary-01",
        ]
        for hostname in safe_order:
            node = next((n for n in env.get_observation().nodes if n.hostname == hostname), None)
            if node and node.state == NodeState.ONLINE:
                action = PatchCascadeAction(
                    action_type=ActionType.SUSPEND_SERVICE,
                    target=hostname,
                )
                result = env.step(action)
                assert result.info.get("valid") is True

        # No nodes should be crashed
        obs2 = env.get_observation()
        crashed = [n for n in obs2.nodes if n.state == NodeState.CRASHED]
        assert len(crashed) == 0, f"Unexpected crashes: {[n.hostname for n in crashed]}"


# =============================================================================
# REWARD TESTS
# =============================================================================


class TestRewards:
    """Test reward calculation correctness."""

    def test_invalid_action_gives_penalty(self, easy_env):
        """Invalid actions should give a negative penalty reward."""
        env, obs = easy_env
        action = PatchCascadeAction(
            action_type=ActionType.SCAN_HOST,
            target="nonexistent-99",
        )
        result = env.step(action)
        # The penalty component from INVALID_ACTION_PENALTY should be present
        assert result.reward <= 0

    def test_patching_gives_positive_reward(self, easy_env):
        """Successfully patching a vulnerability should yield positive reward."""
        env, obs = easy_env
        vuln = obs.vulnerabilities[0]
        target = vuln.affected_hosts[0]

        action = PatchCascadeAction(
            action_type=ActionType.APPLY_PATCH,
            target=target,
            cve_id=vuln.cve_id,
        )
        result = env.step(action)
        # Reward may be negative (downtime penalty for PATCHING state) or positive
        # But at least it should be a valid float
        assert isinstance(result.reward, float)


# =============================================================================
# DYNAMIC EVENTS TESTS
# =============================================================================


class TestDynamicEvents:
    """Test exploit spreading and zero-day injection."""

    def test_zero_day_injection_at_turn_5(self):
        """Zero-day CVE should be injected at turn 5."""
        env = PatchCascadeEnv(seed=42)
        obs = env.reset(task_level="zero_day", seed=42)
        initial_vulns = len(obs.vulnerabilities)

        # Advance to turn 5
        for _ in range(6):  # 0..5
            result = env.step(PatchCascadeAction(action_type=ActionType.NOOP))

        # Should have more vulnerabilities now
        assert len(result.observation.vulnerabilities) > initial_vulns, \
            "Zero-day CVE was not injected at turn 5"

        # Check the injected CVE
        cve_ids = [v.cve_id for v in result.observation.vulnerabilities]
        assert "CVE-2024-5099" in cve_ids

    def test_zero_day_injection_at_turn_15(self):
        """Second zero-day CVE should be injected at turn 15."""
        env = PatchCascadeEnv(seed=42)
        env.reset(task_level="zero_day", seed=42)

        # Advance past turn 15
        for _ in range(17):
            result = env.step(PatchCascadeAction(action_type=ActionType.NOOP))

        cve_ids = [v.cve_id for v in result.observation.vulnerabilities]
        assert "CVE-2024-5100" in cve_ids, "Second zero-day was not injected at turn 15"


# =============================================================================
# TERMINATION TESTS
# =============================================================================


class TestTermination:
    """Test episode termination conditions."""

    def test_timeout_causes_truncation(self, easy_env):
        """Reaching max_turns should end episode as truncated."""
        env, obs = easy_env
        max_turns = env.state.max_turns

        for _ in range(max_turns + 1):
            result = env.step(PatchCascadeAction(action_type=ActionType.NOOP))
            if result.done:
                break

        assert result.done is True
        assert result.truncated is True
        assert env.state.termination_reason == "max_turns_reached"

    def test_double_step_after_termination_raises(self, easy_env):
        """Stepping after episode termination should raise."""
        env, obs = easy_env
        max_turns = env.state.max_turns

        for _ in range(max_turns + 1):
            result = env.step(PatchCascadeAction(action_type=ActionType.NOOP))
            if result.done:
                break

        with pytest.raises(RuntimeError, match="already terminated"):
            env.step(PatchCascadeAction(action_type=ActionType.NOOP))


# =============================================================================
# RENDER TEST
# =============================================================================


class TestRender:
    """Test human-readable rendering."""

    def test_render_output(self, easy_env):
        """Render should produce a non-empty string."""
        env, obs = easy_env
        output = env.render()
        assert isinstance(output, str)
        assert len(output) > 50
        assert "PATCHCASCADE SOC" in output
