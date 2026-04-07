"""
PatchCascade SOC - Pydantic Model Tests
=========================================

Tests for data model validation, serialization, and action validation.

Tests verify:
- Model construction and validation
- JSON serialization/deserialization roundtrips
- Field constraints (min/max values, patterns)
- Enum correctness
- Action validation helper function

Author: PatchCascade SOC Team
License: Apache 2.0
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    ActionType,
    CriticalityTier,
    NodeState,
    SeverityLevel,
    ServerNode,
    Dependency,
    Vulnerability,
    NetworkHealth,
    PatchCascadeObservation,
    PatchCascadeAction,
    PatchCascadeState,
    validate_action_for_observation,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestEnums:
    """Test that all enums have expected values."""

    def test_node_states(self):
        """NodeState should have 5 values."""
        assert len(NodeState) == 5
        assert NodeState.ONLINE.value == "online"
        assert NodeState.CRASHED.value == "crashed"

    def test_criticality_tiers(self):
        """CriticalityTier should have 3 levels."""
        assert CriticalityTier.CRITICAL.value == 1
        assert CriticalityTier.IMPORTANT.value == 2
        assert CriticalityTier.STANDARD.value == 3

    def test_severity_levels(self):
        """SeverityLevel should have 4 values."""
        assert len(SeverityLevel) == 4
        assert SeverityLevel.CRITICAL.value == "critical"

    def test_action_types(self):
        """ActionType should have 5 values."""
        assert len(ActionType) == 5
        expected = {"scan_host", "suspend_service", "apply_patch", "resume_service", "noop"}
        assert {a.value for a in ActionType} == expected


# =============================================================================
# MODEL CONSTRUCTION TESTS
# =============================================================================


class TestServerNode:
    """Test ServerNode model."""

    def test_valid_node(self):
        """Valid node construction should succeed."""
        node = ServerNode(
            hostname="db-primary-01",
            os="Ubuntu 22.04 LTS",
            tier=CriticalityTier.CRITICAL,
            state=NodeState.ONLINE,
            services=["postgresql"],
        )
        assert node.hostname == "db-primary-01"
        assert node.tier == CriticalityTier.CRITICAL
        assert node.patch_turns_remaining == 0

    def test_json_roundtrip(self):
        """Node should survive JSON serialization roundtrip."""
        node = ServerNode(
            hostname="web-01",
            os="RHEL 8.9",
            tier=CriticalityTier.IMPORTANT,
            state=NodeState.PATCHING,
            services=["nginx"],
            patch_turns_remaining=1,
        )
        json_str = node.model_dump_json()
        node2 = ServerNode.model_validate_json(json_str)
        assert node2.hostname == node.hostname
        assert node2.state == node.state
        assert node2.patch_turns_remaining == 1


class TestVulnerability:
    """Test Vulnerability model."""

    def test_valid_vulnerability(self):
        """Valid vulnerability construction should succeed."""
        vuln = Vulnerability(
            cve_id="CVE-2024-1234",
            severity=SeverityLevel.CRITICAL,
            cvss_score=9.8,
            affected_hosts=["db-01"],
            description="Remote code execution",
            patch_available=True,
            exploit_in_wild=True,
        )
        assert vuln.cvss_score == 9.8
        assert vuln.exploit_in_wild is True

    def test_cvss_bounds(self):
        """CVSS score should be between 0.0 and 10.0."""
        # Valid bounds
        Vulnerability(
            cve_id="CVE-2024-0001",
            severity=SeverityLevel.LOW,
            cvss_score=0.0,
            affected_hosts=["host1"],
        )
        Vulnerability(
            cve_id="CVE-2024-0002",
            severity=SeverityLevel.CRITICAL,
            cvss_score=10.0,
            affected_hosts=["host1"],
        )


class TestPatchCascadeAction:
    """Test PatchCascadeAction model."""

    def test_noop_action(self):
        """NOOP action should require no target."""
        action = PatchCascadeAction(action_type=ActionType.NOOP)
        assert action.target == ""
        assert action.cve_id is None

    def test_apply_patch_action(self):
        """APPLY_PATCH action should include cve_id."""
        action = PatchCascadeAction(
            action_type=ActionType.APPLY_PATCH,
            target="web-01",
            cve_id="CVE-2024-1234",
            reason="Patching critical vuln",
        )
        assert action.cve_id == "CVE-2024-1234"

    def test_json_serialization(self):
        """Action should serialize to valid JSON."""
        action = PatchCascadeAction(
            action_type=ActionType.SCAN_HOST,
            target="db-01",
        )
        json_str = action.model_dump_json()
        assert "scan_host" in json_str
        assert "db-01" in json_str


# =============================================================================
# ACTION VALIDATION TESTS
# =============================================================================


class TestActionValidation:
    """Test the validate_action_for_observation helper."""

    def _make_obs(self) -> PatchCascadeObservation:
        """Create a minimal observation for testing validation."""
        return PatchCascadeObservation(
            nodes=[
                ServerNode(
                    hostname="web-01",
                    os="Ubuntu",
                    tier=CriticalityTier.IMPORTANT,
                    state=NodeState.ONLINE,
                    services=["nginx"],
                ),
                ServerNode(
                    hostname="db-01",
                    os="Ubuntu",
                    tier=CriticalityTier.CRITICAL,
                    state=NodeState.ONLINE,
                    services=["postgresql"],
                ),
                ServerNode(
                    hostname="crashed-01",
                    os="Ubuntu",
                    tier=CriticalityTier.STANDARD,
                    state=NodeState.CRASHED,
                    services=[],
                ),
            ],
            vulnerabilities=[
                Vulnerability(
                    cve_id="CVE-2024-0001",
                    severity=SeverityLevel.HIGH,
                    cvss_score=8.0,
                    affected_hosts=["web-01"],
                ),
                Vulnerability(
                    cve_id="CVE-2024-0002",
                    severity=SeverityLevel.CRITICAL,
                    cvss_score=9.5,
                    affected_hosts=["db-01"],
                ),
            ],
            dependencies=[],
            health=NetworkHealth(
                total_nodes=3, nodes_online=2, nodes_crashed=1,
                nodes_patching=0, active_critical_vulns=1,
                active_high_vulns=1, active_medium_vulns=0,
                active_low_vulns=0,
            ),
        )

    def test_noop_always_valid(self):
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(action_type=ActionType.NOOP), obs
        )
        assert valid is True

    def test_scan_existing_host_valid(self):
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(action_type=ActionType.SCAN_HOST, target="web-01"), obs
        )
        assert valid is True

    def test_scan_nonexistent_host_invalid(self):
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(action_type=ActionType.SCAN_HOST, target="ghost-99"), obs
        )
        assert valid is False
        assert "invalid_target" in msg

    def test_suspend_online_valid(self):
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(action_type=ActionType.SUSPEND_SERVICE, target="web-01"), obs
        )
        assert valid is True

    def test_suspend_crashed_invalid(self):
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(action_type=ActionType.SUSPEND_SERVICE, target="crashed-01"), obs
        )
        assert valid is False

    def test_patch_tier1_online_invalid(self):
        """Tier 1 node must be SUSPENDED before patching."""
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(
                action_type=ActionType.APPLY_PATCH,
                target="db-01",
                cve_id="CVE-2024-0002",
            ), obs
        )
        assert valid is False
        assert "dependency_violation" in msg

    def test_patch_tier2_online_valid(self):
        """Tier 2+ nodes can be patched while ONLINE."""
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(
                action_type=ActionType.APPLY_PATCH,
                target="web-01",
                cve_id="CVE-2024-0001",
            ), obs
        )
        assert valid is True

    def test_patch_no_cve_invalid(self):
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(
                action_type=ActionType.APPLY_PATCH,
                target="web-01",
                cve_id=None,
            ), obs
        )
        assert valid is False

    def test_resume_crashed_valid(self):
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(action_type=ActionType.RESUME_SERVICE, target="crashed-01"), obs
        )
        assert valid is True

    def test_resume_online_invalid(self):
        obs = self._make_obs()
        valid, msg = validate_action_for_observation(
            PatchCascadeAction(action_type=ActionType.RESUME_SERVICE, target="web-01"), obs
        )
        assert valid is False
