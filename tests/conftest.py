"""
PatchCascade SOC - Test Fixtures
=================================

Shared pytest fixtures for the test suite.
Provides pre-configured environment instances and common test data.

Author: PatchCascade SOC Team
License: Apache 2.0
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment import PatchCascadeEnv
from models import (
    ActionType,
    PatchCascadeAction,
    PatchCascadeObservation,
)


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def env() -> PatchCascadeEnv:
    """Create a fresh environment with fixed seed for reproducibility."""
    return PatchCascadeEnv(seed=42)


@pytest.fixture
def easy_env(env: PatchCascadeEnv) -> tuple[PatchCascadeEnv, PatchCascadeObservation]:
    """Environment reset to easy mode."""
    obs = env.reset(task_level="easy", seed=42)
    return env, obs


@pytest.fixture
def medium_env(env: PatchCascadeEnv) -> tuple[PatchCascadeEnv, PatchCascadeObservation]:
    """Environment reset to medium mode."""
    obs = env.reset(task_level="medium", seed=42)
    return env, obs


@pytest.fixture
def hard_env(env: PatchCascadeEnv) -> tuple[PatchCascadeEnv, PatchCascadeObservation]:
    """Environment reset to hard mode."""
    obs = env.reset(task_level="hard", seed=42)
    return env, obs


@pytest.fixture
def ir_env(env: PatchCascadeEnv) -> tuple[PatchCascadeEnv, PatchCascadeObservation]:
    """Environment reset to incident_response mode."""
    obs = env.reset(task_level="incident_response", seed=42)
    return env, obs


@pytest.fixture
def zd_env(env: PatchCascadeEnv) -> tuple[PatchCascadeEnv, PatchCascadeObservation]:
    """Environment reset to zero_day mode."""
    obs = env.reset(task_level="zero_day", seed=42)
    return env, obs


# =============================================================================
# ACTION FIXTURES
# =============================================================================


@pytest.fixture
def noop_action() -> PatchCascadeAction:
    """A NOOP action."""
    return PatchCascadeAction(action_type=ActionType.NOOP)


@pytest.fixture
def scan_action() -> PatchCascadeAction:
    """A SCAN_HOST action (target must be set per test)."""
    return PatchCascadeAction(
        action_type=ActionType.SCAN_HOST,
        target="web-server-01",
    )
