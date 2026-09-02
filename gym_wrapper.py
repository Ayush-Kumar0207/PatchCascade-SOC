"""
PatchCascade SOC — Gymnasium-Compatible Wrapper
=================================================

Provides a standard Gymnasium (formerly OpenAI Gym) interface for
PatchCascade SOC, enabling seamless integration with:

- Stable-Baselines3 (PPO, A2C, DQN)
- RLlib
- CleanRL
- Any Gymnasium-compatible RL library

This wrapper handles:
1. Converting complex JSON observations → fixed-size numpy arrays
2. Mapping discrete integer actions → PatchCascadeAction objects
3. Properly defining observation_space and action_space

Usage:
    import gymnasium as gym
    from gym_wrapper import PatchCascadeGymEnv

    env = PatchCascadeGymEnv(task_level="medium")
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment import PatchCascadeEnv
from models import (
    ActionType,
    CriticalityTier,
    NodeState,
    PatchCascadeAction,
    PatchCascadeObservation,
    SeverityLevel,
    validate_action_for_observation,
)


# =============================================================================
# CONSTANTS — Fixed array sizes for observation/action encoding
# =============================================================================

# Maximum capacities (padded with zeros if fewer exist)
MAX_NODES = 15       # Hard mode has up to 15 nodes
MAX_VULNS = 8        # Zero-day can inject up to ~6 CVEs
MAX_DEPS = 20        # Complex graphs can have many edges

# Material observation/action/termination changes are an explicit compatibility
# boundary. Old PPO archives must never be loaded as canonical-v1 models.
ENVIRONMENT_API_VERSION = "patchcascade-gym-v4"
OBSERVATION_SCHEMA_VERSION = "gym-observation-v3-cve-host-incidence"
ACTION_SCHEMA_VERSION = "multidiscrete-v2-joint-validity-penalized"
FLATTENED_ACTION_SCHEMA_VERSION = "discrete-v1-state-masked-joint-validity"

# Feature sizes per element
NODE_FEATURES = 6    # tier, state, patch_turns, has_vuln, is_exploited, is_critical_tier
VULN_FEATURES = 5    # cvss, severity_code, num_affected, exploit_in_wild, patch_available
VULN_HOST_MATRIX_SIZE = MAX_VULNS * MAX_NODES  # explicit CVE-to-host incidence
DEP_FEATURES = 4     # from_node_idx, to_node_idx, is_hard, exists
HEALTH_FEATURES = 11 # all NetworkHealth fields

# Total observation vector size
OBS_SIZE = (
    MAX_NODES * NODE_FEATURES
    + MAX_VULNS * VULN_FEATURES
    + VULN_HOST_MATRIX_SIZE
    + MAX_DEPS * DEP_FEATURES
    + HEALTH_FEATURES
)

# Action types (5 total)
NUM_ACTION_TYPES = 5  # scan, suspend, patch, resume, noop
FLATTENED_ACTION_COUNT = NUM_ACTION_TYPES * MAX_NODES * MAX_VULNS
ACTION_TYPES = (
    ActionType.SCAN_HOST,
    ActionType.SUSPEND_SERVICE,
    ActionType.APPLY_PATCH,
    ActionType.RESUME_SERVICE,
    ActionType.NOOP,
)

# Action encoding: MultiDiscrete([action_type, target_node, target_vuln])
# This allows any combination: "apply_patch node_3 vuln_2"

# State encoding lookup
STATE_ENCODING = {
    NodeState.ONLINE: 0,
    NodeState.OFFLINE: 1,
    NodeState.SUSPENDED: 2,
    NodeState.PATCHING: 3,
    NodeState.CRASHED: 4,
}

SEVERITY_ENCODING = {
    SeverityLevel.LOW: 0,
    SeverityLevel.MEDIUM: 1,
    SeverityLevel.HIGH: 2,
    SeverityLevel.CRITICAL: 3,
}


# =============================================================================
# GYMNASIUM ENVIRONMENT WRAPPER
# =============================================================================


class PatchCascadeGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for PatchCascade SOC.

    Converts the rich JSON-based PatchCascade environment into a
    standard Gymnasium interface with fixed-size numpy arrays for
    observations and a MultiDiscrete action space.

    This enables training with standard RL libraries like
    Stable-Baselines3, RLlib, and CleanRL.

    Args:
        task_level: Difficulty level ("easy", "medium", "hard",
                    "incident_response", "zero_day").
        seed: Random seed for reproducibility.
        normalize_obs: If True, normalize observations to [0, 1] range.
        reward_scale: Multiply rewards by this factor (for training stability).

    Attributes:
        observation_space: Box space of shape (OBS_SIZE,) with float32 values.
        action_space: MultiDiscrete([5, MAX_NODES, MAX_VULNS]) for
                      (action_type, target_node, target_vuln).
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "environment_api_version": ENVIRONMENT_API_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }

    def __init__(
        self,
        task_level: str = "medium",
        seed: int | None = None,
        normalize_obs: bool = True,
        reward_scale: float = 0.01,
        render_mode: str | None = None,
    ):
        super().__init__()

        self._task_level = task_level
        self._seed = seed
        self._normalize_obs = normalize_obs
        self._reward_scale = reward_scale
        self.render_mode = render_mode

        # Internal PatchCascade environment
        self._env = PatchCascadeEnv(seed=seed)

        # Current episode state (populated on reset)
        self._obs: PatchCascadeObservation | None = None
        self._hostname_to_idx: dict[str, int] = {}
        self._cve_to_idx: dict[str, int] = {}
        self._idx_to_hostname: dict[int, str] = {}
        self._idx_to_cve: dict[int, str] = {}
        self._episode_reward: float = 0.0
        self._step_count: int = 0
        self._has_reset: bool = False

        # ── Gymnasium Spaces ──────────────────────────────────────────

        # Observation: fixed-size float32 vector
        self.observation_space = spaces.Box(
            low=-1.0 if normalize_obs else -500.0,
            high=1.0 if normalize_obs else 500.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        # Action: [action_type, target_node_index, target_vuln_index]
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTION_TYPES, MAX_NODES, MAX_VULNS]
        )

    # =====================================================================
    # CORE GYMNASIUM INTERFACE
    # =====================================================================

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to a new episode.

        Args:
            seed: Optional random seed override.
            options: Optional dict with 'task_level' to change difficulty.

        Returns:
            Tuple of (observation_array, info_dict).
        """
        super().reset(seed=seed)

        # Allow changing task level via options
        task_level = self._task_level
        if options and "task_level" in options:
            task_level = options["task_level"]
            self._task_level = task_level

        # The constructor seed initializes a deterministic *sequence*. Reusing
        # it on every reset made all training episodes identical.
        effective_seed = seed if seed is not None else (self._seed if not self._has_reset else None)
        self._obs = self._env.reset(task_level=task_level, seed=effective_seed)
        self._has_reset = True

        # Build hostname ↔ index mappings
        self._hostname_to_idx = {
            n.hostname: i for i, n in enumerate(self._obs.nodes)
        }
        self._idx_to_hostname = {i: h for h, i in self._hostname_to_idx.items()}

        # Build CVE ↔ index mappings
        self._cve_to_idx = {
            v.cve_id: i for i, v in enumerate(self._obs.vulnerabilities)
        }
        self._idx_to_cve = {i: c for c, i in self._cve_to_idx.items()}

        self._episode_reward = 0.0
        self._step_count = 0

        obs_array = self._encode_observation(self._obs)
        if not np.isfinite(obs_array).all():
            raise FloatingPointError("NaN/Inf observation produced during reset")
        info = self._build_info(self._obs)

        return obs_array, info

    def step(
        self, action: np.ndarray | list[int] | int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Array of [action_type_idx, node_idx, vuln_idx].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Decode action from integers → PatchCascadeAction
        patch_action = self._decode_action(action)

        # Execute in underlying environment
        result = self._env.step(patch_action)

        self._obs = result.observation
        self._step_count += 1

        # Rebuild CVE index (vulns can be removed or added dynamically)
        self._cve_to_idx = {
            v.cve_id: i for i, v in enumerate(self._obs.vulnerabilities)
        }
        self._idx_to_cve = {i: c for c, i in self._cve_to_idx.items()}

        # Encode observation
        obs_array = self._encode_observation(self._obs)

        # Scale reward for training stability
        scaled_reward = result.reward * self._reward_scale
        if not np.isfinite(obs_array).all():
            raise FloatingPointError("NaN/Inf observation produced during step")
        if not np.isfinite(scaled_reward):
            raise FloatingPointError("NaN/Inf reward produced during step")
        self._episode_reward += result.reward

        # Build info dict
        info = self._build_info(self._obs, result.info)
        info["raw_reward"] = result.reward
        info["episode_reward"] = self._episode_reward
        info["action_taken"] = patch_action.action_type.value
        info["action_target"] = patch_action.target
        info["action_valid"] = result.info.get("valid", True)

        terminated = bool(result.done and not result.truncated)
        return obs_array, scaled_reward, terminated, bool(result.truncated), info

    def render(self) -> str | None:
        """Render the environment state."""
        if self.render_mode == "human":
            print(self._env.render(mode="ascii"))
        elif self.render_mode == "ansi":
            return self._env.render(mode="ascii")
        return None

    def close(self):
        """Clean up resources."""
        pass

    # =====================================================================
    # OBSERVATION ENCODING — JSON → numpy array
    # =====================================================================

    def _encode_observation(self, obs: PatchCascadeObservation) -> np.ndarray:
        """
        Encode a PatchCascadeObservation into a fixed-size numpy array.

        Layout:
            [0 .. MAX_NODES*NODE_FEATURES)         : Node features
            [.. + MAX_VULNS*VULN_FEATURES)          : Vulnerability features
            [.. + MAX_VULNS*MAX_NODES)              : CVE-to-host incidence
            [.. + MAX_DEPS*DEP_FEATURES)            : Dependency features
            [.. + HEALTH_FEATURES)                  : Health metrics

        All features are normalized to approximately [0, 1] if normalize_obs=True.
        """
        vec = np.zeros(OBS_SIZE, dtype=np.float32)
        offset = 0

        # ── Encode Nodes ──────────────────────────────────────────────
        # Build a quick lookup: which nodes have which vulns/exploits
        vuln_hosts = set()
        exploit_hosts = set()
        for v in obs.vulnerabilities:
            for h in v.affected_hosts:
                vuln_hosts.add(h)
                if v.exploit_in_wild:
                    exploit_hosts.add(h)

        for i, node in enumerate(obs.nodes[:MAX_NODES]):
            base = offset + i * NODE_FEATURES
            # Feature 0: Tier (normalized: 1→1.0, 2→0.66, 3→0.33)
            vec[base + 0] = node.tier.value / 3.0 if self._normalize_obs else float(node.tier.value)
            # Feature 1: State (one-hot-ish encoding)
            vec[base + 1] = STATE_ENCODING.get(node.state, 0) / 4.0 if self._normalize_obs else float(STATE_ENCODING.get(node.state, 0))
            # Feature 2: Patch turns remaining
            vec[base + 2] = node.patch_turns_remaining / 5.0 if self._normalize_obs else float(node.patch_turns_remaining)
            # Feature 3: Has vulnerability (binary)
            vec[base + 3] = 1.0 if node.hostname in vuln_hosts else 0.0
            # Feature 4: Has exploited vulnerability (binary)
            vec[base + 4] = 1.0 if node.hostname in exploit_hosts else 0.0
            # Feature 5: Is critical tier (binary — important for suspend requirement)
            vec[base + 5] = 1.0 if node.tier == CriticalityTier.CRITICAL else 0.0

        offset += MAX_NODES * NODE_FEATURES

        # ── Encode Vulnerabilities ────────────────────────────────────
        for i, vuln in enumerate(obs.vulnerabilities[:MAX_VULNS]):
            base = offset + i * VULN_FEATURES
            # Feature 0: CVSS score (normalized to [0, 1])
            vec[base + 0] = vuln.cvss_score / 10.0 if self._normalize_obs else vuln.cvss_score
            # Feature 1: Severity code
            vec[base + 1] = SEVERITY_ENCODING.get(vuln.severity, 0) / 3.0 if self._normalize_obs else float(SEVERITY_ENCODING.get(vuln.severity, 0))
            # Feature 2: Number of affected hosts (normalized)
            vec[base + 2] = len(vuln.affected_hosts) / MAX_NODES if self._normalize_obs else float(len(vuln.affected_hosts))
            # Feature 3: Exploit in the wild (binary)
            vec[base + 3] = 1.0 if vuln.exploit_in_wild else 0.0
            # Feature 4: Patch available (binary)
            vec[base + 4] = 1.0 if vuln.patch_available else 0.0

        offset += MAX_VULNS * VULN_FEATURES

        # ── Encode CVE-to-host incidence ──────────────────────────────
        # Without this matrix, many distinct states were observationally
        # aliased and the policy could not know which patch target was valid.
        for vuln_idx, vuln in enumerate(obs.vulnerabilities[:MAX_VULNS]):
            for hostname in vuln.affected_hosts:
                node_idx = self._hostname_to_idx.get(hostname)
                if node_idx is not None and node_idx < MAX_NODES:
                    vec[offset + vuln_idx * MAX_NODES + node_idx] = 1.0

        offset += VULN_HOST_MATRIX_SIZE

        # ── Encode Dependencies ───────────────────────────────────────
        for i, dep in enumerate(obs.dependencies[:MAX_DEPS]):
            base = offset + i * DEP_FEATURES
            # Feature 0: Source node index (normalized)
            from_idx = self._hostname_to_idx.get(dep.node, 0)
            vec[base + 0] = from_idx / MAX_NODES if self._normalize_obs else float(from_idx)
            # Feature 1: Target node index (normalized)
            to_idx = self._hostname_to_idx.get(dep.depends_on, 0)
            vec[base + 1] = to_idx / MAX_NODES if self._normalize_obs else float(to_idx)
            # Feature 2: Is hard dependency (binary)
            vec[base + 2] = 1.0 if dep.dependency_type == "hard" else 0.0
            # Feature 3: Edge exists flag
            vec[base + 3] = 1.0

        offset += MAX_DEPS * DEP_FEATURES

        # ── Encode Health Metrics ─────────────────────────────────────
        h = obs.health
        max_nodes = max(h.total_nodes, 1)  # Avoid division by zero
        if self._normalize_obs:
            vec[offset + 0] = h.total_nodes / MAX_NODES
            vec[offset + 1] = h.nodes_online / max_nodes
            vec[offset + 2] = h.nodes_crashed / max_nodes
            vec[offset + 3] = h.nodes_patching / max_nodes
            vec[offset + 4] = h.active_critical_vulns / MAX_VULNS
            vec[offset + 5] = h.active_high_vulns / MAX_VULNS
            vec[offset + 6] = h.active_medium_vulns / MAX_VULNS
            vec[offset + 7] = h.active_low_vulns / MAX_VULNS
            vec[offset + 8] = np.clip(h.cumulative_downtime_penalty / 100.0, -1, 1)
            vec[offset + 9] = np.clip(h.cumulative_risk_penalty / 100.0, -1, 1)
            vec[offset + 10] = h.turn_number / 100.0
        else:
            vec[offset + 0] = float(h.total_nodes)
            vec[offset + 1] = float(h.nodes_online)
            vec[offset + 2] = float(h.nodes_crashed)
            vec[offset + 3] = float(h.nodes_patching)
            vec[offset + 4] = float(h.active_critical_vulns)
            vec[offset + 5] = float(h.active_high_vulns)
            vec[offset + 6] = float(h.active_medium_vulns)
            vec[offset + 7] = float(h.active_low_vulns)
            vec[offset + 8] = h.cumulative_downtime_penalty
            vec[offset + 9] = h.cumulative_risk_penalty
            vec[offset + 10] = float(h.turn_number)

        return vec

    # =====================================================================
    # ACTION DECODING — integer → PatchCascadeAction
    # =====================================================================

    def _decode_action(self, action: np.ndarray | list[int]) -> PatchCascadeAction:
        """
        Decode a MultiDiscrete action [action_type, node_idx, vuln_idx]
        into a PatchCascadeAction.

        Padded/out-of-range choices remain invalid so the environment applies
        its declared invalid-action penalty. They are never silently repaired.
        """
        action_type_idx = int(action[0])
        node_idx = int(action[1])
        vuln_idx = int(action[2])

        # Map action type
        if action_type_idx >= len(ACTION_TYPES):
            action_type_idx = 4  # NOOP fallback

        action_type = ACTION_TYPES[action_type_idx]

        # NOOP needs no target
        if action_type == ActionType.NOOP:
            return PatchCascadeAction(
                action_type=ActionType.NOOP,
                reason="RL agent chose to wait",
            )

        # Resolve target hostname
        hostname = self._idx_to_hostname.get(node_idx, "")
        if not hostname:
            return PatchCascadeAction(
                action_type=action_type,
                target=f"__invalid_node_{node_idx}__",
                cve_id="CVE-0000-0000" if action_type == ActionType.APPLY_PATCH else None,
                reason="RL agent selected padded node index",
            )

        # For APPLY_PATCH, resolve CVE ID
        cve_id = None
        if action_type == ActionType.APPLY_PATCH:
            cve_id = self._idx_to_cve.get(vuln_idx)
            if cve_id is None:
                cve_id = "CVE-0000-0000"

        return PatchCascadeAction(
            action_type=action_type,
            target=hostname,
            cve_id=cve_id,
            reason=f"RL agent: {action_type.value} on {hostname}",
        )

    # =====================================================================
    # INFO BUILDER
    # =====================================================================

    def _build_info(
        self,
        obs: PatchCascadeObservation,
        step_info: dict | None = None,
    ) -> dict:
        """Build the info dict returned with each step/reset."""
        info = {
            "task_level": self._task_level,
            "num_nodes": len(obs.nodes),
            "num_vulns": len(obs.vulnerabilities),
            "num_deps": len(obs.dependencies),
            "nodes_online": obs.health.nodes_online,
            "nodes_crashed": obs.health.nodes_crashed,
            "step_count": self._step_count,
            "episode_seed": self._env.state.episode_seed,
        }
        if step_info:
            info.update(step_info)
        return info

    # =====================================================================
    # CONVENIENCE METHODS
    # =====================================================================

    def get_action_mask(self) -> np.ndarray:
        """
        Generate an action mask for valid actions (for masked policy training).

        Returns the exact joint validity tensor flattened in C order with shape
        ``NUM_ACTION_TYPES * MAX_NODES * MAX_VULNS``. MultiDiscrete factor masks
        cannot express node/CVE pair constraints, so callers must not pass this
        directly to SB3-contrib without a joint-action adapter.
        """
        if self._obs is None:
            return np.zeros(NUM_ACTION_TYPES * MAX_NODES * MAX_VULNS, dtype=bool)

        mask = np.zeros((NUM_ACTION_TYPES, MAX_NODES, MAX_VULNS), dtype=bool)
        mask[4, 0, 0] = True  # one canonical NOOP; other encodings are aliases
        for node_idx, node in enumerate(self._obs.nodes[:MAX_NODES]):
            mask[0, node_idx, 0] = True
            if node.state == NodeState.ONLINE:
                mask[1, node_idx, 0] = True
            if node.state in (NodeState.SUSPENDED, NodeState.CRASHED):
                mask[3, node_idx, 0] = True
            for vuln_idx, vuln in enumerate(self._obs.vulnerabilities[:MAX_VULNS]):
                if (
                    node.hostname in vuln.affected_hosts
                    and node.state in (NodeState.ONLINE, NodeState.SUSPENDED)
                    and (node.tier != CriticalityTier.CRITICAL or node.state == NodeState.SUSPENDED)
                ):
                    mask[2, node_idx, vuln_idx] = True
        return mask.reshape(-1)

    def sync_observation(self, obs: PatchCascadeObservation) -> None:
        """Synchronize index maps for external matched-policy evaluation."""
        self._obs = obs
        self._hostname_to_idx = {node.hostname: idx for idx, node in enumerate(obs.nodes)}
        self._idx_to_hostname = {idx: name for name, idx in self._hostname_to_idx.items()}
        self._cve_to_idx = {vuln.cve_id: idx for idx, vuln in enumerate(obs.vulnerabilities)}
        self._idx_to_cve = {idx: cve for cve, idx in self._cve_to_idx.items()}

    @property
    def unwrapped_env(self) -> PatchCascadeEnv:
        """Access the underlying PatchCascade environment."""
        return self._env


class FlattenedMaskedPatchCascadeEnv(PatchCascadeGymEnv):
    """One-to-one Discrete action adapter for state-dependent MaskablePPO.

    The flattened ID is ``((action_type * MAX_NODES) + node) * MAX_VULNS + cve``.
    Only one encoding is canonical for target-free or CVE-free operations, which
    removes aliases without silently repairing an invalid selection.
    """

    metadata = {
        **PatchCascadeGymEnv.metadata,
        "action_schema_version": FLATTENED_ACTION_SCHEMA_VERSION,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(FLATTENED_ACTION_COUNT)

    @staticmethod
    def flatten_action(action_type_idx: int, node_idx: int, vuln_idx: int) -> int:
        if not 0 <= action_type_idx < NUM_ACTION_TYPES:
            raise ValueError("action_type_idx is outside the flattened action schema")
        if not 0 <= node_idx < MAX_NODES:
            raise ValueError("node_idx is outside the flattened action schema")
        if not 0 <= vuln_idx < MAX_VULNS:
            raise ValueError("vuln_idx is outside the flattened action schema")
        return (action_type_idx * MAX_NODES + node_idx) * MAX_VULNS + vuln_idx

    @staticmethod
    def unflatten_action(action_id: int) -> tuple[int, int, int]:
        if not 0 <= action_id < FLATTENED_ACTION_COUNT:
            raise ValueError("action ID is outside the flattened action schema")
        action_type_idx, remainder = divmod(action_id, MAX_NODES * MAX_VULNS)
        node_idx, vuln_idx = divmod(remainder, MAX_VULNS)
        return action_type_idx, node_idx, vuln_idx

    @staticmethod
    def is_canonical_coordinates(action_type_idx: int, node_idx: int, vuln_idx: int) -> bool:
        if action_type_idx == 4:
            return node_idx == 0 and vuln_idx == 0
        if action_type_idx in {0, 1, 3}:
            return vuln_idx == 0
        return action_type_idx == 2

    def _invalid_flat_action(self, action_id: int) -> PatchCascadeAction:
        return PatchCascadeAction(
            action_type=ActionType.SCAN_HOST,
            target=f"__invalid_flat_action_{action_id}__",
            reason="RL agent selected a noncanonical or out-of-range flattened action",
        )

    def _decode_action(self, action: np.ndarray | list[int] | int) -> PatchCascadeAction:
        try:
            action_id = int(np.asarray(action).item())
            action_type_idx, node_idx, vuln_idx = self.unflatten_action(action_id)
        except (TypeError, ValueError):
            return self._invalid_flat_action(-1)
        if not self.is_canonical_coordinates(action_type_idx, node_idx, vuln_idx):
            return self._invalid_flat_action(action_id)
        return super()._decode_action(np.asarray([action_type_idx, node_idx, vuln_idx]))

    def action_masks(self) -> np.ndarray:
        """Return the exact semantic-validity mask required by SB3-contrib."""
        mask = np.zeros(FLATTENED_ACTION_COUNT, dtype=bool)
        if self._obs is None:
            return mask
        for action_id in range(FLATTENED_ACTION_COUNT):
            action_type_idx, node_idx, vuln_idx = self.unflatten_action(action_id)
            if not self.is_canonical_coordinates(action_type_idx, node_idx, vuln_idx):
                continue
            decoded = super()._decode_action(
                np.asarray([action_type_idx, node_idx, vuln_idx])
            )
            mask[action_id] = validate_action_for_observation(decoded, self._obs)[0]
        if not mask.any():
            raise RuntimeError("flattened action contract produced no valid action")
        return mask

    def get_action_mask(self) -> np.ndarray:
        """Compatibility alias; MaskablePPO consumes ``action_masks`` directly."""
        return self.action_masks()


# =============================================================================
# GYMNASIUM REGISTRATION — enables `gym.make("PatchCascade-v2")`
# =============================================================================


def register_envs():
    """Register all PatchCascade environment variants with Gymnasium."""
    task_levels = ["easy", "medium", "hard", "incident_response", "zero_day"]

    for level in task_levels:
        env_id = f"PatchCascade-{level.replace('_', '-').title()}-v2"
        try:
            gym.register(
                id=env_id,
                entry_point="gym_wrapper:PatchCascadeGymEnv",
                kwargs={"task_level": level},
                max_episode_steps={"easy": 30, "medium": 50, "hard": 100,
                                   "incident_response": 60, "zero_day": 80}[level],
            )
        except gym.error.Error:
            pass  # Already registered

    # Default "PatchCascade-v2" uses medium difficulty
    try:
        gym.register(
            id="PatchCascade-v2",
            entry_point="gym_wrapper:PatchCascadeGymEnv",
            kwargs={"task_level": "medium"},
            max_episode_steps=50,
        )
    except gym.error.Error:
        pass


# Auto-register on import
register_envs()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "PatchCascadeGymEnv",
    "register_envs",
    "OBS_SIZE",
    "MAX_NODES",
    "MAX_VULNS",
    "MAX_DEPS",
]
