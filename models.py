"""
PatchCascade SOC - Pydantic Data Models
========================================

This module defines the core data structures for the PatchCascade SOC reinforcement
learning environment. The environment simulates a Security Operations Center engineer
managing vulnerability patches across a network of interdependent servers.

All models are designed for:
- Strict JSON serialization (no complex types)
- Minimal memory footprint
- Maximum clarity for LLM agents reading schema descriptions
- OpenEnv-compatible structure

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# ENUMERATIONS - Discrete value spaces for type safety
# =============================================================================


class NodeState(str, Enum):
    """
    Represents the operational state of a server node in the network.
    
    State Transition Rules:
    - ONLINE -> SUSPENDED: via 'suspend_service' action
    - ONLINE -> PATCHING: via 'apply_patch' action (for non-critical patches)
    - SUSPENDED -> PATCHING: via 'apply_patch' action (required for database nodes)
    - SUSPENDED -> ONLINE: via 'resume_service' action
    - PATCHING -> ONLINE: automatic after patch completes (1 turn)
    - Any -> CRASHED: when a dependency fails or patch fails
    - CRASHED -> ONLINE: via 'resume_service' after root cause resolved
    """
    ONLINE = "online"
    OFFLINE = "offline"
    SUSPENDED = "suspended"
    PATCHING = "patching"
    CRASHED = "crashed"


class CriticalityTier(int, Enum):
    """
    Criticality tier of a server node, determining penalty weights.
    
    Tier 1 (CRITICAL): Core infrastructure (databases, auth servers).
              Downtime penalty multiplier: 3x. Must suspend before patching.
    Tier 2 (IMPORTANT): Business applications (web servers, APIs).
              Downtime penalty multiplier: 2x. Can patch while online.
    Tier 3 (STANDARD): Non-critical services (dev servers, monitoring).
              Downtime penalty multiplier: 1x. Can patch while online.
    """
    CRITICAL = 1
    IMPORTANT = 2
    STANDARD = 3


class SeverityLevel(str, Enum):
    """
    CVSS-based severity level for vulnerabilities.
    
    Maps to CVSS v3.1 scoring ranges:
    - CRITICAL: CVSS 9.0-10.0 (Remote code execution, zero-click exploits)
    - HIGH: CVSS 7.0-8.9 (Privilege escalation, data exfiltration)
    - MEDIUM: CVSS 4.0-6.9 (Denial of service, information disclosure)
    - LOW: CVSS 0.1-3.9 (Minor information leaks, low-impact bugs)
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionType(str, Enum):
    """
    The set of discrete actions available to the SOC agent each turn.
    
    Action Effects:
    - SCAN_HOST: Reveals detailed vulnerability info for target node. No state change.
    - SUSPEND_SERVICE: Gracefully takes node offline. Required before patching Tier 1 nodes.
    - APPLY_PATCH: Fixes one vulnerability on target. Node enters PATCHING state for 1 turn.
    - RESUME_SERVICE: Brings a SUSPENDED or CRASHED node back ONLINE.
    - NOOP: Skip this turn. Use when waiting for patches to complete or strategically.
    """
    SCAN_HOST = "scan_host"
    SUSPEND_SERVICE = "suspend_service"
    APPLY_PATCH = "apply_patch"
    RESUME_SERVICE = "resume_service"
    NOOP = "noop"


# =============================================================================
# NODE & NETWORK TOPOLOGY MODELS
# =============================================================================


class ServerNode(BaseModel):
    """
    Represents a single server/host in the network topology.
    
    A ServerNode is a compute resource that can host services, have vulnerabilities,
    and depend on other nodes. The agent must balance keeping nodes online for
    uptime while taking them offline to patch vulnerabilities.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hostname": "db-primary-01",
                "os": "Ubuntu 22.04 LTS",
                "tier": 1,
                "state": "online",
                "services": ["postgresql", "pgbouncer"],
                "patch_turns_remaining": 0
            }
        }
    )

    hostname: str = Field(
        ...,
        description="Unique identifier for this server. Use this value as 'target' in actions. "
                    "Format: {role}-{qualifier}-{id}, e.g., 'web-frontend-01', 'db-replica-02'.",
        min_length=1,
        max_length=64,
        examples=["web-frontend-01", "db-primary-01", "cache-redis-01"]
    )

    os: str = Field(
        ...,
        description="Operating system and version. Affects which CVEs apply to this node. "
                    "Common values: 'Ubuntu 22.04 LTS', 'RHEL 8.9', 'Windows Server 2022'.",
        examples=["Ubuntu 22.04 LTS", "RHEL 8.9", "Windows Server 2022"]
    )

    tier: CriticalityTier = Field(
        ...,
        description="Criticality tier (1=CRITICAL, 2=IMPORTANT, 3=STANDARD). "
                    "Tier 1 nodes MUST be suspended before patching. "
                    "Higher tier = higher downtime penalty multiplier."
    )

    state: NodeState = Field(
        default=NodeState.ONLINE,
        description="Current operational state. "
                    "ONLINE: Serving traffic, accruing no penalty. "
                    "SUSPENDED: Gracefully offline, safe to patch, minor penalty. "
                    "PATCHING: Patch in progress, will return to ONLINE next turn. "
                    "CRASHED: Failed due to dependency or error, high penalty until resolved. "
                    "OFFLINE: Intentionally powered down (rare)."
    )

    services: list[str] = Field(
        default_factory=list,
        description="List of services running on this node. Used for dependency resolution "
                    "and context. E.g., ['nginx', 'python3'] or ['postgresql', 'pgbouncer'].",
        examples=[["nginx", "gunicorn"], ["postgresql", "pgbouncer"]]
    )

    patch_turns_remaining: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Number of turns until current patch operation completes. "
                    "0 means no patch in progress. When this decrements to 0, "
                    "the node automatically transitions from PATCHING to ONLINE."
    )


class Dependency(BaseModel):
    """
    Represents a directed dependency edge between two server nodes.
    
    If the 'depends_on' node goes OFFLINE, CRASHED, or SUSPENDED, the 'node'
    will automatically CRASH unless it is also SUSPENDED first. This models
    real-world scenarios like web servers crashing when their database goes down.
    
    CRITICAL: The agent must suspend dependent nodes BEFORE suspending their
    dependencies, or cascade failures will occur.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node": "web-frontend-01",
                "depends_on": "db-primary-01",
                "dependency_type": "hard",
                "description": "Web frontend requires database for all operations"
            }
        }
    )

    node: str = Field(
        ...,
        description="Hostname of the dependent node (the one that will fail if dependency is down).",
        examples=["web-frontend-01", "api-gateway-01"]
    )

    depends_on: str = Field(
        ...,
        description="Hostname of the dependency node (the one that must stay up). "
                    "If this node goes down, 'node' will CRASH.",
        examples=["db-primary-01", "cache-redis-01"]
    )

    dependency_type: Literal["hard", "soft"] = Field(
        default="hard",
        description="'hard': Dependent node CRASHES immediately if dependency fails. "
                    "'soft': Dependent node degrades but stays ONLINE (reduced performance)."
    )

    description: str = Field(
        default="",
        description="Human-readable explanation of why this dependency exists. "
                    "Helps the agent understand the relationship.",
        examples=["Web frontend requires database for all operations"]
    )


# =============================================================================
# VULNERABILITY MODELS
# =============================================================================


class Vulnerability(BaseModel):
    """
    Represents an active CVE (Common Vulnerabilities and Exposures) on the network.
    
    Vulnerabilities are the core problem the agent must solve. Each turn a
    vulnerability remains unpatched on an ONLINE node, it accrues risk penalty.
    The agent must prioritize patching based on severity, affected node criticality,
    and dependency constraints.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cve_id": "CVE-2024-1234",
                "severity": "critical",
                "cvss_score": 9.8,
                "affected_hosts": ["web-frontend-01", "web-frontend-02"],
                "description": "Remote code execution via crafted HTTP request",
                "patch_available": True,
                "exploit_in_wild": True
            }
        }
    )

    cve_id: str = Field(
        ...,
        description="Unique CVE identifier. Format: 'CVE-YYYY-NNNNN'. "
                    "Used to track which vulnerabilities have been patched.",
        pattern=r"^CVE-\d{4}-\d{4,}$",
        examples=["CVE-2024-1234", "CVE-2023-44487"]
    )

    severity: SeverityLevel = Field(
        ...,
        description="CVSS-based severity. CRITICAL and HIGH should be prioritized. "
                    "Severity affects the per-turn risk penalty while unpatched."
    )

    cvss_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="CVSS v3.1 base score (0.0-10.0). Higher = more dangerous. "
                    "Used to calculate exact risk penalty. "
                    "CRITICAL: 9.0-10.0, HIGH: 7.0-8.9, MEDIUM: 4.0-6.9, LOW: 0.1-3.9."
    )

    affected_hosts: list[str] = Field(
        ...,
        min_length=1,
        description="List of hostnames where this CVE is present and unpatched. "
                    "When a host is successfully patched, it is removed from this list. "
                    "When this list becomes empty, the CVE is resolved network-wide."
    )

    description: str = Field(
        default="",
        description="Brief description of the vulnerability and its impact. "
                    "Helps the agent understand the risk.",
        examples=["Remote code execution via crafted HTTP request"]
    )

    patch_available: bool = Field(
        default=True,
        description="Whether a patch exists for this CVE. If False, the agent cannot "
                    "fix it directly and must use other mitigations (out of scope)."
    )

    exploit_in_wild: bool = Field(
        default=False,
        description="Whether this CVE is being actively exploited in the wild. "
                    "If True, risk penalty is DOUBLED. Prioritize these!"
    )


# =============================================================================
# NETWORK HEALTH METRICS
# =============================================================================


class NetworkHealth(BaseModel):
    """
    Aggregate health metrics for the entire network.
    
    These metrics help the agent understand the overall state at a glance
    without needing to iterate through all nodes and vulnerabilities.
    The agent should aim to minimize penalties and maximize uptime.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_nodes": 10,
                "nodes_online": 7,
                "nodes_crashed": 1,
                "nodes_patching": 2,
                "active_critical_vulns": 3,
                "active_high_vulns": 5,
                "active_medium_vulns": 8,
                "active_low_vulns": 2,
                "cumulative_downtime_penalty": 45.5,
                "cumulative_risk_penalty": 120.0,
                "turn_number": 15
            }
        }
    )

    total_nodes: int = Field(
        ...,
        ge=0,
        description="Total number of server nodes in the network topology."
    )

    nodes_online: int = Field(
        ...,
        ge=0,
        description="Count of nodes currently in ONLINE state and serving traffic."
    )

    nodes_crashed: int = Field(
        ...,
        ge=0,
        description="Count of nodes currently in CRASHED state. "
                    "These accrue HIGH penalty each turn. Fix immediately!"
    )

    nodes_patching: int = Field(
        ...,
        ge=0,
        description="Count of nodes currently in PATCHING state. "
                    "These will return to ONLINE when patch completes."
    )

    active_critical_vulns: int = Field(
        ...,
        ge=0,
        description="Count of unpatched CRITICAL severity vulnerabilities (CVSS 9.0+). "
                    "Each accrues ~10 risk penalty per turn per affected host."
    )

    active_high_vulns: int = Field(
        ...,
        ge=0,
        description="Count of unpatched HIGH severity vulnerabilities (CVSS 7.0-8.9). "
                    "Each accrues ~5 risk penalty per turn per affected host."
    )

    active_medium_vulns: int = Field(
        ...,
        ge=0,
        description="Count of unpatched MEDIUM severity vulnerabilities (CVSS 4.0-6.9). "
                    "Each accrues ~2 risk penalty per turn per affected host."
    )

    active_low_vulns: int = Field(
        ...,
        ge=0,
        description="Count of unpatched LOW severity vulnerabilities (CVSS 0.1-3.9). "
                    "Each accrues ~0.5 risk penalty per turn per affected host."
    )

    cumulative_downtime_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description="Total downtime penalty accumulated so far. "
                    "Increases each turn for every non-ONLINE node, weighted by tier. "
                    "GOAL: Minimize this value."
    )

    cumulative_risk_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description="Total risk penalty accumulated so far. "
                    "Increases each turn for every unpatched vulnerability on ONLINE nodes. "
                    "GOAL: Minimize this value."
    )

    turn_number: int = Field(
        default=0,
        ge=0,
        description="Current turn/step number in the episode. "
                    "Episodes typically last 50-100 turns."
    )


# =============================================================================
# OBSERVATION MODEL (What the agent sees each turn)
# =============================================================================


class PatchCascadeObservation(BaseModel):
    """
    The complete observation space for the PatchCascade SOC environment.
    
    This is what the agent receives at the start of each turn. It contains
    all information needed to make an optimal patching decision:
    - Current state of all servers
    - Active vulnerabilities and which hosts they affect
    - Dependency graph (which nodes rely on which)
    - Aggregate health metrics
    
    The agent should analyze this observation to determine the best action
    that balances:
    1. Patching critical vulnerabilities quickly (minimize risk penalty)
    2. Keeping services online (minimize downtime penalty)
    3. Respecting dependencies (avoid cascade failures)
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "nodes": [
                    {
                        "hostname": "db-primary-01",
                        "os": "Ubuntu 22.04 LTS",
                        "tier": 1,
                        "state": "online",
                        "services": ["postgresql"],
                        "patch_turns_remaining": 0
                    }
                ],
                "vulnerabilities": [
                    {
                        "cve_id": "CVE-2024-1234",
                        "severity": "critical",
                        "cvss_score": 9.8,
                        "affected_hosts": ["db-primary-01"],
                        "description": "SQL injection in PostgreSQL",
                        "patch_available": True,
                        "exploit_in_wild": True
                    }
                ],
                "dependencies": [
                    {
                        "node": "web-frontend-01",
                        "depends_on": "db-primary-01",
                        "dependency_type": "hard"
                    }
                ],
                "health": {
                    "total_nodes": 5,
                    "nodes_online": 5,
                    "nodes_crashed": 0,
                    "nodes_patching": 0,
                    "active_critical_vulns": 1,
                    "active_high_vulns": 0,
                    "active_medium_vulns": 0,
                    "active_low_vulns": 0,
                    "cumulative_downtime_penalty": 0.0,
                    "cumulative_risk_penalty": 0.0,
                    "turn_number": 0
                },
                "last_action_result": None,
                "messages": ["Episode started. 1 critical vulnerability detected."]
            }
        }
    )

    nodes: list[ServerNode] = Field(
        ...,
        description="List of all server nodes in the network. "
                    "Iterate through this to find patch targets and check states. "
                    "Use hostname as the 'target' argument in actions."
    )

    vulnerabilities: list[Vulnerability] = Field(
        ...,
        description="List of all active (unpatched) vulnerabilities on the network. "
                    "When all hosts in 'affected_hosts' are patched, the CVE is removed. "
                    "Prioritize CRITICAL and HIGH severity, especially if exploit_in_wild=True."
    )

    dependencies: list[Dependency] = Field(
        ...,
        description="List of all dependency edges in the network topology. "
                    "CRITICAL: Before suspending or patching a node, check if other nodes "
                    "depend on it. Suspend dependents FIRST to avoid cascade crashes."
    )

    health: NetworkHealth = Field(
        ...,
        description="Aggregate health metrics for quick assessment. "
                    "Check nodes_crashed > 0 to detect cascade failures that need fixing."
    )

    last_action_result: str | None = Field(
        default=None,
        description="Result message from the previous action. "
                    "'success': Action completed as expected. "
                    "'invalid_target': Target hostname does not exist. "
                    "'invalid_state': Target is in wrong state for this action. "
                    "'dependency_violation': Action would cause cascade failure. "
                    "None if this is the first turn."
    )

    messages: list[str] = Field(
        default_factory=list,
        description="System messages and alerts for this turn. "
                    "May include warnings about imminent threats, patch completions, etc."
    )


# =============================================================================
# ACTION MODEL (What the agent can do)
# =============================================================================


class PatchCascadeAction(BaseModel):
    """
    The action space for the PatchCascade SOC environment.
    
    Each turn, the agent must choose exactly ONE action. Actions are atomic
    and take effect immediately (except APPLY_PATCH which takes 1 turn).
    
    Action Selection Guide:
    - SCAN_HOST: Use to get detailed info. Useful early in episode.
    - SUSPEND_SERVICE: Use before patching Tier 1 (CRITICAL) nodes.
    - APPLY_PATCH: The core action. Fixes one CVE on one host.
    - RESUME_SERVICE: Use after patching or to recover CRASHED nodes.
    - NOOP: Use when waiting for patches or no good action available.
    
    Target Rules:
    - SCAN_HOST: Target must exist.
    - SUSPEND_SERVICE: Target must be ONLINE.
    - APPLY_PATCH: Target must be ONLINE (Tier 2-3) or SUSPENDED (Tier 1).
    - RESUME_SERVICE: Target must be SUSPENDED or CRASHED.
    - NOOP: Target is ignored (use empty string or any value).
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action_type": "apply_patch",
                "target": "db-primary-01",
                "cve_id": "CVE-2024-1234",
                "reason": "Patching critical SQLi vulnerability on primary database"
            }
        }
    )

    action_type: ActionType = Field(
        ...,
        description="The type of action to perform this turn. "
                    "SCAN_HOST: Read detailed state (no state change). "
                    "SUSPEND_SERVICE: Take node offline gracefully. "
                    "APPLY_PATCH: Fix vulnerability (requires cve_id). "
                    "RESUME_SERVICE: Bring node back online. "
                    "NOOP: Skip turn."
    )

    target: str = Field(
        default="",
        description="Hostname of the target server for this action. "
                    "Must match a hostname from observation.nodes exactly. "
                    "Leave empty or any value for NOOP action.",
        examples=["db-primary-01", "web-frontend-01", ""]
    )

    cve_id: str | None = Field(
        default=None,
        description="CVE identifier to patch. REQUIRED when action_type is APPLY_PATCH. "
                    "Must match a cve_id from observation.vulnerabilities that affects the target. "
                    "Ignored for other action types.",
        pattern=r"^CVE-\d{4}-\d{4,}$|^$",
        examples=["CVE-2024-1234", None]
    )

    reason: str = Field(
        default="",
        description="Optional reasoning for this action. Not used by the environment, "
                    "but useful for logging, debugging, and training analysis. "
                    "Good agents explain their decisions.",
        max_length=256
    )


# =============================================================================
# STATE MODEL (Internal simulation state - not directly visible to agent)
# =============================================================================


class PatchCascadeState(BaseModel):
    """
    Complete internal state of the PatchCascade SOC environment.
    
    This model contains everything needed to:
    1. Generate observations for the agent
    2. Process actions and compute next state
    3. Calculate rewards and penalties
    4. Determine episode termination
    
    The State is more detailed than the Observation. Some fields (like
    random seeds, reward history) are not exposed to the agent.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "nodes": [],
                "vulnerabilities": [],
                "dependencies": [],
                "health": {},
                "turn_number": 0,
                "max_turns": 50,
                "episode_seed": 42,
                "reward_history": [],
                "action_history": [],
                "is_terminated": False,
                "termination_reason": None
            }
        }
    )

    # --- Core State (mirrored in Observation) ---

    nodes: list[ServerNode] = Field(
        ...,
        description="All server nodes and their current states."
    )

    vulnerabilities: list[Vulnerability] = Field(
        ...,
        description="All active vulnerabilities. Updated when patches succeed."
    )

    dependencies: list[Dependency] = Field(
        ...,
        description="Dependency graph edges. Immutable during episode."
    )

    health: NetworkHealth = Field(
        ...,
        description="Aggregate health metrics. Recalculated each turn."
    )

    # --- Episode Tracking ---

    turn_number: int = Field(
        default=0,
        ge=0,
        description="Current turn number (0-indexed)."
    )

    max_turns: int = Field(
        default=50,
        ge=1,
        description="Maximum turns before episode ends (timeout)."
    )

    episode_seed: int | None = Field(
        default=None,
        description="Random seed used to generate this episode. For reproducibility."
    )

    # --- History (for analysis and grading) ---

    reward_history: list[float] = Field(
        default_factory=list,
        description="Reward received at each turn. Sum = total episode reward."
    )

    action_history: list[PatchCascadeAction] = Field(
        default_factory=list,
        description="All actions taken by the agent this episode."
    )

    # --- Termination ---

    is_terminated: bool = Field(
        default=False,
        description="Whether the episode has ended."
    )

    termination_reason: Literal[
        "all_patched",
        "max_turns_reached", 
        "all_crashed",
        "agent_quit",
        None
    ] = Field(
        default=None,
        description="Why the episode ended. "
                    "'all_patched': Agent successfully patched all CVEs (VICTORY). "
                    "'max_turns_reached': Timeout. "
                    "'all_crashed': All nodes crashed (CATASTROPHIC FAILURE). "
                    "'agent_quit': Agent sent quit signal. "
                    "None: Episode still running."
    )


# =============================================================================
# CONVENIENCE TYPE ALIASES
# =============================================================================

# Type aliases for clearer function signatures
Observation = PatchCascadeObservation
Action = PatchCascadeAction
State = PatchCascadeState


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_action_for_observation(action: PatchCascadeAction, obs: PatchCascadeObservation) -> tuple[bool, str]:
    """
    Validate an action against the current observation.
    
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    
    This is a helper for pre-validation. The environment.step() function
    performs the authoritative validation.
    """
    # NOOP is always valid
    if action.action_type == ActionType.NOOP:
        return True, ""
    
    # Check target exists
    hostnames = {node.hostname for node in obs.nodes}
    if action.target not in hostnames:
        return False, f"invalid_target: '{action.target}' not found in network"
    
    # Get target node
    target_node = next(n for n in obs.nodes if n.hostname == action.target)
    
    # Validate based on action type
    if action.action_type == ActionType.SCAN_HOST:
        return True, ""  # Can scan any existing host
    
    elif action.action_type == ActionType.SUSPEND_SERVICE:
        if target_node.state != NodeState.ONLINE:
            return False, f"invalid_state: Cannot suspend '{action.target}' (state={target_node.state}, requires ONLINE)"
        return True, ""
    
    elif action.action_type == ActionType.APPLY_PATCH:
        # Check CVE ID provided
        if not action.cve_id:
            return False, "missing_cve_id: APPLY_PATCH requires cve_id"
        
        # Check CVE exists and affects target
        matching_cves = [v for v in obs.vulnerabilities if v.cve_id == action.cve_id]
        if not matching_cves:
            return False, f"invalid_cve: '{action.cve_id}' not found in active vulnerabilities"
        
        cve = matching_cves[0]
        if action.target not in cve.affected_hosts:
            return False, f"invalid_target_for_cve: '{action.target}' not affected by {action.cve_id}"
        
        # Check node state
        valid_states = {NodeState.ONLINE, NodeState.SUSPENDED}
        if target_node.state not in valid_states:
            return False, f"invalid_state: Cannot patch '{action.target}' (state={target_node.state})"
        
        # Tier 1 must be suspended
        if target_node.tier == CriticalityTier.CRITICAL and target_node.state != NodeState.SUSPENDED:
            return False, f"dependency_violation: Tier 1 node '{action.target}' must be SUSPENDED before patching"
        
        return True, ""
    
    elif action.action_type == ActionType.RESUME_SERVICE:
        valid_states = {NodeState.SUSPENDED, NodeState.CRASHED}
        if target_node.state not in valid_states:
            return False, f"invalid_state: Cannot resume '{action.target}' (state={target_node.state}, requires SUSPENDED or CRASHED)"
        return True, ""
    
    return False, f"unknown_action_type: {action.action_type}"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "NodeState",
    "CriticalityTier", 
    "SeverityLevel",
    "ActionType",
    # Core Models
    "ServerNode",
    "Dependency",
    "Vulnerability",
    "NetworkHealth",
    # Main Environment Models
    "PatchCascadeObservation",
    "PatchCascadeAction",
    "PatchCascadeState",
    # Aliases
    "Observation",
    "Action",
    "State",
    # Helpers
    "validate_action_for_observation",
]
