"""
PatchCascade SOC - Environment Core Logic
==========================================

This module implements the PatchCascadeEnv class, the core reinforcement learning
environment for the PatchCascade SOC simulation. It provides:

- `reset(task_level)`: Initialize episodes at 5 difficulty levels:
    - easy: 3-5 nodes, no dependencies, 1 vulnerability
    - medium: 5-8 nodes, linear dependency chain, 2 vulnerabilities
    - hard: 10-15 nodes, complex graph, multiple critical vulns
    - incident_response: Active breach scenario with pre-crashed nodes
    - zero_day: Dynamic threat injection with exploit spreading
- `step(action)`: Process agent actions and advance simulation state
- `get_observation()`: Generate agent-visible observation from internal state

Advanced Features:
- Dynamic Event System: Events fire at configurable turns (new CVEs, exploit spreading)
- Exploit Spreading: Unpatched exploited CVEs can spread to connected nodes
- Multi-objective Optimization: Balance risk, downtime, and cascade avoidance

The environment follows OpenEnv conventions and returns standard
(observation, reward, done, info) tuples from step().

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Literal

from models import (
    # Enums
    ActionType,
    CriticalityTier,
    NodeState,
    SeverityLevel,
    # Models
    Dependency,
    NetworkHealth,
    PatchCascadeAction,
    PatchCascadeObservation,
    PatchCascadeState,
    ServerNode,
    Vulnerability,
    # Helpers
    validate_action_for_observation,
)


# =============================================================================
# CONSTANTS - Penalty weights and environment parameters
# =============================================================================

# Downtime penalty multipliers by criticality tier
DOWNTIME_PENALTY_MULTIPLIER: dict[CriticalityTier, float] = {
    CriticalityTier.CRITICAL: 3.0,   # Tier 1: Core infrastructure
    CriticalityTier.IMPORTANT: 2.0,  # Tier 2: Business applications
    CriticalityTier.STANDARD: 1.0,   # Tier 3: Non-critical services
}

# Additional multiplier for CRASHED state (doubles the penalty)
CRASHED_PENALTY_MULTIPLIER: float = 2.0

# Multiplier for vulnerabilities being actively exploited
EXPLOIT_IN_WILD_MULTIPLIER: float = 2.0

# Penalty for invalid actions (discourages random/invalid moves)
INVALID_ACTION_PENALTY: float = -0.5

# Bonus reward for completing all patches (victory condition)
VICTORY_BONUS: float = 50.0

# Penalty for catastrophic failure (all nodes crashed)
CATASTROPHIC_FAILURE_PENALTY: float = -100.0

# Small per-turn time pressure — ensures dense (non-zero) reward every step,
# even when state doesn't change (e.g. noop). Incentivizes efficient patching.
TIME_PRESSURE_PENALTY: float = -0.1

# Discount used by the canonical PPO configuration and by potential-based
# shaping. Keeping these equal is required for the policy-invariance theorem.
SHAPING_GAMMA: float = 0.99

# Default max turns by difficulty
MAX_TURNS_BY_DIFFICULTY: dict[str, int] = {
    "easy": 30,
    "medium": 50,
    "hard": 100,
    "incident_response": 60,
    "zero_day": 80,
}

# Exploit spreading: after this many turns unpatched on an ONLINE node,
# an exploit_in_wild vulnerability spreads to a connected node
EXPLOIT_SPREAD_THRESHOLD: int = 4

# Dynamic event injection turns for zero_day task
ZERO_DAY_EVENT_TURNS: list[int] = [5, 15]


# =============================================================================
# STEP RESULT DATACLASS
# =============================================================================

@dataclass
class StepResult:
    """
    Standard return type from environment.step().
    
    Compatible with OpenEnv and Gymnasium conventions.
    """
    observation: PatchCascadeObservation
    reward: float
    done: bool
    truncated: bool  # True if episode ended due to max_turns, not terminal state
    info: dict
    
    def as_tuple(self) -> tuple:
        """Convert to standard (obs, reward, done, truncated, info) tuple."""
        return (self.observation, self.reward, self.done, self.truncated, self.info)


# =============================================================================
# ENVIRONMENT CLASS
# =============================================================================

class PatchCascadeEnv:
    """
    PatchCascade SOC Reinforcement Learning Environment.
    
    Simulates a Security Operations Center engineer managing vulnerability
    patches across a network of interdependent servers. The agent must
    balance patching vulnerabilities (reducing risk) with keeping services
    online (reducing downtime).
    
    Usage:
        env = PatchCascadeEnv()
        obs = env.reset(task_level="medium")
        
        while not done:
            action = agent.decide(obs)
            result = env.step(action)
            obs, reward, done, truncated, info = result.as_tuple()
    """
    
    def __init__(self, seed: int | None = None):
        """
        Initialize the environment.
        
        Args:
            seed: Random seed for reproducibility. If None, uses system entropy.
        """
        self._rng = random.Random(seed)
        self._state: PatchCascadeState | None = None
        self._last_total_penalty: float = 0.0
        self._pending_patches: dict[str, str] = {}  # hostname -> cve_id being patched
        self._last_action_result: str | None = None
        self._messages: list[str] = []
        self._task_level: str = "easy"
        self._exploit_turn_tracker: dict[str, int] = {}  # cve_id -> turns unpatched on online node
        self._cascade_failure_count: int = 0  # Total cascades this episode
        self._invalid_action_count: int = 0  # Total invalid actions this episode
        self._initial_vuln_count: int = 0  # Vulnerabilities at episode start
        self._dynamic_events_fired: list[str] = []  # Track which events have fired
    
    @property
    def state(self) -> PatchCascadeState:
        """Access internal state (for debugging/grading only)."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state
    
    # =========================================================================
    # RESET - Episode Initialization
    # =========================================================================
    
    def reset(
        self,
        task_level: Literal["easy", "medium", "hard", "incident_response", "zero_day"] = "easy",
        seed: int | None = None,
    ) -> PatchCascadeObservation:
        """
        Reset the environment to a new episode.
        
        Args:
            task_level: Difficulty level determining network complexity.
                - "easy": 3-5 nodes, no dependencies, 1 vulnerability
                - "medium": 5-8 nodes, linear dependency chain, 2 vulnerabilities
                - "hard": 10-15 nodes, complex graph, multiple critical vulns
                - "incident_response": Active breach with pre-crashed nodes, exploit spreading
                - "zero_day": Dynamic CVE injection at turns 5 and 15
            seed: Optional seed for this episode (overrides constructor seed).
        
        Returns:
            Initial observation for the agent.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        
        episode_seed = self._rng.randint(0, 2**31 - 1)
        self._task_level = task_level
        
        # Generate scenario based on difficulty
        if task_level == "easy":
            nodes, dependencies, vulnerabilities = self._generate_easy_scenario()
            max_turns = MAX_TURNS_BY_DIFFICULTY["easy"]
        elif task_level == "medium":
            nodes, dependencies, vulnerabilities = self._generate_medium_scenario()
            max_turns = MAX_TURNS_BY_DIFFICULTY["medium"]
        elif task_level == "hard":
            nodes, dependencies, vulnerabilities = self._generate_hard_scenario()
            max_turns = MAX_TURNS_BY_DIFFICULTY["hard"]
        elif task_level == "incident_response":
            nodes, dependencies, vulnerabilities = self._generate_incident_response_scenario()
            max_turns = MAX_TURNS_BY_DIFFICULTY["incident_response"]
        elif task_level == "zero_day":
            nodes, dependencies, vulnerabilities = self._generate_zero_day_scenario()
            max_turns = MAX_TURNS_BY_DIFFICULTY["zero_day"]
        else:
            raise ValueError(
                f"Invalid task_level: {task_level}. "
                f"Must be one of: easy, medium, hard, incident_response, zero_day."
            )
        
        # A reset starts a new episode.  Clear the prior state before deriving
        # health so cumulative risk/downtime cannot leak across seeded resets.
        self._state = None

        # Calculate initial health metrics
        health = self._calculate_health_metrics(nodes, vulnerabilities, turn_number=0)
        
        # Initialize state
        self._state = PatchCascadeState(
            nodes=nodes,
            vulnerabilities=vulnerabilities,
            dependencies=dependencies,
            health=health,
            turn_number=0,
            max_turns=max_turns,
            episode_seed=episode_seed,
            reward_history=[],
            action_history=[],
            is_terminated=False,
            termination_reason=None,
            messages=[],  # Initialize messages for dynamic events
        )
        
        # Reset tracking variables
        self._last_total_penalty = self._calculate_total_penalty(nodes, vulnerabilities)
        self._pending_patches = {}
        self._last_action_result = None
        self._exploit_turn_tracker = {}
        self._cascade_failure_count = 0
        self._invalid_action_count = 0
        self._initial_vuln_count = len(vulnerabilities)
        self._dynamic_events_fired = []
        
        # Build task-specific intro messages
        intro_msgs = [f"Episode started ({task_level} difficulty). {len(vulnerabilities)} vulnerabilities detected."]
        if task_level == "incident_response":
            intro_msgs.append("⚠️ ACTIVE BREACH: Multiple nodes are already compromised. Triage immediately!")
            intro_msgs.append("⚠️ Exploits are spreading — unpatched exploited CVEs will infect connected nodes.")
        elif task_level == "zero_day":
            intro_msgs.append("🔍 Intel reports suggest undisclosed zero-day vulnerabilities may emerge during this operation.")
            intro_msgs.append("📡 Stay alert for dynamic threat advisories.")
        
        self._messages = intro_msgs
        
        return self.get_observation()
    
    def _generate_easy_scenario(self) -> tuple[list[ServerNode], list[Dependency], list[Vulnerability]]:
        """
        Generate an easy scenario: 3-5 nodes, no dependencies, 1 vulnerability.
        
        Perfect for learning basic patch mechanics without cascade complexity.
        """
        num_nodes = self._rng.randint(3, 5)
        
        nodes = [
            ServerNode(
                hostname="web-server-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "gunicorn"],
            ),
            ServerNode(
                hostname="api-server-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["python3", "fastapi"],
            ),
            ServerNode(
                hostname="dev-server-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["nodejs", "npm"],
            ),
        ]
        
        # Add extra nodes if needed
        if num_nodes >= 4:
            nodes.append(ServerNode(
                hostname="monitoring-01",
                os="RHEL 8.9",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["prometheus", "grafana"],
            ))
        if num_nodes >= 5:
            nodes.append(ServerNode(
                hostname="backup-server-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["rsync", "cron"],
            ))
        
        # No dependencies in easy mode
        dependencies: list[Dependency] = []
        
        # Single medium/high vulnerability affecting one or two nodes
        severity = self._rng.choice([SeverityLevel.MEDIUM, SeverityLevel.HIGH])
        cvss = self._rng.uniform(5.0, 8.5) if severity == SeverityLevel.MEDIUM else self._rng.uniform(7.0, 8.9)
        
        affected = self._rng.sample([n.hostname for n in nodes], k=min(2, len(nodes)))
        
        vulnerabilities = [
            Vulnerability(
                cve_id="CVE-2024-1001",
                severity=severity,
                cvss_score=round(cvss, 1),
                affected_hosts=affected,
                description="Remote code execution in web framework",
                patch_available=True,
                exploit_in_wild=False,
            )
        ]
        
        return nodes, dependencies, vulnerabilities
    
    def _generate_medium_scenario(self) -> tuple[list[ServerNode], list[Dependency], list[Vulnerability]]:
        """
        Generate a medium scenario: 5-8 nodes, linear dependency chain, 2 vulnerabilities.
        
        Introduces dependency management: Web -> App -> DB pattern.
        One vulnerability on a Tier 1 node requires suspend-patch-resume workflow.
        """
        nodes = [
            # Tier 1 - Database layer (must suspend before patching)
            ServerNode(
                hostname="db-primary-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.CRITICAL,
                state=NodeState.ONLINE,
                services=["postgresql", "pgbouncer"],
            ),
            # Tier 2 - Application layer
            ServerNode(
                hostname="app-server-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["python3", "django"],
            ),
            ServerNode(
                hostname="app-server-02",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["python3", "django"],
            ),
            # Tier 2 - Web layer
            ServerNode(
                hostname="web-frontend-01",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "nodejs"],
            ),
            ServerNode(
                hostname="web-frontend-02",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "nodejs"],
            ),
            # Tier 3 - Supporting services
            ServerNode(
                hostname="cache-redis-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["redis-server"],
            ),
        ]
        
        # Add 1-2 extra nodes randomly
        extra_count = self._rng.randint(0, 2)
        if extra_count >= 1:
            nodes.append(ServerNode(
                hostname="worker-queue-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["celery", "rabbitmq"],
            ))
        if extra_count >= 2:
            nodes.append(ServerNode(
                hostname="logging-elk-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["elasticsearch", "logstash", "kibana"],
            ))
        
        # Linear dependency chain: Web -> App -> DB
        dependencies = [
            Dependency(
                node="web-frontend-01",
                depends_on="app-server-01",
                dependency_type="hard",
                description="Web frontend proxies to app server",
            ),
            Dependency(
                node="web-frontend-02",
                depends_on="app-server-02",
                dependency_type="hard",
                description="Web frontend proxies to app server",
            ),
            Dependency(
                node="app-server-01",
                depends_on="db-primary-01",
                dependency_type="hard",
                description="App server requires database",
            ),
            Dependency(
                node="app-server-02",
                depends_on="db-primary-01",
                dependency_type="hard",
                description="App server requires database",
            ),
        ]
        
        # Two vulnerabilities: one on Tier 1 (critical), one on Tier 2/3
        vulnerabilities = [
            Vulnerability(
                cve_id="CVE-2024-2001",
                severity=SeverityLevel.CRITICAL,
                cvss_score=9.1,
                affected_hosts=["db-primary-01"],
                description="SQL injection in PostgreSQL stored procedures",
                patch_available=True,
                exploit_in_wild=False,
            ),
            Vulnerability(
                cve_id="CVE-2024-2002",
                severity=SeverityLevel.HIGH,
                cvss_score=7.5,
                affected_hosts=["web-frontend-01", "web-frontend-02"],
                description="XSS vulnerability in Nginx configuration",
                patch_available=True,
                exploit_in_wild=False,
            ),
        ]
        
        return nodes, dependencies, vulnerabilities
    
    def _generate_hard_scenario(self) -> tuple[list[ServerNode], list[Dependency], list[Vulnerability]]:
        """
        Generate a hard scenario: 10-15 nodes, complex dependency graph, multiple critical vulns.
        
        Features:
        - Multiple Tier 1 nodes (database cluster, auth server)
        - Load balancer -> multiple web servers -> multiple app servers -> DB cluster
        - Some vulnerabilities actively exploited (doubled penalty)
        """
        nodes = [
            # Tier 1 - Database cluster
            ServerNode(
                hostname="db-primary-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.CRITICAL,
                state=NodeState.ONLINE,
                services=["postgresql", "patroni"],
            ),
            ServerNode(
                hostname="db-replica-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.CRITICAL,
                state=NodeState.ONLINE,
                services=["postgresql", "patroni"],
            ),
            # Tier 1 - Auth server
            ServerNode(
                hostname="auth-server-01",
                os="RHEL 8.9",
                tier=CriticalityTier.CRITICAL,
                state=NodeState.ONLINE,
                services=["keycloak", "java"],
            ),
            # Tier 2 - Load balancers
            ServerNode(
                hostname="lb-primary-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["haproxy", "keepalived"],
            ),
            ServerNode(
                hostname="lb-secondary-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["haproxy", "keepalived"],
            ),
            # Tier 2 - Web servers
            ServerNode(
                hostname="web-frontend-01",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "nodejs"],
            ),
            ServerNode(
                hostname="web-frontend-02",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "nodejs"],
            ),
            ServerNode(
                hostname="web-frontend-03",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "nodejs"],
            ),
            # Tier 2 - App servers
            ServerNode(
                hostname="app-server-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["python3", "django", "gunicorn"],
            ),
            ServerNode(
                hostname="app-server-02",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["python3", "django", "gunicorn"],
            ),
            # Tier 3 - Supporting infrastructure
            ServerNode(
                hostname="cache-redis-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["redis-cluster"],
            ),
            ServerNode(
                hostname="mq-rabbitmq-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["rabbitmq", "erlang"],
            ),
            ServerNode(
                hostname="monitoring-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["prometheus", "grafana", "alertmanager"],
            ),
        ]
        
        # Add 0-2 extra nodes
        extra_count = self._rng.randint(0, 2)
        if extra_count >= 1:
            nodes.append(ServerNode(
                hostname="logging-elk-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["elasticsearch", "logstash", "kibana"],
            ))
        if extra_count >= 2:
            nodes.append(ServerNode(
                hostname="ci-jenkins-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["jenkins", "docker"],
            ))
        
        # Complex dependency graph
        dependencies = [
            # Web servers depend on app servers (distributed)
            Dependency(node="web-frontend-01", depends_on="app-server-01", dependency_type="hard"),
            Dependency(node="web-frontend-02", depends_on="app-server-01", dependency_type="hard"),
            Dependency(node="web-frontend-02", depends_on="app-server-02", dependency_type="hard"),
            Dependency(node="web-frontend-03", depends_on="app-server-02", dependency_type="hard"),
            # App servers depend on database
            Dependency(node="app-server-01", depends_on="db-primary-01", dependency_type="hard"),
            Dependency(node="app-server-02", depends_on="db-primary-01", dependency_type="hard"),
            # App servers depend on auth
            Dependency(node="app-server-01", depends_on="auth-server-01", dependency_type="hard"),
            Dependency(node="app-server-02", depends_on="auth-server-01", dependency_type="hard"),
            # Load balancers depend on web servers (soft - can degrade)
            Dependency(node="lb-primary-01", depends_on="web-frontend-01", dependency_type="soft"),
            Dependency(node="lb-primary-01", depends_on="web-frontend-02", dependency_type="soft"),
            Dependency(node="lb-secondary-01", depends_on="web-frontend-02", dependency_type="soft"),
            Dependency(node="lb-secondary-01", depends_on="web-frontend-03", dependency_type="soft"),
            # DB replica depends on primary
            Dependency(node="db-replica-01", depends_on="db-primary-01", dependency_type="hard"),
        ]
        
        # Multiple vulnerabilities with varying severity
        vulnerabilities = [
            Vulnerability(
                cve_id="CVE-2024-3001",
                severity=SeverityLevel.CRITICAL,
                cvss_score=9.8,
                affected_hosts=["db-primary-01", "db-replica-01"],
                description="Remote code execution in PostgreSQL replication protocol",
                patch_available=True,
                exploit_in_wild=True,  # ACTIVELY EXPLOITED!
            ),
            Vulnerability(
                cve_id="CVE-2024-3002",
                severity=SeverityLevel.CRITICAL,
                cvss_score=9.1,
                affected_hosts=["auth-server-01"],
                description="Authentication bypass in Keycloak SAML parser",
                patch_available=True,
                exploit_in_wild=True,  # ACTIVELY EXPLOITED!
            ),
            Vulnerability(
                cve_id="CVE-2024-3003",
                severity=SeverityLevel.HIGH,
                cvss_score=8.2,
                affected_hosts=["web-frontend-01", "web-frontend-02", "web-frontend-03"],
                description="Server-side request forgery in Nginx",
                patch_available=True,
                exploit_in_wild=False,
            ),
            Vulnerability(
                cve_id="CVE-2024-3004",
                severity=SeverityLevel.HIGH,
                cvss_score=7.5,
                affected_hosts=["app-server-01", "app-server-02"],
                description="Deserialization vulnerability in Django REST framework",
                patch_available=True,
                exploit_in_wild=False,
            ),
            Vulnerability(
                cve_id="CVE-2024-3005",
                severity=SeverityLevel.MEDIUM,
                cvss_score=5.3,
                affected_hosts=["mq-rabbitmq-01"],
                description="Information disclosure in RabbitMQ management API",
                patch_available=True,
                exploit_in_wild=False,
            ),
        ]
        
        return nodes, dependencies, vulnerabilities
    
    def _generate_incident_response_scenario(
        self,
    ) -> tuple[list[ServerNode], list[Dependency], list[Vulnerability]]:
        """
        Generate an incident response scenario: active breach in progress.
        
        Key Features:
        - 8 nodes with 2 already CRASHED (simulating ongoing attack)
        - Complex dependency graph with both hard and soft dependencies
        - 3 vulnerabilities, 2 actively exploited
        - Exploit spreading: unpatched exploited CVEs infect connected nodes
        - Agent must triage: recover crashed nodes AND patch vulnerabilities
        
        This tests the agent's ability to:
        1. Assess damage and prioritize recovery vs. patching
        2. Isolate compromised nodes to prevent further spread
        3. Work under pressure with degraded infrastructure
        """
        nodes = [
            # Tier 1 - Database (CRASHED - breach entry point)
            ServerNode(
                hostname="db-primary-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.CRITICAL,
                state=NodeState.CRASHED,  # Already compromised!
                services=["postgresql", "patroni"],
            ),
            # Tier 1 - Auth server (still online but vulnerable)
            ServerNode(
                hostname="auth-server-01",
                os="RHEL 8.9",
                tier=CriticalityTier.CRITICAL,
                state=NodeState.ONLINE,
                services=["keycloak", "java"],
            ),
            # Tier 2 - App servers (one crashed from DB cascade)
            ServerNode(
                hostname="app-server-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.CRASHED,  # Cascaded from DB crash
                services=["python3", "django", "gunicorn"],
            ),
            ServerNode(
                hostname="app-server-02",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["python3", "django", "gunicorn"],
            ),
            # Tier 2 - Web servers
            ServerNode(
                hostname="web-frontend-01",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "nodejs"],
            ),
            ServerNode(
                hostname="web-frontend-02",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "nodejs"],
            ),
            # Tier 3 - Supporting
            ServerNode(
                hostname="cache-redis-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["redis-cluster"],
            ),
            ServerNode(
                hostname="monitoring-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["prometheus", "grafana"],
            ),
        ]
        
        dependencies = [
            Dependency(node="app-server-01", depends_on="db-primary-01", dependency_type="hard",
                       description="App server requires database for queries"),
            Dependency(node="app-server-02", depends_on="db-primary-01", dependency_type="hard",
                       description="App server requires database for queries"),
            Dependency(node="app-server-01", depends_on="auth-server-01", dependency_type="hard",
                       description="App requires auth for request validation"),
            Dependency(node="app-server-02", depends_on="auth-server-01", dependency_type="hard",
                       description="App requires auth for request validation"),
            Dependency(node="web-frontend-01", depends_on="app-server-01", dependency_type="hard",
                       description="Frontend proxies to app backend"),
            Dependency(node="web-frontend-02", depends_on="app-server-02", dependency_type="hard",
                       description="Frontend proxies to app backend"),
            Dependency(node="cache-redis-01", depends_on="app-server-01", dependency_type="soft",
                       description="Cache serves app queries (degraded without)"),
        ]
        
        vulnerabilities = [
            Vulnerability(
                cve_id="CVE-2024-4001",
                severity=SeverityLevel.CRITICAL,
                cvss_score=9.8,
                affected_hosts=["db-primary-01"],
                description="Remote code execution in PostgreSQL — BREACH ENTRY POINT",
                patch_available=True,
                exploit_in_wild=True,
            ),
            Vulnerability(
                cve_id="CVE-2024-4002",
                severity=SeverityLevel.CRITICAL,
                cvss_score=9.4,
                affected_hosts=["auth-server-01"],
                description="Authentication bypass in Keycloak SAML — ACTIVELY EXPLOITED",
                patch_available=True,
                exploit_in_wild=True,
            ),
            Vulnerability(
                cve_id="CVE-2024-4003",
                severity=SeverityLevel.HIGH,
                cvss_score=7.8,
                affected_hosts=["web-frontend-01", "web-frontend-02"],
                description="Server-side request forgery in Nginx reverse proxy",
                patch_available=True,
                exploit_in_wild=False,
            ),
        ]
        
        return nodes, dependencies, vulnerabilities
    
    def _generate_zero_day_scenario(
        self,
    ) -> tuple[list[ServerNode], list[Dependency], list[Vulnerability]]:
        """
        Generate a zero-day cascade scenario: dynamic CVE injection mid-episode.
        
        Key Features:
        - 10 nodes with moderate dependency graph
        - 2 initial vulnerabilities (manageable)
        - At turn 5: A new CRITICAL zero-day CVE is injected
        - At turn 15: Another HIGH severity CVE appears on newly patched nodes
        - Agent must dynamically adapt strategy when new threats appear
        
        This tests the agent's ability to:
        1. Plan ahead while remaining adaptable
        2. Reprioritize when new critical threats emerge
        3. Handle strategy disruption gracefully
        4. Manage increasingly complex state
        """
        nodes = [
            # Tier 1 - Core infrastructure
            ServerNode(
                hostname="db-primary-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.CRITICAL,
                state=NodeState.ONLINE,
                services=["postgresql", "pgbouncer"],
            ),
            ServerNode(
                hostname="auth-server-01",
                os="RHEL 8.9",
                tier=CriticalityTier.CRITICAL,
                state=NodeState.ONLINE,
                services=["keycloak", "java"],
            ),
            # Tier 2 - Application layer
            ServerNode(
                hostname="app-server-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["python3", "django", "celery"],
            ),
            ServerNode(
                hostname="app-server-02",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["python3", "django", "celery"],
            ),
            ServerNode(
                hostname="api-gateway-01",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["kong", "nginx"],
            ),
            # Tier 2 - Web layer
            ServerNode(
                hostname="web-frontend-01",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "react"],
            ),
            ServerNode(
                hostname="web-frontend-02",
                os="RHEL 8.9",
                tier=CriticalityTier.IMPORTANT,
                state=NodeState.ONLINE,
                services=["nginx", "react"],
            ),
            # Tier 3 - Infrastructure
            ServerNode(
                hostname="cache-redis-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["redis-cluster"],
            ),
            ServerNode(
                hostname="mq-rabbitmq-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["rabbitmq", "erlang"],
            ),
            ServerNode(
                hostname="monitoring-01",
                os="Ubuntu 22.04 LTS",
                tier=CriticalityTier.STANDARD,
                state=NodeState.ONLINE,
                services=["prometheus", "grafana", "alertmanager"],
            ),
        ]
        
        dependencies = [
            # Web -> API Gateway -> App -> DB/Auth
            Dependency(node="web-frontend-01", depends_on="api-gateway-01", dependency_type="hard",
                       description="Frontend routes through API gateway"),
            Dependency(node="web-frontend-02", depends_on="api-gateway-01", dependency_type="hard",
                       description="Frontend routes through API gateway"),
            Dependency(node="api-gateway-01", depends_on="app-server-01", dependency_type="hard",
                       description="Gateway proxies to app server"),
            Dependency(node="api-gateway-01", depends_on="app-server-02", dependency_type="soft",
                       description="Gateway can degrade with single app server"),
            Dependency(node="app-server-01", depends_on="db-primary-01", dependency_type="hard",
                       description="App requires database"),
            Dependency(node="app-server-02", depends_on="db-primary-01", dependency_type="hard",
                       description="App requires database"),
            Dependency(node="app-server-01", depends_on="auth-server-01", dependency_type="hard",
                       description="App requires authentication service"),
            Dependency(node="app-server-02", depends_on="auth-server-01", dependency_type="hard",
                       description="App requires authentication service"),
            Dependency(node="app-server-01", depends_on="cache-redis-01", dependency_type="soft",
                       description="App uses cache for performance"),
        ]
        
        # Initial vulnerabilities (manageable — but more will come)
        vulnerabilities = [
            Vulnerability(
                cve_id="CVE-2024-5001",
                severity=SeverityLevel.HIGH,
                cvss_score=8.1,
                affected_hosts=["app-server-01", "app-server-02"],
                description="Deserialization RCE in Django REST framework",
                patch_available=True,
                exploit_in_wild=False,
            ),
            Vulnerability(
                cve_id="CVE-2024-5002",
                severity=SeverityLevel.MEDIUM,
                cvss_score=5.5,
                affected_hosts=["cache-redis-01"],
                description="Information disclosure in Redis MONITOR command",
                patch_available=True,
                exploit_in_wild=False,
            ),
        ]
        
        return nodes, dependencies, vulnerabilities
    
    # =========================================================================
    # STEP - Main State Machine
    # =========================================================================
    
    def step(self, action: PatchCascadeAction) -> StepResult:
        """
        Process an agent action and advance the environment by one turn.
        
        The step function applies rules in strict order:
        1. Validation - Check if action is legal
        2. Action Application - Execute the action
        3. Time Progression - Advance patch timers, complete patches
        4. Dependency Cascade - Check for cascade failures
        5. Health Calculation - Update metrics and compute reward
        6. Termination Check - Determine if episode is over
        
        Args:
            action: The action the agent wants to take this turn.
        
        Returns:
            StepResult containing (observation, reward, done, truncated, info).
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self._state.is_terminated:
            raise RuntimeError("Episode already terminated. Call reset() to start a new episode.")
        
        self._messages = []
        info: dict = {"action": action.model_dump(), "turn": self._state.turn_number}
        
        # ---------------------------------------------------------------------
        # PHASE 1: Validation
        # ---------------------------------------------------------------------
        obs = self.get_observation()
        is_valid, error_msg = validate_action_for_observation(action, obs)
        
        if not is_valid:
            # Invalid action: apply penalty, don't change state
            self._last_action_result = error_msg
            self._messages.append(f"Invalid action: {error_msg}")
            self._state.action_history.append(action)
            self._state.turn_number += 1
            self._invalid_action_count += 1
            
            # Still need to run time progression for pending patches
            self._process_time_progression()
            self._process_dynamic_events()
            self._process_dependency_cascade()
            
            # Calculate state transition before reward so terminal potential is
            # handled consistently even if a pending patch completes.
            previous_penalty = self._last_total_penalty
            current_penalty = self._calculate_total_penalty(self._state.nodes, self._state.vulnerabilities)
            self._last_total_penalty = current_penalty
            self._update_health_metrics()
            done, truncated = self._check_termination()
            reward, reward_components = self._calculate_shaped_reward(
                previous_penalty, current_penalty, done and not truncated,
                invalid_action=True,
            )
            self._state.reward_history.append(reward)
            
            info["valid"] = False
            info["error"] = error_msg
            info["reward_components"] = reward_components
            info["termination_reason"] = self._state.termination_reason
            
            return StepResult(
                observation=self.get_observation(),
                reward=reward,
                done=done,
                truncated=truncated,
                info=info,
            )
        
        # ---------------------------------------------------------------------
        # PHASE 2: Action Application
        # ---------------------------------------------------------------------
        self._apply_action(action)
        self._state.action_history.append(action)
        
        # ---------------------------------------------------------------------
        # PHASE 3: Time Progression (The "Tick")
        # ---------------------------------------------------------------------
        self._process_time_progression()

        # A consumed action advances the turn before turn-indexed dynamic events.
        # Invalid actions already increment at validation time; both paths now
        # have identical event timing.
        self._state.turn_number += 1
        
        # ---------------------------------------------------------------------
        # PHASE 3.5: Dynamic Events (Zero-Day injection, exploit spreading)
        # ---------------------------------------------------------------------
        self._process_dynamic_events()
        
        # ---------------------------------------------------------------------
        # PHASE 4: Dependency Cascade
        # ---------------------------------------------------------------------
        cascade_count = self._process_dependency_cascade()
        if cascade_count > 0:
            self._cascade_failure_count += cascade_count
            self._messages.append(f"⚠️ CASCADE FAILURE: {cascade_count} node(s) crashed due to dependency failures!")
        
        # ---------------------------------------------------------------------
        # PHASE 5: Health Calculation & Reward
        # ---------------------------------------------------------------------
        previous_penalty = self._last_total_penalty
        current_penalty = self._calculate_total_penalty(self._state.nodes, self._state.vulnerabilities)
        self._last_total_penalty = current_penalty
        self._update_health_metrics()
        
        # ---------------------------------------------------------------------
        # PHASE 6: Termination Check
        # ---------------------------------------------------------------------
        done, truncated = self._check_termination()
        
        reward, reward_components = self._calculate_shaped_reward(
            previous_penalty, current_penalty, done and not truncated,
            invalid_action=False,
        )
        self._state.reward_history.append(reward)
        if self._state.termination_reason == "all_patched":
            self._messages.append("VICTORY! All vulnerabilities patched successfully.")
        elif self._state.termination_reason == "all_crashed":
            self._messages.append("CATASTROPHIC FAILURE! All nodes have crashed.")
        
        info["valid"] = True
        info["cascade_failures"] = cascade_count
        info["total_cascade_failures"] = self._cascade_failure_count
        info["invalid_actions"] = self._invalid_action_count
        info["initial_vulnerability_count"] = self._initial_vuln_count
        info["patches_completed"] = len([m for m in self._messages if "Patch completed" in m])
        info["reward_components"] = reward_components
        info["termination_reason"] = self._state.termination_reason
        
        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def _calculate_shaped_reward(
        self,
        previous_penalty: float,
        current_penalty: float,
        terminated: bool,
        *,
        invalid_action: bool,
    ) -> tuple[float, dict[str, float]]:
        """Return base reward plus gamma*Phi(next)-Phi(previous).

        Phi(s) is negative total penalty. True terminal states use zero
        potential; time-limit truncations retain their state potential.
        """
        previous_potential = -previous_penalty
        next_potential = 0.0 if terminated else -current_penalty
        shaping = SHAPING_GAMMA * next_potential - previous_potential
        base = TIME_PRESSURE_PENALTY
        if invalid_action:
            base += INVALID_ACTION_PENALTY
        if terminated and self._state is not None:
            if self._state.termination_reason == "all_patched":
                base += VICTORY_BONUS
            elif self._state.termination_reason == "all_crashed":
                base += CATASTROPHIC_FAILURE_PENALTY
        components = {"base": base, "potential_shaping": shaping, "gamma": SHAPING_GAMMA}
        return base + shaping, components
    
    def _apply_action(self, action: PatchCascadeAction) -> None:
        """Apply a validated action to the environment state."""
        assert self._state is not None
        
        if action.action_type == ActionType.NOOP:
            self._last_action_result = "success"
            self._messages.append("Agent chose to wait this turn.")
            return
        
        if action.action_type == ActionType.SCAN_HOST:
            # Scan doesn't change state, just provides info (already in observation)
            self._last_action_result = "success"
            node = self._get_node_by_hostname(action.target)
            vulns_on_host = [v.cve_id for v in self._state.vulnerabilities if action.target in v.affected_hosts]
            self._messages.append(
                f"Scan of {action.target}: state={node.state.value}, tier={node.tier.value}, "
                f"vulns={vulns_on_host or 'none'}"
            )
            return
        
        if action.action_type == ActionType.SUSPEND_SERVICE:
            node = self._get_node_by_hostname(action.target)
            node.state = NodeState.SUSPENDED
            self._last_action_result = "success"
            self._messages.append(f"Suspended service on {action.target}.")
            return
        
        if action.action_type == ActionType.RESUME_SERVICE:
            node = self._get_node_by_hostname(action.target)
            node.state = NodeState.ONLINE
            self._last_action_result = "success"
            self._messages.append(f"Resumed service on {action.target}.")
            return
        
        if action.action_type == ActionType.APPLY_PATCH:
            node = self._get_node_by_hostname(action.target)
            node.state = NodeState.PATCHING
            node.patch_turns_remaining = 1
            self._pending_patches[action.target] = action.cve_id  # type: ignore
            self._last_action_result = "success"
            self._messages.append(f"Started patching {action.cve_id} on {action.target}. Will complete next turn.")
            return
    
    def _process_time_progression(self) -> None:
        """
        Advance time: decrement patch timers and complete patches.
        
        When a patch completes:
        1. Node returns to ONLINE state
        2. Node is removed from vulnerability's affected_hosts
        3. If affected_hosts is empty, vulnerability is fully resolved
        """
        assert self._state is not None
        
        completed_patches: list[tuple[str, str]] = []  # (hostname, cve_id)
        
        for node in self._state.nodes:
            if node.state == NodeState.PATCHING and node.patch_turns_remaining > 0:
                node.patch_turns_remaining -= 1
                
                if node.patch_turns_remaining == 0:
                    # Patch completed!
                    node.state = NodeState.ONLINE
                    cve_id = self._pending_patches.pop(node.hostname, None)
                    if cve_id:
                        completed_patches.append((node.hostname, cve_id))
                        self._messages.append(f"Patch completed: {cve_id} on {node.hostname}.")
        
        # Remove patched hosts from vulnerability affected_hosts
        for hostname, cve_id in completed_patches:
            for vuln in self._state.vulnerabilities:
                if vuln.cve_id == cve_id and hostname in vuln.affected_hosts:
                    vuln.affected_hosts.remove(hostname)
        
        # Remove fully resolved vulnerabilities (keep reference for grading)
        self._state.vulnerabilities = [
            v for v in self._state.vulnerabilities if len(v.affected_hosts) > 0
        ]
    
    def _process_dynamic_events(self) -> None:
        """
        Process dynamic events: exploit spreading and zero-day CVE injection.
        
        This system makes the environment dynamic and tests adaptive planning:
        
        1. Exploit Spreading (incident_response & hard tasks):
           If a vulnerability with exploit_in_wild=True remains unpatched on
           an ONLINE node for EXPLOIT_SPREAD_THRESHOLD turns, it spreads to
           a randomly selected connected node (via dependency graph).
        
        2. Zero-Day Injection (zero_day task):
           At predefined turns (5 and 15), new CVEs are injected into the
           environment, forcing the agent to adapt its strategy mid-episode.
        """
        assert self._state is not None
        
        # --- Exploit Spreading ---
        if self._task_level in ("incident_response", "hard", "zero_day"):
            self._process_exploit_spreading()
        
        # --- Zero-Day CVE Injection ---
        if self._task_level == "zero_day":
            self._process_zero_day_injection()
        
        # --- Stochastic Node Degradation (hard/incident_response) ---
        # Adds real-world randomness: overloaded nodes may degrade under stress
        if self._task_level in ("hard", "incident_response"):
            self._process_stochastic_degradation()
    
    def _process_stochastic_degradation(self) -> None:
        """
        Simulate random node degradation under stress conditions.
        
        In real SOCs, heavily loaded or compromised nodes may experience
        degraded performance or spontaneous failures. This mechanic:
        
        - 3% chance per turn per vulnerable ONLINE node to degrade
        - Degradation: node becomes CRASHED if it has an exploited CVE
        - Creates realistic uncertainty and rewards proactive patching
        """
        assert self._state is not None
        
        # Only trigger if there are unpatched exploited vulnerabilities
        exploited_hosts = set()
        for vuln in self._state.vulnerabilities:
            if vuln.exploit_in_wild:
                exploited_hosts.update(vuln.affected_hosts)
        
        if not exploited_hosts:
            return
        
        DEGRADATION_CHANCE = 0.03  # 3% per vulnerable node per turn
        
        for node in self._state.nodes:
            if node.state != NodeState.ONLINE:
                continue
            if node.hostname not in exploited_hosts:
                continue
            
            # Roll for degradation
            if self._rng.random() < DEGRADATION_CHANCE:
                # Node experiences stress-induced failure
                node.state = NodeState.CRASHED
                self._state.health.nodes_online -= 1
                self._state.health.nodes_crashed += 1
                
                self._messages.append(
                    f"⚠️ ALERT: {node.hostname} crashed due to exploit-induced stress!"
                )
    
    def _process_exploit_spreading(self) -> None:
        """
        Spread actively exploited vulnerabilities to connected nodes.
        
        Mechanic: If an exploit_in_wild CVE remains on an ONLINE node for
        EXPLOIT_SPREAD_THRESHOLD consecutive turns, it spreads to one
        connected node (selected via dependency graph). This creates urgency
        and rewards proactive patching.
        """
        assert self._state is not None
        
        online_hosts = {n.hostname for n in self._state.nodes if n.state == NodeState.ONLINE}
        
        for vuln in self._state.vulnerabilities:
            if not vuln.exploit_in_wild:
                continue
            
            # Check if any affected host is still online (vulnerability active)
            active_online = [h for h in vuln.affected_hosts if h in online_hosts]
            
            if active_online:
                # Increment tracker
                tracker_key = vuln.cve_id
                self._exploit_turn_tracker[tracker_key] = (
                    self._exploit_turn_tracker.get(tracker_key, 0) + 1
                )
                
                # Check if threshold reached
                if self._exploit_turn_tracker[tracker_key] >= EXPLOIT_SPREAD_THRESHOLD:
                    # Find a connected node to spread to
                    spread_target = self._find_spread_target(vuln, online_hosts)
                    if spread_target and spread_target not in vuln.affected_hosts:
                        vuln.affected_hosts.append(spread_target)
                        self._exploit_turn_tracker[tracker_key] = 0  # Reset timer
                        self._messages.append(
                            f"🔴 EXPLOIT SPREAD: {vuln.cve_id} has spread to {spread_target}! "
                            f"Patch immediately to contain the threat."
                        )
            else:
                # Reset tracker if no online hosts affected
                self._exploit_turn_tracker.pop(vuln.cve_id, None)
    
    def _find_spread_target(
        self, vuln: Vulnerability, online_hosts: set[str]
    ) -> str | None:
        """
        Find a connected node for exploit to spread to.
        
        Looks at dependency graph edges from affected hosts
        to find a connected, online, unaffected node.
        """
        assert self._state is not None
        
        candidates: set[str] = set()
        affected = set(vuln.affected_hosts)
        
        for dep in self._state.dependencies:
            # Spread along dependency edges (both directions)
            if dep.node in affected and dep.depends_on in online_hosts:
                candidates.add(dep.depends_on)
            if dep.depends_on in affected and dep.node in online_hosts:
                candidates.add(dep.node)
        
        # Remove already-affected hosts
        candidates -= affected
        
        if candidates:
            # Sets have process-randomized iteration order; sorting is required
            # for cross-process deterministic seeds.
            return self._rng.choice(sorted(candidates))
        return None
    
    def _process_zero_day_injection(self) -> None:
        """
        Inject zero-day CVEs at predefined turns.
        
        Turn 5: CRITICAL zero-day on auth infrastructure (exploit_in_wild=True)
        Turn 15: HIGH severity CVE on web layer
        
        These events force the agent to dynamically reprioritize,
        testing adaptive planning and strategy revision capabilities.
        """
        assert self._state is not None
        turn = self._state.turn_number
        
        if turn == 5 and "zero_day_turn_5" not in self._dynamic_events_fired:
            self._dynamic_events_fired.append("zero_day_turn_5")
            
            new_cve = Vulnerability(
                cve_id="CVE-2024-5099",
                severity=SeverityLevel.CRITICAL,
                cvss_score=9.9,
                affected_hosts=["auth-server-01", "db-primary-01"],
                description="ZERO-DAY: Critical authentication bypass — actively exploited in the wild",
                patch_available=True,
                exploit_in_wild=True,
            )
            self._state.vulnerabilities.append(new_cve)
            self._messages.append(
                "🚨 ZERO-DAY ALERT: CVE-2024-5099 (CVSS 9.9) discovered! "
                "Critical authentication bypass affecting auth-server-01 and db-primary-01. "
                "ACTIVELY EXPLOITED — immediate patching required!"
            )
        
        if turn == 15 and "zero_day_turn_15" not in self._dynamic_events_fired:
            self._dynamic_events_fired.append("zero_day_turn_15")
            
            new_cve = Vulnerability(
                cve_id="CVE-2024-5100",
                severity=SeverityLevel.HIGH,
                cvss_score=8.4,
                affected_hosts=["web-frontend-01", "web-frontend-02", "api-gateway-01"],
                description="ZERO-DAY: HTTP request smuggling in reverse proxy configuration",
                patch_available=True,
                exploit_in_wild=False,
            )
            self._state.vulnerabilities.append(new_cve)
            self._messages.append(
                "⚠️ NEW THREAT: CVE-2024-5100 (CVSS 8.4) discovered! "
                "HTTP request smuggling affecting web-frontend-01, web-frontend-02, api-gateway-01. "
                "Patch available — prioritize based on current strategy."
            )
    
    def _process_dependency_cascade(self) -> int:
        """
        Check dependencies and crash nodes whose dependencies are down.
        
        A node crashes if:
        - It has a HARD dependency on another node
        - The dependency is OFFLINE, CRASHED, or SUSPENDED
        - The dependent node is NOT already SUSPENDED (safe state)
        
        Returns:
            Number of nodes that crashed due to cascade.
        """
        assert self._state is not None
        
        # Build lookup for node states
        node_states = {n.hostname: n.state for n in self._state.nodes}
        
        # States that cause cascade (dependency is down)
        down_states = {NodeState.OFFLINE, NodeState.CRASHED, NodeState.SUSPENDED}
        
        # States that are safe from cascade (node already protected)
        safe_states = {NodeState.SUSPENDED, NodeState.CRASHED, NodeState.OFFLINE}
        
        cascade_count = 0
        changed = True
        
        # Iterate until no more cascades (handles multi-level dependencies)
        while changed:
            changed = False
            for dep in self._state.dependencies:
                if dep.dependency_type != "hard":
                    continue  # Soft dependencies don't cause crashes
                
                dep_state = node_states.get(dep.depends_on)
                node_state = node_states.get(dep.node)
                
                if dep_state is None or node_state is None:
                    continue  # Skip invalid references
                
                # Check if cascade should occur
                if dep_state in down_states and node_state not in safe_states:
                    # Crash the dependent node
                    node = self._get_node_by_hostname(dep.node)
                    node.state = NodeState.CRASHED
                    node_states[dep.node] = NodeState.CRASHED
                    cascade_count += 1
                    changed = True
        
        return cascade_count
    
    # =========================================================================
    # PENALTY & REWARD CALCULATION
    # =========================================================================
    
    def _calculate_total_penalty(
        self,
        nodes: list[ServerNode],
        vulnerabilities: list[Vulnerability],
    ) -> float:
        """
        Calculate the total penalty for the current state.
        
        Total Penalty = Risk Penalty + Downtime Penalty
        
        Risk Penalty:
        - Sum of CVSS scores for vulns on ONLINE nodes
        - Doubled if exploit_in_wild
        
        Downtime Penalty:
        - Per non-ONLINE node: tier_multiplier * base_penalty
        - CRASHED nodes: doubled penalty
        """
        risk_penalty = self._calculate_risk_penalty(nodes, vulnerabilities)
        downtime_penalty = self._calculate_downtime_penalty(nodes)
        return risk_penalty + downtime_penalty
    
    def _calculate_risk_penalty(
        self,
        nodes: list[ServerNode],
        vulnerabilities: list[Vulnerability],
    ) -> float:
        """
        Calculate risk penalty: sum of CVSS scores for vulnerabilities on ONLINE nodes.
        
        Only ONLINE nodes contribute to risk (offline nodes aren't reachable).
        Actively exploited vulnerabilities have doubled impact.
        """
        online_hosts = {n.hostname for n in nodes if n.state == NodeState.ONLINE}
        
        total = 0.0
        for vuln in vulnerabilities:
            affected_online = [h for h in vuln.affected_hosts if h in online_hosts]
            base_score = vuln.cvss_score * len(affected_online)
            
            if vuln.exploit_in_wild:
                base_score *= EXPLOIT_IN_WILD_MULTIPLIER
            
            total += base_score
        
        return total
    
    def _calculate_downtime_penalty(self, nodes: list[ServerNode]) -> float:
        """
        Calculate downtime penalty for non-ONLINE nodes.
        
        Penalty per node = tier_multiplier * (2 if CRASHED else 1)
        """
        total = 0.0
        
        for node in nodes:
            if node.state == NodeState.ONLINE:
                continue
            
            base_penalty = DOWNTIME_PENALTY_MULTIPLIER[node.tier]
            
            if node.state == NodeState.CRASHED:
                base_penalty *= CRASHED_PENALTY_MULTIPLIER
            
            total += base_penalty
        
        return total
    
    # =========================================================================
    # HEALTH METRICS
    # =========================================================================
    
    def _calculate_health_metrics(
        self,
        nodes: list[ServerNode],
        vulnerabilities: list[Vulnerability],
        turn_number: int,
    ) -> NetworkHealth:
        """Calculate aggregate health metrics."""
        severity_counts = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 0,
            SeverityLevel.MEDIUM: 0,
            SeverityLevel.LOW: 0,
        }
        for vuln in vulnerabilities:
            severity_counts[vuln.severity] += 1
        
        return NetworkHealth(
            total_nodes=len(nodes),
            nodes_online=sum(1 for n in nodes if n.state == NodeState.ONLINE),
            nodes_crashed=sum(1 for n in nodes if n.state == NodeState.CRASHED),
            nodes_patching=sum(1 for n in nodes if n.state == NodeState.PATCHING),
            active_critical_vulns=severity_counts[SeverityLevel.CRITICAL],
            active_high_vulns=severity_counts[SeverityLevel.HIGH],
            active_medium_vulns=severity_counts[SeverityLevel.MEDIUM],
            active_low_vulns=severity_counts[SeverityLevel.LOW],
            cumulative_downtime_penalty=self._state.health.cumulative_downtime_penalty if self._state else 0.0,
            cumulative_risk_penalty=self._state.health.cumulative_risk_penalty if self._state else 0.0,
            turn_number=turn_number,
        )
    
    def _update_health_metrics(self) -> None:
        """Update health metrics in state, including cumulative penalties."""
        assert self._state is not None
        
        risk = self._calculate_risk_penalty(self._state.nodes, self._state.vulnerabilities)
        downtime = self._calculate_downtime_penalty(self._state.nodes)
        
        self._state.health = NetworkHealth(
            total_nodes=len(self._state.nodes),
            nodes_online=sum(1 for n in self._state.nodes if n.state == NodeState.ONLINE),
            nodes_crashed=sum(1 for n in self._state.nodes if n.state == NodeState.CRASHED),
            nodes_patching=sum(1 for n in self._state.nodes if n.state == NodeState.PATCHING),
            active_critical_vulns=sum(1 for v in self._state.vulnerabilities if v.severity == SeverityLevel.CRITICAL),
            active_high_vulns=sum(1 for v in self._state.vulnerabilities if v.severity == SeverityLevel.HIGH),
            active_medium_vulns=sum(1 for v in self._state.vulnerabilities if v.severity == SeverityLevel.MEDIUM),
            active_low_vulns=sum(1 for v in self._state.vulnerabilities if v.severity == SeverityLevel.LOW),
            cumulative_downtime_penalty=self._state.health.cumulative_downtime_penalty + downtime,
            cumulative_risk_penalty=self._state.health.cumulative_risk_penalty + risk,
            turn_number=self._state.turn_number,
        )
    
    # =========================================================================
    # TERMINATION
    # =========================================================================
    
    def _check_termination(self) -> tuple[bool, bool]:
        """
        Check if episode should terminate.
        
        Returns:
            Tuple of (done, truncated).
            - done: True if episode is over for any reason.
            - truncated: True if episode ended due to max_turns (not terminal state).
        """
        assert self._state is not None
        
        # Victory: All vulnerabilities patched
        if len(self._state.vulnerabilities) == 0:
            self._state.is_terminated = True
            self._state.termination_reason = "all_patched"
            return True, False
        
        # Catastrophic failure: All nodes crashed
        all_crashed = all(n.state == NodeState.CRASHED for n in self._state.nodes)
        if all_crashed:
            self._state.is_terminated = True
            self._state.termination_reason = "all_crashed"
            return True, False
        
        # Timeout: Max turns reached
        if self._state.turn_number >= self._state.max_turns:
            self._state.is_terminated = True
            self._state.termination_reason = "max_turns_reached"
            return True, True  # Truncated, not terminal
        
        return False, False
    
    # =========================================================================
    # OBSERVATION
    # =========================================================================
    
    def get_observation(self) -> PatchCascadeObservation:
        """
        Generate the agent-visible observation from internal state.
        
        The observation is a filtered/formatted view of the state,
        excluding internal tracking data like reward_history.
        """
        assert self._state is not None
        
        return PatchCascadeObservation(
            nodes=copy.deepcopy(self._state.nodes),
            vulnerabilities=copy.deepcopy(self._state.vulnerabilities),
            dependencies=copy.deepcopy(self._state.dependencies),
            health=copy.deepcopy(self._state.health),
            last_action_result=self._last_action_result,
            messages=list(self._messages),
        )
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _get_node_by_hostname(self, hostname: str) -> ServerNode:
        """Get a node by hostname. Raises ValueError if not found."""
        assert self._state is not None
        for node in self._state.nodes:
            if node.hostname == hostname:
                return node
        raise ValueError(f"Node '{hostname}' not found")
    
    def render(self, mode: str = "ascii") -> str:
        """
        Render a human-readable visualization of the current state.
        
        Args:
            mode: "ascii" for network diagram, "text" for simple list
        
        Returns:
            Formatted string visualization suitable for terminal or logs.
        """
        if self._state is None:
            return "Environment not initialized. Call reset() first."
        
        if mode == "text":
            return self._render_text()
        return self._render_ascii()
    
    def _render_ascii(self) -> str:
        """
        Render an ASCII network diagram with visual node states.
        
        Example output:
        ╔══════════════════════════════════════════════════════════════════╗
        ║  🛡️ PatchCascade SOC — Turn 3/30 (Easy)                          ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  NETWORK TOPOLOGY                                                ║
        ║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          ║
        ║  │ web-svr-01  │───►│ app-svr-01  │───►│ db-main-01  │          ║
        ║  │   ONLINE    │    │  PATCHING   │    │  SUSPENDED  │          ║
        ║  │ ⚠️ CVE-1001 │    │   T2 ████░  │    │   T1 🔒     │          ║
        ║  └─────────────┘    └─────────────┘    └─────────────┘          ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  VULNS: 2 active (1 CRITICAL, 1 HIGH) | HEALTH: 2/3 online      ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        W = 70  # Box width
        
        # State icons
        STATE_ICONS = {
            "online": "✓",
            "offline": "○",
            "suspended": "⏸",
            "patching": "⟳",
            "crashed": "✗",
        }
        
        STATE_COLORS = {
            "online": "🟢",
            "offline": "⚪",
            "suspended": "🟡",
            "patching": "🔵",
            "crashed": "🔴",
        }
        
        # Build header
        task_name = self._task_level.replace("_", " ").title()
        header = f"🛡️ PATCHCASCADE SOC — Turn {self._state.turn_number}/{self._state.max_turns} ({task_name})"
        
        lines = [
            "╔" + "═" * (W - 2) + "╗",
            "║" + header.center(W - 2) + "║",
            "╠" + "═" * (W - 2) + "╣",
        ]
        
        # Network topology section
        lines.append("║" + "  NETWORK TOPOLOGY".ljust(W - 2) + "║")
        lines.append("║" + " " * (W - 2) + "║")
        
        # Build dependency graph visualization
        deps = {d.node: d.depends_on for d in self._state.dependencies}
        
        # Find root nodes (no incoming edges)
        all_nodes = {n.hostname for n in self._state.nodes}
        dependent_nodes = set(deps.keys())
        root_nodes = all_nodes - dependent_nodes
        
        # Build node boxes
        vuln_map = {}
        for v in self._state.vulnerabilities:
            for h in v.affected_hosts:
                if h not in vuln_map:
                    vuln_map[h] = []
                vuln_map[h].append(v)
        
        # Render nodes in rows
        node_strs = []
        for node in self._state.nodes:
            icon = STATE_COLORS.get(node.state.value, "⚪")
            vuln_indicator = ""
            if node.hostname in vuln_map:
                vulns = vuln_map[node.hostname]
                if any(v.exploit_in_wild for v in vulns):
                    vuln_indicator = " 🔥"
                else:
                    vuln_indicator = " ⚠️"
            
            tier_str = f"T{node.tier.value}"
            state_str = node.state.value.upper()[:8]
            
            # Build compact node representation
            node_repr = f"{icon} {node.hostname[:12]:<12} [{state_str:^8}] {tier_str}{vuln_indicator}"
            node_strs.append(node_repr)
        
        # Add nodes (2 per row for readability)
        for i in range(0, len(node_strs), 2):
            row = "  " + node_strs[i]
            if i + 1 < len(node_strs):
                row += "    " + node_strs[i + 1]
            lines.append("║" + row.ljust(W - 2) + "║")
        
        lines.append("║" + " " * (W - 2) + "║")
        
        # Dependencies section (if any)
        if self._state.dependencies:
            lines.append("║" + "  DEPENDENCIES".ljust(W - 2) + "║")
            dep_strs = []
            for dep in self._state.dependencies[:4]:  # Show max 4
                arrow = "━━►" if dep.dependency_type == "hard" else "┄┄►"
                dep_strs.append(f"    {dep.node[:10]} {arrow} {dep.depends_on[:10]}")
            for ds in dep_strs:
                lines.append("║" + ds.ljust(W - 2) + "║")
            if len(self._state.dependencies) > 4:
                lines.append("║" + f"    ... and {len(self._state.dependencies) - 4} more".ljust(W - 2) + "║")
            lines.append("║" + " " * (W - 2) + "║")
        
        # Vulnerabilities section
        lines.append("╠" + "═" * (W - 2) + "╣")
        if not self._state.vulnerabilities:
            vuln_summary = "  ✅ ALL VULNERABILITIES PATCHED!"
        else:
            crit = sum(1 for v in self._state.vulnerabilities if v.severity.value == "CRITICAL")
            high = sum(1 for v in self._state.vulnerabilities if v.severity.value == "HIGH")
            med = sum(1 for v in self._state.vulnerabilities if v.severity.value == "MEDIUM")
            exploited = sum(1 for v in self._state.vulnerabilities if v.exploit_in_wild)
            
            parts = []
            if crit: parts.append(f"{crit} CRIT")
            if high: parts.append(f"{high} HIGH")
            if med: parts.append(f"{med} MED")
            vuln_str = ", ".join(parts) if parts else "0"
            exploit_str = f" ({exploited} exploited!)" if exploited else ""
            vuln_summary = f"  VULNS: {len(self._state.vulnerabilities)} active ({vuln_str}){exploit_str}"
        
        lines.append("║" + vuln_summary.ljust(W - 2) + "║")
        
        # Health metrics
        h = self._state.health
        health_str = f"  HEALTH: {h.nodes_online}/{h.total_nodes} online"
        if h.nodes_crashed > 0:
            health_str += f" | {h.nodes_crashed} crashed"
        health_str += f" | Risk: {h.cumulative_risk_penalty:.1f} | Downtime: {h.cumulative_downtime_penalty:.1f}"
        
        lines.append("║" + health_str[:W-2].ljust(W - 2) + "║")
        
        # Reward
        if self._state.reward_history:
            total_reward = sum(self._state.reward_history)
            reward_str = f"  REWARD: {total_reward:+.2f} (last: {self._state.reward_history[-1]:+.2f})"
            lines.append("║" + reward_str.ljust(W - 2) + "║")
        
        # Footer
        lines.append("╚" + "═" * (W - 2) + "╝")
        
        return "\n".join(lines)
    
    def _render_text(self) -> str:
        """Simple text-based render (original implementation)."""
        lines = [
            "=" * 60,
            f"PATCHCASCADE SOC - Turn {self._state.turn_number}/{self._state.max_turns}",
            "=" * 60,
            "",
            "NODES:",
        ]
        
        for node in self._state.nodes:
            status = f"[{node.state.value.upper():^10}]"
            tier_str = f"T{node.tier.value}"
            patch_info = f" (patching: {node.patch_turns_remaining}t)" if node.patch_turns_remaining > 0 else ""
            lines.append(f"  {tier_str} {node.hostname:<24} {status}{patch_info}")
        
        lines.append("")
        lines.append("VULNERABILITIES:")
        
        if not self._state.vulnerabilities:
            lines.append("  (none - ALL PATCHED!)")
        else:
            for vuln in self._state.vulnerabilities:
                exploit = " [EXPLOITED!]" if vuln.exploit_in_wild else ""
                lines.append(f"  {vuln.cve_id} ({vuln.severity.value}, CVSS {vuln.cvss_score}){exploit}")
                lines.append(f"    Affects: {', '.join(vuln.affected_hosts)}")
        
        lines.append("")
        lines.append("HEALTH METRICS:")
        lines.append(f"  Online: {self._state.health.nodes_online}/{self._state.health.total_nodes}")
        lines.append(f"  Crashed: {self._state.health.nodes_crashed}")
        lines.append(f"  Cumulative Risk Penalty: {self._state.health.cumulative_risk_penalty:.1f}")
        lines.append(f"  Cumulative Downtime Penalty: {self._state.health.cumulative_downtime_penalty:.1f}")
        
        if self._state.reward_history:
            lines.append(f"  Total Reward: {sum(self._state.reward_history):.2f}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "PatchCascadeEnv",
    "StepResult",
    "DOWNTIME_PENALTY_MULTIPLIER",
    "CRASHED_PENALTY_MULTIPLIER",
    "EXPLOIT_IN_WILD_MULTIPLIER",
    "INVALID_ACTION_PENALTY",
    "VICTORY_BONUS",
    "CATASTROPHIC_FAILURE_PENALTY",
    "MAX_TURNS_BY_DIFFICULTY",
]
