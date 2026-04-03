"""
PatchCascade SOC - Environment Core Logic
==========================================

This module implements the PatchCascadeEnv class, the core reinforcement learning
environment for the PatchCascade SOC simulation. It provides:

- `reset(task_level)`: Initialize episodes at easy/medium/hard difficulty
- `step(action)`: Process agent actions and advance simulation state
- `get_observation()`: Generate agent-visible observation from internal state

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

# Default max turns by difficulty
MAX_TURNS_BY_DIFFICULTY: dict[str, int] = {
    "easy": 30,
    "medium": 50,
    "hard": 100,
}


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
        task_level: Literal["easy", "medium", "hard"] = "easy",
        seed: int | None = None,
    ) -> PatchCascadeObservation:
        """
        Reset the environment to a new episode.
        
        Args:
            task_level: Difficulty level determining network complexity.
                - "easy": 3-5 nodes, no dependencies, 1 vulnerability
                - "medium": 5-8 nodes, linear dependency chain, 2 vulnerabilities
                - "hard": 10-15 nodes, complex graph, multiple critical vulns
            seed: Optional seed for this episode (overrides constructor seed).
        
        Returns:
            Initial observation for the agent.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        
        episode_seed = self._rng.randint(0, 2**31 - 1)
        
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
        else:
            raise ValueError(f"Invalid task_level: {task_level}. Must be 'easy', 'medium', or 'hard'.")
        
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
        )
        
        # Reset tracking variables
        self._last_total_penalty = self._calculate_total_penalty(nodes, vulnerabilities)
        self._pending_patches = {}
        self._last_action_result = None
        self._messages = [f"Episode started ({task_level} difficulty). {len(vulnerabilities)} vulnerabilities detected."]
        
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
            
            # Still need to run time progression for pending patches
            self._process_time_progression()
            self._process_dependency_cascade()
            
            # Calculate reward (penalty for invalid action)
            current_penalty = self._calculate_total_penalty(self._state.nodes, self._state.vulnerabilities)
            reward = (self._last_total_penalty - current_penalty) + INVALID_ACTION_PENALTY
            self._last_total_penalty = current_penalty
            self._state.reward_history.append(reward)
            
            # Update health and check termination
            self._update_health_metrics()
            done, truncated = self._check_termination()
            
            info["valid"] = False
            info["error"] = error_msg
            
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
        
        # ---------------------------------------------------------------------
        # PHASE 4: Dependency Cascade
        # ---------------------------------------------------------------------
        cascade_count = self._process_dependency_cascade()
        if cascade_count > 0:
            self._messages.append(f"CASCADE FAILURE: {cascade_count} node(s) crashed due to dependency failures!")
        
        # ---------------------------------------------------------------------
        # PHASE 5: Health Calculation & Reward
        # ---------------------------------------------------------------------
        current_penalty = self._calculate_total_penalty(self._state.nodes, self._state.vulnerabilities)
        
        # Reward = improvement in penalty (lower is better, so we want positive reward for decrease)
        reward = self._last_total_penalty - current_penalty
        self._last_total_penalty = current_penalty
        self._state.reward_history.append(reward)
        
        self._update_health_metrics()
        self._state.turn_number += 1
        
        # ---------------------------------------------------------------------
        # PHASE 6: Termination Check
        # ---------------------------------------------------------------------
        done, truncated = self._check_termination()
        
        # Apply terminal bonuses/penalties
        if done and not truncated:
            if self._state.termination_reason == "all_patched":
                reward += VICTORY_BONUS
                self._messages.append("VICTORY! All vulnerabilities patched successfully.")
            elif self._state.termination_reason == "all_crashed":
                reward += CATASTROPHIC_FAILURE_PENALTY
                self._messages.append("CATASTROPHIC FAILURE! All nodes have crashed.")
        
        info["valid"] = True
        info["cascade_failures"] = cascade_count
        info["patches_completed"] = len([m for m in self._messages if "Patch completed" in m])
        
        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )
    
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
        
        # Remove fully resolved vulnerabilities
        self._state.vulnerabilities = [
            v for v in self._state.vulnerabilities if len(v.affected_hosts) > 0
        ]
    
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
    
    def render(self) -> str:
        """
        Render a human-readable summary of the current state.
        
        Useful for debugging and visualization.
        """
        if self._state is None:
            return "Environment not initialized. Call reset() first."
        
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
