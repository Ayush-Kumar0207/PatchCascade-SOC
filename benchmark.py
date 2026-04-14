#!/usr/bin/env python3
"""
PatchCascade SOC — Benchmarking & Evaluation System
====================================================

Systematically evaluates and compares different agents across all 5
PatchCascade task levels. Generates the 'Source of Truth' comparison
table for the hackathon submission.

Agents compared:
1. Random (Baseline)
2. Heuristic (Rule-based)
3. RL Agent (Trained PPO - if model exists)
4. LLM Agent (Baseline inference.py logic)

Usage:
    # Run a full benchmark (10 episodes per level per agent)
    python benchmark.py --episodes 10

    # Benchmark a specific level
    python benchmark.py --task medium

    # Generate the comparison table only
    python benchmark.py --table-only

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
from tqdm import tqdm

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Environment imports
from environment import PatchCascadeEnv
from grader import grade_episode
from models import ActionType, PatchCascadeAction, PatchCascadeObservation, NodeState, CriticalityTier

# Agent imports
try:
    from stable_baselines3 import PPO
    from gym_wrapper import PatchCascadeGymEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


# =============================================================================
# AGENT IMPLEMENTATIONS
# =============================================================================


class BaseAgent:
    """Base class for benchmark agents."""
    def act(self, obs: PatchCascadeObservation) -> PatchCascadeAction:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Basest baseline: picks random valid-ish actions."""
    def act(self, obs: PatchCascadeObservation) -> PatchCascadeAction:
        import random as _random
        action_type = _random.choice(list(ActionType))
        target = ""
        if obs.nodes:
            target = _random.choice(obs.nodes).hostname
        
        cve_id = None
        if action_type == ActionType.APPLY_PATCH and obs.vulnerabilities:
            v = _random.choice(obs.vulnerabilities)
            cve_id = v.cve_id
            if v.affected_hosts:
                target = _random.choice(v.affected_hosts)
        
        return PatchCascadeAction(
            action_type=action_type,
            target=target,
            cve_id=cve_id,
            reason="Random baseline agent",
        )


class HeuristicAgent(BaseAgent):
    """
    Advanced rule-based agent with proper dependency-aware logic.
    
    Strategy:
    1. Patch non-critical (Tier 2-3) vulnerabilities first (safe, no suspend needed)
    2. Handle Tier 1 nodes: suspend dependents -> suspend node -> patch -> resume all
    3. Never try to resume a crashed node whose dependency is still down
    """
    def act(self, obs: PatchCascadeObservation) -> PatchCascadeAction:
        # Build helper lookups
        dep_map = {}  # node -> list of (dependency, type)
        reverse_dep_map = {}  # dependency -> list of (dependent_node, type)
        for dep in obs.dependencies:
            dep_map.setdefault(dep.node, []).append((dep.depends_on, dep.dependency_type))
            reverse_dep_map.setdefault(dep.depends_on, []).append((dep.node, dep.dependency_type))
        
        node_by_name = {n.hostname: n for n in obs.nodes}
        
        # 1. Resume crashed nodes — but ONLY if their hard dependencies are ONLINE
        for node in obs.nodes:
            if node.state == NodeState.CRASHED:
                hard_deps = [(d, t) for d, t in dep_map.get(node.hostname, []) if t == "hard"]
                deps_ok = all(
                    node_by_name.get(d) and node_by_name[d].state == NodeState.ONLINE
                    for d, _ in hard_deps
                )
                if deps_ok:
                    return PatchCascadeAction(
                        action_type=ActionType.RESUME_SERVICE,
                        target=node.hostname,
                        reason="Recover crashed node (deps online)",
                    )
        
        # 2. Patch NON-CRITICAL vulns first (Tier 2-3, patch while online — zero cascade risk)
        sorted_vulns = sorted(
            obs.vulnerabilities,
            key=lambda v: (v.exploit_in_wild, v.cvss_score),
            reverse=True,
        )
        
        for vuln in sorted_vulns:
            for host in vuln.affected_hosts:
                node = node_by_name.get(host)
                if not node or node.state in (NodeState.PATCHING, NodeState.CRASHED, NodeState.OFFLINE):
                    continue
                if node.tier != CriticalityTier.CRITICAL and node.state == NodeState.ONLINE:
                    return PatchCascadeAction(
                        action_type=ActionType.APPLY_PATCH,
                        target=host,
                        cve_id=vuln.cve_id,
                        reason=f"Patch {vuln.cve_id} on non-critical {host}",
                    )
        
        # 3. Handle Tier 1 (CRITICAL) vulns — suspend-patch-resume workflow
        for vuln in sorted_vulns:
            for host in vuln.affected_hosts:
                node = node_by_name.get(host)
                if not node or node.tier != CriticalityTier.CRITICAL:
                    continue
                if node.state in (NodeState.PATCHING, NodeState.CRASHED, NodeState.OFFLINE):
                    continue
                    
                if node.state == NodeState.ONLINE:
                    # Suspend hard dependents first
                    dependents = reverse_dep_map.get(host, [])
                    for dep_name, dep_type in dependents:
                        if dep_type == "hard":
                            dep_node = node_by_name.get(dep_name)
                            if dep_node and dep_node.state == NodeState.ONLINE:
                                return PatchCascadeAction(
                                    action_type=ActionType.SUSPEND_SERVICE,
                                    target=dep_name,
                                    reason=f"Suspend {dep_name} before its dep {host}",
                                )
                    return PatchCascadeAction(
                        action_type=ActionType.SUSPEND_SERVICE,
                        target=host,
                        reason=f"Suspend Tier 1 {host} for patching",
                    )
                elif node.state == NodeState.SUSPENDED:
                    return PatchCascadeAction(
                        action_type=ActionType.APPLY_PATCH,
                        target=host,
                        cve_id=vuln.cve_id,
                        reason=f"Patch {vuln.cve_id} on suspended {host}",
                    )
        
        # 4. Resume suspended nodes (bottom-up: resume deps-ready nodes first)
        for node in obs.nodes:
            if node.state == NodeState.SUSPENDED:
                hard_deps = [(d, t) for d, t in dep_map.get(node.hostname, []) if t == "hard"]
                deps_online = all(
                    node_by_name.get(d) and node_by_name[d].state == NodeState.ONLINE
                    for d, _ in hard_deps
                )
                if deps_online:
                    return PatchCascadeAction(
                        action_type=ActionType.RESUME_SERVICE,
                        target=node.hostname,
                        reason=f"Resume {node.hostname} (deps ready)",
                    )
        
        # 5. Fallback resume
        for node in obs.nodes:
            if node.state == NodeState.SUSPENDED:
                return PatchCascadeAction(
                    action_type=ActionType.RESUME_SERVICE,
                    target=node.hostname,
                    reason="Resume suspended node",
                )

        return PatchCascadeAction(action_type=ActionType.NOOP, reason="Heuristic: waiting")


class RLAgent(BaseAgent):
    """Wrapper for trained SB3 models."""
    def __init__(self, model_path: str, env: PatchCascadeGymEnv):
        self.model = PPO.load(model_path)
        self.env = env
    
    def act(self, obs: PatchCascadeObservation) -> PatchCascadeAction:
        # RLAgent uses the Gym wrapper's encoding
        obs_array = self.env._encode_observation(obs)
        action_idx, _ = self.model.predict(obs_array, deterministic=True)
        return self.env._decode_action(action_idx)


# =============================================================================
# BENCHMARK ENGINE
# =============================================================================


@dataclass
class BenchmarkResult:
    agent_name: str
    task_level: str
    mean_score: float
    mean_reward: float
    success_rate: float
    completion: float
    efficiency: float
    safety: float
    strategy: float


def evaluate_agent(
    agent: BaseAgent, 
    task_level: str, 
    episodes: int = 10, 
    max_steps: int = 100
) -> BenchmarkResult:
    """Run an agent through multiple episodes of a task level."""
    env = PatchCascadeEnv()
    
    scores = []
    rewards = []
    successes = []
    dims = {"completion": [], "efficiency": [], "safety": [], "strategy": []}
    
    for _ in range(episodes):
        obs = env.reset(task_level=task_level)
        initial_vuln_count = len(obs.vulnerabilities)
        episode_rewards = []
        done = False
        steps = 0
        cascades = 0
        invalid = 0
        
        while not done and steps < max_steps:
            action = agent.act(obs)
            result = env.step(action)
            obs = result.observation
            episode_rewards.append(result.reward)
            done = result.done or result.truncated
            steps += 1
            
            if not result.info.get("valid", True): invalid += 1
            cascades = result.info.get("total_cascade_failures", cascades)
        
        success = len(obs.vulnerabilities) == 0
        
        # Grading
        episode_data = {
            "total_reward": sum(episode_rewards),
            "rewards": episode_rewards,
            "success": success,
            "steps": steps,
            "state": {
                "vulnerabilities": [v.model_dump() for v in obs.vulnerabilities],
                "nodes": [n.model_dump() for n in obs.nodes],
            },
            "cascade_failures": cascades,
            "invalid_actions": invalid,
            "initial_vulnerability_count": initial_vuln_count,
        }
        
        grader_result = grade_episode(task_level, episode_data)
        scores.append(grader_result.score)
        rewards.append(sum(episode_rewards))
        successes.append(success)
        for d in dims:
            dims[d].append(grader_result.dimensions.get(d, 0.0))
            
    return BenchmarkResult(
        agent_name=agent.agent_name if hasattr(agent, "agent_name") else agent.__class__.__name__,
        task_level=task_level,
        mean_score=np.mean(scores),
        mean_reward=np.mean(rewards),
        success_rate=np.mean(successes),
        completion=np.mean(dims["completion"]),
        efficiency=np.mean(dims["efficiency"]),
        safety=np.mean(dims["safety"]),
        strategy=np.mean(dims["strategy"]),
    )


def main():
    parser = argparse.ArgumentParser(description="PatchCascade SOC Benchmark")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per test")
    parser.add_argument("--task", default="all", help="Task level or 'all'")
    parser.add_argument("--rl-model", help="Path to trained RL model")
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    
    tasks = ["easy", "medium", "hard", "incident_response", "zero_day"]
    if args.task != "all":
        tasks = [args.task]
        
    agents = [
        RandomAgent(),
        HeuristicAgent()
    ]
    
    # Try to add RL agent if path provided or default exists
    if HAS_SB3:
        model_path = args.rl_model or "results/ppo_medium_final.zip"
        if os.path.exists(model_path):
            print(f"  🤖 Loading RL Agent from {model_path}")
            gym_env = PatchCascadeGymEnv() # For encoding
            rl_agent = RLAgent(model_path, gym_env)
            rl_agent.agent_name = "PPO (RL)"
            agents.append(rl_agent)
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"  🚀 PATCHCASCADE SOC BENCHMARK — {args.episodes} eps/task")
    print(f"{'='*70}\n")
    
    for task in tasks:
        print(f"  📈 Benchmarking {task.upper()}...")
        for agent in agents:
            res = evaluate_agent(agent, task, episodes=args.episodes)
            results.append(res)
            print(f"     {res.agent_name:<15}: Score={res.mean_score:.3f} | Success={res.success_rate:.1%}")
            
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
        
    # Generate Markdown Table for README
    print(f"\n{'='*70}")
    print(f"  📝 GENERATED RESULTS TABLE FOR README")
    print(f"{'='*70}\n")
    
    table = "| Agent | Easy | Medium | Hard | IR | Zero-Day | Avg |\n"
    table += "|-------|------|--------|------|----|----------|-----|\n"
    
    agent_names = sorted(list(set(r.agent_name for r in results)))
    for name in agent_names:
        row = f"| {name} | "
        task_scores = []
        for t in ["easy", "medium", "hard", "incident_response", "zero_day"]:
            r = next((x for x in results if x.agent_name == name and x.task_level == t), None)
            val = f"{r.mean_score:.2f}" if r else "---"
            row += f"{val} | "
            if r: task_scores.append(r.mean_score)
        
        avg = np.mean(task_scores) if task_scores else 0
        row += f"**{avg:.2f}** |"
        print(row)
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
