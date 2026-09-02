#!/usr/bin/env python3
"""
PatchCascade SOC — Benchmarking & Evaluation System
====================================================

Systematically evaluates and compares different agents across all 5
PatchCascade task levels. Generates the 'Source of Truth' comparison
table for reproducible policy comparison.

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
import csv
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
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
from training_repro import (
    ReproducibilityError, canonical_json, file_identity, git_metadata, load_spec,
    run_fingerprint, spec_hash, spec_reference,
)

# Agent imports
try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    from gym_wrapper import (
        FLATTENED_ACTION_SCHEMA_VERSION,
        FlattenedMaskedPatchCascadeEnv,
        PatchCascadeGymEnv,
    )
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


# =============================================================================
# AGENT IMPLEMENTATIONS
# =============================================================================


class BaseAgent:
    """Base class for benchmark agents."""
    agent_name = "agent"

    def begin_episode(self, seed: int, obs: PatchCascadeObservation) -> None:
        """Reset agent-local stochastic state for a matched episode."""

    def act(self, obs: PatchCascadeObservation) -> PatchCascadeAction:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Basest baseline: picks random valid-ish actions."""
    agent_name = "random"

    def __init__(self):
        self._rng = random.Random(0)

    def begin_episode(self, seed: int, obs: PatchCascadeObservation) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: PatchCascadeObservation) -> PatchCascadeAction:
        action_type = self._rng.choice(list(ActionType))
        target = ""
        if obs.nodes:
            target = self._rng.choice(obs.nodes).hostname
        
        cve_id = None
        if action_type == ActionType.APPLY_PATCH and obs.vulnerabilities:
            v = self._rng.choice(obs.vulnerabilities)
            cve_id = v.cve_id
            if v.affected_hosts:
                target = self._rng.choice(v.affected_hosts)
        
        return PatchCascadeAction(
            action_type=action_type,
            target=target,
            cve_id=cve_id,
            reason="Random baseline agent",
        )


class HeuristicAgent(BaseAgent):
    """
    agent_name = "heuristic"
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
        self.masked = isinstance(env, FlattenedMaskedPatchCascadeEnv)
        self.model = (MaskablePPO if self.masked else PPO).load(model_path)
        self.env = env
        self.agent_name = "ppo"
        if self.model.observation_space != env.observation_space or self.model.action_space != env.action_space:
            raise ReproducibilityError("RL model observation/action spaces do not match the corrected environment")

    def begin_episode(self, seed: int, obs: PatchCascadeObservation) -> None:
        self.env.sync_observation(obs)
    
    def act(self, obs: PatchCascadeObservation) -> PatchCascadeAction:
        # RLAgent uses the Gym wrapper's encoding
        self.env.sync_observation(obs)
        obs_array = self.env._encode_observation(obs)
        predict_kwargs = {"action_masks": self.env.action_masks()} if self.masked else {}
        action_idx, _ = self.model.predict(obs_array, deterministic=True, **predict_kwargs)
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
    score_std: float = 0.0
    score_median: float = 0.0
    score_ci95_low: float = 0.0
    score_ci95_high: float = 0.0
    reward_std: float = 0.0
    episodes: int = 0
    catastrophic_failures: int = 0
    raw_episodes: list[dict] = field(default_factory=list)


def _bootstrap_mean_ci(values: list[float], seed: int, samples: int = 5000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def evaluate_agent(
    agent: BaseAgent, 
    task_level: str, 
    episodes: int = 10, 
    max_steps: int = 100,
    seeds: list[int] | None = None,
    bootstrap_samples: int = 5000,
) -> BenchmarkResult:
    """Run an agent through multiple episodes of a task level."""
    env = PatchCascadeEnv()
    
    scores = []
    rewards = []
    successes = []
    dims = {"completion": [], "efficiency": [], "safety": [], "strategy": []}
    
    episode_seeds = list(seeds) if seeds is not None else list(range(episodes))
    if len(episode_seeds) != len(set(episode_seeds)):
        raise ReproducibilityError(f"Duplicate evaluation seeds for {task_level}")
    raw_episodes = []
    catastrophic_failures = 0

    for episode_index, episode_seed in enumerate(episode_seeds, start=1):
        obs = env.reset(task_level=task_level, seed=episode_seed)
        agent.begin_episode(episode_seed, obs)
        initial_vuln_count = len(obs.vulnerabilities)
        episode_rewards = []
        done = False
        environment_terminated = False
        environment_truncated = False
        steps = 0
        cascades = 0
        invalid = 0
        
        while not done and steps < max_steps:
            action = agent.act(obs)
            result = env.step(action)
            obs = result.observation
            episode_rewards.append(result.reward)
            done = result.done or result.truncated
            environment_terminated = bool(result.done and not result.truncated)
            environment_truncated = bool(result.truncated)
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
        catastrophic = bool(obs.nodes) and all(node.state == NodeState.CRASHED for node in obs.nodes)
        catastrophic_failures += int(catastrophic)
        raw_episodes.append({
            "episode_id": f"{agent.agent_name}:{task_level}:{episode_seed}",
            "episode_index": episode_index,
            "seed": episode_seed,
            "task_level": task_level,
            "agent": agent.agent_name,
            "steps": steps,
            "terminated": environment_terminated,
            "environment_truncated": environment_truncated,
            "externally_truncated": bool(not done and steps >= max_steps),
            "success": bool(success),
            "total_reward": float(sum(episode_rewards)),
            "score": float(grader_result.score),
            "dimensions": {key: float(value) for key, value in grader_result.dimensions.items()},
            "cascade_failures": int(cascades),
            "invalid_actions": int(invalid),
            "catastrophic_failure": catastrophic,
        })

    low, high = _bootstrap_mean_ci(scores, seed=sum(episode_seeds) + len(task_level), samples=bootstrap_samples)
            
    return BenchmarkResult(
        agent_name=agent.agent_name,
        task_level=task_level,
        mean_score=float(np.mean(scores)),
        mean_reward=float(np.mean(rewards)),
        success_rate=float(np.mean(successes)),
        completion=float(np.mean(dims["completion"])),
        efficiency=float(np.mean(dims["efficiency"])),
        safety=float(np.mean(dims["safety"])),
        strategy=float(np.mean(dims["strategy"])),
        score_std=float(np.std(scores)),
        score_median=float(np.median(scores)),
        score_ci95_low=float(low),
        score_ci95_high=float(high),
        reward_std=float(np.std(rewards)),
        episodes=len(episode_seeds),
        catastrophic_failures=catastrophic_failures,
        raw_episodes=raw_episodes,
    )


def _paired_gate(results: list[BenchmarkResult], task: str, baseline: str, bootstrap_samples: int = 5000) -> dict:
    ppo = next((item for item in results if item.agent_name == "ppo" and item.task_level == task), None)
    base = next((item for item in results if item.agent_name == baseline and item.task_level == task), None)
    if not ppo or not base:
        return {"task": task, "baseline": baseline, "available": False}
    ppo_by_seed = {row["seed"]: row["score"] for row in ppo.raw_episodes}
    base_by_seed = {row["seed"]: row["score"] for row in base.raw_episodes}
    seeds = sorted(set(ppo_by_seed) & set(base_by_seed))
    deltas = [ppo_by_seed[seed] - base_by_seed[seed] for seed in seeds]
    low, high = _bootstrap_mean_ci(deltas, seed=sum(seeds) + len(baseline), samples=bootstrap_samples)
    mean_delta = statistics.mean(deltas)
    return {
        "task": task, "baseline": baseline, "available": True,
        "paired_episodes": len(seeds), "mean_score_delta": mean_delta,
        "delta_ci95": [low, high], "evidence_exceeds_baseline": low > 0,
        "regression_flag": mean_delta < 0,
    }


def write_outputs(output_dir: Path, payload: dict, results: list[BenchmarkResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "agent_name", "task_level", "episodes", "mean_score", "score_std", "score_median",
            "score_ci95_low", "score_ci95_high", "success_rate", "mean_reward", "reward_std",
            "completion", "efficiency", "safety", "strategy", "catastrophic_failures",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fields})
    lines = ["# Matched benchmark summary", "", f"Split: `{payload['config']['split']}`", "", "| Agent | Task | n | Score mean ± std | Median | 95% bootstrap CI | Success | Safety failures |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in results:
        lines.append(
            f"| {item.agent_name} | {item.task_level} | {item.episodes} | {item.mean_score:.3f} ± {item.score_std:.3f} | "
            f"{item.score_median:.3f} | [{item.score_ci95_low:.3f}, {item.score_ci95_high:.3f}] | "
            f"{item.success_rate:.1%} | {item.catastrophic_failures} |"
        )
    lines.extend(["", "## Baseline gates", ""])
    for gate in payload["baseline_gates"]:
        if not gate["available"]:
            lines.append(f"- {gate['task']} vs {gate['baseline']}: not available")
        else:
            status = "evidence exceeds" if gate["evidence_exceeds_baseline"] else ("regression" if gate["regression_flag"] else "inconclusive")
            lines.append(f"- {gate['task']} vs {gate['baseline']}: **{status}**, paired mean delta {gate['mean_score_delta']:.3f}, 95% CI {gate['delta_ci95']}")
    lines.append("\nBelow-Heuristic or negative results remain valid scientific outcomes and are never hidden or auto-rejected.\n")
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Matched, seed-frozen PatchCascade benchmark")
    parser.add_argument("--spec", default="training_specs/canonical_v1.json")
    parser.add_argument("--split", choices=["validation", "canonical", "confirmation"], default="validation")
    parser.add_argument("--run-fingerprint", help="Required for canonical/confirmation evidence")
    parser.add_argument("--task", default="all")
    parser.add_argument("--rl-model")
    parser.add_argument("--output-dir", default="results/benchmark-development")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    spec, resolved_spec = load_spec(args.spec)
    split_key = {"validation": "validation", "canonical": "canonical_test", "confirmation": "confirmation_test"}[args.split]
    seeds = list(spec["seeds"][split_key])
    git = git_metadata()
    if git["dirty"]:
        raise ReproducibilityError("evaluation requires the clean locked source commit")
    expected_fingerprint = run_fingerprint(spec, git["commit"])
    if args.rl_model and args.run_fingerprint != expected_fingerprint:
        raise ReproducibilityError("Evaluation requires the current locked run fingerprint")
    tasks = spec["environment"]["task_levels"] if args.task == "all" else [args.task]
    if not set(tasks).issubset(spec["environment"]["task_levels"]):
        raise ReproducibilityError("Unknown task level")
    agents: list[BaseAgent] = [RandomAgent(), HeuristicAgent()]
    if args.rl_model and not args.baseline_only:
        env_class = (
            FlattenedMaskedPatchCascadeEnv
            if spec["environment"]["action_schema_version"] == FLATTENED_ACTION_SCHEMA_VERSION
            else PatchCascadeGymEnv
        )
        agents.append(RLAgent(args.rl_model, env_class()))
    results: list[BenchmarkResult] = []
    for task in tasks:
        for agent in agents:
            result = evaluate_agent(
                agent, task, seeds=seeds,
                max_steps=spec["evaluation"]["max_steps_by_task"][task],
                bootstrap_samples=spec["evaluation"]["bootstrap_samples"],
            )
            results.append(result)
            print(f"{agent.agent_name}/{task}: score={result.mean_score:.3f} success={result.success_rate:.1%}")
    summaries = []
    raw = []
    for result in results:
        item = asdict(result)
        raw.extend(item.pop("raw_episodes"))
        summaries.append(item)
    expected_ids = {(agent.agent_name, task, seed) for agent in agents for task in tasks for seed in seeds}
    actual_ids = {(row["agent"], row["task_level"], row["seed"]) for row in raw}
    if actual_ids != expected_ids or len(raw) != len(expected_ids):
        raise ReproducibilityError("Benchmark episode matrix is incomplete or duplicated")
    gates = [_paired_gate(results, task, baseline, spec["evaluation"]["bootstrap_samples"]) for task in tasks for baseline in ("random", "heuristic")]
    model_identity = file_identity(args.rl_model) if args.rl_model and not args.baseline_only else None
    payload = {
        "schema_version": 1, "status": "complete",
        "config": {
            "split": args.split, "spec_path": spec_reference(resolved_spec), "source_commit": git["commit"],
            "run_fingerprint": args.run_fingerprint, "seeds": seeds, "tasks": tasks,
            "max_steps_by_task": spec["evaluation"]["max_steps_by_task"],
            "bootstrap_samples": spec["evaluation"]["bootstrap_samples"],
            "deterministic_policy": spec["evaluation"]["deterministic_policy"],
            "spec_sha256": spec_hash(spec),
            "environment_schema_version": spec["environment"]["schema_version"],
            "reward_schema_version": spec["environment"]["reward_schema_version"],
            "grader_source_commit": git["commit"],
            "model_identity": model_identity,
        },
        "summaries": summaries, "raw_episodes": raw, "baseline_gates": gates,
    }
    write_outputs(Path(args.output_dir), payload, results)
    print(canonical_json({"complete": True, "episodes": len(raw), "output_dir": args.output_dir}))

if __name__ == "__main__":
    main()
