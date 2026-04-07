#!/usr/bin/env python3
"""
PatchCascade SOC - End-to-End Smoke Test
==========================================

Self-validating script that runs a heuristic agent through ALL 5 task levels
without requiring an LLM. Verifies that:

1. All 5 task levels initialize correctly
2. Actions execute and produce valid step results
3. Grading produces valid multi-dimensional scores
4. No runtime errors or crashes

This serves as a quick validation before submission.

Usage:
    python smoke_test.py

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

# Ensure project root is importable
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment import PatchCascadeEnv
from grader import grade_episode, list_graders, GRADERS
from models import (
    ActionType,
    CriticalityTier,
    NodeState,
    PatchCascadeAction,
    PatchCascadeObservation,
)
from tasks import list_tasks, list_tasks_with_graders


# =============================================================================
# HEURISTIC AGENT — No LLM needed
# =============================================================================


def heuristic_action(obs: PatchCascadeObservation) -> PatchCascadeAction:
    """
    Simple heuristic agent that makes reasonable decisions.

    Strategy:
    1. If there are crashed nodes, resume them
    2. If there's a Tier 1 node with a vuln that needs suspending, suspend dependents first
    3. If there's a patchable vulnerability, patch it
    4. Otherwise, noop
    """
    # Priority 1: Resume crashed nodes
    for node in obs.nodes:
        if node.state == NodeState.CRASHED:
            return PatchCascadeAction(
                action_type=ActionType.RESUME_SERVICE,
                target=node.hostname,
                reason="Recover crashed node",
            )

    # Priority 2: Patch vulnerabilities
    for vuln in sorted(obs.vulnerabilities, key=lambda v: -v.cvss_score):
        for host in vuln.affected_hosts:
            node = next((n for n in obs.nodes if n.hostname == host), None)
            if node is None:
                continue

            # Tier 1: needs to be SUSPENDED first
            if node.tier == CriticalityTier.CRITICAL:
                if node.state == NodeState.ONLINE:
                    # Suspend dependents first
                    for dep in obs.dependencies:
                        if dep.depends_on == host and dep.dependency_type == "hard":
                            dep_node = next(
                                (n for n in obs.nodes if n.hostname == dep.node), None
                            )
                            if dep_node and dep_node.state == NodeState.ONLINE:
                                return PatchCascadeAction(
                                    action_type=ActionType.SUSPEND_SERVICE,
                                    target=dep.node,
                                    reason=f"Suspend dependent before patching {host}",
                                )
                    # All dependents handled, suspend the critical node
                    return PatchCascadeAction(
                        action_type=ActionType.SUSPEND_SERVICE,
                        target=host,
                        reason=f"Suspend Tier 1 node for patching",
                    )
                elif node.state == NodeState.SUSPENDED:
                    return PatchCascadeAction(
                        action_type=ActionType.APPLY_PATCH,
                        target=host,
                        cve_id=vuln.cve_id,
                        reason=f"Patch {vuln.cve_id} on suspended Tier 1",
                    )
            # Tier 2-3: can patch while ONLINE
            elif node.state == NodeState.ONLINE:
                return PatchCascadeAction(
                    action_type=ActionType.APPLY_PATCH,
                    target=host,
                    cve_id=vuln.cve_id,
                    reason=f"Patch {vuln.cve_id} on {host}",
                )

    # Priority 3: Resume suspended nodes (cleanup)
    for node in obs.nodes:
        if node.state == NodeState.SUSPENDED:
            return PatchCascadeAction(
                action_type=ActionType.RESUME_SERVICE,
                target=node.hostname,
                reason="Resume suspended node",
            )

    return PatchCascadeAction(action_type=ActionType.NOOP, reason="Nothing to do")


# =============================================================================
# SMOKE TEST RUNNER
# =============================================================================


@dataclass
class TaskResult:
    task_id: str
    steps: int
    total_reward: float
    success: bool
    score: float
    dimensions: dict
    passed: bool
    elapsed_ms: float


def run_task(task_level: str, seed: int = 42, max_steps: int = 120) -> TaskResult:
    """Run a single task with the heuristic agent and grade it."""
    env = PatchCascadeEnv(seed=seed)
    obs = env.reset(task_level=task_level, seed=seed)

    rewards: list[float] = []
    cascade_failures = 0
    invalid_actions = 0
    done = False
    step_num = 0

    t0 = time.perf_counter()

    while not done and step_num < max_steps:
        action = heuristic_action(obs)
        result = env.step(action)

        obs = result.observation
        rewards.append(result.reward)
        done = result.done
        step_num += 1

        if not result.info.get("valid", True):
            invalid_actions += 1
        cascade_failures = result.info.get("total_cascade_failures", cascade_failures)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    success = len(obs.vulnerabilities) == 0

    # Grade the episode
    episode_data = {
        "total_reward": sum(rewards),
        "rewards": rewards,
        "success": success,
        "steps": step_num,
        "state": {
            "vulnerabilities": [v.model_dump() for v in obs.vulnerabilities],
            "nodes": [n.model_dump() for n in obs.nodes],
        },
        "cascade_failures": cascade_failures,
        "invalid_actions": invalid_actions,
        "initial_vulnerability_count": env._initial_vuln_count,
    }

    grader_result = grade_episode(task_level, episode_data)

    return TaskResult(
        task_id=task_level,
        steps=step_num,
        total_reward=sum(rewards),
        success=success,
        score=grader_result.score,
        dimensions=grader_result.dimensions,
        passed=grader_result.passed,
        elapsed_ms=elapsed_ms,
    )


def main():
    """Run smoke tests across all 5 task levels."""
    print("=" * 70)
    print("  PatchCascade SOC — End-to-End Smoke Test")
    print("=" * 70)
    print()

    # ---- Check 1: Task Registry ----
    print("📋 Check 1: Task Registry")
    all_tasks = list_tasks()
    tasks_with_graders = list_tasks_with_graders()
    print(f"   Total tasks: {len(all_tasks)}")
    print(f"   Tasks with graders: {len(tasks_with_graders)}")
    assert len(all_tasks) == 5, f"Expected 5 tasks, got {len(all_tasks)}"
    assert len(tasks_with_graders) == 5, f"Expected 5 graded tasks, got {len(tasks_with_graders)}"
    print("   ✅ PASS — 5 tasks with 5 graders")
    print()

    # ---- Check 2: Grader Registry ----
    print("📊 Check 2: Grader Registry")
    graders = list_graders()
    print(f"   Total graders: {len(graders)}")
    for g in graders:
        print(f"   - {g['task_id']}: {g['class']} (threshold={g['success_threshold']})")
    assert len(graders) == 5, f"Expected 5 graders, got {len(graders)}"
    print("   ✅ PASS — 5 graders registered")
    print()

    # ---- Check 3: Run All 5 Tasks ----
    print("🎮 Check 3: Running all 5 tasks with heuristic agent")
    print()

    task_levels = ["easy", "medium", "hard", "incident_response", "zero_day"]
    results: list[TaskResult] = []
    all_passed = True

    for level in task_levels:
        try:
            result = run_task(level)
            results.append(result)

            status = "✅ PASS" if result.success else "⚠️ PARTIAL"
            print(f"   {level:>20s}: score={result.score:.3f} steps={result.steps:>3d} "
                  f"reward={result.total_reward:>8.1f} {status} ({result.elapsed_ms:.0f}ms)")
            print(f"   {'':>20s}  completion={result.dimensions['completion']:.2f} "
                  f"efficiency={result.dimensions['efficiency']:.2f} "
                  f"safety={result.dimensions['safety']:.2f} "
                  f"strategy={result.dimensions['strategy']:.2f}")
        except Exception as e:
            all_passed = False
            print(f"   {level:>20s}: ❌ ERROR — {e}")

    print()

    # ---- Check 4: Score Validation ----
    print("🔍 Check 4: Score Validation")
    for r in results:
        assert 0.0 < r.score < 1.0, f"Score {r.score} not in (0, 1) for {r.task_id}"
        for dim, val in r.dimensions.items():
            assert 0.0 <= val <= 1.0, f"Dimension {dim}={val} not in [0, 1] for {r.task_id}"
    print("   ✅ PASS — All scores in valid range (0, 1)")
    print()

    # ---- Check 5: Import Validation ----
    print("📦 Check 5: Import Validation")
    try:
        from server import app
        from client import PatchCascadeLocalClient, PatchCascadeClient
        from grader import TaskGrader, EasyGrader, MediumGrader, HardGrader
        from grader import IncidentResponseGrader, ZeroDayGrader
        from models import PatchCascadeObservation, PatchCascadeAction, PatchCascadeState
        print("   ✅ PASS — All modules import successfully")
    except ImportError as e:
        print(f"   ❌ FAIL — Import error: {e}")
        all_passed = False
    print()

    # ---- Summary ----
    print("=" * 70)
    if all_passed:
        print("  ✅ ALL CHECKS PASSED — Submission is ready!")
    else:
        print("  ❌ SOME CHECKS FAILED — Fix errors before submitting.")
    print("=" * 70)
    print()

    # Print summary table
    print("📊 Results Summary:")
    print(f"  {'Task':<20s} {'Score':>6s} {'Steps':>5s} {'Completion':>10s} {'Grader':>10s}")
    print(f"  {'-'*20} {'-'*6} {'-'*5} {'-'*10} {'-'*10}")
    for r in results:
        grader_status = "PASS" if r.passed else "FAIL"
        print(f"  {r.task_id:<20s} {r.score:>6.3f} {r.steps:>5d} "
              f"{r.dimensions['completion']:>10.2f} {grader_status:>10s}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
