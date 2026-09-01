#!/usr/bin/env python3
"""Run every CPU-feasible gate before PatchCascade optimizer steps."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_repro import (
    ReproducibilityError, declared_dependency_mismatches, git_metadata, load_spec,
    run_fingerprint, runtime_info, spec_hash,
)

PATTERNS = {
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def scan_secrets() -> list[dict]:
    findings = []
    excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "results", "runs", "checkpoints"}
    suffixes = {".py", ".json", ".jsonl", ".md", ".txt", ".toml", ".yaml", ".yml", ".env", ".log"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or excluded.intersection(path.parts) or path.name == Path(__file__).name:
            continue
        if path.suffix.lower() not in suffixes and not path.name.startswith(".env"):
            continue
        try:
            if path.stat().st_size > 10 * 1024 * 1024:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append({"path": str(path.relative_to(ROOT)), "line": line_number, "kind": kind})
    return findings


def run_command(command: list[str], label: str) -> dict:
    process = subprocess.run(command, cwd=ROOT)
    if process.returncode:
        raise ReproducibilityError(f"{label} failed — training has not started")
    return {"label": label, "command": command, "returncode": process.returncode}


def environment_checks(spec: dict) -> None:
    import numpy as np
    from gymnasium.utils.env_checker import check_env
    from benchmark import HeuristicAgent, RandomAgent, evaluate_agent
    from gym_wrapper import PatchCascadeGymEnv

    for level in spec["environment"]["task_levels"]:
        env = PatchCascadeGymEnv(task_level=level, seed=123)
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=123)
        second, _ = env.reset(seed=123)
        if not np.array_equal(first, second):
            raise ReproducibilityError(f"deterministic reset failed for {level}")
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise ReproducibilityError(f"non-finite rollout value on {level}")
            if terminated or truncated:
                break
    seeds = spec["seeds"]["validation"][:2]
    for level in spec["environment"]["task_levels"]:
        max_steps = spec["evaluation"]["max_steps_by_task"][level]
        evaluate_agent(RandomAgent(), level, seeds=seeds, max_steps=max_steps, bootstrap_samples=100)
        evaluate_agent(HeuristicAgent(), level, seeds=seeds, max_steps=max_steps, bootstrap_samples=100)


def optimizer_shape_probe(spec: dict) -> dict:
    """Exercise one exact-size PPO rollout/update on CPU before costly training."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gym_wrapper import PatchCascadeGymEnv

    cfg = spec["algorithm"]
    count = int(cfg["parallel_environments"])
    seed = int(spec["seeds"]["global_training_seed"])
    env = DummyVecEnv([
        (lambda rank=rank: Monitor(PatchCascadeGymEnv(task_level="easy", seed=seed + rank)))
        for rank in range(count)
    ])
    try:
        model = PPO(
            cfg["policy"], env, seed=seed, learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
            ent_coef=cfg["entropy_coefficient"], vf_coef=cfg["value_coefficient"],
            max_grad_norm=cfg["max_grad_norm"], n_steps=cfg["rollout_steps"],
            batch_size=cfg["batch_size"], n_epochs=cfg["epochs_per_update"], device="cpu",
            policy_kwargs={"net_arch": {"pi": cfg["architecture"]["policy"], "vf": cfg["architecture"]["value"]}},
            verbose=0,
        )
        before = [parameter.detach().clone() for parameter in model.policy.parameters()]
        rollout = int(cfg["rollout_steps"]) * count
        model.learn(total_timesteps=rollout, progress_bar=False)
        after = list(model.policy.parameters())
        if not all(torch.isfinite(parameter).all().item() for parameter in after):
            raise ReproducibilityError("canonical optimizer-shape probe produced non-finite parameters")
        changed = sum(not torch.equal(left, right.detach()) for left, right in zip(before, after))
        if changed == 0:
            raise ReproducibilityError("canonical optimizer-shape probe made no parameter update")
        return {"label": "exact-shape CPU PPO rollout/update", "returncode": 0, "timesteps": rollout, "changed_parameter_tensors": changed}
    finally:
        env.close()


def perform_preflight(spec_path: str | Path, *, run_tests: bool = True, enforce_lock: bool = True) -> dict:
    spec, resolved = load_spec(spec_path)
    git = git_metadata()
    if git["dirty"]:
        raise ReproducibilityError("source commit does not match a clean run identity; changed paths: " + ", ".join(git["status_paths"][:20]))
    missing = [item for item in spec["required_source_files"] if not (ROOT / item).exists()]
    if missing:
        raise ReproducibilityError(f"required canonical files are missing: {missing}")
    secrets = scan_secrets()
    if secrets:
        raise ReproducibilityError(f"potential credential material detected (values hidden): {secrets[:20]}")
    runtime = runtime_info(spec["dependencies"])
    mismatches = declared_dependency_mismatches(spec, runtime)
    if enforce_lock and mismatches:
        raise ReproducibilityError(f"dependency lock mismatch: {json.dumps(mismatches, sort_keys=True)}")
    validations = []
    if run_tests:
        validations.append(run_command([sys.executable, "-m", "pytest", "-q"], "test suite"))
        validations.append(run_command([sys.executable, "smoke_test.py"], "smoke test"))
        environment_checks(spec)
        validations.append({"label": "Gymnasium/determinism/finite/baseline checks", "returncode": 0})
        validations.append(optimizer_shape_probe(spec))
    canonical_pass = bool(run_tests and enforce_lock)
    return {
        "passed": canonical_pass, "diagnostic_only": not canonical_pass,
        "source_commit": git["commit"], "branch": git["branch"],
        "spec_path": resolved.relative_to(ROOT).as_posix(), "spec_sha256": spec_hash(spec),
        "run_fingerprint": run_fingerprint(spec, git["commit"]),
        "runtime": runtime, "dependency_mismatches": mismatches, "validations": validations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="training_specs/canonical_v1.json")
    parser.add_argument("--skip-tests", action="store_true", help="Development diagnostics only; canonical trainer never uses this")
    parser.add_argument("--allow-unlocked-dependencies", action="store_true", help="Development diagnostics only")
    args = parser.parse_args()
    try:
        report = perform_preflight(args.spec, run_tests=not args.skip_tests, enforce_lock=not args.allow_unlocked_dependencies)
    except ReproducibilityError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["passed"]:
        print("PREFLIGHT PASS: optimizer steps remain disabled until the canonical trainer creates the run lock.")
    else:
        print("DIAGNOSTIC COMPLETE — NOT A CANONICAL PREFLIGHT PASS; training remains prohibited.")


if __name__ == "__main__":
    main()
