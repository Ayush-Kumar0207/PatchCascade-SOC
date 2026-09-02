#!/usr/bin/env python3
"""Prove CPU continuation equivalence across realistic vector/stage boundaries.

This is a deterministic CPU regression proof, not a claim of cross-GPU bitwise
equivalence. Each continuation is restored in a new Python process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_canonical import (  # noqa: E402
    TRUSTED_RUNTIME_STATE_FORMAT,
    BoundaryCheckpointMaskablePPO,
    BoundaryCheckpointPPO,
    load_resumable_checkpoint,
    make_vec_env,
    model_algorithm_class,
    model_from_spec,
    save_resumable_checkpoint,
)
from training_repro import atomic_json, canonical_json, load_spec, seed_everything  # noqa: E402


class PlannedInterruption(RuntimeError):
    pass


class TraceCallback(BaseCallback):
    def __init__(self, path: Path):
        super().__init__(verbose=0)
        self.path = path

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", []))
        rewards = np.asarray(self.locals.get("rewards", []))
        dones = np.asarray(self.locals.get("dones", []))
        infos = self.locals.get("infos", [])
        with self.path.open("a", encoding="utf-8") as handle:
            for worker in range(len(rewards)):
                row = {
                    "timestep": int(self.model.num_timesteps),
                    "worker": worker,
                    "task": infos[worker].get("task_level"),
                    "action": np.asarray(actions[worker]).astype(int).tolist(),
                    "reward": float(rewards[worker]),
                    "done": bool(dones[worker]),
                    "episode_seed": infos[worker].get("episode_seed"),
                    "action_valid": bool(infos[worker].get("action_valid", True)),
                }
                handle.write(canonical_json(row) + "\n")
        return True


def probe_spec() -> dict:
    spec, _ = load_spec(ROOT / "training_specs" / "canonical_v1.json")
    probe = copy.deepcopy(spec)
    probe["algorithm"].update({
        "rollout_steps": 4,
        "batch_size": 16,
        "epochs_per_update": 1,
        "parallel_environments": 4,
        "device": "cpu",
        "architecture": {"policy": [16], "value": [16]},
    })
    return probe


def scenario_stages(spec: dict, scenario: str) -> list[dict]:
    tasks = list(spec["environment"]["task_levels"])
    if scenario == "mixed-mid-stage":
        return [{"task": "mixed", "tasks": tasks, "timesteps": 32}]
    if scenario == "easy-to-mixed-stage-boundary":
        return [
            {"task": "easy", "timesteps": 16},
            {"task": "mixed", "tasks": tasks, "timesteps": 16},
        ]
    raise ValueError(f"unknown probe scenario: {scenario}")


def parameter_digest(model: BoundaryCheckpointPPO | BoundaryCheckpointMaskablePPO) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_result(model, trace: Path, output: Path) -> None:
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    atomic_json(output, {
        "num_timesteps": int(model.num_timesteps),
        "parameter_sha256": parameter_digest(model),
        "trajectory_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "trajectory_rows": len(rows),
        "worker_ids": sorted({row["worker"] for row in rows}),
        "task_counts": dict(sorted(Counter(row["task"] for row in rows).items())),
    })


def transition_stage(model, spec: dict, stage: dict, stage_index: int):
    new_env = make_vec_env(spec, stage, stage_index)
    old_env = model.get_env()
    model.set_env(new_env, force_reset=True)
    if old_env is not None:
        old_env.close()
    return new_env


def save_probe_checkpoint(model, directory: Path, spec: dict) -> None:
    model_identity, runtime_identity = save_resumable_checkpoint(
        model, directory / "checkpoint.zip", directory
    )
    atomic_json(directory / "checkpoint.metadata.json", {
        "algorithm": spec["algorithm"],
        "total_timesteps": int(model.num_timesteps),
        "model_identity": model_identity,
        "runtime_state_identity": runtime_identity,
        "safe_boundary": "after-complete-rollout-and-optimizer-update",
        "runtime_state_format": TRUSTED_RUNTIME_STATE_FORMAT,
        "algorithm_class": model_algorithm_class(model),
    })


def worker(mode: str, scenario: str, directory: Path) -> None:
    seed_everything(42)
    spec = probe_spec()
    stages = scenario_stages(spec, scenario)
    trace = directory / ("baseline.trace.jsonl" if mode == "baseline" else "resumed.trace.jsonl")

    if mode == "resume":
        metadata = json.loads((directory / "checkpoint.metadata.json").read_text(encoding="utf-8"))
        model = load_resumable_checkpoint(
            directory / "checkpoint.zip", metadata, directory, "cpu", trusted=True
        )
        if scenario == "mixed-mid-stage":
            remaining = 16
        else:
            transition_stage(model, spec, stages[1], 1)
            remaining = 16
        model.learn(
            total_timesteps=remaining, reset_num_timesteps=False,
            callback=TraceCallback(trace), progress_bar=False,
        )
        write_result(model, trace, directory / "resumed.result.json")
        model.get_env().close()
        return

    env = make_vec_env(spec, stages[0], 0)
    model = model_from_spec(spec, env)
    callback = TraceCallback(trace)

    if mode == "interrupt":
        if scenario == "mixed-mid-stage":
            def save_and_stop(current) -> None:
                save_probe_checkpoint(current, directory, spec)
                raise PlannedInterruption("simulated process loss after a mixed-task update")

            model.boundary_hook = save_and_stop
            try:
                model.learn(total_timesteps=32, callback=callback, progress_bar=False)
            except PlannedInterruption:
                pass
            else:  # pragma: no cover
                raise RuntimeError("planned mixed-task interruption did not occur")
        else:
            model.learn(total_timesteps=16, callback=callback, progress_bar=False)
            save_probe_checkpoint(model, directory, spec)
        model.get_env().close()
        return

    model.learn(total_timesteps=16, callback=callback, progress_bar=False)
    if scenario == "mixed-mid-stage":
        model.learn(total_timesteps=16, reset_num_timesteps=False, callback=callback, progress_bar=False)
    else:
        transition_stage(model, spec, stages[1], 1)
        model.learn(total_timesteps=16, reset_num_timesteps=False, callback=callback, progress_bar=False)
    write_result(model, trace, directory / "baseline.result.json")
    model.get_env().close()


def orchestrate() -> dict:
    comparisons = {}
    scenarios = ("mixed-mid-stage", "easy-to-mixed-stage-boundary")
    with tempfile.TemporaryDirectory(prefix="patchcascade-resume-probe-") as raw:
        root = Path(raw)
        for scenario in scenarios:
            directory = root / scenario
            directory.mkdir()
            for mode in ("baseline", "interrupt", "resume"):
                subprocess.run([
                    sys.executable, str(Path(__file__).resolve()),
                    "--worker", mode, "--scenario", scenario, "--directory", str(directory),
                ], cwd=ROOT, check=True)
            baseline = json.loads((directory / "baseline.result.json").read_text(encoding="utf-8"))
            resumed = json.loads((directory / "resumed.result.json").read_text(encoding="utf-8"))
            if baseline != resumed:
                raise RuntimeError(
                    f"{scenario} continuation diverged: "
                    + canonical_json({"baseline": baseline, "resumed": resumed})
                )
            if baseline["worker_ids"] != [0, 1, 2, 3] or not baseline["task_counts"]:
                raise RuntimeError(f"{scenario} did not exercise all four vector workers/tasks")
            comparisons[scenario] = baseline
    return {
        "passed": True,
        "scope": "deterministic CPU only; no cross-GPU bitwise-equivalence claim",
        "processes": len(scenarios) * 3,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["baseline", "interrupt", "resume"])
    parser.add_argument("--scenario", choices=["mixed-mid-stage", "easy-to-mixed-stage-boundary"])
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.directory is None or args.scenario is None:
            raise SystemExit("--scenario and --directory are required with --worker")
        worker(args.worker, args.scenario, args.directory)
        return
    try:
        report = orchestrate()
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"STOP: resume-equivalence probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("RESUME EQUIVALENCE PASS: four-worker mixed-task and stage-boundary continuations matched.")


if __name__ == "__main__":
    main()
