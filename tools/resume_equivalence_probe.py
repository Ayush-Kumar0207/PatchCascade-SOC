#!/usr/bin/env python3
"""Compare uninterrupted PPO with a safe-checkpoint continuation in a new process."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_canonical import (  # noqa: E402
    BoundaryCheckpointPPO,
    load_resumable_checkpoint,
    make_vec_env,
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
        rows = []
        actions = np.asarray(self.locals.get("actions", []))
        rewards = np.asarray(self.locals.get("rewards", []))
        dones = np.asarray(self.locals.get("dones", []))
        infos = self.locals.get("infos", [])
        for index in range(len(rewards)):
            rows.append({
                "timestep": int(self.model.num_timesteps),
                "worker": index,
                "action": np.asarray(actions[index]).astype(int).tolist(),
                "reward": float(rewards[index]),
                "done": bool(dones[index]),
                "episode_seed": infos[index].get("episode_seed"),
                "action_valid": bool(infos[index].get("action_valid", True)),
            })
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        return True


def probe_spec() -> tuple[dict, dict]:
    spec, _ = load_spec(ROOT / "training_specs" / "canonical_v1.json")
    probe = copy.deepcopy(spec)
    probe["algorithm"].update({
        "rollout_steps": 8,
        "batch_size": 8,
        "epochs_per_update": 1,
        "parallel_environments": 1,
        "device": "cpu",
        "architecture": {"policy": [16], "value": [16]},
    })
    return probe, {"task": "easy", "timesteps": 16}


def parameter_digest(model: BoundaryCheckpointPPO) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_result(model: BoundaryCheckpointPPO, trace: Path, output: Path) -> None:
    atomic_json(output, {
        "num_timesteps": int(model.num_timesteps),
        "parameter_sha256": parameter_digest(model),
        "trajectory_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "trajectory_rows": len(trace.read_text(encoding="utf-8").splitlines()),
    })


def worker(mode: str, directory: Path) -> None:
    seed_everything(42)
    spec, stage = probe_spec()
    trace = directory / ("baseline.trace.jsonl" if mode == "baseline" else "resumed.trace.jsonl")
    if mode == "resume":
        metadata = json.loads((directory / "checkpoint.metadata.json").read_text(encoding="utf-8"))
        model = load_resumable_checkpoint(directory / "checkpoint.zip", metadata, directory, "cpu")
        model.learn(
            total_timesteps=8, reset_num_timesteps=False,
            callback=TraceCallback(trace), progress_bar=False,
        )
        write_result(model, trace, directory / "resumed.result.json")
        model.get_env().close()
        return

    env = make_vec_env(spec, stage, 0)
    model = model_from_spec(spec, env)
    callback = TraceCallback(trace)
    if mode == "interrupt":
        def save_and_stop(current: BoundaryCheckpointPPO) -> None:
            model_identity, state_identity = save_resumable_checkpoint(
                current, directory / "checkpoint.zip", directory
            )
            atomic_json(directory / "checkpoint.metadata.json", {
                "total_timesteps": int(current.num_timesteps),
                "model_identity": model_identity,
                "runtime_state_identity": state_identity,
                "safe_boundary": "after-complete-rollout-and-optimizer-update",
            })
            raise PlannedInterruption("simulated process loss after a safe update boundary")

        model.boundary_hook = save_and_stop
        try:
            model.learn(total_timesteps=16, callback=callback, progress_bar=False)
        except PlannedInterruption:
            pass
        else:  # pragma: no cover - fail-closed guard
            raise RuntimeError("planned safe-boundary interruption did not occur")
        model.get_env().close()
        return

    model.learn(total_timesteps=16, callback=callback, progress_bar=False)
    write_result(model, trace, directory / "baseline.result.json")
    model.get_env().close()


def orchestrate() -> dict:
    with tempfile.TemporaryDirectory(prefix="patchcascade-resume-probe-") as raw:
        directory = Path(raw)
        for mode in ("baseline", "interrupt", "resume"):
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", mode, "--directory", str(directory)],
                cwd=ROOT, check=True,
            )
        baseline = json.loads((directory / "baseline.result.json").read_text(encoding="utf-8"))
        resumed = json.loads((directory / "resumed.result.json").read_text(encoding="utf-8"))
        if baseline != resumed:
            raise RuntimeError(
                "safe-checkpoint continuation diverged from uninterrupted CPU training: "
                + canonical_json({"baseline": baseline, "resumed": resumed})
            )
        return {"passed": True, "comparison": baseline, "processes": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["baseline", "interrupt", "resume"])
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.directory is None:
            raise SystemExit("--directory is required with --worker")
        worker(args.worker, args.directory)
        return
    try:
        report = orchestrate()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"STOP: resume-equivalence probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("RESUME EQUIVALENCE PASS: new-process continuation matched uninterrupted CPU training.")


if __name__ == "__main__":
    main()
