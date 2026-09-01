#!/usr/bin/env python3
"""One safe, resume-aware entry point for the frozen PatchCascade experiment."""

from __future__ import annotations

import argparse
import cloudpickle
import json
import math
import os
import random
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_wrapper import PatchCascadeGymEnv
from training_repro import (
    ReproducibilityError, append_event, atomic_json, ensure_external_run_dir,
    ensure_run_lock, file_identity, load_spec, provenance, resolve_run_path,
    seed_everything, spec_hash, utc_now, validate_file_identity, validate_resume,
)
from tools.training_preflight import perform_preflight


def save_model_atomic(model: PPO, path: Path) -> None:
    """Prevent an interrupted SB3 write from replacing a durable model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.zip")
    model.save(temporary)
    os.replace(temporary, path)


class BoundaryCheckpointPPO(PPO):
    """PPO with a hook that runs only after a complete rollout update."""

    boundary_hook = None

    def train(self) -> None:
        super().train()
        if self.boundary_hook is not None:
            self.boundary_hook(self)


def runtime_state_path(model_path: Path) -> Path:
    return model_path.with_suffix(".runtime.pkl")


def _runtime_state(model: PPO) -> dict[str, Any]:
    """Capture stochastic and environment state excluded by SB3 model.save()."""
    import torch

    env = model.get_env()
    if env is None:
        raise ReproducibilityError("cannot checkpoint PPO without its vectorized environment")
    return {
        "schema_version": 1,
        "safe_boundary": "after-complete-rollout-and-optimizer-update",
        "num_timesteps": int(model.num_timesteps),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "vector_environment": env,
        "last_observation": model._last_obs,
        "last_episode_starts": model._last_episode_starts,
        "last_original_observation": getattr(model, "_last_original_obs", None),
    }


def save_runtime_state_atomic(model: PPO, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        cloudpickle.dump(_runtime_state(model), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_resumable_checkpoint(model: PPO, model_path: Path, run_dir: Path) -> tuple[dict, dict]:
    """Persist an SB3 model and its exact boundary runtime state before metadata."""
    hook = getattr(model, "boundary_hook", None)
    if hasattr(model, "boundary_hook"):
        model.boundary_hook = None
    try:
        save_model_atomic(model, model_path)
    finally:
        if hasattr(model, "boundary_hook"):
            model.boundary_hook = hook
    state_path = runtime_state_path(model_path)
    save_runtime_state_atomic(model, state_path)
    return (
        file_identity(model_path, relative_to=run_dir),
        file_identity(state_path, relative_to=run_dir),
    )


def load_resumable_checkpoint(
    model_path: Path, metadata: dict[str, Any], run_dir: Path, device: str
) -> BoundaryCheckpointPPO:
    """Validate and restore the complete safe-boundary continuation state."""
    import torch

    state_identity = metadata.get("runtime_state_identity", {})
    state_path = resolve_run_path(run_dir, str(state_identity.get("path", "")))
    validate_file_identity(state_path, state_identity, relative_to=run_dir)
    try:
        with state_path.open("rb") as handle:
            state = cloudpickle.load(handle)
    except Exception as exc:
        raise ReproducibilityError("checkpoint runtime state is missing, corrupt, or incompatible") from exc
    if (
        state.get("schema_version") != 1
        or state.get("safe_boundary") != "after-complete-rollout-and-optimizer-update"
        or state.get("num_timesteps") != metadata.get("total_timesteps")
    ):
        raise ReproducibilityError("checkpoint runtime state does not match its safe-boundary metadata")
    env = state.get("vector_environment")
    if not isinstance(env, DummyVecEnv):
        raise ReproducibilityError("checkpoint vectorized environment state is incompatible")
    model = BoundaryCheckpointPPO.load(model_path, env=env, device=device, force_reset=False)
    model._last_obs = state.get("last_observation")
    model._last_episode_starts = state.get("last_episode_starts")
    model._last_original_obs = state.get("last_original_observation")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = state.get("torch_cuda_rng_states", [])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ReproducibilityError("checkpoint CUDA RNG state does not match this runtime")
        torch.cuda.set_rng_state_all(cuda_states)
    return model


class MixedTaskEnv(PatchCascadeGymEnv):
    def __init__(self, tasks: list[str], seed: int, **kwargs: Any):
        self._mixed_tasks = list(tasks)
        self._task_rng = random.Random(seed)
        super().__init__(task_level=self._mixed_tasks[0], seed=seed, **kwargs)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._task_rng = random.Random(seed)
        chosen = self._task_rng.choice(self._mixed_tasks)
        merged = dict(options or {})
        merged["task_level"] = chosen
        return super().reset(seed=seed, options=merged)


def make_vec_env(spec: dict, stage: dict, stage_index: int) -> DummyVecEnv:
    global_seed = int(spec["seeds"]["global_training_seed"])
    count = int(spec["algorithm"]["parallel_environments"])

    def factory(rank: int):
        worker_seed = global_seed + 1000 * stage_index + rank

        def create():
            kwargs = {
                "seed": worker_seed,
                "normalize_obs": spec["environment"]["normalize_observations"],
                "reward_scale": spec["environment"]["reward_scale"],
            }
            if stage["task"] == "mixed":
                env = MixedTaskEnv(stage["tasks"], **kwargs)
            else:
                env = PatchCascadeGymEnv(task_level=stage["task"], **kwargs)
            return Monitor(env)

        return create

    return DummyVecEnv([factory(rank) for rank in range(count)])


def model_from_spec(spec: dict, env: DummyVecEnv) -> BoundaryCheckpointPPO:
    cfg = spec["algorithm"]
    return BoundaryCheckpointPPO(
        cfg["policy"], env, seed=spec["seeds"]["global_training_seed"],
        learning_rate=cfg["learning_rate"], gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"], ent_coef=cfg["entropy_coefficient"], vf_coef=cfg["value_coefficient"],
        max_grad_norm=cfg["max_grad_norm"], n_steps=cfg["rollout_steps"], batch_size=cfg["batch_size"],
        n_epochs=cfg["epochs_per_update"], device=cfg["device"], verbose=0,
        policy_kwargs={"net_arch": {"pi": cfg["architecture"]["policy"], "vf": cfg["architecture"]["value"]}},
    )


class SafetyCheckpointCallback(BaseCallback):
    def __init__(self, run_dir: Path, spec: dict, lock: dict, progress: dict, stage: dict, stage_index: int):
        super().__init__(verbose=0)
        self.run_dir, self.spec, self.lock, self.progress = run_dir, spec, lock, progress
        self.stage, self.stage_index = stage, stage_index
        self.checkpoint_every = int(spec["checkpoint"]["frequency_timesteps"])
        self.last_checkpoint = int(progress.get("total_timesteps", 0))
        window = int(spec["runtime_guards"]["diagnostic_window"])
        self.actions: deque[str] = deque(maxlen=window)
        self.validity: deque[bool] = deque(maxlen=window)
        self.warning_emitted: set[str] = set()

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        observations = np.asarray(self.locals.get("new_obs", []), dtype=float)
        if not np.isfinite(rewards).all() or not np.isfinite(observations).all():
            raise FloatingPointError("NaN/Inf observation or reward detected")
        for info in self.locals.get("infos", []):
            self.actions.append(str(info.get("action_taken", "unknown")))
            self.validity.append(bool(info.get("action_valid", True)))
        self._guard_policy_collapse()
        return True

    def _on_rollout_end(self) -> None:
        diagnostics = {key: value for key, value in self.model.logger.name_to_value.items() if any(token in key for token in ("loss", "approx_kl", "entropy", "explained_variance", "clip_fraction"))}
        for key, value in diagnostics.items():
            if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"non-finite model diagnostic: {key}")
        path = self.run_dir / "training_diagnostics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "run_fingerprint": self.lock["run_fingerprint"],
                "source_commit": self.lock["source_commit"],
                "timesteps": self.num_timesteps,
                "stage": self.stage["task"],
                "stage_index": self.stage_index,
                "metrics": diagnostics,
            }, default=float, sort_keys=True) + "\n")

    def _guard_policy_collapse(self) -> None:
        window = int(self.spec["runtime_guards"]["diagnostic_window"])
        if len(self.actions) < window:
            return
        invalid_rate = 1.0 - sum(self.validity) / len(self.validity)
        noop_rate = sum(action == "noop" for action in self.actions) / len(self.actions)
        limits = self.spec["runtime_guards"]
        for name, value, threshold in (
            ("invalid_action_rate", invalid_rate, limits["invalid_action_warning_rate"]),
            ("noop_rate", noop_rate, limits["degenerate_noop_warning_rate"]),
        ):
            if value >= threshold and name not in self.warning_emitted:
                append_event(self.run_dir / self.spec["outputs"]["events"], "warning", self.lock, self.stage["task"], guard=name, value=value, threshold=threshold)
                self.warning_emitted.add(name)
        if invalid_rate >= 0.9:
            raise ReproducibilityError("invalid-action explosion detected (>=90% over diagnostic window)")
        if noop_rate >= 0.995:
            raise ReproducibilityError("degenerate no-op policy detected (>=99.5% over diagnostic window)")

    def save_checkpoint(self, *, force: bool = False) -> None:
        if not force and self.num_timesteps - self.last_checkpoint < self.checkpoint_every:
            return
        if self.num_timesteps == self.last_checkpoint:
            return
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / f"checkpoint_{self.num_timesteps}.zip"
        identity, runtime_identity = save_resumable_checkpoint(
            self.model, model_path, self.run_dir
        )
        stage_start = int(self.progress.get("stage_start_total_timesteps", 0))
        metadata = {
            **self.lock, "stage_index": self.stage_index, "stage": self.stage,
            "stage_timesteps_completed": self.num_timesteps - stage_start,
            "total_timesteps": self.num_timesteps, "model_file": model_path.name,
            "model_identity": identity, "runtime_state_identity": runtime_identity,
            "safe_boundary": "after-complete-rollout-and-optimizer-update",
        }
        atomic_json(model_path.with_suffix(".metadata.json"), metadata)
        self.progress.update(
            stage_index=self.stage_index, stage=self.stage,
            stage_timesteps_completed=metadata["stage_timesteps_completed"],
            total_timesteps=self.num_timesteps, latest_checkpoint=identity["path"],
            latest_checkpoint_sha256=identity["sha256"], status="training",
            latest_runtime_state=runtime_identity["path"],
            latest_runtime_state_sha256=runtime_identity["sha256"],
        )
        atomic_json(self.run_dir / self.spec["outputs"]["progress"], self.progress)
        append_event(
            self.run_dir / self.spec["outputs"]["events"], "checkpoint_saved",
            self.lock, self.stage["task"], checkpoint=identity["path"],
            checkpoint_sha256=identity["sha256"], runtime_state=runtime_identity["path"],
            runtime_state_sha256=runtime_identity["sha256"],
            safe_boundary=metadata["safe_boundary"], total_timesteps=self.num_timesteps,
        )
        self.last_checkpoint = self.num_timesteps
        pairs = sorted(checkpoint_dir.glob("checkpoint_*.metadata.json"), key=lambda path: int(path.stem.split("_")[-1].split(".")[0]))
        for old_metadata in pairs[:-int(self.spec["checkpoint"]["retain_last"])]:
            old_model = old_metadata.with_name(old_metadata.name.replace(".metadata.json", ".zip"))
            old_runtime = runtime_state_path(old_model)
            if old_model.parent.resolve() == checkpoint_dir.resolve():
                try:
                    pruned_metadata = json.loads(old_metadata.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ReproducibilityError(
                        "checkpoint metadata is unreadable during retention pruning"
                    ) from exc
                pruned_model_identity = pruned_metadata.get("model_identity", {})
                pruned_runtime_identity = pruned_metadata.get("runtime_state_identity", {})
                validate_file_identity(
                    old_model, pruned_model_identity, relative_to=self.run_dir
                )
                validate_file_identity(
                    old_runtime, pruned_runtime_identity, relative_to=self.run_dir
                )
                old_model.unlink(missing_ok=True)
                old_runtime.unlink(missing_ok=True)
                old_metadata.unlink(missing_ok=True)
                append_event(
                    self.run_dir / self.spec["outputs"]["events"],
                    "checkpoint_pruned", self.lock, self.stage["task"],
                    checkpoint=pruned_model_identity["path"],
                    checkpoint_sha256=pruned_model_identity["sha256"],
                    runtime_state=pruned_runtime_identity["path"],
                    runtime_state_sha256=pruned_runtime_identity["sha256"],
                    reason="configured-retention-limit",
                )


def load_progress(run_dir: Path, spec: dict, lock: dict) -> dict:
    path = run_dir / spec["outputs"]["progress"]
    if not path.exists():
        return {**lock, "status": "created", "stage_index": 0, "completed_stages": [], "total_timesteps": 0, "stage_timesteps_completed": 0}
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError("training progress is corrupt; refusing resume") from exc
    validate_resume(progress, lock)
    expected = [stage["task"] for stage in spec["methodology"]["stages"]]
    completed = progress.get("completed_stages", [])
    if completed != expected[:len(completed)]:
        raise ReproducibilityError("completed stages are not an exact canonical curriculum prefix")
    return progress


def capture_provenance_files(run_dir: Path, spec: dict, spec_path: Path, command: list[str], lock: dict) -> None:
    json_path = run_dir / spec["outputs"]["provenance"]
    markdown_path = run_dir / spec["outputs"]["provenance_markdown"]
    if json_path.exists() or markdown_path.exists():
        if not json_path.is_file() or not markdown_path.is_file():
            raise ReproducibilityError("initial provenance is incomplete; refusing to overwrite it")
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproducibilityError("initial provenance is corrupt; refusing to overwrite it") from exc
        if (
            existing.get("run_fingerprint") != lock["run_fingerprint"]
            or existing.get("spec_sha256") != lock["spec_sha256"]
            or existing.get("git", {}).get("commit") != lock["source_commit"]
            or existing.get("git", {}).get("dirty")
        ):
            raise ReproducibilityError("initial provenance belongs to another or dirty run")
        return
    payload = provenance(spec, spec_path, run_dir, command)
    atomic_json(json_path, payload)
    runtime = payload["runtime"]
    gpu = runtime["gpu"]
    markdown = f"""# Run provenance

- Source commit: `{payload['git']['commit']}`
- Clean source: `{not payload['git']['dirty']}`
- Spec SHA-256: `{payload['spec_sha256']}`
- Run fingerprint: `{payload['run_fingerprint']}`
- Python: `{runtime['python'].splitlines()[0]}`
- GPU: `{gpu.get('name') or 'none'}`
- CUDA: `{gpu.get('cuda')}`

The adjacent JSON is authoritative and contains only allowlisted environment variables.
"""
    markdown_path.write_text(markdown, encoding="utf-8")


def persist_preflight_report(run_dir: Path, spec: dict, lock: dict, report: dict) -> Path:
    initial = run_dir / spec["outputs"]["preflight_report"]
    if not initial.exists():
        atomic_json(initial, report)
        return initial
    try:
        existing = json.loads(initial.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError("initial preflight report is corrupt; refusing to overwrite it") from exc
    for candidate in (existing, report):
        if (
            candidate.get("passed") is not True
            or candidate.get("source_commit") != lock["source_commit"]
            or candidate.get("spec_sha256") != spec_hash(spec)
            or candidate.get("run_fingerprint") != lock["run_fingerprint"]
            or candidate.get("dependency_mismatches")
        ):
            raise ReproducibilityError("preflight report belongs to another or failed run")
    directory = run_dir / "resume_preflight_reports"
    name = "preflight_" + utc_now().replace(":", "").replace("-", "").replace(".", "") + ".json"
    path = directory / name
    if path.exists():
        raise ReproducibilityError("resume preflight report name collision")
    atomic_json(path, report)
    return path


def train(spec: dict, spec_path: Path, run_dir: Path, lock: dict) -> None:
    seed_everything(spec["seeds"]["global_training_seed"])
    progress = load_progress(run_dir, spec, lock)
    events = run_dir / spec["outputs"]["events"]
    stages = spec["methodology"]["stages"]
    if progress.get("status") == "trained":
        final_path = run_dir / spec["outputs"]["final_model"]
        metadata_path = run_dir / spec["outputs"]["final_model_metadata"]
        try:
            final_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproducibilityError("trained progress exists but final model metadata is missing/corrupt") from exc
        validate_resume(final_metadata, lock)
        validate_file_identity(final_path, final_metadata.get("model_identity", {}), relative_to=run_dir)
        return
    model: PPO | None = None
    checkpoint_metadata: dict[str, Any] | None = None
    latest = progress.get("latest_checkpoint")
    if latest:
        latest_path = resolve_run_path(run_dir, latest)
        metadata_path = latest_path.with_suffix(".metadata.json")
        try:
            checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproducibilityError("this checkpoint has missing/corrupt identity metadata") from exc
        validate_resume(checkpoint_metadata, lock)
        validate_file_identity(latest_path, checkpoint_metadata.get("model_identity", {}), relative_to=run_dir)
        runtime_identity = checkpoint_metadata.get("runtime_state_identity", {})
        runtime_path = resolve_run_path(run_dir, str(runtime_identity.get("path", "")))
        validate_file_identity(runtime_path, runtime_identity, relative_to=run_dir)
        checkpoint_identity = checkpoint_metadata["model_identity"]
        if (
            checkpoint_identity.get("sha256") != progress.get("latest_checkpoint_sha256")
            or checkpoint_metadata.get("total_timesteps") != progress.get("total_timesteps")
            or runtime_identity.get("path") != progress.get("latest_runtime_state")
            or runtime_identity.get("sha256") != progress.get("latest_runtime_state_sha256")
            or checkpoint_metadata.get("safe_boundary") != "after-complete-rollout-and-optimizer-update"
        ):
            raise ReproducibilityError("this checkpoint does not match durable training progress")
        metadata_stage_index = checkpoint_metadata.get("stage_index")
        if not isinstance(metadata_stage_index, int) or not 0 <= metadata_stage_index < len(stages):
            raise ReproducibilityError("this checkpoint has an invalid canonical stage index")
        if latest_path.name.startswith("checkpoint_"):
            if (
                metadata_stage_index != progress.get("stage_index")
                or checkpoint_metadata.get("stage_timesteps_completed") != progress.get("stage_timesteps_completed")
                or checkpoint_metadata.get("stage") != stages[metadata_stage_index]
            ):
                raise ReproducibilityError("this checkpoint stage/progress identity is inconsistent")
        elif latest_path.name.startswith("stage_"):
            if (
                progress.get("stage_index") != metadata_stage_index + 1
                or progress.get("stage_timesteps_completed") != 0
                or checkpoint_metadata.get("stage") != stages[metadata_stage_index]
            ):
                raise ReproducibilityError("this completed-stage checkpoint does not match durable progress")
        else:
            raise ReproducibilityError("latest checkpoint name is not repository-generated")
        append_event(events, "training_resumed", lock, checkpoint_metadata["stage"]["task"], checkpoint=latest)

    for stage_index, stage in enumerate(stages):
        if stage_index < len(progress.get("completed_stages", [])):
            continue
        env = make_vec_env(spec, stage, stage_index)
        if model is None:
            if latest:
                if checkpoint_metadata is None:
                    raise ReproducibilityError("checkpoint metadata was not loaded")
                model = load_resumable_checkpoint(
                    resolve_run_path(run_dir, latest), checkpoint_metadata, run_dir,
                    spec["algorithm"]["device"],
                )
                restored_env = model.get_env()
                if latest_path.name.startswith("checkpoint_"):
                    env.close()
                    if restored_env is None:
                        raise ReproducibilityError("checkpoint did not restore its vectorized environment")
                    env = restored_env
                else:
                    if restored_env is not None:
                        restored_env.close()
                    model.set_env(env, force_reset=True)
            else:
                model = model_from_spec(spec, env)
        else:
            model.set_env(env)
        model.set_logger(configure(str(run_dir / "sb3" / f"stage_{stage_index}_{stage['task']}"), ["stdout", "csv", "json"] ))
        already = int(progress.get("stage_timesteps_completed", 0)) if progress.get("stage_index") == stage_index else 0
        remaining = int(stage["timesteps"]) - already
        if remaining < 0:
            raise ReproducibilityError("checkpoint progress exceeds the frozen stage target")
        if remaining:
            progress["stage_start_total_timesteps"] = int(model.num_timesteps) - already
            append_event(events, "stage_started", lock, stage["task"], stage_index=stage_index, remaining_timesteps=remaining)
            callback = SafetyCheckpointCallback(run_dir, spec, lock, progress, stage, stage_index)
            if not isinstance(model, BoundaryCheckpointPPO):
                raise ReproducibilityError("canonical trainer loaded a PPO class without boundary checkpoint support")
            model.boundary_hook = lambda _model: callback.save_checkpoint()
            model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=callback, progress_bar=False)
            model.boundary_hook = None
            callback.save_checkpoint(force=True)
        stage_model = run_dir / "checkpoints" / f"stage_{stage_index}_{stage['task']}.zip"
        stage_identity, stage_runtime_identity = save_resumable_checkpoint(model, stage_model, run_dir)
        atomic_json(stage_model.with_suffix(".metadata.json"), {
            **lock, "stage_index": stage_index, "stage": stage,
            "total_timesteps": model.num_timesteps, "model_identity": stage_identity,
            "runtime_state_identity": stage_runtime_identity,
            "safe_boundary": "after-complete-rollout-and-optimizer-update",
        })
        progress.setdefault("completed_stages", []).append(stage["task"])
        progress.update(
            stage_index=stage_index + 1, stage_timesteps_completed=0,
            total_timesteps=model.num_timesteps, latest_checkpoint=stage_identity["path"],
            latest_checkpoint_sha256=stage_identity["sha256"],
            latest_runtime_state=stage_runtime_identity["path"],
            latest_runtime_state_sha256=stage_runtime_identity["sha256"], status="training",
        )
        atomic_json(run_dir / spec["outputs"]["progress"], progress)
        append_event(events, "stage_completed", lock, stage["task"], total_timesteps=model.num_timesteps)
        env.close()

    if model is None:
        if not latest or checkpoint_metadata is None:
            raise ReproducibilityError("training progress has no resumable model")
        model = load_resumable_checkpoint(
            resolve_run_path(run_dir, latest), checkpoint_metadata, run_dir,
            spec["algorithm"]["device"],
        )
    final_path = run_dir / spec["outputs"]["final_model"]
    save_model_atomic(model, final_path)
    final_identity = file_identity(final_path, relative_to=run_dir)
    atomic_json(run_dir / spec["outputs"]["final_model_metadata"], {**lock, "status": "frozen", "total_timesteps": model.num_timesteps, "completed_stages": progress["completed_stages"], "model_identity": final_identity})
    progress.update(status="trained", final_model=final_identity["path"])
    atomic_json(run_dir / spec["outputs"]["progress"], progress)
    append_event(events, "final_model_created", lock, "final", model=final_identity["path"], model_sha256=final_identity["sha256"], total_timesteps=model.num_timesteps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="training_specs/canonical_v1.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        report = perform_preflight(args.spec, run_tests=True, enforce_lock=True)
        if args.preflight_only:
            print(json.dumps(report, indent=2))
            return
        spec, spec_path = load_spec(args.spec)
        run_dir = ensure_external_run_dir(args.run_dir, float(spec["runtime"]["minimum_free_disk_gib"]))
        lock = ensure_run_lock(run_dir, spec, report["source_commit"])
        report_path = persist_preflight_report(run_dir, spec, lock, report)
        capture_provenance_files(run_dir, spec, spec_path, ["python", "train_canonical.py", "--spec", spec_path.relative_to(ROOT).as_posix(), "--run-dir", "<RUN_DIR>"], lock)
        events = run_dir / spec["outputs"]["events"]
        if not events.exists():
            append_event(events, "run_created", lock, "preflight")
        report_identity = file_identity(report_path, relative_to=run_dir)
        append_event(events, "preflight_passed", lock, "preflight", report=report_identity["path"], report_sha256=report_identity["sha256"])
        train(spec, spec_path, run_dir, lock)
        from tools.run_evaluation import run_evaluation
        run_evaluation(run_dir, "validation", spec_path)
        subprocess.run([
            sys.executable, "tools/plot_training_diagnostics.py", str(run_dir),
            "--spec", str(spec_path),
        ], cwd=ROOT, check=True)
        print(f"TRAINING COMPLETE: frozen model is {run_dir / spec['outputs']['final_model']}")
        print(f"Next: python tools/run_evaluation.py {run_dir} --split all")
    except (ReproducibilityError, FloatingPointError, subprocess.CalledProcessError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
