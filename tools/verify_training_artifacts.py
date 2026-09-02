#!/usr/bin/env python3
"""Verify a complete PatchCascade run and regenerate its immutable manifest."""

from __future__ import annotations

import argparse
import math
import json
import random
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_repro import (
    ReproducibilityError, atomic_json, build_lock, canonical_json,
    declared_dependency_mismatches, ensure_external_run_dir, file_identity,
    git_metadata, load_spec, runtime_info, sha256_file, spec_hash,
    resolve_run_path, validate_file_identity, validate_resume,
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"corrupt or missing JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReproducibilityError(f"expected JSON object: {path}")
    return value


def _bootstrap_mean_ci(values: list[float], seed: int, samples: int) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def _same_number(actual: Any, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)) or abs(float(actual) - expected) > 1e-12:
        raise ReproducibilityError(f"benchmark summary mismatch: {label}")


def _recomputed_summary(agent: str, task: str, rows: list[dict], spec: dict) -> dict[str, Any]:
    scores = [float(row["score"]) for row in rows]
    rewards = [float(row["total_reward"]) for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    low, high = _bootstrap_mean_ci(scores, sum(seeds) + len(task), int(spec["evaluation"]["bootstrap_samples"]))
    return {
        "agent_name": agent, "task_level": task,
        "mean_score": statistics.mean(scores), "mean_reward": statistics.mean(rewards),
        "success_rate": statistics.mean(float(row["success"]) for row in rows),
        "completion": statistics.mean(float(row["dimensions"]["completion"]) for row in rows),
        "efficiency": statistics.mean(float(row["dimensions"]["efficiency"]) for row in rows),
        "safety": statistics.mean(float(row["dimensions"]["safety"]) for row in rows),
        "strategy": statistics.mean(float(row["dimensions"]["strategy"]) for row in rows),
        "score_std": statistics.pstdev(scores), "score_median": statistics.median(scores),
        "score_ci95_low": low, "score_ci95_high": high,
        "reward_std": statistics.pstdev(rewards), "episodes": len(rows),
        "catastrophic_failures": sum(bool(row["catastrophic_failure"]) for row in rows),
    }


def _recomputed_gate(raw: list[dict], task: str, baseline: str, seeds: list[int], samples: int) -> dict[str, Any]:
    ppo = {row["seed"]: row["score"] for row in raw if row["agent"] == "ppo" and row["task_level"] == task}
    base = {row["seed"]: row["score"] for row in raw if row["agent"] == baseline and row["task_level"] == task}
    deltas = [float(ppo[seed]) - float(base[seed]) for seed in seeds]
    low, high = _bootstrap_mean_ci(deltas, sum(seeds) + len(baseline), samples)
    mean_delta = statistics.mean(deltas)
    return {
        "task": task, "baseline": baseline, "available": True,
        "paired_episodes": len(seeds), "mean_score_delta": mean_delta,
        "delta_ci95": [low, high], "evidence_exceeds_baseline": low > 0,
        "regression_flag": mean_delta < 0,
    }


def _compare_gate(actual: dict, expected: dict, split: str) -> None:
    for field in ("task", "baseline", "available", "paired_episodes", "evidence_exceeds_baseline", "regression_flag"):
        if actual.get(field) != expected[field]:
            raise ReproducibilityError(f"{split} baseline gate mismatch: {expected['task']} vs {expected['baseline']} ({field})")
    _same_number(actual.get("mean_score_delta"), expected["mean_score_delta"], "baseline mean delta")
    interval = actual.get("delta_ci95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ReproducibilityError(f"{split} baseline gate confidence interval is invalid")
    _same_number(interval[0], expected["delta_ci95"][0], "baseline CI low")
    _same_number(interval[1], expected["delta_ci95"][1], "baseline CI high")


def verify_benchmark(
    path: Path, split: str, spec: dict, lock: dict,
    model_identity: dict[str, Any] | None = None, *, require_rendered: bool = True,
) -> dict[str, Any]:
    payload = read_json(path / "benchmark.json")
    config = payload.get("config", {})
    seed_key = {"validation": "validation", "canonical": "canonical_test", "confirmation": "confirmation_test"}[split]
    expected_seeds = spec["seeds"][seed_key]
    expected_tasks = spec["environment"]["task_levels"]
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ReproducibilityError(f"{split} benchmark is not marked complete")
    if config.get("split") != split or config.get("seeds") != expected_seeds or config.get("tasks") != expected_tasks:
        raise ReproducibilityError(f"{split} benchmark seed/task configuration mismatch")
    if config.get("source_commit") != lock["source_commit"] or config.get("grader_source_commit") != lock["source_commit"]:
        raise ReproducibilityError(f"{split} benchmark used different source/grader code")
    if config.get("run_fingerprint") != lock["run_fingerprint"]:
        raise ReproducibilityError(f"{split} benchmark belongs to another experiment")
    expected_config = {
        "max_steps_by_task": spec["evaluation"]["max_steps_by_task"],
        "bootstrap_samples": spec["evaluation"]["bootstrap_samples"],
        "deterministic_policy": spec["evaluation"]["deterministic_policy"],
        "spec_sha256": lock["spec_sha256"],
        "environment_schema_version": spec["environment"]["schema_version"],
        "reward_schema_version": spec["environment"]["reward_schema_version"],
        "model_identity": model_identity,
    }
    for field, wanted in expected_config.items():
        if config.get(field) != wanted:
            raise ReproducibilityError(f"{split} benchmark configuration mismatch: {field}")
    raw = payload.get("raw_episodes", [])
    agents = spec["evaluation"]["agents"]
    expected = {(agent, task, seed) for agent in agents for task in expected_tasks for seed in expected_seeds}
    actual = [(row.get("agent"), row.get("task_level"), row.get("seed")) for row in raw]
    if len(actual) != len(set(actual)):
        raise ReproducibilityError(f"{split} benchmark has duplicate episode IDs")
    missing = expected - set(actual)
    extra = set(actual) - expected
    if missing or extra:
        raise ReproducibilityError(
            f"{split} evaluation is incomplete ({len(actual)}/{len(expected)} episodes); "
            f"missing={len(missing)}, unexpected={len(extra)}"
        )
    seed_position = {seed: index + 1 for index, seed in enumerate(expected_seeds)}
    dimensions = {"completion", "efficiency", "safety", "strategy"}
    for row in raw:
        agent, task, seed = row["agent"], row["task_level"], row["seed"]
        if row.get("episode_id") != f"{agent}:{task}:{seed}" or row.get("episode_index") != seed_position[seed]:
            raise ReproducibilityError(f"{split} raw episode identity/index mismatch")
        if not isinstance(row.get("success"), bool) or not isinstance(row.get("catastrophic_failure"), bool):
            raise ReproducibilityError(f"{split} raw episode boolean fields are invalid")
        if set(row.get("dimensions", {})) != dimensions:
            raise ReproducibilityError(f"{split} raw episode grader dimensions are incomplete")
        numeric = [row.get(name) for name in ("score", "total_reward", "steps", "cascade_failures", "invalid_actions")]
        numeric.extend(row["dimensions"].values())
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in numeric):
            raise ReproducibilityError(f"{split} raw episode contains non-finite/non-numeric evidence")
        if not 0 <= int(row["steps"]) <= int(spec["evaluation"]["max_steps_by_task"][task]):
            raise ReproducibilityError(f"{split} raw episode step count exceeds the frozen limit")
        endings = [bool(row.get("terminated")), bool(row.get("environment_truncated")), bool(row.get("externally_truncated"))]
        if sum(endings) > 1 or (row.get("externally_truncated") and row["steps"] != spec["evaluation"]["max_steps_by_task"][task]):
            raise ReproducibilityError(f"{split} raw episode termination evidence is inconsistent")
    summaries = payload.get("summaries", [])
    if len(summaries) != len(agents) * len(expected_tasks):
        raise ReproducibilityError(f"{split} benchmark summary matrix is incomplete")
    summary_ids = [(summary.get("agent_name"), summary.get("task_level")) for summary in summaries]
    expected_summary_ids = [(agent, task) for task in expected_tasks for agent in agents]
    if summary_ids != expected_summary_ids:
        raise ReproducibilityError(f"{split} benchmark summary order/identity matrix is invalid")
    for summary in summaries:
        rows = [row for row in raw if row["agent"] == summary["agent_name"] and row["task_level"] == summary["task_level"]]
        recomputed = _recomputed_summary(summary["agent_name"], summary["task_level"], rows, spec)
        for field, wanted in recomputed.items():
            if field in {"agent_name", "task_level", "episodes", "catastrophic_failures"}:
                if summary.get(field) != wanted:
                    raise ReproducibilityError(f"{split} summary mismatch: {field}")
            else:
                _same_number(summary.get(field), wanted, f"{split}/{summary['agent_name']}/{summary['task_level']}/{field}")
    expected_gates = [
        _recomputed_gate(raw, task, baseline, expected_seeds, int(spec["evaluation"]["bootstrap_samples"]))
        for task in expected_tasks for baseline in ("random", "heuristic")
    ]
    gates = payload.get("baseline_gates", [])
    if len(gates) != len(expected_gates):
        raise ReproducibilityError(f"{split} baseline gate matrix is incomplete")
    for actual_gate, expected_gate in zip(gates, expected_gates):
        _compare_gate(actual_gate, expected_gate, split)
    if require_rendered:
        try:
            from benchmark import BenchmarkResult, write_outputs
            results = []
            for summary in summaries:
                rows = [row for row in raw if row["agent"] == summary["agent_name"] and row["task_level"] == summary["task_level"]]
                results.append(BenchmarkResult(**summary, raw_episodes=rows))
            with tempfile.TemporaryDirectory() as temporary:
                rendered = Path(temporary) / "rendered"
                write_outputs(rendered, payload, results)
                for name in ("summary.csv", "SUMMARY.md"):
                    if (path / name).read_bytes() != (rendered / name).read_bytes():
                        raise ReproducibilityError(f"{split} derived report was edited or does not match raw evidence: {name}")
        except OSError as exc:
            raise ReproducibilityError(f"{split} derived evaluation report is missing") from exc
    return payload


def verify_diagnostics(root: Path, spec: dict, lock: dict) -> list[dict[str, Any]]:
    path = root / "training_diagnostics.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReproducibilityError("training diagnostics are missing") from exc
    if not lines:
        raise ReproducibilityError("training diagnostics are empty")
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReproducibilityError(f"training diagnostics are corrupt at line {line_number}") from exc
        if row.get("run_fingerprint") != lock["run_fingerprint"]:
            raise ReproducibilityError(f"training diagnostic at line {line_number} belongs to another experiment")
        if row.get("source_commit") != lock["source_commit"]:
            raise ReproducibilityError(f"training diagnostic at line {line_number} used different source")
        if not isinstance(row.get("metrics"), dict) or not row["metrics"]:
            raise ReproducibilityError(f"training diagnostic at line {line_number} has no metrics")
        for key, value in row["metrics"].items():
            if not isinstance(value, (int, float)) or not __import__("math").isfinite(value):
                raise ReproducibilityError(f"non-finite training diagnostic at line {line_number}: {key}")
        rows.append(row)
    if any(int(right.get("timesteps", -1)) < int(left.get("timesteps", -1)) for left, right in zip(rows, rows[1:])):
        raise ReproducibilityError("training diagnostic timesteps are not monotonic")
    plot = root / spec["outputs"]["training_plots_dir"] / "diagnostics.png"
    if not plot.is_file() or plot.stat().st_size == 0:
        raise ReproducibilityError("training diagnostics plot is missing")
    if plot.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ReproducibilityError("training diagnostics plot is not a valid PNG")
    expected_stages = {stage["task"] for stage in spec["methodology"]["stages"]}
    observed_stages = {row.get("stage") for row in rows}
    if not expected_stages.issubset(observed_stages):
        raise ReproducibilityError(f"training diagnostics omit stages: {sorted(expected_stages - observed_stages)}")
    return rows


def verify_run(
    run_dir: str | Path, spec_path: str | Path, *, require_evaluations: bool = True,
    load_model: bool = True, enforce_source: bool = True, enforce_runtime: bool = True,
) -> dict[str, Any]:
    spec, _ = load_spec(spec_path)
    root = ensure_external_run_dir(run_dir, float(spec["runtime"]["minimum_free_disk_gib"]))
    outputs = spec["outputs"]
    lock = read_json(root / outputs["run_lock"])
    expected_lock = build_lock(spec, lock.get("source_commit", ""))
    if canonical_json(lock) != canonical_json(expected_lock) or lock.get("spec_sha256") != spec_hash(spec):
        raise ReproducibilityError("run lock does not match the supplied canonical specification")
    if enforce_source:
        git = git_metadata()
        if git["dirty"] or git["commit"] != lock["source_commit"]:
            raise ReproducibilityError("source commit does not match run identity")
    if enforce_runtime:
        mismatches = declared_dependency_mismatches(spec, runtime_info(spec["dependencies"]))
        if mismatches:
            raise ReproducibilityError(f"dependency lock mismatch during verification: {canonical_json(mismatches)}")
    preflight = read_json(root / outputs["preflight_report"])
    if (
        preflight.get("passed") is not True
        or preflight.get("source_commit") != lock["source_commit"]
        or preflight.get("spec_sha256") != lock["spec_sha256"]
        or preflight.get("run_fingerprint") != lock["run_fingerprint"]
        or preflight.get("dependency_mismatches")
    ):
        raise ReproducibilityError("preflight report is missing, failed, or belongs to another run")
    provenance = read_json(root / outputs["provenance"])
    if provenance.get("run_fingerprint") != lock["run_fingerprint"] or provenance.get("git", {}).get("dirty"):
        raise ReproducibilityError("provenance is dirty or belongs to another run")
    if provenance.get("spec_sha256") != lock["spec_sha256"] or provenance.get("git", {}).get("commit") != lock["source_commit"]:
        raise ReproducibilityError("provenance source/spec identity mismatch")
    provenance_runtime = provenance.get("runtime", {})
    if declared_dependency_mismatches(spec, provenance_runtime):
        raise ReproducibilityError("recorded training runtime did not satisfy the dependency lock")
    freeze = provenance_runtime.get("package_freeze")
    if not isinstance(freeze, list) or not freeze:
        raise ReproducibilityError("full training package inventory is missing")
    expected_freeze_hash = __import__("hashlib").sha256("\n".join(freeze).encode()).hexdigest()
    if provenance_runtime.get("package_freeze_sha256") != expected_freeze_hash:
        raise ReproducibilityError("recorded package inventory hash is invalid")
    progress = read_json(root / outputs["progress"])
    validate_resume(progress, lock)
    expected_stages = [stage["task"] for stage in spec["methodology"]["stages"]]
    expected_timesteps = sum(stage["timesteps"] for stage in spec["methodology"]["stages"])
    if progress.get("completed_stages") != expected_stages or progress.get("status") != "trained":
        raise ReproducibilityError(f"training is incomplete: completed={progress.get('completed_stages')} expected={expected_stages}")
    if progress.get("total_timesteps") != expected_timesteps:
        raise ReproducibilityError(f"training timestep count mismatch: {progress.get('total_timesteps')}/{expected_timesteps}")
    model_path = root / outputs["final_model"]
    if not model_path.is_file():
        raise ReproducibilityError("final model archive is missing")
    model_metadata = read_json(root / outputs["final_model_metadata"])
    validate_resume(model_metadata, lock)
    if model_metadata.get("status") != "frozen" or model_metadata.get("completed_stages") != expected_stages:
        raise ReproducibilityError("final model metadata is incomplete")
    if model_metadata.get("total_timesteps") != expected_timesteps:
        raise ReproducibilityError("final model timestep count does not match the frozen methodology")
    model_identity = model_metadata.get("model_identity", {})
    validate_file_identity(model_path, model_identity, relative_to=root)
    expected_algorithm_class = {
        "PPO": "BoundaryCheckpointPPO",
        "MaskablePPO": "BoundaryCheckpointMaskablePPO",
    }.get(spec["algorithm"]["name"])
    if model_metadata.get("algorithm_class") != expected_algorithm_class:
        raise ReproducibilityError("final model algorithm class does not match the frozen specification")
    if load_model:
        try:
            from sb3_contrib import MaskablePPO
            from stable_baselines3 import PPO
            from gym_wrapper import (
                FLATTENED_ACTION_SCHEMA_VERSION,
                FlattenedMaskedPatchCascadeEnv,
                PatchCascadeGymEnv,
            )
            masked = spec["environment"]["action_schema_version"] == FLATTENED_ACTION_SCHEMA_VERSION
            model = (MaskablePPO if masked else PPO).load(model_path)
            env = (FlattenedMaskedPatchCascadeEnv if masked else PatchCascadeGymEnv)()
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ReproducibilityError("model observation/action spaces do not match the environment")
        except ReproducibilityError:
            raise
        except Exception as exc:
            raise ReproducibilityError(f"final model archive cannot be loaded: {type(exc).__name__}") from exc
    events_path = root / outputs["events"]
    events = []
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReproducibilityError("event log is missing") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReproducibilityError(f"event log is corrupt at line {line_number}") from exc
        if event.get("run_fingerprint") != lock["run_fingerprint"]:
            raise ReproducibilityError(f"event at line {line_number} belongs to another experiment")
        if event.get("source_commit") != lock["source_commit"]:
            raise ReproducibilityError(f"event at line {line_number} used different source")
        events.append(event)
    names = [event.get("event") for event in events]
    allowed_events = {
        "run_created", "preflight_passed", "training_resumed", "stage_started",
            "checkpoint_saved", "checkpoint_pruned", "warning", "stage_completed", "final_model_created",
        "evaluation_started", "evaluation_failed", "evaluation_interrupted_quarantined",
        "evaluation_completed",
    }
    unknown_events = sorted({str(name) for name in names if name not in allowed_events})
    if unknown_events:
        raise ReproducibilityError(f"unrecognized run lifecycle events: {unknown_events}")
    if names.count("run_created") != 1 or not names.count("preflight_passed") or names.count("final_model_created") != 1:
        raise ReproducibilityError("required run lifecycle event counts are invalid")
    for event in (row for row in events if row.get("event") == "preflight_passed"):
        report_path = resolve_run_path(root, str(event.get("report", "")))
        report_identity = file_identity(report_path, relative_to=root)
        if event.get("report_sha256") != report_identity["sha256"]:
            raise ReproducibilityError("preflight lifecycle event does not match its report bytes")
        recorded_report = read_json(report_path)
        if (
            recorded_report.get("passed") is not True
            or recorded_report.get("source_commit") != lock["source_commit"]
            or recorded_report.get("spec_sha256") != lock["spec_sha256"]
            or recorded_report.get("run_fingerprint") != lock["run_fingerprint"]
            or recorded_report.get("dependency_mismatches")
        ):
            raise ReproducibilityError("a recorded resume preflight is failed or incompatible")
    checkpoint_artifacts: list[Path] = []
    saved_checkpoints = {
        str(event.get("checkpoint")): (index, event)
        for index, event in enumerate(events)
        if event.get("event") == "checkpoint_saved"
    }
    pruned_checkpoints: dict[str, tuple[int, dict]] = {}
    for index, event in enumerate(events):
        if event.get("event") != "checkpoint_pruned":
            continue
        checkpoint = str(event.get("checkpoint", ""))
        saved = saved_checkpoints.get(checkpoint)
        if saved is None or checkpoint in pruned_checkpoints or index <= saved[0]:
            raise ReproducibilityError("checkpoint pruning lifecycle is inconsistent")
        saved_event = saved[1]
        if (
            event.get("reason") != "configured-retention-limit"
            or event.get("checkpoint_sha256") != saved_event.get("checkpoint_sha256")
            or event.get("runtime_state") != saved_event.get("runtime_state")
            or event.get("runtime_state_sha256") != saved_event.get("runtime_state_sha256")
            or event.get("runtime_state_format") != saved_event.get("runtime_state_format")
            or event.get("algorithm_class") != saved_event.get("algorithm_class")
        ):
            raise ReproducibilityError("checkpoint pruning event does not match the saved checkpoint")
        pruned_checkpoints[checkpoint] = (index, event)
    for event in (row for row in events if row.get("event") == "checkpoint_saved"):
        if event.get("safe_boundary") != "after-complete-rollout-and-optimizer-update":
            raise ReproducibilityError("checkpoint lifecycle event is not a safe PPO update boundary")
        if event.get("runtime_state_format") != "python-cloudpickle-trusted-run-only-v1":
            raise ReproducibilityError("checkpoint lifecycle event has an unknown runtime-state format")
        if event.get("algorithm_class") != expected_algorithm_class:
            raise ReproducibilityError("checkpoint lifecycle event has the wrong algorithm class")
        if str(event.get("checkpoint")) in pruned_checkpoints:
            continue
        checkpoint_path = resolve_run_path(root, str(event.get("checkpoint", "")))
        runtime_path = resolve_run_path(root, str(event.get("runtime_state", "")))
        metadata_path = checkpoint_path.with_suffix(".metadata.json")
        if not checkpoint_path.is_file() or not runtime_path.is_file() or not metadata_path.is_file():
            raise ReproducibilityError("checkpoint model/runtime/metadata triplet is incomplete")
        if sha256_file(checkpoint_path) != event.get("checkpoint_sha256"):
            raise ReproducibilityError("checkpoint lifecycle event does not match model bytes")
        if sha256_file(runtime_path) != event.get("runtime_state_sha256"):
            raise ReproducibilityError("checkpoint lifecycle event does not match runtime-state bytes")
        checkpoint_metadata = read_json(metadata_path)
        validate_resume(checkpoint_metadata, lock)
        if (
            checkpoint_metadata.get("safe_boundary") != event.get("safe_boundary")
            or checkpoint_metadata.get("total_timesteps") != event.get("total_timesteps")
            or checkpoint_metadata.get("runtime_state_format") != event.get("runtime_state_format")
            or checkpoint_metadata.get("algorithm_class") != event.get("algorithm_class")
        ):
            raise ReproducibilityError("checkpoint metadata/lifecycle boundary mismatch")
        validate_file_identity(
            checkpoint_path, checkpoint_metadata.get("model_identity", {}), relative_to=root
        )
        validate_file_identity(
            runtime_path, checkpoint_metadata.get("runtime_state_identity", {}), relative_to=root
        )
        # Runtime state is executable cloudpickle. Verification intentionally
        # hashes its opaque bytes and validates metadata without deserializing it.
        checkpoint_artifacts.extend([checkpoint_path, runtime_path, metadata_path])
    final_event = next(event for event in events if event.get("event") == "final_model_created")
    if final_event.get("model_sha256") != model_identity["sha256"]:
        raise ReproducibilityError("final-model lifecycle event does not match frozen bytes")
    attempt_starts = [event for event in events if event.get("event") == "evaluation_started"]
    attempt_ids = [event.get("attempt_id") for event in attempt_starts]
    if None in attempt_ids or len(attempt_ids) != len(set(attempt_ids)):
        raise ReproducibilityError("evaluation attempt IDs are missing or duplicated")
    terminal_attempt_ids = {
        event.get("attempt_id") for event in events
        if event.get("event") in {"evaluation_failed", "evaluation_interrupted_quarantined", "evaluation_completed"}
    }
    if not set(attempt_ids).issubset(terminal_attempt_ids):
        raise ReproducibilityError("evaluation event log contains an unaccounted interrupted attempt")
    if any(event.get("attempt_id") not in set(attempt_ids) for event in events if event.get("event") in {"evaluation_failed", "evaluation_interrupted_quarantined", "evaluation_completed"}):
        raise ReproducibilityError("evaluation terminal event has no matching start event")
    for stage in expected_stages:
        completions = [event for event in events if event.get("event") == "stage_completed" and event.get("stage") == stage]
        if len(completions) != 1:
            raise ReproducibilityError(f"stage completion event count is invalid: {stage}")
    verify_diagnostics(root, spec, lock)

    benchmark_payloads = {}
    critical_findings = []
    acceptance_failures = []
    policy_accepted: bool | None = None
    if require_evaluations:
        for split, key in (("validation", "validation_dir"), ("canonical", "canonical_dir"), ("confirmation", "confirmation_dir")):
            evaluation_dir = root / outputs[key]
            benchmark_payloads[split] = verify_benchmark(evaluation_dir, split, spec, lock, model_identity)
            marker = read_json(evaluation_dir / outputs["evaluation_marker"])
            benchmark_identity = file_identity(evaluation_dir / "benchmark.json", relative_to=root)
            if (
                marker.get("status") != "complete" or marker.get("split") != split
                or marker.get("run_fingerprint") != lock["run_fingerprint"]
                or marker.get("model_identity") != model_identity
                or marker.get("benchmark_identity") != benchmark_identity
            ):
                raise ReproducibilityError(f"{split} evaluation completion marker is invalid")
            completion_events = [event for event in events if event.get("event") == "evaluation_completed" and event.get("split") == split]
            if len(completion_events) != 1:
                raise ReproducibilityError(f"{split} evaluation completion event count is invalid")
            completion = completion_events[0]
            if completion.get("attempt_id") != marker.get("attempt_id") or completion.get("model_sha256") != model_identity["sha256"] or completion.get("benchmark_sha256") != benchmark_identity["sha256"]:
                raise ReproducibilityError(f"{split} evaluation lifecycle identity mismatch")
        for split in ("canonical", "confirmation"):
            for summary in benchmark_payloads[split]["summaries"]:
                if summary.get("catastrophic_failures", 0):
                    critical_findings.append({"split": split, "agent": summary["agent_name"], "task": summary["task_level"], "catastrophic_failures": summary["catastrophic_failures"]})
        for split in ("canonical", "confirmation"):
            for gate in benchmark_payloads[split].get("baseline_gates", []):
                if gate.get("regression_flag"):
                    critical_findings.append({"split": split, "type": "baseline_regression", **gate})
        acceptance = spec["evaluation"]["policy_acceptance"]
        for split in acceptance["held_out_splits"]:
            payload = benchmark_payloads[split]
            for task in spec["environment"]["task_levels"]:
                ppo_rows = [row for row in payload["raw_episodes"] if row["agent"] == "ppo" and row["task_level"] == task]
                catastrophic = sum(bool(row["catastrophic_failure"]) for row in ppo_rows)
                cascades = sum(int(row["cascade_failures"]) for row in ppo_rows)
                invalid = sum(int(row["invalid_actions"]) for row in ppo_rows)
                if acceptance["require_zero_ppo_catastrophic_failures"] and catastrophic:
                    acceptance_failures.append({"split": split, "task": task, "gate": "zero_ppo_catastrophic_failures", "observed": catastrophic})
                if acceptance["require_zero_ppo_cascade_failures"] and cascades:
                    acceptance_failures.append({"split": split, "task": task, "gate": "zero_ppo_cascade_failures", "observed": cascades})
                if acceptance["require_zero_ppo_invalid_actions"] and invalid:
                    acceptance_failures.append({"split": split, "task": task, "gate": "zero_ppo_invalid_actions", "observed": invalid})
                for baseline, setting in (
                    ("random", "require_paired_ci_above_random_per_task"),
                    ("heuristic", "require_paired_ci_above_heuristic_per_task"),
                ):
                    gate = next(item for item in payload["baseline_gates"] if item["task"] == task and item["baseline"] == baseline)
                    if acceptance[setting] and not gate["evidence_exceeds_baseline"]:
                        acceptance_failures.append({
                            "split": split, "task": task, "gate": f"paired_ci_above_{baseline}",
                            "mean_score_delta": gate["mean_score_delta"], "delta_ci95": gate["delta_ci95"],
                        })
        policy_accepted = not acceptance_failures

    candidates = [
        root / outputs["run_lock"], root / outputs["preflight_report"], root / outputs["provenance"],
        root / outputs["provenance_markdown"], root / outputs["events"],
        root / outputs["progress"], model_path, root / outputs["final_model_metadata"],
        root / "training_diagnostics.jsonl",
        root / outputs["training_plots_dir"] / "diagnostics.png",
        *checkpoint_artifacts,
    ]
    if require_evaluations:
        for key in ("validation_dir", "canonical_dir", "confirmation_dir"):
            candidates.extend(sorted(path for path in (root / outputs[key]).rglob("*") if path.is_file()))
    resume_reports = root / "resume_preflight_reports"
    if resume_reports.exists():
        candidates.extend(sorted(path for path in resume_reports.rglob("*") if path.is_file()))
    files = []
    for path in sorted(set(path for path in candidates if path.is_file())):
        files.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 1, "verified": True, "run_fingerprint": lock["run_fingerprint"],
        "spec_sha256": lock["spec_sha256"], "files": files,
        "critical_findings": critical_findings, "acceptance_failures": acceptance_failures,
        "policy_accepted": policy_accepted,
        "scientific_outcome": (
            "accepted_policy_evidence" if policy_accepted is True
            else "rejected_policy_evidence" if policy_accepted is False
            else "training_integrity_only"
        ),
        "interpretation": "Integrity verification means complete/auditable; policy acceptance additionally requires every frozen quality and safety gate.",
    }
    atomic_json(root / outputs["manifest"], manifest)
    checksum_files = [*files, {"path": outputs["manifest"], "size": (root / outputs["manifest"]).stat().st_size, "sha256": sha256_file(root / outputs["manifest"])}]
    (root / outputs["checksums"]).write_text("\n".join(f"{item['sha256']}  {item['path']}" for item in checksum_files) + "\n", encoding="utf-8")
    return {
        "valid": True, "run_fingerprint": lock["run_fingerprint"],
        "files_hashed": len(checksum_files), "critical_findings": critical_findings,
        "acceptance_failures": acceptance_failures, "policy_accepted": policy_accepted,
        "scientific_outcome": manifest["scientific_outcome"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--spec", default="training_specs/canonical_v1.json")
    args = parser.parse_args()
    try:
        result = verify_run(args.run_dir, args.spec)
    except ReproducibilityError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
    if result["policy_accepted"] is not True:
        print("POLICY ACCEPTANCE REJECTED: integrity evidence is retained, but one or more frozen quality/security gates failed.", file=sys.stderr)
        raise SystemExit(2)
    print("POLICY ACCEPTANCE PASS: integrity, held-out quality, and safety gates all passed.")


if __name__ == "__main__":
    main()
