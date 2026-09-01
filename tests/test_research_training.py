"""Regression tests for the research-grade training and evaluation contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from environment import INVALID_ACTION_PENALTY, TIME_PRESSURE_PENALTY
from gym_wrapper import MAX_NODES, NODE_FEATURES, PatchCascadeGymEnv, VULN_FEATURES
from training_repro import (
    ReproducibilityError,
    build_lock,
    canonical_json,
    ensure_external_run_dir,
    file_identity,
    load_spec,
    run_fingerprint,
    resolve_run_path,
    scrub,
    validate_resume,
)
from tools.verify_training_artifacts import verify_run


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "training_specs" / "canonical_v1.json"


def test_explicit_reset_is_reproducible_but_constructor_seed_is_a_sequence():
    left = PatchCascadeGymEnv(task_level="hard", seed=777)
    right = PatchCascadeGymEnv(task_level="hard", seed=777)
    left_sequence = [left.reset()[1]["episode_seed"] for _ in range(4)]
    right_sequence = [right.reset()[1]["episode_seed"] for _ in range(4)]
    assert left_sequence == right_sequence
    assert len(set(left_sequence)) > 1
    obs_a, info_a = left.reset(seed=1234)
    obs_b, info_b = left.reset(seed=1234)
    assert info_a["episode_seed"] == info_b["episode_seed"]
    assert np.array_equal(obs_a, obs_b)


def test_observation_contains_cve_to_host_incidence():
    env = PatchCascadeGymEnv(task_level="easy", seed=9)
    vector, _ = env.reset(seed=9)
    rich = env._obs
    assert rich is not None and rich.vulnerabilities
    matrix_offset = MAX_NODES * NODE_FEATURES + 8 * VULN_FEATURES
    vuln = rich.vulnerabilities[0]
    expected = {env._hostname_to_idx[name] for name in vuln.affected_hosts}
    encoded = {
        node_index
        for node_index in range(MAX_NODES)
        if vector[matrix_offset + node_index] == pytest.approx(1.0)
    }
    assert encoded == expected


def test_padded_action_is_penalized_and_never_silently_repaired():
    env = PatchCascadeGymEnv(task_level="easy", seed=3, reward_scale=1.0)
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.array([2, MAX_NODES - 1, 7]))
    assert info["action_valid"] is False
    assert info["action_target"].startswith("__invalid_node_")
    assert info["reward_components"]["base"] == pytest.approx(
        INVALID_ACTION_PENALTY + TIME_PRESSURE_PENALTY
    )


def test_time_limit_is_truncated_not_terminated_and_reward_history_is_final():
    env = PatchCascadeGymEnv(task_level="easy", seed=4, reward_scale=1.0)
    env.reset(seed=4)
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step(np.array([4, 0, 0]))
        assert not (terminated and truncated)
        if terminated or truncated:
            assert truncated is True
            assert terminated is False
            assert env.unwrapped_env.state.reward_history[-1] == pytest.approx(reward)
            assert reward == pytest.approx(
                info["reward_components"]["base"] + info["reward_components"]["potential_shaping"]
            )
            break
    else:
        pytest.fail("episode did not reach its declared limit")


def test_dynamic_event_turn_is_identical_for_valid_and_invalid_actions():
    valid = PatchCascadeGymEnv(task_level="zero_day", seed=42)
    invalid = PatchCascadeGymEnv(task_level="zero_day", seed=42)
    valid.reset(seed=42)
    invalid.reset(seed=42)
    counts = []
    for step in range(5):
        valid.step(np.array([4, 0, 0]))
        invalid.step(np.array([0, MAX_NODES - 1, 0]))
        counts.append((len(valid._obs.vulnerabilities), len(invalid._obs.vulnerabilities)))
        if step < 4:
            assert "zero_day_turn_5" not in valid.unwrapped_env._dynamic_events_fired
            assert "zero_day_turn_5" not in invalid.unwrapped_env._dynamic_events_fired
    assert "zero_day_turn_5" in valid.unwrapped_env._dynamic_events_fired
    assert "zero_day_turn_5" in invalid.unwrapped_env._dynamic_events_fired
    assert counts[-1][0] == counts[-1][1]


def test_fingerprint_is_stable_sensitive_and_resume_fails_closed():
    spec, _ = load_spec(SPEC_PATH)
    commit = "a" * 40
    assert run_fingerprint(spec, commit) == run_fingerprint(copy.deepcopy(spec), commit)
    changed = copy.deepcopy(spec)
    changed["seeds"]["global_training_seed"] += 1
    assert run_fingerprint(changed, commit) != run_fingerprint(spec, commit)
    expected = build_lock(spec, commit)
    incompatible = copy.deepcopy(expected)
    incompatible["run_fingerprint"] = "b" * 64
    with pytest.raises(ReproducibilityError, match="Incompatible resume"):
        validate_resume(incompatible, expected)


def test_provenance_scrubber_redacts_secret_fields_recursively():
    payload = {"TOKEN": "visible", "nested": {"api_key": "visible", "safe": "kept"}}
    assert scrub(payload) == {"TOKEN": "<redacted>", "nested": {"api_key": "<redacted>", "safe": "kept"}}


def test_run_directory_and_metadata_paths_fail_closed():
    with pytest.raises(ReproducibilityError, match="outside the source checkout"):
        ensure_external_run_dir(ROOT / "not-created-canonical-run")
    with pytest.raises(ReproducibilityError, match="outside the run directory"):
        resolve_run_path(ROOT.parent, "../foreign-checkpoint.zip")


def test_optimizer_shape_probe_updates_finite_parameters():
    from tools.training_preflight import optimizer_shape_probe

    spec, _ = load_spec(SPEC_PATH)
    tiny = copy.deepcopy(spec)
    tiny["algorithm"].update({
        "architecture": {"policy": [16], "value": [16]},
        "rollout_steps": 8, "batch_size": 8, "epochs_per_update": 1,
        "parallel_environments": 2,
    })
    result = optimizer_shape_probe(tiny)
    assert result["timesteps"] == 16
    assert result["changed_parameter_tensors"] > 0


def test_tiny_ppo_save_load_roundtrip(tmp_path):
    from stable_baselines3 import PPO

    env = PatchCascadeGymEnv(task_level="easy", seed=11)
    model = PPO(
        "MlpPolicy", env, seed=11, n_steps=8, batch_size=8, n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}}, verbose=0,
    )
    model.learn(total_timesteps=16)
    path = tmp_path / "tiny_model.zip"
    model.save(path)
    loaded = PPO.load(path)
    assert loaded.observation_space == env.observation_space
    assert loaded.action_space == env.action_space


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_benchmark(spec: dict, lock: dict, split: str, model_identity: dict) -> dict:
    seed_key = {"validation": "validation", "canonical": "canonical_test", "confirmation": "confirmation_test"}[split]
    seeds = spec["seeds"][seed_key]
    tasks = spec["environment"]["task_levels"]
    raw = []
    summaries = []
    for task in tasks:
        for agent in spec["evaluation"]["agents"]:
            rows = []
            for episode_index, seed in enumerate(seeds, start=1):
                row = {
                    "episode_id": f"{agent}:{task}:{seed}", "episode_index": episode_index,
                    "agent": agent, "task_level": task, "seed": seed,
                    "score": 0.5, "total_reward": 1.0, "steps": 1,
                    "terminated": True, "environment_truncated": False,
                    "externally_truncated": False, "success": True,
                    "dimensions": {"completion": 0.5, "efficiency": 0.5, "safety": 0.5, "strategy": 0.5},
                    "cascade_failures": 0, "invalid_actions": 0,
                    "catastrophic_failure": False,
                }
                rows.append(row)
                raw.append(row)
            summaries.append({
                "agent_name": agent, "task_level": task, "episodes": len(rows),
                "mean_score": 0.5, "mean_reward": 1.0, "success_rate": 1.0,
                "completion": 0.5, "efficiency": 0.5, "safety": 0.5, "strategy": 0.5,
                "score_std": 0.0, "score_median": 0.5,
                "score_ci95_low": 0.5, "score_ci95_high": 0.5,
                "reward_std": 0.0, "catastrophic_failures": 0,
            })
    gates = [{
        "task": task, "baseline": baseline, "available": True,
        "paired_episodes": len(seeds), "mean_score_delta": 0.0,
        "delta_ci95": [0.0, 0.0], "evidence_exceeds_baseline": False,
        "regression_flag": False,
    } for task in tasks for baseline in ("random", "heuristic")]
    return {
        "schema_version": 1,
        "status": "complete",
        "config": {
            "split": split, "seeds": seeds, "tasks": tasks,
            "source_commit": lock["source_commit"],
            "grader_source_commit": lock["source_commit"],
            "run_fingerprint": lock["run_fingerprint"],
            "spec_path": "training_specs/canonical_v1.json",
            "max_steps_by_task": spec["evaluation"]["max_steps_by_task"],
            "bootstrap_samples": spec["evaluation"]["bootstrap_samples"],
            "deterministic_policy": spec["evaluation"]["deterministic_policy"],
            "spec_sha256": lock["spec_sha256"],
            "environment_schema_version": spec["environment"]["schema_version"],
            "reward_schema_version": spec["environment"]["reward_schema_version"],
            "model_identity": model_identity,
        },
        "summaries": summaries, "raw_episodes": raw, "baseline_gates": gates,
    }


def _artifact_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    spec, _ = load_spec(SPEC_PATH)
    lock = build_lock(spec, "c" * 40)
    outputs = spec["outputs"]
    stages = [stage["task"] for stage in spec["methodology"]["stages"]]
    total_timesteps = sum(stage["timesteps"] for stage in spec["methodology"]["stages"])
    _write_json(tmp_path / outputs["run_lock"], lock)
    _write_json(tmp_path / outputs["preflight_report"], {
        "passed": True, "source_commit": lock["source_commit"], "spec_sha256": lock["spec_sha256"],
        "run_fingerprint": lock["run_fingerprint"], "dependency_mismatches": {},
    })
    preflight_identity = file_identity(tmp_path / outputs["preflight_report"], relative_to=tmp_path)
    freeze = ["fixture==1"]
    _write_json(tmp_path / outputs["provenance"], {
        "run_fingerprint": lock["run_fingerprint"], "spec_sha256": lock["spec_sha256"],
        "git": {"dirty": False, "commit": lock["source_commit"]},
        "runtime": {
            "python_major_minor": "3.11", "packages": spec["dependencies"],
            "package_freeze": freeze,
            "package_freeze_sha256": hashlib.sha256("\n".join(freeze).encode()).hexdigest(),
        },
    })
    (tmp_path / outputs["provenance_markdown"]).write_text("provenance", encoding="utf-8")
    _write_json(tmp_path / outputs["progress"], {**lock, "status": "trained", "completed_stages": stages, "total_timesteps": total_timesteps})
    (tmp_path / outputs["final_model"]).write_bytes(b"fixture model")
    model_identity = file_identity(tmp_path / outputs["final_model"], relative_to=tmp_path)
    _write_json(tmp_path / outputs["final_model_metadata"], {**lock, "status": "frozen", "completed_stages": stages, "total_timesteps": total_timesteps, "model_identity": model_identity})
    event_lines = [
        {"event": "run_created", "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"]},
        {"event": "preflight_passed", "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"], "report": preflight_identity["path"], "report_sha256": preflight_identity["sha256"]},
        *({"event": "stage_completed", "stage": stage, "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"]} for stage in stages),
        {"event": "final_model_created", "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"], "model_sha256": model_identity["sha256"]},
    ]
    (tmp_path / outputs["events"]).write_text("\n".join(canonical_json(row) for row in event_lines) + "\n", encoding="utf-8")
    diagnostic_rows = [{
        "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"],
        "timesteps": (index + 1) * 20480, "stage": stage, "stage_index": index,
        "metrics": {"train/loss": 0.25},
    } for index, stage in enumerate(stages)]
    (tmp_path / "training_diagnostics.jsonl").write_text(
        "\n".join(canonical_json(row) for row in diagnostic_rows) + "\n", encoding="utf-8"
    )
    plot = tmp_path / outputs["training_plots_dir"] / "diagnostics.png"
    plot.parent.mkdir(parents=True, exist_ok=True)
    plot.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    from benchmark import BenchmarkResult, write_outputs
    for split, key in (("validation", "validation_dir"), ("canonical", "canonical_dir"), ("confirmation", "confirmation_dir")):
        payload = _complete_benchmark(spec, lock, split, model_identity)
        results = []
        for summary in payload["summaries"]:
            rows = [row for row in payload["raw_episodes"] if row["agent"] == summary["agent_name"] and row["task_level"] == summary["task_level"]]
            results.append(BenchmarkResult(**summary, raw_episodes=rows))
        evaluation_dir = tmp_path / outputs[key]
        write_outputs(evaluation_dir, payload, results)
        benchmark_identity = file_identity(evaluation_dir / "benchmark.json", relative_to=tmp_path)
        attempt_id = f"fixture-{split}"
        _write_json(evaluation_dir / outputs["evaluation_marker"], {
            "schema_version": 1, "status": "complete", "split": split,
            "attempt_id": attempt_id, "run_fingerprint": lock["run_fingerprint"],
            "model_identity": model_identity, "benchmark_identity": benchmark_identity,
        })
        event_lines.extend([
            {"event": "evaluation_started", "stage": split, "split": split, "attempt_id": attempt_id, "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"]},
            {"event": "evaluation_completed", "stage": split, "split": split, "attempt_id": attempt_id, "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"], "model_sha256": model_identity["sha256"], "benchmark_sha256": benchmark_identity["sha256"]},
        ])
    (tmp_path / outputs["events"]).write_text("\n".join(canonical_json(row) for row in event_lines) + "\n", encoding="utf-8")
    return tmp_path, spec, lock


def test_artifact_verifier_accepts_complete_identity_matched_fixture(tmp_path):
    root, _, _ = _artifact_fixture(tmp_path)
    result = verify_run(root, SPEC_PATH, load_model=False, enforce_source=False, enforce_runtime=False)
    assert result["valid"] is True
    assert result["policy_accepted"] is False
    assert result["scientific_outcome"] == "rejected_policy_evidence"
    assert (root / "artifact_manifest.json").is_file()
    assert (root / "SHA256SUMS.txt").is_file()


def test_artifact_verifier_rejects_partial_evaluation(tmp_path):
    root, spec, _ = _artifact_fixture(tmp_path)
    path = root / spec["outputs"]["canonical_dir"] / "benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["raw_episodes"].pop()
    _write_json(path, payload)
    with pytest.raises(ReproducibilityError, match="incomplete"):
        verify_run(root, SPEC_PATH, load_model=False, enforce_source=False, enforce_runtime=False)


def test_artifact_verifier_rejects_wrong_run_identity(tmp_path):
    root, spec, _ = _artifact_fixture(tmp_path)
    path = root / spec["outputs"]["confirmation_dir"] / "benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["run_fingerprint"] = "wrong"
    _write_json(path, payload)
    with pytest.raises(ReproducibilityError, match="another experiment"):
        verify_run(root, SPEC_PATH, load_model=False, enforce_source=False, enforce_runtime=False)


def test_artifact_verifier_rejects_model_byte_tampering(tmp_path):
    root, spec, _ = _artifact_fixture(tmp_path)
    (root / spec["outputs"]["final_model"]).write_bytes(b"different model bytes")
    with pytest.raises(ReproducibilityError, match="artifact identity mismatch"):
        verify_run(root, SPEC_PATH, load_model=False, enforce_source=False, enforce_runtime=False)


def test_artifact_verifier_recomputes_all_summary_metrics(tmp_path):
    root, spec, _ = _artifact_fixture(tmp_path)
    path = root / spec["outputs"]["canonical_dir"] / "benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summaries"][0]["score_ci95_low"] = 0.49
    _write_json(path, payload)
    with pytest.raises(ReproducibilityError, match="summary mismatch"):
        verify_run(root, SPEC_PATH, load_model=False, enforce_source=False, enforce_runtime=False)


def test_artifact_verifier_rejects_edited_derived_report(tmp_path):
    root, spec, _ = _artifact_fixture(tmp_path)
    report = root / spec["outputs"]["confirmation_dir"] / "SUMMARY.md"
    report.write_text("misleading summary", encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="derived report"):
        verify_run(root, SPEC_PATH, load_model=False, enforce_source=False, enforce_runtime=False)


def test_artifact_verifier_rejects_duplicate_evaluation_completion(tmp_path):
    root, spec, lock = _artifact_fixture(tmp_path)
    events = root / spec["outputs"]["events"]
    with events.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json({
            "event": "evaluation_completed", "stage": "canonical", "split": "canonical",
            "attempt_id": "fixture-canonical", "run_fingerprint": lock["run_fingerprint"],
            "source_commit": lock["source_commit"],
        }) + "\n")
    with pytest.raises(ReproducibilityError, match="completion event count"):
        verify_run(root, SPEC_PATH, load_model=False, enforce_source=False, enforce_runtime=False)


def test_artifact_verifier_rejects_failed_preflight_evidence(tmp_path):
    root, spec, _ = _artifact_fixture(tmp_path)
    path = root / spec["outputs"]["preflight_report"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["passed"] = False
    _write_json(path, payload)
    with pytest.raises(ReproducibilityError, match="preflight report"):
        verify_run(root, SPEC_PATH, load_model=False, enforce_source=False, enforce_runtime=False)
