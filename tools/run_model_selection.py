#!/usr/bin/env python3
"""Execute or fixture-test the locked, resume-aware validation-only campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_repro import (  # noqa: E402
    ReproducibilityError,
    atomic_json,
    canonical_json,
    ensure_external_run_dir,
    file_identity,
    git_metadata,
    load_spec,
    sha256_file,
    spec_hash,
)
from tools.validate_model_selection import validate as validate_protocol  # noqa: E402
from tools.verify_training_artifacts import read_json, verify_benchmark  # noqa: E402

PROTOCOL_PATH = ROOT / "training_specs" / "model_selection_v1.json"
BASELINE_PATH = ROOT / "training_specs" / "canonical_v1.json"


def protocol_payload() -> dict[str, Any]:
    validate_protocol()
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproducibilityError(f"existing campaign file is corrupt: {path.name}") from exc
        if canonical_json(actual) != canonical_json(payload):
            raise ReproducibilityError(f"existing campaign file changed identity: {path.name}")
        return
    atomic_json(path, payload)


def candidate_spec(
    baseline: dict[str, Any], protocol: dict[str, Any], candidate: dict[str, Any],
    round_: dict[str, Any], training_seed: int, interface: dict[str, Any],
    *, phase: str = "hyperparameter-selection",
    interface_decision_sha256: str | None = None,
) -> dict[str, Any]:
    spec = copy.deepcopy(baseline)
    spec["experiment_id"] = (
        f"{protocol['campaign_id']}-{phase}-r{round_['round']}-"
        f"{interface['id']}-{candidate['id']}-seed{training_seed}"
    )
    spec["status"] = "development-selection-candidate"
    spec["algorithm"]["learning_rate"] = candidate["learning_rate"]
    spec["algorithm"]["entropy_coefficient"] = candidate["entropy_coefficient"]
    spec["algorithm"]["architecture"] = {
        "policy": list(candidate["network"]), "value": list(candidate["network"]),
    }
    spec["algorithm"]["name"] = interface["algorithm"]
    spec["environment"]["action_schema_version"] = interface["action_schema_version"]
    spec["seeds"]["global_training_seed"] = training_seed
    for stage in spec["methodology"]["stages"]:
        stage["timesteps"] = round_["timesteps_per_stage"]
    spec["selection_identity"] = {
        "campaign_id": protocol["campaign_id"],
        "round": round_["round"],
        "candidate_id": candidate["id"],
        "training_seed": training_seed,
        "phase": phase,
        "interface_id": interface["id"],
        "algorithm": interface["algorithm"],
        "action_schema_version": interface["action_schema_version"],
        "interface_decision_sha256": interface_decision_sha256,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    return spec


def result_metrics(payload: dict[str, Any], candidate: dict[str, Any], round_: dict[str, Any]) -> dict[str, Any]:
    ppo_summaries = [row for row in payload["summaries"] if row["agent_name"] == "ppo"]
    ppo_episodes = [row for row in payload["raw_episodes"] if row["agent"] == "ppo"]
    heuristic_gates = [row for row in payload["baseline_gates"] if row["baseline"] == "heuristic"]
    if not ppo_summaries or not ppo_episodes or not heuristic_gates:
        raise ReproducibilityError("validation evidence lacks PPO summaries/episodes/heuristic gates")
    catastrophic = sum(int(bool(row["catastrophic_failure"])) for row in ppo_episodes)
    cascades = sum(int(row["cascade_failures"]) for row in ppo_episodes)
    invalid = sum(int(row["invalid_actions"]) for row in ppo_episodes)
    return {
        "candidate_id": candidate["id"],
        "eligible": catastrophic == 0 and cascades == 0 and invalid == 0,
        "catastrophic_failures": catastrophic,
        "cascade_failures": cascades,
        "invalid_actions": invalid,
        "worst_task_paired_bootstrap_lower_bound_vs_heuristic": min(
            float(row["delta_ci95"][0]) for row in heuristic_gates
        ),
        "worst_task_success_rate": min(float(row["success_rate"]) for row in ppo_summaries),
        "macro_validation_score": statistics.mean(float(row["mean_score"]) for row in ppo_summaries),
        "training_timesteps": int(round_["timesteps_per_stage"]) * len(payload["config"]["tasks"]),
        "network_size_proxy": sum(int(value) for value in candidate["network"]),
        "paired_validation_scores": [
            {
                "task_level": row["task_level"],
                "validation_seed": int(row["seed"]),
                "score": float(row["score"]),
            }
            for row in ppo_episodes
        ],
    }


def aggregate_candidate(
    candidate: dict[str, Any], records: list[dict[str, Any]], round_: dict[str, Any],
) -> dict[str, Any]:
    if len(records) != len(round_["training_seeds"]):
        raise ReproducibilityError(
            f"candidate {candidate['id']} has incomplete seed evidence "
            f"({len(records)}/{len(round_['training_seeds'])})"
        )
    metrics = [row["metrics"] for row in records]
    return {
        "candidate_id": candidate["id"],
        "eligible": all(row["eligible"] for row in metrics),
        "catastrophic_failures": sum(row["catastrophic_failures"] for row in metrics),
        "cascade_failures": sum(row["cascade_failures"] for row in metrics),
        "invalid_actions": sum(row["invalid_actions"] for row in metrics),
        "worst_task_paired_bootstrap_lower_bound_vs_heuristic": min(
            row["worst_task_paired_bootstrap_lower_bound_vs_heuristic"] for row in metrics
        ),
        "worst_task_success_rate": min(row["worst_task_success_rate"] for row in metrics),
        "macro_validation_score": statistics.mean(row["macro_validation_score"] for row in metrics),
        "training_timesteps": sum(row["training_timesteps"] for row in metrics),
        "network_size_proxy": sum(int(value) for value in candidate["network"]),
        "evidence": [row["evidence"] for row in records],
    }


def rank_candidates(rows: list[dict[str, Any]], advance: int) -> tuple[list[dict[str, Any]], list[str]]:
    ordered = sorted(rows, key=lambda row: (
        not row["eligible"],
        -row["worst_task_paired_bootstrap_lower_bound_vs_heuristic"],
        -row["worst_task_success_rate"],
        -row["macro_validation_score"],
        row["training_timesteps"],
        row["network_size_proxy"],
        row["candidate_id"],
    ))
    eligible = [row for row in ordered if row["eligible"]]
    if len(eligible) < advance:
        raise ReproducibilityError(
            f"only {len(eligible)} safety-eligible candidates remain; {advance} are required. "
            "STOP rather than advancing an unsafe candidate"
        )
    return ordered, [row["candidate_id"] for row in eligible[:advance]]


def _bootstrap_interval(values: list[float], *, seed: int, samples: int) -> list[float]:
    if not values:
        raise ReproducibilityError("paired interface comparison has no observations")
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def synthetic_interface_record(
    interface: dict[str, Any], protocol: dict[str, Any], training_seed: int,
) -> dict[str, Any]:
    tasks = tuple(protocol["action_interface_selection"]["task_levels"])
    is_masked = interface["id"] == "flattened-discrete-maskableppo"
    scores = [
        {
            "task_level": task,
            "validation_seed": validation_seed,
            "score": 0.60 + task_index / 100 + (0.05 if is_masked else 0.0),
        }
        for task_index, task in enumerate(tasks)
        for validation_seed in protocol["action_interface_selection"]["validation_seeds"]
    ]
    return {
        "interface_id": interface["id"],
        "training_seed": training_seed,
        "metrics": {
            "eligible": True,
            "catastrophic_failures": 0,
            "cascade_failures": 0,
            "invalid_actions": 0,
            "paired_validation_scores": scores,
        },
        "evidence": {
            "kind": "synthetic-fixture-not-scientific-evidence",
            "identity": hashlib.sha256(canonical_json({
                "phase": "interface-first", "interface": interface["id"], "seed": training_seed,
            }).encode()).hexdigest(),
        },
    }


def select_interface(
    protocol: dict[str, Any], records: list[dict[str, Any]],
) -> dict[str, Any]:
    stage = protocol["action_interface_selection"]
    interfaces = {row["id"]: row for row in stage["interfaces"]}
    expected_seeds = list(stage["training_seeds"])
    aggregates: dict[str, dict[str, Any]] = {}
    score_maps: dict[str, dict[tuple[int, str, int], float]] = {}
    for interface_id, interface in interfaces.items():
        rows = [row for row in records if row["interface_id"] == interface_id]
        if [row["training_seed"] for row in rows] != expected_seeds:
            raise ReproducibilityError(f"interface {interface_id} has incomplete training-seed evidence")
        score_map: dict[tuple[int, str, int], float] = {}
        for row in rows:
            for score in row["metrics"]["paired_validation_scores"]:
                key = (row["training_seed"], score["task_level"], score["validation_seed"])
                if key in score_map:
                    raise ReproducibilityError(f"interface {interface_id} has duplicate paired evidence")
                score_map[key] = float(score["score"])
        expected_keys = {
            (seed, task, validation_seed)
            for seed in expected_seeds
            for task in stage["task_levels"]
            for validation_seed in stage["validation_seeds"]
        }
        if set(score_map) != expected_keys:
            raise ReproducibilityError(
                f"interface {interface_id} paired validation is incomplete "
                f"({len(score_map)}/{len(expected_keys)} observations)"
            )
        score_maps[interface_id] = score_map
        metrics = [row["metrics"] for row in rows]
        aggregates[interface_id] = {
            "interface_id": interface_id,
            "algorithm": interface["algorithm"],
            "action_schema_version": interface["action_schema_version"],
            "complexity_priority": interface["complexity_priority"],
            "eligible": all(row["eligible"] for row in metrics),
            "catastrophic_failures": sum(row["catastrophic_failures"] for row in metrics),
            "cascade_failures": sum(row["cascade_failures"] for row in metrics),
            "invalid_actions": sum(row["invalid_actions"] for row in metrics),
            "evidence": [row["evidence"] for row in rows],
        }
    baseline_id = "multidiscrete-ppo"
    masked_id = "flattened-discrete-maskableppo"
    if set(score_maps[baseline_id]) != set(score_maps[masked_id]):
        raise ReproducibilityError("interface comparison pair keys differ")
    paired = stage["paired_comparison"]
    task_intervals = {}
    for task_index, task in enumerate(stage["task_levels"]):
        keys = sorted(key for key in score_maps[baseline_id] if key[1] == task)
        differences = [score_maps[masked_id][key] - score_maps[baseline_id][key] for key in keys]
        interval = _bootstrap_interval(
            differences,
            seed=int(paired["bootstrap_seed"]) + task_index,
            samples=int(paired["bootstrap_samples"]),
        )
        task_intervals[task] = {
            "paired_observations": len(differences),
            "mean_score_difference": statistics.mean(differences),
            "bootstrap_ci95": interval,
        }
    baseline_eligible = aggregates[baseline_id]["eligible"]
    masked_eligible = aggregates[masked_id]["eligible"]
    if not baseline_eligible and not masked_eligible:
        raise ReproducibilityError("neither preregistered interface passed every safety gate")
    if masked_eligible and not baseline_eligible:
        selected = masked_id
        reason = "only-maskable-interface-safety-eligible"
    elif baseline_eligible and not masked_eligible:
        selected = baseline_id
        reason = "only-multidiscrete-interface-safety-eligible"
    elif all(row["bootstrap_ci95"][0] > 0 for row in task_intervals.values()):
        selected = masked_id
        reason = "maskable-positive-paired-lower-bound-on-every-task"
    else:
        selected = baseline_id
        reason = "both-safe-no-uniform-paired-advantage-default-to-lower-complexity"
    return {
        "schema_version": 1,
        "status": "mechanical-interface-decision-complete",
        "selected_interface": selected,
        "selected_algorithm": interfaces[selected]["algorithm"],
        "selected_action_schema_version": interfaces[selected]["action_schema_version"],
        "reason": reason,
        "aggregates": [aggregates[row["id"]] for row in stage["interfaces"]],
        "paired_maskable_minus_multidiscrete": task_intervals,
        "paired_bootstrap_samples": paired["bootstrap_samples"],
        "manual_override": False,
        "held_out_splits_used": False,
    }


def synthetic_record(
    candidate: dict[str, Any], round_: dict[str, Any], seed: int, interface: dict[str, Any],
) -> dict[str, Any]:
    index = int(candidate["id"][1:])
    unsafe = candidate["id"] == "c01"
    metrics = {
        "candidate_id": candidate["id"],
        "eligible": not unsafe,
        "catastrophic_failures": int(unsafe),
        "cascade_failures": 0,
        "invalid_actions": 0,
        "worst_task_paired_bootstrap_lower_bound_vs_heuristic": index / 100.0,
        "worst_task_success_rate": 0.70 + index / 100.0,
        "macro_validation_score": 0.60 + index / 100.0,
        "training_timesteps": round_["timesteps_per_stage"] * 6,
        "network_size_proxy": sum(candidate["network"]),
    }
    return {
        "candidate_id": candidate["id"], "interface_id": interface["id"],
        "training_seed": seed,
        "metrics": metrics,
        "evidence": {
            "kind": "synthetic-fixture-not-scientific-evidence",
            "identity": hashlib.sha256(
                canonical_json({"candidate": candidate["id"], "round": round_["round"], "seed": seed}).encode()
            ).hexdigest(),
        },
    }


def real_record(
    root: Path, baseline: dict[str, Any], protocol: dict[str, Any],
    candidate: dict[str, Any], round_: dict[str, Any], seed: int,
    interface: dict[str, Any], *, phase: str = "hyperparameter-selection",
    interface_decision_sha256: str | None = None,
) -> dict[str, Any]:
    name = f"{phase}-r{round_['round']}-{interface['id']}-{candidate['id']}-seed{seed}"
    spec = candidate_spec(
        baseline, protocol, candidate, round_, seed, interface, phase=phase,
        interface_decision_sha256=interface_decision_sha256,
    )
    spec_path = root / "generated_specs" / f"{name}.json"
    spec_path.parent.mkdir(exist_ok=True)
    write_once(spec_path, spec)
    run_dir = root / "runs" / name
    run_dir.parent.mkdir(exist_ok=True)
    subprocess.run([
        sys.executable, str(ROOT / "train_canonical.py"),
        "--spec", str(spec_path), "--run-dir", str(run_dir),
    ], cwd=ROOT, check=True)
    lock = read_json(run_dir / spec["outputs"]["run_lock"])
    model_metadata = read_json(run_dir / spec["outputs"]["final_model_metadata"])
    evaluation = run_dir / spec["outputs"]["validation_dir"]
    payload = verify_benchmark(
        evaluation, "validation", spec, lock, model_metadata["model_identity"]
    )
    benchmark_identity = file_identity(evaluation / "benchmark.json", relative_to=run_dir)
    metrics = result_metrics(payload, candidate, round_)
    return {
        "candidate_id": candidate["id"], "interface_id": interface["id"],
        "training_seed": seed,
        "metrics": metrics,
        "evidence": {
            "run_fingerprint": lock["run_fingerprint"],
            "spec_sha256": spec_hash(spec),
            "benchmark": benchmark_identity,
            "model": model_metadata["model_identity"],
        },
    }


def orchestrate(campaign_dir: str | Path, *, synthetic_fixture: bool = False) -> dict[str, Any]:
    protocol = protocol_payload()
    baseline, _ = load_spec(BASELINE_PATH)
    if not synthetic_fixture and protocol.get("status") != "preregistered-compute-authorized":
        raise ReproducibilityError(
            "model-selection compute is not authorized by the committed protocol; no training started"
        )
    git = git_metadata()
    if not synthetic_fixture and git["dirty"]:
        raise ReproducibilityError("source commit does not match clean campaign identity; no training started")
    root = ensure_external_run_dir(campaign_dir, float(baseline["runtime"]["minimum_free_disk_gib"] if not synthetic_fixture else 0))
    root.mkdir(parents=True, exist_ok=True)
    lock = {
        "schema_version": 1,
        "campaign_id": protocol["campaign_id"],
        "mode": "synthetic-fixture" if synthetic_fixture else "real-validation-only",
        "source_commit": "synthetic-fixture" if synthetic_fixture else git["commit"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "baseline_spec_sha256": spec_hash(baseline),
    }
    lock["campaign_fingerprint"] = hashlib.sha256(canonical_json(lock).encode()).hexdigest()
    lock_path = root / "campaign_lock.json"
    if not lock_path.exists() and any(root.iterdir()):
        raise ReproducibilityError("new campaign requires an empty directory or its exact campaign lock")
    write_once(lock_path, lock)
    state_path = root / "campaign_state.json"
    prior_state = read_json(state_path) if state_path.exists() else None
    if prior_state and prior_state.get("campaign_fingerprint") != lock["campaign_fingerprint"]:
        raise ReproducibilityError("campaign state belongs to another source/protocol identity")
    prior_interface_records = prior_state.get("interface_records", []) if prior_state else []
    prior_interface_decision = prior_state.get("interface_decision") if prior_state else None
    prior_records = prior_state.get("records", []) if prior_state else []
    prior_decisions = prior_state.get("rounds", prior_state.get("decisions", [])) if prior_state else []
    if (
        not isinstance(prior_interface_records, list)
        or not isinstance(prior_records, list)
        or not isinstance(prior_decisions, list)
    ):
        raise ReproducibilityError("campaign state has invalid resume structure")

    interface_stage = protocol["action_interface_selection"]
    interfaces = {item["id"]: item for item in interface_stage["interfaces"]}
    reference = {
        "id": "interface-reference",
        **interface_stage["reference_hyperparameters"],
    }
    interface_round = {
        "round": 0,
        "training_seeds": interface_stage["training_seeds"],
        "timesteps_per_stage": interface_stage["timesteps_per_stage"],
    }
    interface_records: list[dict[str, Any]] = []
    for interface in interface_stage["interfaces"]:
        for seed in interface_stage["training_seeds"]:
            if synthetic_fixture:
                record = synthetic_interface_record(interface, protocol, seed)
            else:
                raw = real_record(
                    root, baseline, protocol, reference, interface_round, seed,
                    interface, phase="interface-first",
                )
                record = {
                    "interface_id": interface["id"],
                    "training_seed": seed,
                    "metrics": raw["metrics"],
                    "evidence": raw["evidence"],
                }
            if (
                len(interface_records) < len(prior_interface_records)
                and canonical_json(prior_interface_records[len(interface_records)]) != canonical_json(record)
            ):
                raise ReproducibilityError("existing interface result does not match revalidated evidence")
            interface_records.append(record)
            atomic_json(state_path, {
                **lock, "status": "interface-selection-running",
                "interface_records": [
                    *interface_records, *prior_interface_records[len(interface_records):]
                ],
                "interface_decision": prior_interface_decision,
                "records": prior_records,
                "decisions": prior_decisions,
            })
    interface_decision = select_interface(protocol, interface_records)
    interface_decision = {**lock, **interface_decision}
    if prior_interface_decision and canonical_json(prior_interface_decision) != canonical_json(interface_decision):
        raise ReproducibilityError("existing interface decision does not match deterministic paired comparison")
    interface_decision_path = root / protocol["orchestration"]["interface_decision"]
    write_once(interface_decision_path, interface_decision)
    interface_decision_sha256 = sha256_file(interface_decision_path)
    selected_interface = interfaces[interface_decision["selected_interface"]]

    candidates = {item["id"]: item for item in protocol["bounded_search"]["candidates"]}
    survivors = list(candidates)
    decisions = []
    all_records = []
    for round_ in protocol["successive_halving"]:
        if len(survivors) != round_["candidate_count"]:
            raise ReproducibilityError("campaign survivor count drifted from preregistration")
        aggregates = []
        for candidate_id in survivors:
            candidate = candidates[candidate_id]
            records = []
            for seed in round_["training_seeds"]:
                record = (
                    synthetic_record(candidate, round_, seed, selected_interface)
                    if synthetic_fixture else
                    real_record(
                        root, baseline, protocol, candidate, round_, seed,
                        selected_interface,
                        interface_decision_sha256=interface_decision_sha256,
                    )
                )
                records.append(record)
                durable_record = {"round": round_["round"], **record}
                if len(all_records) < len(prior_records) and canonical_json(prior_records[len(all_records)]) != canonical_json(durable_record):
                    raise ReproducibilityError("existing campaign result does not match revalidated evidence")
                all_records.append(durable_record)
                atomic_json(state_path, {
                    **lock, "status": "running",
                    "interface_records": interface_records,
                    "interface_decision": interface_decision,
                    "records": [*all_records, *prior_records[len(all_records):]],
                    "decisions": [*decisions, *prior_decisions[len(decisions):]],
                })
            aggregates.append(aggregate_candidate(candidate, records, round_))
        ranking, survivors = rank_candidates(aggregates, int(round_["advance"]))
        decision = {
            "round": round_["round"], "ranking": ranking,
            "advanced": survivors, "manual_override": False,
        }
        if len(decisions) < len(prior_decisions) and canonical_json(prior_decisions[len(decisions)]) != canonical_json(decision):
            raise ReproducibilityError("existing survivor decision does not match deterministic reranking")
        decisions.append(decision)
        write_once(root / f"round_{round_['round']}_decision.json", {**lock, **decision})
        atomic_json(state_path, {
            **lock, "status": "running",
            "interface_records": interface_records,
            "interface_decision": interface_decision,
            "records": [*all_records, *prior_records[len(all_records):]],
            "decisions": [*decisions, *prior_decisions[len(decisions):]],
        })

    winner = survivors[0]
    decision = {
        **lock,
        "status": "synthetic-fixture-complete" if synthetic_fixture else "development-selection-complete",
        "winner": winner,
        "held_out_splits_used": False,
        "manual_override": False,
        "interface_decision": interface_decision,
        "interface_records": interface_records,
        "records": all_records,
        "rounds": decisions,
        "action_interface_decision_required_before_final_freeze": False,
    }
    write_once(root / "selection_decision.json", decision)
    proposal = candidate_spec(
        baseline, protocol, candidates[winner], protocol["successive_halving"][-1],
        protocol["data_policy"]["training_seeds"][0], selected_interface,
        interface_decision_sha256=interface_decision_sha256,
    )
    proposal["experiment_id"] = f"{protocol['campaign_id']}-{winner}-proposed-final"
    proposal["status"] = (
        "synthetic-fixture-not-evidence" if synthetic_fixture
        else "proposed-final-selected-review-required"
    )
    proposal["selection_evidence"] = {
        "campaign_fingerprint": lock["campaign_fingerprint"],
        "decision_file": "selection_decision.json",
        "interface_decision_file": protocol["orchestration"]["interface_decision"],
        "interface_decision_sha256": interface_decision_sha256,
        "selected_interface": selected_interface["id"],
        "selected_algorithm": selected_interface["algorithm"],
        "selected_action_schema_version": selected_interface["action_schema_version"],
        "held_out_splits_used": False,
        "action_interface_decision_required": False,
    }
    write_once(root / "proposed_final_spec.json", proposal)
    atomic_json(state_path, {**decision, "status": decision["status"]})
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument(
        "--synthetic-fixture", action="store_true",
        help="Exercise orchestration/ranking only; emits clearly non-evidence fake results and runs no training",
    )
    args = parser.parse_args()
    try:
        report = orchestrate(args.campaign_dir, synthetic_fixture=args.synthetic_fixture)
    except (OSError, ValueError, json.JSONDecodeError, ReproducibilityError, subprocess.CalledProcessError) as exc:
        print(f"STOP: model-selection campaign failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({
        "status": report["status"], "winner": report["winner"],
        "campaign_fingerprint": report["campaign_fingerprint"],
        "held_out_splits_used": report["held_out_splits_used"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
