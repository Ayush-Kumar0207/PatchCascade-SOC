#!/usr/bin/env python3
"""Execute or fixture-test the locked, resume-aware validation-only campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    round_: dict[str, Any], training_seed: int,
) -> dict[str, Any]:
    spec = copy.deepcopy(baseline)
    spec["experiment_id"] = (
        f"{protocol['campaign_id']}-r{round_['round']}-{candidate['id']}-seed{training_seed}"
    )
    spec["status"] = "development-selection-candidate"
    spec["algorithm"]["learning_rate"] = candidate["learning_rate"]
    spec["algorithm"]["entropy_coefficient"] = candidate["entropy_coefficient"]
    spec["algorithm"]["architecture"] = {
        "policy": list(candidate["network"]), "value": list(candidate["network"]),
    }
    spec["seeds"]["global_training_seed"] = training_seed
    for stage in spec["methodology"]["stages"]:
        stage["timesteps"] = round_["timesteps_per_stage"]
    spec["selection_identity"] = {
        "campaign_id": protocol["campaign_id"],
        "round": round_["round"],
        "candidate_id": candidate["id"],
        "training_seed": training_seed,
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


def synthetic_record(candidate: dict[str, Any], round_: dict[str, Any], seed: int) -> dict[str, Any]:
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
        "candidate_id": candidate["id"], "training_seed": seed,
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
) -> dict[str, Any]:
    name = f"r{round_['round']}-{candidate['id']}-seed{seed}"
    spec = candidate_spec(baseline, protocol, candidate, round_, seed)
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
        "candidate_id": candidate["id"], "training_seed": seed,
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
    prior_records = prior_state.get("records", []) if prior_state else []
    prior_decisions = prior_state.get("rounds", prior_state.get("decisions", [])) if prior_state else []
    if not isinstance(prior_records, list) or not isinstance(prior_decisions, list):
        raise ReproducibilityError("campaign state has invalid resume structure")

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
                    synthetic_record(candidate, round_, seed)
                    if synthetic_fixture else
                    real_record(root, baseline, protocol, candidate, round_, seed)
                )
                records.append(record)
                durable_record = {"round": round_["round"], **record}
                if len(all_records) < len(prior_records) and canonical_json(prior_records[len(all_records)]) != canonical_json(durable_record):
                    raise ReproducibilityError("existing campaign result does not match revalidated evidence")
                all_records.append(durable_record)
                atomic_json(state_path, {
                    **lock, "status": "running",
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
        "records": all_records,
        "rounds": decisions,
        "action_interface_decision_required_before_final_freeze": True,
    }
    write_once(root / "selection_decision.json", decision)
    proposal = candidate_spec(
        baseline, protocol, candidates[winner], protocol["successive_halving"][-1],
        protocol["data_policy"]["training_seeds"][0],
    )
    proposal["experiment_id"] = f"{protocol['campaign_id']}-{winner}-proposed-final"
    proposal["status"] = (
        "synthetic-fixture-not-evidence" if synthetic_fixture
        else "proposed-final-selected-review-required"
    )
    proposal["selection_evidence"] = {
        "campaign_fingerprint": lock["campaign_fingerprint"],
        "decision_file": "selection_decision.json",
        "held_out_splits_used": False,
        "action_interface_decision_required": True,
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
