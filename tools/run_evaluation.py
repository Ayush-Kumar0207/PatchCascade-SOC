#!/usr/bin/env python3
"""Run one frozen split through an atomic, identity-bound evaluation entry point."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_repro import (
    ReproducibilityError, append_event, atomic_json, build_lock, canonical_json,
    declared_dependency_mismatches, ensure_external_run_dir, file_identity,
    git_metadata, load_spec, runtime_info, utc_now, validate_file_identity,
)
from tools.verify_training_artifacts import read_json, verify_benchmark


def _events(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReproducibilityError("run event log is missing") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ReproducibilityError(f"run event log is corrupt at line {line_number}") from exc
    return rows


def run_evaluation(run_dir: str | Path, split: str, spec_path: str | Path = "training_specs/canonical_v1.json") -> dict:
    if split not in {"validation", "canonical", "confirmation"}:
        raise ReproducibilityError(f"unsupported evaluation split: {split}")
    spec, resolved_spec = load_spec(spec_path)
    root = ensure_external_run_dir(run_dir, float(spec["runtime"]["minimum_free_disk_gib"]))
    git = git_metadata()
    if git["dirty"]:
        raise ReproducibilityError("source commit does not match run identity; evaluation has not started")
    mismatches = declared_dependency_mismatches(spec, runtime_info(spec["dependencies"]))
    if mismatches:
        raise ReproducibilityError(f"dependency lock mismatch; evaluation has not started: {json.dumps(mismatches, sort_keys=True)}")
    lock = read_json(root / spec["outputs"]["run_lock"])
    expected = build_lock(spec, git["commit"])
    if canonical_json(lock) != canonical_json(expected):
        raise ReproducibilityError("source commit does not match run identity")
    model = root / spec["outputs"]["final_model"]
    metadata = read_json(root / spec["outputs"]["final_model_metadata"])
    model_identity = metadata.get("model_identity", {})
    validate_file_identity(model, model_identity, relative_to=root)

    output_key = {"validation": "validation_dir", "canonical": "canonical_dir", "confirmation": "confirmation_dir"}[split]
    output = root / spec["outputs"][output_key]
    in_progress = output.with_name(output.name + ".inprogress")
    event_path = root / spec["outputs"]["events"]

    predecessor = {"canonical": ("validation", "validation_dir"), "confirmation": ("canonical", "canonical_dir")}.get(split)
    if predecessor:
        predecessor_split, predecessor_key = predecessor
        predecessor_dir = root / spec["outputs"][predecessor_key]
        if not predecessor_dir.is_dir():
            raise ReproducibilityError(f"{predecessor_split} evaluation must complete before {split}; evaluation has not started")
        verify_benchmark(predecessor_dir, predecessor_split, spec, lock, model_identity)
        predecessor_marker = read_json(predecessor_dir / spec["outputs"]["evaluation_marker"])
        predecessor_benchmark = file_identity(predecessor_dir / "benchmark.json", relative_to=root)
        predecessor_completions = [
            row for row in _events(event_path)
            if row.get("event") == "evaluation_completed" and row.get("split") == predecessor_split
        ]
        if (
            predecessor_marker.get("status") != "complete"
            or predecessor_marker.get("model_identity") != model_identity
            or predecessor_marker.get("benchmark_identity") != predecessor_benchmark
            or len(predecessor_completions) != 1
            or predecessor_completions[0].get("attempt_id") != predecessor_marker.get("attempt_id")
        ):
            raise ReproducibilityError(f"{predecessor_split} evaluation identity/lifecycle is invalid; {split} has not started")

    if output.exists():
        payload = verify_benchmark(output, split, spec, lock, model_identity)
        marker = read_json(output / spec["outputs"]["evaluation_marker"])
        benchmark_identity = file_identity(output / "benchmark.json", relative_to=root)
        if (
            marker.get("split") != split
            or marker.get("run_fingerprint") != lock["run_fingerprint"]
            or marker.get("model_identity") != model_identity
            or marker.get("benchmark_identity") != benchmark_identity
        ):
            raise ReproducibilityError(f"{split} durable completion marker is missing or incompatible")
        completions = [row for row in _events(event_path) if row.get("event") == "evaluation_completed" and row.get("split") == split]
        if len(completions) > 1:
            raise ReproducibilityError(f"{split} has duplicate completion events")
        if not completions:
            append_event(
                event_path, "evaluation_completed", lock, split, split=split,
                attempt_id=marker["attempt_id"], output=output.relative_to(root).as_posix(),
                benchmark_sha256=benchmark_identity["sha256"], model_sha256=model_identity["sha256"],
                reconciled_after_interruption=True,
            )
        return {"complete": True, "split": split, "output": str(output), "episodes": len(payload["raw_episodes"])}

    if in_progress.exists():
        prior_starts = [row for row in _events(event_path) if row.get("event") == "evaluation_started" and row.get("split") == split]
        if not prior_starts or not prior_starts[-1].get("attempt_id"):
            raise ReproducibilityError(f"{split} in-progress directory has no auditable evaluation-start event")
        interrupted_attempt = prior_starts[-1]["attempt_id"]
        quarantine = root / "evaluation_quarantine"
        quarantine.mkdir(exist_ok=True)
        suffix = utc_now().replace(":", "").replace("-", "").replace(".", "")
        destination = quarantine / f"{output.name}.interrupted.{suffix}"
        os.replace(in_progress, destination)
        append_event(
            event_path, "evaluation_interrupted_quarantined", lock, split, split=split,
            attempt_id=interrupted_attempt, quarantined_path=destination.relative_to(root).as_posix(),
        )

    attempt_id = str(uuid.uuid4())
    append_event(
        event_path, "evaluation_started", lock, split, split=split, attempt_id=attempt_id,
        output=output.relative_to(root).as_posix(), model_sha256=model_identity["sha256"],
    )
    try:
        subprocess.run([
            sys.executable, "benchmark.py", "--spec", str(resolved_spec),
            "--split", split, "--run-fingerprint", lock["run_fingerprint"],
            "--rl-model", str(model), "--output-dir", str(in_progress),
        ], cwd=ROOT, check=True)
        payload = verify_benchmark(in_progress, split, spec, lock, model_identity, require_rendered=True)
        benchmark_identity = file_identity(in_progress / "benchmark.json", relative_to=root)
        benchmark_identity["path"] = (output.relative_to(root) / "benchmark.json").as_posix()
        marker = {
            "schema_version": 1, "status": "complete", "completed_at": utc_now(),
            "split": split, "attempt_id": attempt_id, "run_fingerprint": lock["run_fingerprint"],
            "model_identity": model_identity, "benchmark_identity": benchmark_identity,
            "episodes": len(payload["raw_episodes"]),
        }
        atomic_json(in_progress / spec["outputs"]["evaluation_marker"], marker)
        os.replace(in_progress, output)
        append_event(
            event_path, "evaluation_completed", lock, split, split=split,
            attempt_id=attempt_id, output=output.relative_to(root).as_posix(),
            benchmark_sha256=benchmark_identity["sha256"], model_sha256=model_identity["sha256"],
        )
        return {"complete": True, "split": split, "output": str(output), "episodes": len(payload["raw_episodes"])}
    except (subprocess.CalledProcessError, ReproducibilityError) as exc:
        append_event(event_path, "evaluation_failed", lock, split, split=split, attempt_id=attempt_id, error_type=type(exc).__name__)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--split", default="all", choices=["all", "canonical", "confirmation"], help="Use 'all' for the canonical contributor flow")
    parser.add_argument("--spec", default="training_specs/canonical_v1.json")
    args = parser.parse_args()
    try:
        if args.split == "all":
            result = {split: run_evaluation(args.run_dir, split, args.spec) for split in ("canonical", "confirmation")}
        else:
            result = run_evaluation(args.run_dir, args.split, args.spec)
    except (ReproducibilityError, subprocess.CalledProcessError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
