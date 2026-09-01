#!/usr/bin/env python3
"""Build a small, model-free submission bundle after artifact verification."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_repro import ReproducibilityError, load_spec
from tools.verify_training_artifacts import verify_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--spec", default="training_specs/canonical_v1.json")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    try:
        result = verify_run(root, args.spec)
        spec, _ = load_spec(args.spec)
    except ReproducibilityError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
    bundle = root / "submission_bundle"
    archive_path = root / "patchcascade_submission_bundle.zip"
    if bundle.exists() or archive_path.exists():
        print("STOP: submission bundle directory/archive already exists; move both aside before rebuilding", file=sys.stderr)
        raise SystemExit(1)
    bundle.mkdir()
    outputs = spec["outputs"]
    for name in (
        outputs["run_lock"], outputs["preflight_report"], outputs["provenance"], outputs["provenance_markdown"],
        outputs["events"], outputs["progress"], outputs["final_model_metadata"],
        outputs["manifest"], outputs["checksums"], "training_diagnostics.jsonl",
    ):
        source = root / name
        if source.is_file():
            shutil.copy2(source, bundle / source.name)
    for key in ("validation_dir", "canonical_dir", "confirmation_dir"):
        shutil.copytree(root / outputs[key], bundle / outputs[key])
    shutil.copytree(root / outputs["training_plots_dir"], bundle / outputs["training_plots_dir"])
    metadata = json.loads((root / outputs["final_model_metadata"]).read_text(encoding="utf-8"))
    external = {
        "schema_version": 1, "required_external_artifacts": [metadata["model_identity"]],
        "instruction": "Publish the byte-identical model separately; verification must match this SHA-256 and size.",
    }
    (bundle / "EXTERNAL_ARTIFACTS.json").write_text(json.dumps(external, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = result["scientific_outcome"]
    checklist = f"""# Submission bundle

Run fingerprint: `{result['run_fingerprint']}`
Scientific outcome label: `{status}`

This bundle deliberately excludes `final_model.zip`. Publish that archive with
your own artifact-hosting account and provide the exact identity recorded in
`EXTERNAL_ARTIFACTS.json` and `SHA256SUMS.txt`.
Do not edit any generated metric. Link the model, issue, logs, and this bundle in
the training-reproduction PR template.

An integrity-valid bundle with critical findings is canonical negative/qualified
evidence, not a successful-policy or superiority claim.
"""
    (bundle / "README.md").write_text(checklist, encoding="utf-8")
    archive = shutil.make_archive(str(archive_path.with_suffix("")), "zip", bundle)
    print(json.dumps({"bundle": archive, "run_fingerprint": result["run_fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
