#!/usr/bin/env python3
"""Fail closed if the bounded validation-only model-selection contract drifts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate() -> dict[str, object]:
    selection = json.loads((ROOT / "training_specs/model_selection_v1.json").read_text(encoding="utf-8"))
    canonical = json.loads((ROOT / "training_specs/canonical_v1.json").read_text(encoding="utf-8"))
    policy = selection["data_policy"]
    if selection["status"] not in {
        "preregistered-no-results-compute-not-authorized",
        "preregistered-compute-authorized",
    }:
        raise ValueError("campaign authorization status is unsupported")
    if policy["allowed_splits"] != ["training", "validation"]:
        raise ValueError("selection may access only training and validation splits")
    if set(policy["forbidden_splits"]) != {"canonical_test", "confirmation_test"}:
        raise ValueError("held-out split prohibition is incomplete")
    if policy["validation_seeds"] != canonical["seeds"]["validation"]:
        raise ValueError("selection validation seeds drift from the versioned experiment")
    held_out = set(canonical["seeds"]["canonical_test"] + canonical["seeds"]["confirmation_test"])
    used = set(policy["validation_seeds"] + policy["training_seeds"])
    if held_out & used:
        raise ValueError("model selection leaks a canonical/confirmation seed")
    search = selection["bounded_search"]
    candidates = search["candidates"]
    if len(candidates) != search["maximum_candidates"] or len(candidates) > 8:
        raise ValueError("model-selection candidate bound is invalid")
    if len({item["id"] for item in candidates}) != len(candidates):
        raise ValueError("model-selection candidate IDs are duplicated")
    rounds = selection["successive_halving"]
    if [item["candidate_count"] for item in rounds] != [8, 3, 2] or [item["advance"] for item in rounds] != [3, 2, 1]:
        raise ValueError("successive-halving topology drifted")
    if selection["freeze_policy"]["selected_spec_required_status"] != "frozen-final-selected":
        raise ValueError("final selected-spec status is not fail closed")
    interfaces = selection["action_interface_investigation"]["interfaces"]
    if [item["id"] for item in interfaces] != ["multidiscrete-ppo", "flattened-discrete-maskableppo"]:
        raise ValueError("predeclared action-interface comparison drifted")
    masked = interfaces[1]
    if (
        masked.get("status") != "implemented-contract-tested-no-comparison-results"
        or masked.get("algorithm") != "MaskablePPO"
        or masked.get("action_schema_version") != "discrete-v1-state-masked-joint-validity"
        or masked.get("dependency") != "sb3-contrib==2.8.0"
    ):
        raise ValueError("flattened MaskablePPO candidate contract is incomplete")
    if selection["selection_rule"]["safety_eligibility"] != [
        "zero_catastrophic_failures", "zero_cascade_failures", "zero_invalid_actions"
    ]:
        raise ValueError("selection safety eligibility gates drifted")
    orchestration = selection.get("orchestration", {})
    if (
        orchestration.get("entrypoint") != "tools/run_model_selection.py"
        or orchestration.get("real_execution_requires_status") != "preregistered-compute-authorized"
        or orchestration.get("synthetic_fixture_is_never_evidence") is not True
    ):
        raise ValueError("model-selection orchestration contract is incomplete")
    return {
        "passed": True,
        "campaign_id": selection["campaign_id"],
        "candidates": len(candidates),
        "held_out_seed_count_prohibited": len(held_out),
    }


def main() -> None:
    try:
        report = validate()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"STOP: model-selection protocol is invalid: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
