---
name: Canonical RL reproduction
about: Submit complete external PatchCascade training evidence
---

## Links

- Related issue:
- Frozen model artifact/revision:
- Submission bundle:
- Full logs:
- Provenance:
- SHA256SUMS:

## Identity and integrity

- [ ] I used the recorded clean source commit.
- [ ] Preflight and every required test passed before optimizer steps.
- [ ] The spec SHA-256 and run fingerprint match all checkpoints/results.
- [ ] All curriculum and mixed-consolidation stages completed.
- [ ] Every resume used repository-produced compatible metadata.
- [ ] Model load/space validation and artifact verification passed.
- [ ] Validation, canonical, and confirmation outputs are separate and complete.
- [ ] Random, Heuristic, and PPO saw the exact same seeds/max steps/grader.
- [ ] Every raw episode is included; no episode was selected, deleted, or rerun.
- [ ] Metrics and plots were generated, not manually edited.
- [ ] Baseline regressions, high variance, and catastrophic safety failures are disclosed.
- [ ] I copied the verifier's exact `policy_accepted` and `scientific_outcome` labels; I did not describe rejected evidence as successful.
- [ ] If accepted: every task on canonical and confirmation exceeded Random and Heuristic by the frozen paired-CI gate, with zero PPO catastrophic/cascade/invalid-action failures.
- [ ] I did not tune after viewing canonical or confirmation results.
- [ ] I used my own accounts/credentials and committed no secret or large model.
- [ ] Every interruption, warning, restart, and deviation is disclosed below.

## Frozen configuration

- Source commit:
- Spec SHA-256:
- Run fingerprint:
- Runtime / GPU / VRAM:
- Model artifact SHA-256:

## Outcome and limitations

Report PPO versus Random and Heuristic for every task, including paired confidence
intervals. A below-baseline result is valid evidence and must be stated plainly.

## Interruptions, warnings, and deviations

None / complete disclosure.
