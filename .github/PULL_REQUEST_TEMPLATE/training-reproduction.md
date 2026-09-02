---
name: Research training reproduction
about: Submit development-selection or later frozen reproduction evidence
---

## Links

- Related issue:
- Phase: development selection / later frozen canonical reproduction
- Track: environment preflight / compute / independent verification / research
- Frozen model artifact/revision:
- Submission bundle:
- Full logs:
- Provenance:
- SHA256SUMS:

## Identity and integrity

- [ ] The issue assigned or acknowledged this work before significant compute.
- [ ] I used the recorded clean source commit.
- [ ] Preflight and every required test passed before optimizer steps.
- [ ] The spec SHA-256 and run fingerprint match all checkpoints/results.
- [ ] Every planned interface/candidate, seed, task, round, curriculum, and mixed stage for the assigned phase is present, or the evidence is explicitly labelled incomplete/negative.
- [ ] Every resume used repository-produced compatible metadata.
- [ ] Model load/space validation and artifact verification passed.
- [ ] Development selection used only training/validation splits and produced no canonical/confirmation artifact.
- [ ] If this is a later frozen reproduction, held-out access was authorized by a separately reviewed committed spec and its outputs are separate and complete.
- [ ] Random, Heuristic, and PPO saw the exact same seeds/max steps/grader.
- [ ] Every raw episode is included; no episode was selected, deleted, or rerun.
- [ ] Metrics and plots were generated, not manually edited.
- [ ] Baseline regressions, high variance, and catastrophic safety failures are disclosed.
- [ ] I copied the verifier's exact `policy_accepted` and `scientific_outcome` labels; I did not describe rejected evidence as successful.
- [ ] Interface and survivor decisions are repository-generated and were not manually overridden after results.
- [ ] If later accepted as final evidence: every task on canonical and confirmation exceeded Random and Heuristic by the frozen paired-CI gate, with zero PPO catastrophic/cascade/invalid-action failures.
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

For development selection, link the complete campaign state, interface records and
decision, candidate/seed/task records, paired intervals, round decisions, final
selection decision, proposed spec, logs, manifests, and checksums. For a later
authorized frozen reproduction, also report PPO versus Random and Heuristic for
every held-out task. A below-baseline or incomplete result is useful evidence and
must be stated plainly.

## Interruptions, warnings, and deviations

None / complete disclosure.
