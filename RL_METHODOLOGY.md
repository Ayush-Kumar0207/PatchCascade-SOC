# Canonical RL methodology decision

The canonical-v1 protocol is a **progressive five-task curriculum followed by a
balanced mixed-task consolidation stage**. It is a frozen, defensible baseline,
not a claim that these hyperparameters or this method are optimal.

It is not the final contributor-training protocol. The bounded, validation-only
selection and action-interface investigation in
[`MODEL_SELECTION_PROTOCOL.md`](MODEL_SELECTION_PROTOCOL.md) must complete, and
the winner must be committed as a new `frozen-final-selected` spec, before either
canonical or confirmation evaluation is unsealed.

## Options considered

1. **Separate PPO per level** avoids catastrophic forgetting but produces five
   policies, multiplies compute, and creates checkpoint/model-selection ambiguity.
2. **Pure curriculum** matches the increasing mechanics but can forget earlier
   tasks and lets curriculum order determine the final policy.
3. **Mixed-task training** directly optimizes broad coverage but exposes a new
   policy to the hardest sparse/unsafe regimes before it learns basic patching.
4. **Curriculum plus per-level fine-tuning** is useful as a preregistered ablation,
   but five final policies and added tuning increase selection risk.

Because the corrected wrapper has one stable observation/action schema for every
task, canonical-v1 uses curriculum to introduce skills, then a uniform mixed stage
to consolidate them. Per-level and pure-mixed modes should be separate specs and
must be compared on validation seeds before any canonical test is opened.

## Correctness audit findings addressed

- Fixed-seed Gym environments previously reset to the same scenario every episode.
- The numeric observation omitted CVE-to-host incidence even though patch actions
  require a node/CVE pair, creating observational aliasing.
- Padded action indices were silently converted to NOOP or a valid CVE, hiding
  invalid behavior and, in one path, using privileged environment knowledge.
- Time-limit episodes returned both `terminated=True` and `truncated=True` from
  the Gym wrapper.
- Valid and invalid actions advanced turn-indexed dynamic events at different times.
- Exploit-spread target selection depended on unordered set iteration, so the same
  seed could differ across Python processes.
- The old shaping equation omitted the learner's discount and terminal-potential
  handling, so documentation overstated policy-invariance theory.
- The old benchmark did not supply matched, predeclared episode seeds and the RL
  adapter did not synchronize observation index maps before decoding actions.

These changes invalidate old unaccepted PPO archives as canonical-v1 models. They
do not alter or delete historical artifacts.

The compatibility boundary is explicit: environment API `patchcascade-gym-v4`,
observation `gym-observation-v3-cve-host-incidence`, action
`multidiscrete-v2-joint-validity-penalized`, and reward
`pbrs-v2-gamma-0.99-terminal-zero`. All four values enter the run fingerprint;
older PPO archives are pre-canonical and incompatible.

Ordinary MultiDiscrete PPO remains a provisional baseline because node/CVE
validity is joint. Independent factor masks cannot express that relation. A
flattened Discrete plus state-dependent MaskablePPO interface is therefore a
predeclared validation-only ablation, not an unreviewed change in this PR.

## Evidence contract

Methodological correctness comes before scale. Canonical-v1 therefore rejects a
dirty source tree, dependency drift, run directories inside the checkout, foreign
or byte-modified checkpoints, non-finite training telemetry, incomplete episode
matrices, edited summaries, model changes between held-out splits, and lifecycle
logs with missing or duplicate completions. Final model bytes are frozen before
validation and rechecked before canonical and confirmation evaluation. Evaluation
publishes atomically from an in-progress directory only after raw evidence has
been independently recomputed.

“Verified” means the evidence is complete, identity-bound, and auditable. It does
not by itself mean PPO is superior. Policy acceptance additionally requires, on
both held-out splits and every task, a positive paired-bootstrap lower confidence
bound versus Random and Heuristic, zero PPO catastrophic failures, zero cascade
failures, and zero invalid actions. Any failure is retained as canonical negative
evidence and explicitly labeled rejected. More timesteps, epochs, or hardware do
not repair a failed evidence contract.

Checkpoints are written only after a complete rollout and PPO optimizer update.
Each model checkpoint is paired with a hashed runtime snapshot containing Python,
NumPy, Torch CPU/CUDA, vector-environment, worker, MixedTask task-selection,
current observation, and episode-start state. CI compares uninterrupted CPU PPO
against save → new process/load → continue and requires identical trajectory and
parameter hashes. Mid-rollout work after the latest safe boundary is intentionally
discarded after an interruption; it is never labelled durable progress.
