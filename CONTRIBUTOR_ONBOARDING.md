# Contributor onboarding: development training and reproducibility

This runbook turns an expression of interest into one bounded, reviewable
PatchCascade contribution. The current authorization covers the frozen
**interface-first development/model-selection campaign**. Final training and
canonical and confirmation evaluation remain sealed.

## 1. Choose one bounded track

| Track | Suitable for | Completion evidence |
|---|---|---|
| A — preflight | A contributor checking an environment or the resume guarantees | Exact dependency/preflight output and the six-process resume-equivalence result |
| B — development compute | A contributor with stable compute and persistent storage | The complete interface-first plus 8→3→2→1 campaign directory |
| C — independent review | A contributor verifying an existing campaign without training | Completed review checklist, recomputed hashes, and discrepancies |
| D — research | Methodology, analysis, interpretation, or writing beyond execution | A separately reviewed proposal, analysis, or PR under a new identity when methods change |

Tracks A–C can receive public technical attribution when accepted. Authorship is
never promised for compute alone; it depends on a substantial scholarly
contribution if a manuscript results.

## 2. Claim work before spending compute

Comment on [intake issue #2](https://github.com/Ayush-Kumar0207/PatchCascade-SOC/issues/2)
and wait for a public acknowledgement. Include only:

- the desired track;
- CPU/GPU model, RAM/VRAM, OS, and Python version;
- approximate uninterrupted availability and expected interruption pattern;
- persistent-storage plan with at least 2 GiB free at preflight;
- intended artifact host and retention period; and
- the exact commit you intend to run.

The repository does not claim a minimum GPU or VRAM for this campaign. Suitability
is established by the exact preflight, not by guessing from a model name.

Do not publish email addresses, phone numbers, tokens, account identifiers,
private paths, cloud-console screenshots, or credentials. Use your own accounts;
maintainer credentials are never needed.

## 3. Qualify the environment before expensive work

From a clean checkout of the acknowledged commit, create a fresh Python 3.11
environment and run:

```bash
python -m pip install -r requirements-training.txt
python -m pip install -e . --no-deps
python -m pip check
python tools/training_preflight.py --spec training_specs/canonical_v1.json
python tools/resume_equivalence_probe.py
```

The preflight checks the exact dependency set and a real optimizer update. The
resume proof covers four vector workers, mixed-task continuation, and an
easy-to-mixed stage boundary in six new processes. It is not a claim of bitwise
equivalence across different GPU stacks.

Stop on any error. Do not change dependencies, optimizer shape, action schema,
seeds, thresholds, or environment semantics to make a check pass. A passing
preflight does not authorize the provisional `canonical_v1.json` for final
training or open a held-out split.

## 4. Run only the authorized campaign

Track B uses a new empty directory on persistent storage outside the source
checkout:

```bash
python tools/run_model_selection.py --campaign-dir /persistent/path/patchcascade-selection
```

Rerun that exact command after an interruption. The runner locks the source and
protocol identities, validates existing evidence, and resumes compatible work.
Never edit a run lock or campaign state, reuse another experiment directory,
delete an unfavorable record, or add flags that alter candidates, seeds, budgets,
ranking, interfaces, or evaluation.

The runner first compares MultiDiscrete PPO and flattened Discrete MaskablePPO
under identical reference settings: two interfaces times three training seeds,
or six interface records. Its mechanical decision then drives the 8→3→2→1
candidate campaign: 8 + 9 + 6 = 23 candidate-seed records. A complete campaign
therefore contains 29 registered training/evaluation records plus the decisions.
It may use only training and validation data. Do not attempt canonical or
confirmation evaluation.

## 5. Preserve every deliverable

Submit the complete directory, including unsuccessful and interrupted attempts.
At minimum, reviewers must be able to locate and verify:

- `campaign_lock.json` and `campaign_state.json`;
- all six paired interface records and `interface_decision.json`;
- every generated candidate spec and its exact source/spec/interface identity;
- preflight reports and dependency, CPU/GPU, RAM/VRAM, OS, Python, and runtime
  information;
- training logs, diagnostics, checkpoints, interruption/resume events, and
  failure records;
- every raw validation episode and all pair keys;
- per-seed and aggregate metrics, safety results, and bootstrap inputs/results;
- all three round decisions and survivor lists;
- `selection_decision.json` and `proposed_final_spec.json`; and
- a recursive file inventory with byte sizes and SHA-256 hashes.

Do not rename, prune, rewrite, or manually “repair” runner outputs. A negative or
incomplete run is useful evidence when its status and failure are preserved.

### Executable checkpoint warning

Any `.runtime.pkl` sidecar is executable trusted-run material, not a passive data
file. A reviewer must not deserialize a contributor's sidecar on a workstation.
Hash and inventory it as opaque bytes. If third-party resume is ever necessary,
first audit it and use a disposable, credential-free, network-disabled isolated
environment with no host mounts. See
[`RUNTIME_STATE_SECURITY.md`](RUNTIME_STATE_SECURITY.md).

## 6. Submission without large files in Git

1. Keep checkpoints, model ZIPs, runtime pickles, raw episode directories,
   archives, and credentials out of Git.
2. Upload the untouched campaign directory or a lossless archive to a durable
   artifact host under your own account. Prefer an immutable/versioned link.
3. Open a PR containing only a concise Markdown report and any small
   machine-readable decisions/manifests appropriate for review.
4. Link issue #2, the acknowledged commit, the artifact URL, the archive hash,
   the recursive SHA-256 manifest, and the exact reproduction command.
5. Disclose every interruption, warning, deviation, missing file, and retention
   limit. Never put a token or private path in the PR or logs.

The proposed selected spec is review material only. A maintainer must review and
commit a new versioned spec with status `frozen-final-selected` before final
training or either held-out split can run.

## 7. Acceptance checklist

A reviewer should mark each item explicitly:

- [ ] The claimed commit is clean and matches `campaign_lock.json`.
- [ ] Python and all training-critical dependencies match the frozen files; the
      preflight and optimizer-shape check passed.
- [ ] The six-process/four-worker resume-equivalence proof passed in its stated
      CPU scope.
- [ ] All six interface and 23 candidate-seed records are present, or the
      campaign is clearly labelled incomplete with its failure evidence.
- [ ] Pair keys, action schemas, interfaces, candidates, seeds, budgets,
      validation episodes, gates, ranking, and decisions match
      `model_selection_v1.json` exactly.
- [ ] Every registered seed is included; no seed shopping, manual override,
      deleted failure, or favorable rerun occurred.
- [ ] Every interface/candidate is free of catastrophic, cascade, and invalid
      action failures before it is treated as safety-eligible.
- [ ] No canonical or confirmation split or result was accessed.
- [ ] Raw episodes, summaries, decisions, hashes, and provenance agree.
- [ ] Runtime pickles were treated as opaque untrusted bytes during review.
- [ ] The external archive hash and recursive manifest verify.
- [ ] The outcome is labelled correctly using the definitions below.

### Status vocabulary

- **Campaign complete:** all 29 registered development records and both
  mechanical decisions exist. This alone is not a successful policy claim.
- **Evidence valid:** identities, completeness, provenance, hashes, pairings, and
  calculations verify. Valid evidence may still be negative.
- **Development winner proposed:** the mechanical validation rule selected one
  configuration. It is not yet a frozen final policy.
- **Scientifically accepted policy:** reserved for a later frozen model whose
  canonical and confirmation evidence both pass every integrity, per-task
  paired-bootstrap, quality, and safety gate. No current development run may use
  this label.

## 8. Safety boundary

Use only the repository's synthetic, authorized environment. Do not connect the
agent to real systems, accounts, networks, targets, production data, vulnerability
scanners, or private datasets. Do not weaken action validation, approval gates,
logging, deterministic identities, seed separation, or evaluation isolation.
Stop and report ambiguity before spending additional compute.

For the detailed experimental rules, read
[`TRAINING_CONTRIBUTION_GUIDE.md`](TRAINING_CONTRIBUTION_GUIDE.md),
[`MODEL_SELECTION_PROTOCOL.md`](MODEL_SELECTION_PROTOCOL.md),
[`KNOWN_TRAINING_FAILURE_MODES.md`](KNOWN_TRAINING_FAILURE_MODES.md), and
[`RUNTIME_STATE_SECURITY.md`](RUNTIME_STATE_SECURITY.md).
