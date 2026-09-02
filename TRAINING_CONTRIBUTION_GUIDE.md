# Research-grade training contribution guide

The safe workflow is intentionally easy to execute correctly: values come from
the repository spec, while errors stop before GPU time is spent.

> **Current state:** development/model-selection compute is authorized by the
> committed `model_selection_v1.json` status `preregistered-compute-authorized`.
> External compute and reproducibility contributors are being recruited now.
>
> **Still sealed:** `canonical_v1.json` remains a provisional corrected baseline,
> not the final experiment. Final training plus canonical and confirmation
> evaluation remain unauthorized until the bounded selection campaign finishes and
> a separate reviewed `frozen-final-selected` spec is committed.

## Before compute

1. Comment on [issue #2](https://github.com/Ayush-Kumar0207/PatchCascade-SOC/issues/2)
   with GPU/VRAM, platform, persistent-storage plan, and your artifact host. Wait
   for acknowledgement to avoid duplicate compute.
2. Fork/clone the approved commit and create your own branch. Use only your own
   GitHub, cloud, Drive, and artifact-hosting credentials.
3. Create a fresh Python 3.11 environment, then install the repository and exact
   frozen research stack (use the official PyTorch selector if your platform needs
   a CUDA-specific index; the resolved versions must still match):

   ```bash
   python -m pip install -r requirements-training.txt
   python -m pip install -e . --no-deps
   python -m pip check
   ```

   Never put a token in a notebook, config, log, or PR.
4. Run preflight without creating a run:

   ```bash
   python tools/training_preflight.py --spec training_specs/canonical_v1.json
   ```

   This qualifies the exact environment while final training remains sealed.
   Running `train_canonical.py` with the provisional baseline stops before training.
   The authorized development campaign uses only:

   ```bash
   python tools/run_model_selection.py --campaign-dir /persistent/path/patchcascade-selection
   ```

   The orchestrator first gives MultiDiscrete PPO and MaskablePPO identical
   reference hyperparameters, budgets, training seeds, and validation episodes;
   it applies the preregistered safety gates and per-task paired-bootstrap rule,
   writes `interface_decision.json`, and only then runs the bounded 8→3→2→1
   hyperparameter campaign on that mechanical winner. It supplies every candidate
   spec, path, budget, seed, validation setting, rank, automatic resume identity,
   and durable decision. There is no manual post-result interface override.

The repository does not encode a minimum GPU model or VRAM threshold for this
campaign. It requires Python 3.11, the exact declared dependencies, at least 2 GiB
free disk at preflight, stable compute, and persistent campaign storage. Report the
actual CPU/GPU, RAM/VRAM, platform, and expected interruptions when claiming the
run; compatibility is established by repository preflight, not by a guessed
hardware promise.

## Choose a contribution track

- **Track A — environment/preflight verifier:** reproduce the exact dependency,
  environment, optimizer-shape, and six-process resume-equivalence checks. This is
  a small commitment and does not claim a campaign result.
- **Track B — development compute contributor:** run the authorized interface-first
  and 8→3→2→1 development campaign and return the complete evidence directory.
- **Track C — independent artifact/reproducibility reviewer:** verify campaign
  identity, paired-interface evidence, completeness, resume lineage, gates, and
  decisions without choosing results or opening held-out splits.
- **Track D — research contributor:** propose substantive methodology, analysis,
  or writing as a separately reviewed contribution under a new experiment identity.

Tracks A–C receive public technical credit appropriate to accepted work but do not
carry promised authorship. Authorship, if a manuscript results, follows actual
scholarly contribution and the applicable venue/authorship standards.

## One final training command (after selection and freeze)

Choose a new empty persistent directory **outside the source checkout**, then run:

```bash
python train_canonical.py \
  --spec training_specs/<reviewed-selected-spec>.json \
  --run-dir /persistent/path/patchcascade-final-selected
```

The command reruns preflight (including a CPU exact-shape PPO update), persists
the preflight evidence, creates the immutable identity/provenance, trains
all frozen stages, records SB3 diagnostics, checkpoints at the first completed
PPO update boundary after each 5,000 timesteps, resumes compatible interruptions,
rejects foreign checkpoints, and runs only the validation split. A checkpoint
includes hashed Python/NumPy/Torch/CUDA/vector-worker/environment/MixedTask state
that normal SB3 saves omit. The six-process CPU proof covers four vector workers,
mixed-task continuation, and an easy→mixed stage boundary; it does not claim
bitwise equivalence across GPU stacks. Rerun the identical command after a disconnect. Never edit a
run lock or point a new experiment at the old directory.

Runtime `.pkl` sidecars are executable trusted-run material. Do not resume a
downloaded contributor checkpoint on a maintainer machine. Verification hashes
them without deserialization; resume only your own run, or use a disposable
credential-free isolated environment after audit. See
[`RUNTIME_STATE_SECURITY.md`](RUNTIME_STATE_SECURITY.md).

## Frozen evaluation and verification

Only after validation selection is complete and a **new** spec with status
`frozen-final-selected` is reviewed and committed may its held-out canonical and
confirmation evaluations run with one command:
The wrapper reads the model path and fingerprint from the locked run, so nothing
training-critical is entered manually:

```bash
python tools/run_evaluation.py /persistent/path/patchcascade-final-selected --split all \
  --spec training_specs/<selected-spec>.json
```

Do not tune after either held-out result. Seeds, episode counts, output paths, max
steps, grader, fingerprint, model and agent matrix are injected or derived from
the immutable run and spec. Evaluation is written to an in-progress directory and
published atomically only after verification. An interrupted directory is retained
under `evaluation_quarantine/`, disclosed in the event log, and the entire
deterministic split is safely rerun; partial evidence can never look complete.

Running that command with `canonical_v1.json` stops immediately because the
provisional spec deliberately seals held-out seeds.

Then run:

```bash
python tools/verify_training_artifacts.py /persistent/path/patchcascade-canonical-v1
python tools/build_submission_bundle.py /persistent/path/patchcascade-canonical-v1
```

Verification recomputes summaries from raw episodes, rejects missing/duplicate
episodes or a wrong grader/source/fingerprint, rejects dirty source or dependency
drift, verifies the exact frozen model bytes before both splits, loads the model
against the current spaces, regenerates derived reports, hashes artifacts, and
flags baseline regressions/catastrophic failures.
Integrity passing does not mean PPO beat Heuristic. The verifier returns exit code
`2` and labels the policy rejected unless **both** held-out splits show a paired
bootstrap lower bound above Random and Heuristic on every task, with zero PPO
catastrophic failures, cascade failures, and invalid actions. Negative outcomes
remain valid, publishable scientific evidence, but cannot be called a successful
policy reproduction. The bundle records `accepted_policy_evidence` or
`rejected_policy_evidence` prominently.

Typical mistakes stop in plain language before compute or publication, for example:

- `STOP: source commit does not match run identity`
- `STOP: artifact identity mismatch: final_model.zip`
- `STOP: canonical evaluation is incomplete (149/750 episodes)`
- `STOP: dependency lock mismatch; evaluation has not started`

## Contribution types and credit

- **Compute reproduction:** execute the frozen protocol and return all evidence.
- **Code/infrastructure:** improve correctness or safeguards without claiming a run.
- **Methodology:** propose a separate preregistered spec and validation comparison.

Accepted contributions keep PR/commit attribution and may be named under
**Training & Reproducibility Contributors** with immutable artifact links. This is
unpaid open-source work; no employment relationship or GitHub badge is promised.
