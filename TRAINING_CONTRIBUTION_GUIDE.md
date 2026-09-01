# Research-grade training contribution guide

The safe workflow is intentionally easy to execute correctly: values come from
the repository spec, while errors stop before GPU time is spent.

> **Current review state:** expensive contributor training is not authorized.
> `canonical_v1.json` is a provisional corrected baseline, not the highest-quality
> final experiment. Complete the bounded validation-only protocol in
> [`MODEL_SELECTION_PROTOCOL.md`](MODEL_SELECTION_PROTOCOL.md), then commit the
> selected configuration as a new `frozen-final-selected` spec before opening
> canonical or confirmation seeds.

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

## One canonical training command

Choose a new empty persistent directory **outside the source checkout**, then run:

```bash
python train_canonical.py \
  --spec training_specs/canonical_v1.json \
  --run-dir /persistent/path/patchcascade-canonical-v1
```

The command reruns preflight (including a CPU exact-shape PPO update), persists
the preflight evidence, creates the immutable identity/provenance, trains
all frozen stages, records SB3 diagnostics, checkpoints at the first completed
PPO update boundary after each 5,000 timesteps, resumes compatible interruptions,
rejects foreign checkpoints, and runs only the validation split. A checkpoint
includes hashed Python/NumPy/Torch/CUDA/vector-worker/environment/MixedTask state
that normal SB3 saves omit. Rerun the identical command after a disconnect. Never edit a
run lock or point a new experiment at the old directory.

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
