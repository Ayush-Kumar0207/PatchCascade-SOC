# Validation-only model selection before final canonical training

`canonical_v1.json` is a corrected, frozen **provisional baseline**. Its 122,880
timesteps are not claimed to be the highest-quality configuration, and its held-out
canonical/confirmation seeds are intentionally blocked. Expensive compute remains
unauthorized in this PR.

The machine-readable preregistration is
[`training_specs/model_selection_v1.json`](training_specs/model_selection_v1.json).
It permits training and the ten validation seeds only. Fifty canonical and fifty
confirmation seeds remain inaccessible to selection.

`tools/run_model_selection.py` is the mechanical entry point. It derives every
candidate spec, seed, budget, path, resume identity, safety decision, rank, and
survivor from the preregistration. A campaign directory is locked to source and
protocol hashes; reruns revalidate existing evidence and resume the same candidate
run. It refuses real compute while the committed protocol status says
`preregistered-no-results-compute-not-authorized`.

The orchestration/ranking path can be tested without training:

```bash
python tools/run_model_selection.py --campaign-dir /tmp/patchcascade-selection-fixture --synthetic-fixture
```

Synthetic output is labelled `synthetic-fixture-not-evidence` and can never be
promoted. After explicit review authorizes compute in the version-controlled
protocol, omit `--synthetic-fixture`; there are no contributor-entered candidates,
seeds, budgets, ranking flags, or evaluator settings.

## Bounded selection

Eight declared PPO candidates vary only learning rate, entropy coefficient, and
128×128 versus 256×256 networks. Gamma, GAE, clipping, optimizer epochs, batch
size, and curriculum-plus-mixed schedule stay fixed. Three successive-halving
rounds use 10,240, 20,480, then 40,960 timesteps per stage and at most three
training seeds. The topology advances 8 → 3 → 2 → 1 and cannot expand after
results appear.

Candidates first need zero catastrophic and cascade failures. Ranking is
lexicographic and preregistered: worst-task paired-bootstrap lower bound versus
Heuristic, worst-task success, macro score, then lower compute, smaller network,
and candidate ID. There is no discretionary post-result override.

Training duration is therefore selected by evidence rather than assumed from the
122,880-step baseline. Network size is included only as one bounded factor; larger
models or more steps do not automatically win.

## Joint-action interface investigation

The current MultiDiscrete PPO is an implemented baseline, not a final quality
claim. Joint node/CVE validity cannot be represented by independent factor masks.
The flattened Discrete representation and state-dependent MaskablePPO candidate
are now implemented and contract-tested, but have **no comparison results**. Its
600 IDs have an exact bijection, non-semantic aliases are invalid rather than
repaired, and every mask bit is derived from the authoritative action validator.
`sb3-contrib==2.8.0` is exact-locked. The two interfaces are compared on validation
only. The more complex masked interface
is selected only if safety passes and it has a positive paired-bootstrap lower
bound in sample efficiency; otherwise the simpler safe interface wins. A
hierarchical policy is considered only if both preregistered interfaces fail.

This infrastructure PR does not run that comparison or declare either interface
superior. The provisional baseline stays MultiDiscrete PPO until the registered
development evidence supports a decision.

## Freeze boundary

The winner must be written into a new versioned experiment specification with
status `frozen-final-selected`, reviewed, and committed before either held-out
split can run. `tools/run_evaluation.py` rejects canonical or confirmation runs
from a provisional spec. Results from the selection campaign and its decision
artifact must show that no held-out result existed at freeze time.

Run `python tools/validate_model_selection.py` to validate the preregistration.
