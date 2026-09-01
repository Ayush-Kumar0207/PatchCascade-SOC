# Known PatchCascade training failure modes

No PPO archive produced before environment API `patchcascade-gym-v4` is compatible
with the corrected research environment or accepted as canonical evidence.
Historical files are preserved; absence of complete immutable provenance means
their exact source/config/seed and failure causes are `unknown / insufficient
evidence` unless stated below.

| Failure mode | Scientific risk | Automated safeguard |
|---|---|---|
| Observation omitted CVE→host incidence | Different action-critical states looked identical | Observation v3 encodes the full incidence matrix; API/schema versions enter the run fingerprint |
| Padded/joint-invalid actions were silently repaired | Policy received privileged help and misleading scores | Action v2 penalizes without repair; evaluation recomputes exact joint validity and requires zero PPO invalid actions |
| Reward shaping used the wrong discount/terminal potential | Policy-invariance claim was not justified | Reward v2 fixes `gamma=0.99` and zero true-terminal potential; spec loader rejects drift |
| Reset carried cumulative risk/downtime while event timing, set iteration, or termination semantics drifted | Same seed or Gym trajectory could differ | New episodes clear prior state; explicit seeding, same-seed step equivalence, sorted choices, aligned timing, and terminated/truncated regression tests |
| Development CI did not install the canonical lock | Green CI could coexist with an unresolvable training environment | Required CI installs `requirements-training.txt`, editable code with `--no-deps`, runs `pip check`, exact preflight, and a real PPO update |
| SB3 checkpoint was saved inside a rollout with no environment/RNG state | Resume could silently become a different trajectory | Checkpoints occur only after a complete PPO update and pair the model with hashed Python/NumPy/Torch/CUDA/vector-environment/MixedTask state; CI requires new-process equivalence |
| 122,880 timesteps were mistaken for “highest quality” | Scale/hyperparameters were frozen without validation evidence | `canonical_v1` remains provisional; a bounded validation-only 8→3→2→1 protocol selects duration/LR/entropy/network before a new final spec |
| Factorized MultiDiscrete PPO wastes mass on jointly invalid node/CVE pairs | Learning may be unnecessarily handicapped | A flattened Discrete + MaskablePPO interface is a preregistered validation-only ablation; held-out seeds remain sealed |
| Old PPO artifacts were loaded against corrected semantics | Incompatible observations/actions could look like a valid resume | API, observation, action, and reward versions are immutable run identity; old archives are explicitly pre-canonical |

A positive aggregate metric cannot override an identity, integrity, safety, or
per-task acceptance failure. Unknown historical causes are not retroactively
invented; the strongest fail-closed detector is used instead.
