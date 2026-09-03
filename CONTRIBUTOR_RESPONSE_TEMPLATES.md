# Contributor response templates

These are maintainer aids, not automatic replies. Re-read the person's message,
verify the current repository state, personalize the opening, and send only what
is relevant. Never promise payment, employment, publication, or authorship.

## Interested contributor

> Thank you for your interest in PatchCascade. The currently authorized work is
> the frozen interface-first development/model-selection campaign; final training
> and held-out evaluation remain sealed. Please read the
> [onboarding guide](CONTRIBUTOR_ONBOARDING.md) and tell me which track interests
> you. Before substantial compute, please add the non-sensitive preflight details
> requested in [issue #2](https://github.com/Ayush-Kumar0207/PatchCascade-SOC/issues/2)
> so we can avoid duplicate work. Please do not post credentials, private paths,
> phone numbers, email addresses, or account identifiers.

## “What task should I take?”

> A good first step is Track A: reproduce the exact preflight and six-process
> resume-equivalence proof. If your environment qualifies and you have stable
> persistent storage, we can then acknowledge Track B, the complete interface-first
> plus 8→3→2→1 campaign. If you prefer not to train, Track C is an independent
> artifact review. Which of those best matches your time and skills?

## Limited compute

> Thanks for being clear about the limit. PatchCascade does not publish a guessed
> GPU/VRAM minimum; the exact preflight determines compatibility. A bounded
> preflight, code/test review, documentation, methodology, or independent evidence
> review may fit even when the full campaign does not. If you share only your
> CPU/GPU, RAM/VRAM, OS, Python version, approximate availability, and storage
> constraints, I can suggest the smallest useful track.

## Substantial compute

> That environment may be suitable. Before committing expensive compute, please
> post the CPU/GPU, RAM/VRAM, OS, Python version, stability/interruption plan,
> persistent storage, artifact host, and intended commit in issue #2. After the
> claim is acknowledged, start with the exact preflight and resume proof in the
> onboarding guide. Stop on any fail-closed error; do not modify dependencies,
> interfaces, candidates, seeds, budgets, or gates. A passing preflight does not
> authorize final training or held-out evaluation.

## Failed or interrupted run

> Thank you for reporting it—please preserve the directory exactly as it is.
> Do not delete failed records, edit locks/state, or start a favorable replacement.
> Share the exact command, commit, last successful stage, error text, relevant
> environment report, interruption timeline, and a hash manifest, with secrets and
> private paths removed. If the runner identifies the state as compatible, resume
> by rerunning the identical campaign command; otherwise stop for review. Do not
> send or deserialize a `.runtime.pkl` outside its trusted-run boundary. A
> well-preserved negative or incomplete run is still valuable evidence.

## Completed results

> Thank you. Before describing this as successful, please provide the full
> unedited campaign directory through a durable artifact link, its archive hash,
> a recursive SHA-256 manifest, the exact commit and command, and a disclosure of
> interruptions or deviations. Open a small PR linking those materials and issue
> #2; do not commit models, raw large artifacts, or runtime pickles. We will verify
> all six interface records, all 23 candidate-seed records, pair keys, identities,
> gates, decisions, and absence of held-out access. “Campaign complete” and
> “development winner proposed” do not yet mean “scientifically accepted policy.”

## Paper or authorship question

> There is no finished PatchCascade paper yet. The intention is to develop a
> paper only if the frozen process produces defensible reproducible evidence.
> Accepted execution or verification work receives public technical attribution.
> Authorship is considered only for a substantial scholarly contribution—such as
> methodology, analysis, interpretation, or writing—under the applicable venue's
> authorship standards; compute alone does not guarantee authorship.

## Maintainer acknowledgement of a run claim

> Acknowledged for **[track and bounded scope]** against commit **[commit]**.
> Please use a new persistent directory and only the exact command in the
> onboarding guide. Preserve all failures and interruptions, keep held-out splits
> sealed, and stop if any identity or preflight check fails. Treat `.runtime.pkl`
> as trusted-run-only executable material. This acknowledgement covers only the
> stated development work and expires if the commit, protocol, or runtime changes;
> report a change before continuing.
