# Runtime checkpoint trust boundary

PatchCascade's `.runtime.pkl` checkpoint sidecar is executable Python
serialization (`cloudpickle`). It is necessary to preserve Python, NumPy, Torch,
CUDA, vector-worker, environment, and `MixedTaskEnv` state that a normal SB3 ZIP
does not contain. It is **trusted run material**, not a safe interchange format.

- Never resume a runtime sidecar downloaded from a PR, issue, artifact host, or
  unknown contributor on a maintainer workstation.
- Verify hashes, metadata, source identity, model ZIP, and evidence without
  loading the pickle. The artifact verifier deliberately treats runtime bytes as
  opaque and tests that it never calls `cloudpickle.load`.
- Resume only a run you created and controlled. If third-party resume is required,
  first audit it and use a disposable, isolated VM/container with no credentials,
  secrets, host mounts, or network access.
- `load_resumable_checkpoint` refuses deserialization unless the caller makes the
  explicit trusted-run decision. Metadata must declare
  `python-cloudpickle-trusted-run-only-v1`, the registered algorithm class, exact
  file identity, safe update boundary, and matching run lock.

Submission bundles may include or externally index runtime sidecars for complete
reproducibility, but their presence never grants trust or causes verification to
deserialize them.
