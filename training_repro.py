"""Standard-library-only identity and provenance helpers for canonical RL runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
SECRET_RE = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|credential|private[_-]?key)")
SAFE_ENV = {"CI", "COLAB_RELEASE_TAG", "COLAB_GPU", "CUDA_VISIBLE_DEVICES", "LANG", "PYTHONHASHSEED"}


class ReproducibilityError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)


def load_spec(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    try:
        spec = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"Cannot read specification {resolved}: {exc}") from exc
    required = {"schema_version", "experiment_id", "methodology", "algorithm", "environment", "seeds", "evaluation"}
    if spec.get("schema_version") != 1 or not required.issubset(spec):
        raise ReproducibilityError(f"Invalid or unsupported canonical specification: {resolved}")
    if spec["algorithm"]["gamma"] != 0.99:
        raise ReproducibilityError("Canonical gamma must match environment SHAPING_GAMMA=0.99")
    expected_environment_versions = {
        "api_version": "patchcascade-gym-v4",
        "schema_version": "gym-observation-v3-cve-host-incidence",
        "reward_schema_version": "pbrs-v2-gamma-0.99-terminal-zero",
    }
    drift = {
        key: {"expected": value, "actual": spec["environment"].get(key)}
        for key, value in expected_environment_versions.items()
        if spec["environment"].get(key) != value
    }
    if drift:
        raise ReproducibilityError(
            "Canonical environment/API compatibility version mismatch: " + canonical_json(drift)
        )
    action_schema = spec["environment"].get("action_schema_version")
    algorithm_name = spec["algorithm"].get("name")
    allowed_action_algorithms = {
        "multidiscrete-v2-joint-validity-penalized": "PPO",
        "discrete-v1-state-masked-joint-validity": "MaskablePPO",
    }
    if action_schema not in allowed_action_algorithms:
        raise ReproducibilityError(f"Unsupported action schema: {action_schema}")
    if algorithm_name != allowed_action_algorithms[action_schema]:
        raise ReproducibilityError(
            f"Action schema {action_schema} requires algorithm {allowed_action_algorithms[action_schema]}, "
            f"not {algorithm_name}"
        )
    if action_schema.startswith("discrete-") and spec.get("dependencies", {}).get("sb3-contrib") != "2.8.0":
        raise ReproducibilityError("Flattened MaskablePPO requires exact sb3-contrib==2.8.0")
    task_levels = spec["environment"]["task_levels"]
    if len(task_levels) != len(set(task_levels)) or not task_levels:
        raise ReproducibilityError("Canonical task levels must be non-empty and unique")
    stage_tasks = [stage["task"] for stage in spec["methodology"]["stages"]]
    if stage_tasks[:-1] != task_levels or stage_tasks[-1:] != ["mixed"]:
        raise ReproducibilityError("Canonical curriculum must contain every task once followed by mixed consolidation")
    if spec["methodology"]["stages"][-1].get("tasks") != task_levels:
        raise ReproducibilityError("Mixed consolidation must use the exact canonical task order")
    seed_groups = spec["seeds"]
    evaluation_seeds = [*seed_groups["validation"], *seed_groups["canonical_test"], *seed_groups["confirmation_test"]]
    if len(evaluation_seeds) != len(set(evaluation_seeds)):
        raise ReproducibilityError("Validation, canonical, and confirmation seeds must be mutually disjoint and unique")
    if len(seed_groups["validation"]) != 10 or len(seed_groups["canonical_test"]) != 50 or len(seed_groups["confirmation_test"]) != 50:
        raise ReproducibilityError("Canonical seed-set cardinalities must be validation=10, canonical=50, confirmation=50")
    rollout_batch = spec["algorithm"]["rollout_steps"] * spec["algorithm"]["parallel_environments"]
    if any(stage["timesteps"] % rollout_batch for stage in spec["methodology"]["stages"]):
        raise ReproducibilityError(
            f"Every stage target must be divisible by one vectorized PPO rollout ({rollout_batch})"
        )
    return spec, resolved


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(spec).encode()).hexdigest()


def spec_reference(path: str | Path) -> str:
    """Return a portable, non-secret display reference for repo or generated specs."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"external-selection-spec/{resolved.name}"


def _git(*args: str, check: bool = True) -> str:
    process = subprocess.run(["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and process.returncode:
        raise ReproducibilityError(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def git_metadata() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current", check=False) or "DETACHED",
        "dirty": bool(status),
        "status_paths": [line[3:] for line in status.splitlines() if len(line) >= 4],
    }


def run_fingerprint(spec: dict[str, Any], commit: str) -> str:
    identity = {
        "fingerprint_schema": 1, "source_commit": commit,
        "spec_sha256": spec_hash(spec), "experiment_id": spec["experiment_id"],
        "methodology": spec["methodology"], "algorithm": spec["algorithm"],
        "environment": spec["environment"], "seeds": spec["seeds"],
    }
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def build_lock(spec: dict[str, Any], commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1, "experiment_id": spec["experiment_id"],
        "source_commit": commit, "spec_sha256": spec_hash(spec),
        "run_fingerprint": run_fingerprint(spec, commit),
        "methodology": spec["methodology"], "algorithm": spec["algorithm"],
        "environment": spec["environment"], "seeds": spec["seeds"],
    }


def ensure_run_lock(run_dir: str | Path, spec: dict[str, Any], commit: str) -> dict[str, Any]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = build_lock(spec, commit)
    path = root / spec["outputs"]["run_lock"]
    if path.exists():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproducibilityError(f"Unreadable run lock: {path}") from exc
        if canonical_json(actual) != canonical_json(expected):
            raise ReproducibilityError("Run directory belongs to an incompatible source/spec; nothing was modified")
    else:
        if any(root.iterdir()):
            raise ReproducibilityError("A new canonical run requires an empty run directory")
        atomic_json(path, expected)
    return expected


def validate_resume(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    fields = ("run_fingerprint", "source_commit", "spec_sha256", "methodology", "algorithm", "environment", "seeds")
    mismatch = {field: {"expected": expected.get(field), "actual": actual.get(field)} for field in fields if actual.get(field) != expected.get(field)}
    if mismatch:
        raise ReproducibilityError(f"Incompatible resume metadata: {canonical_json(mismatch)}")


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "<redacted>" if SECRET_RE.search(str(key)) else scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def versions(names: Iterable[str]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def package_freeze() -> list[str]:
    """Return a stable full distribution inventory without invoking a shell."""
    rows = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            rows.append(f"{name}=={distribution.version}")
    return sorted(set(rows), key=str.casefold)


def runtime_info(names: Iterable[str]) -> dict[str, Any]:
    gpu = {"cuda_available": False, "cuda": None, "name": None, "vram_gb": None}
    try:
        import torch
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.update(name=props.name, vram_gb=round(props.total_memory / 1024**3, 3))
    except Exception as exc:
        gpu["probe_error"] = type(exc).__name__
    frozen = package_freeze()
    return {
        "python": sys.version, "python_executable": sys.executable,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.platform(), "hostname": socket.gethostname(),
        "packages": versions(names), "package_freeze": frozen,
        "package_freeze_sha256": hashlib.sha256("\n".join(frozen).encode()).hexdigest(),
        "gpu": gpu,
    }


def declared_dependency_mismatches(spec: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    observed = runtime or runtime_info(spec.get("dependencies", {}))
    mismatches: dict[str, Any] = {}
    wanted_python = str(spec.get("runtime", {}).get("python", ""))
    if wanted_python and observed.get("python_major_minor") != wanted_python:
        mismatches["python"] = {"required": wanted_python, "installed": observed.get("python_major_minor")}
    for name, wanted in spec.get("dependencies", {}).items():
        installed = observed.get("packages", {}).get(name)
        if installed != wanted:
            mismatches[name] = {"required": wanted, "installed": installed}
    return mismatches


def file_identity(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise ReproducibilityError(f"artifact is missing: {target.name}")
    if relative_to is None:
        display = target.name
    else:
        try:
            display = target.relative_to(Path(relative_to).resolve()).as_posix()
        except ValueError as exc:
            raise ReproducibilityError("artifact path escapes its run directory") from exc
    return {"path": display, "size": target.stat().st_size, "sha256": sha256_file(target)}


def validate_file_identity(path: str | Path, identity: dict[str, Any], *, relative_to: str | Path | None = None) -> None:
    actual = file_identity(path, relative_to=relative_to)
    if canonical_json(actual) != canonical_json(identity):
        raise ReproducibilityError(f"artifact identity mismatch: {actual['path']}")


def resolve_run_path(run_dir: str | Path, relative_path: str) -> Path:
    root = Path(run_dir).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReproducibilityError("run metadata contains a path outside the run directory") from exc
    return candidate


def ensure_external_run_dir(run_dir: str | Path, minimum_free_gib: float = 2.0) -> Path:
    root = Path(run_dir).resolve()
    try:
        root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ReproducibilityError("run directory must be outside the source checkout")
    probe = root if root.exists() else root.parent
    if not probe.exists():
        raise ReproducibilityError(f"run-directory parent does not exist: {probe}")
    free_gib = shutil.disk_usage(probe).free / 1024**3
    if free_gib < minimum_free_gib:
        raise ReproducibilityError(f"only {free_gib:.2f} GiB free; at least {minimum_free_gib:.1f} GiB is required")
    return root


def provenance(spec: dict[str, Any], spec_path: Path, run_dir: Path, command: list[str]) -> dict[str, Any]:
    git = git_metadata()
    return scrub({
        "schema_version": 1, "captured_at": utc_now(), "git": git,
        "spec_path": spec_reference(spec_path), "spec_sha256": spec_hash(spec),
        "run_fingerprint": run_fingerprint(spec, git["commit"]),
        "runtime": runtime_info(spec.get("dependencies", {})),
        "command": command, "run_directory": ".",
        "safe_environment": {key: value for key, value in os.environ.items() if key in SAFE_ENV},
    })


def append_event(path: str | Path, event: str, lock: dict[str, Any], stage: str, **fields: Any) -> None:
    payload = scrub({
        "schema_version": 1, "timestamp": utc_now(), "event": event,
        "run_fingerprint": lock["run_fingerprint"], "source_commit": lock["source_commit"],
        "stage": stage, **fields,
    })
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
