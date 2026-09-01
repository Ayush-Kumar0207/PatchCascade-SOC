#!/usr/bin/env python3
"""Plot raw and lightly smoothed SB3 diagnostics without hiding instability."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_repro import ReproducibilityError, load_spec


def rolling_mean(values: list[float], window: int = 5) -> list[float]:
    return [sum(values[max(0, index - window + 1):index + 1]) / min(window, index + 1) for index in range(len(values))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--spec", default="training_specs/canonical_v1.json")
    args = parser.parse_args()
    try:
        spec, _ = load_spec(args.spec)
        root = Path(args.run_dir).resolve()
        lock = json.loads((root / spec["outputs"]["run_lock"]).read_text(encoding="utf-8"))
        rows = []
        for line_number, line in enumerate((root / "training_diagnostics.jsonl").read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if row.get("run_fingerprint") != lock.get("run_fingerprint"):
                raise ReproducibilityError(f"diagnostic line {line_number} belongs to another experiment")
            rows.append(row)
        keys = sorted({key for row in rows for key, value in row.get("metrics", {}).items() if isinstance(value, (int, float)) and math.isfinite(value)})
        if not rows or not keys:
            raise ReproducibilityError("no finite training diagnostics are available to plot")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        columns = 2
        figure, axes = plt.subplots(math.ceil(len(keys) / columns), columns, figsize=(12, 3.2 * math.ceil(len(keys) / columns)), squeeze=False)
        for axis, key in zip(axes.flat, keys):
            points = [(row["timesteps"], float(row["metrics"][key])) for row in rows if key in row.get("metrics", {})]
            x = [item[0] for item in points]
            y = [item[1] for item in points]
            axis.plot(x, y, alpha=0.35, linewidth=1, label="raw")
            axis.plot(x, rolling_mean(y), linewidth=1.6, label="rolling mean (5)")
            axis.set_title(key)
            axis.set_xlabel("total timesteps")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(keys):]:
            axis.set_visible(False)
        figure.suptitle("PatchCascade canonical training diagnostics (raw retained)")
        figure.tight_layout()
        output = root / spec["outputs"]["training_plots_dir"] / "diagnostics.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)
        print(output)
    except (OSError, json.JSONDecodeError, ReproducibilityError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
