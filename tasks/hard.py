"""
PatchCascade SOC - Hard Task Definition
=========================================

Hard mode: 10-15 nodes, complex dependency graph, multiple critical vulnerabilities.
Expert level requiring strategic planning and dependency-aware patching.

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from grader import HardGrader

_grader = HardGrader()

HARD_TASK = {
    "id": "hard",
    "name": "Hard Mode",
    "description": (
        "10-15 nodes, complex dependency graph, multiple critical vulnerabilities. "
        "Expert level requiring strategic planning and dependency-aware patching."
    ),
    "max_turns": 100,
    "difficulty": 3,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_vulnerabilities_patched": True,
        "no_catastrophic_failures": True,
        "respect_dependencies": True,
        "minimize_downtime": True,
    },
}
