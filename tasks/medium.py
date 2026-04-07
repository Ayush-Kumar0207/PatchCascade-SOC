"""
PatchCascade SOC - Medium Task Definition
============================================

Medium mode: 5-8 nodes, linear dependency chain, 2 vulnerabilities.
Requires dependency awareness and suspend-patch-resume workflow.

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from grader import MediumGrader

_grader = MediumGrader()

MEDIUM_TASK = {
    "id": "medium",
    "name": "Medium Mode",
    "description": (
        "5-8 nodes, linear dependency chain, 2 vulnerabilities. "
        "Requires dependency awareness and suspend-patch-resume workflow."
    ),
    "max_turns": 50,
    "difficulty": 2,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_vulnerabilities_patched": True,
        "no_catastrophic_failures": True,
        "respect_dependencies": True,
    },
}
