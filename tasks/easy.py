"""
PatchCascade SOC - Easy Task Definition
=========================================

Easy mode: 3-5 nodes, no dependencies, 1 vulnerability.
Beginner-friendly scenario for learning basic patch mechanics.

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from grader import EasyGrader

_grader = EasyGrader()

EASY_TASK = {
    "id": "easy",
    "name": "Easy Mode",
    "description": (
        "3-5 nodes, no dependencies, 1 vulnerability. "
        "Beginner-friendly scenario for learning basic patch mechanics."
    ),
    "max_turns": 30,
    "difficulty": 1,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_vulnerabilities_patched": True,
        "no_catastrophic_failures": True,
    },
}
