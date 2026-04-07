"""
PatchCascade SOC - Incident Response Task Definition
======================================================

Incident Response mode: Active breach scenario with pre-crashed nodes,
exploit spreading, and triage-under-pressure mechanics.

Difficulty: 4/5
Nodes: 8 (2 pre-crashed)
Dependencies: Complex with hard and soft edges
Vulnerabilities: 3 (2 actively exploited)
Max Turns: 60
Key Mechanic: Exploit spreading — unpatched exploited CVEs infect connected nodes

This task tests:
- Damage assessment and triage prioritization
- Recovery-while-patching workflow
- Threat containment under degraded infrastructure
- Decision-making under time pressure

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from grader import IncidentResponseGrader

_grader = IncidentResponseGrader()

INCIDENT_RESPONSE_TASK = {
    "id": "incident_response",
    "name": "Incident Response",
    "description": (
        "Active breach scenario: 8 nodes (2 pre-crashed), 3 vulnerabilities "
        "(2 actively exploited), with exploit spreading mechanic. "
        "The agent must triage an ongoing attack — recovering crashed nodes, "
        "patching exploited CVEs before they spread, and stabilizing the network "
        "under degraded conditions. Tests damage assessment, threat containment, "
        "and recovery-under-pressure decision making."
    ),
    "max_turns": 60,
    "difficulty": 4,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_vulnerabilities_patched": True,
        "all_crashed_nodes_recovered": True,
        "no_additional_cascade_failures": True,
        "exploit_spreading_contained": True,
    },
}
