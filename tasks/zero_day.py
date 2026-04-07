"""
PatchCascade SOC - Zero-Day Cascade Task Definition
=====================================================

Zero-Day Cascade mode: Dynamic threat injection mid-episode with
adaptive strategy requirements and escalating complexity.

Difficulty: 5/5 (Hardest)
Nodes: 10
Dependencies: Multi-layer graph (Web → Gateway → App → DB/Auth)
Vulnerabilities: 2 initial + 2 dynamically injected (4 total)
Max Turns: 80
Key Mechanic: Zero-day CVE injection at turns 5 and 15

This task tests:
- Adaptive planning: Strategy must change when new threats emerge
- Reprioritization: A CRITICAL zero-day at turn 5 demands immediate attention
- Multi-objective optimization: Balance initial plan with new requirements
- Resilience: Agent must handle strategy disruption gracefully

Dynamic Events:
- Turn 5:  CVE-2024-5099 (CVSS 9.9, CRITICAL, exploit_in_wild=True)
           Affects auth-server-01 and db-primary-01
- Turn 15: CVE-2024-5100 (CVSS 8.4, HIGH)
           Affects web-frontend-01, web-frontend-02, api-gateway-01

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from grader import ZeroDayGrader

_grader = ZeroDayGrader()

ZERO_DAY_TASK = {
    "id": "zero_day",
    "name": "Zero-Day Cascade",
    "description": (
        "Dynamic threat injection scenario: 10 nodes, 2 initial vulnerabilities, "
        "with zero-day CVEs injected at turns 5 (CRITICAL, CVSS 9.9) and 15 "
        "(HIGH, CVSS 8.4). The agent must adapt its patching strategy mid-episode "
        "when new critical threats emerge. Features multi-layer dependency graph "
        "(Web → Gateway → App → DB/Auth) requiring careful dependency-aware "
        "patching. Tests adaptive planning, strategy revision, and multi-objective "
        "optimization under evolving threat conditions."
    ),
    "max_turns": 80,
    "difficulty": 5,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_vulnerabilities_patched": True,
        "no_catastrophic_failures": True,
        "dynamic_cves_handled": True,
        "strategy_adapted": True,
    },
}
