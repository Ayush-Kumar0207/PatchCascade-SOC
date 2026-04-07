# 📚 PatchCascade SOC — Example Walkthroughs

This document provides detailed step-by-step walkthroughs for each task level, showing optimal agent strategies and explaining the reasoning behind each action.

---

## 🟢 Easy Mode: The Basics

**Scenario**: 3 nodes, 0 dependencies, 1 vulnerability (CVE-2024-1001, CVSS 7.5, HIGH)

```
Initial State:
  web-server-01  [ONLINE] Tier 2 — affected by CVE-2024-1001
  api-server-01  [ONLINE] Tier 2 — affected by CVE-2024-1001
  dev-server-01  [ONLINE] Tier 3

Strategy: Direct patching — no dependencies to worry about.
```

| Turn | Action | Reasoning | Reward |
|------|--------|-----------|--------|
| 1 | `apply_patch(web-server-01, CVE-2024-1001)` | Tier 2 nodes can be patched while ONLINE | +7.50 |
| 2 | `apply_patch(api-server-01, CVE-2024-1001)` | Patch completes on web-server-01, start next | +7.50 |
| 3 | *(Patch auto-completes)* | CVE resolved! | +50.00 |

**Total Reward**: +65.0 → **Normalized Score: 0.999**

---

## 🟡 Medium Mode: Dependency Awareness

**Scenario**: 6 nodes, linear dependencies, 2 vulnerabilities

```
Dependency Graph:
  web-frontend-01 ──hard──► app-server-01 ──hard──► db-primary-01 (Tier 1!)
  web-frontend-02 ──hard──► app-server-02 ──hard──► db-primary-01 (Tier 1!)

Vulnerabilities:
  CVE-2024-2001 (CRITICAL, CVSS 9.1) on db-primary-01
  CVE-2024-2002 (HIGH, CVSS 7.5) on web-frontend-01, web-frontend-02

Challenge: db-primary-01 is Tier 1 — MUST be SUSPENDED before patching.
           But suspending it will CASCADE CRASH all dependent nodes!
```

### Optimal Strategy: Bottom-up suspend, then patch

| Turn | Action | Reasoning |
|------|--------|-----------|
| 1 | `suspend_service(web-frontend-01)` | Protect from cascade when DB goes down |
| 2 | `suspend_service(web-frontend-02)` | Protect second frontend |
| 3 | `suspend_service(app-server-01)` | Protect app layer |
| 4 | `suspend_service(app-server-02)` | All dependents now safe |
| 5 | `suspend_service(db-primary-01)` | Required: Tier 1 must be SUSPENDED |
| 6 | `apply_patch(db-primary-01, CVE-2024-2001)` | Patch critical DB vulnerability |
| 7 | `resume_service(app-server-01)` | DB online → safe to resume apps |
| 8 | `resume_service(app-server-02)` | Resume second app |
| 9 | `resume_service(web-frontend-01)` | Resume web (still has CVE-2024-2002) |
| 10 | `apply_patch(web-frontend-01, CVE-2024-2002)` | Patch web vuln |
| 11 | `resume_service(web-frontend-02)` | Resume second web |
| 12 | `apply_patch(web-frontend-02, CVE-2024-2002)` | Patch last affected host |

**Score**: completion=1.0, efficiency=0.85, safety=1.0, strategy=1.0 → **Final: 0.95**

### Common Mistake: Skipping suspend order

```
❌ WRONG: suspend_service(db-primary-01) first
   → app-server-01 CRASHES (hard dependency!)
   → web-frontend-01 CRASHES (cascade!)
   → 4 nodes crashed, massive penalty
```

---

## 🟣 Incident Response: Active Breach Triage

**Scenario**: Active breach — 2 nodes already CRASHED, exploits actively spreading

```
Initial State:
  db-primary-01   [CRASHED!]  — Breach entry point
  app-server-01   [CRASHED!]  — Cascaded from DB
  auth-server-01  [ONLINE]    — Tier 1, has CVE-2024-4002 (EXPLOITED!)
  app-server-02   [ONLINE]    — Depends on crashed DB!
  web-frontend-01 [ONLINE]
  web-frontend-02 [ONLINE]
  cache-redis-01  [ONLINE]
  monitoring-01   [ONLINE]

⚠️ CVE-2024-4002 is exploit_in_wild — if not patched within 4 turns,
   it will SPREAD to connected nodes!
```

### Optimal Triage Strategy

| Turn | Action | Reasoning |
|------|--------|-----------|
| 1 | `suspend_service(auth-server-01)` | Isolate exploited auth (Tier 1, must suspend) |
| 2 | `apply_patch(auth-server-01, CVE-2024-4002)` | Patch FIRST — it's actively exploited and spreading |
| 3 | `resume_service(db-primary-01)` | Recover DB — but we need to patch it first |
| 4 | `suspend_service(db-primary-01)` | Tier 1, must suspend before patching |
| 5 | `apply_patch(db-primary-01, CVE-2024-4001)` | Patch breach entry point |
| 6 | `resume_service(auth-server-01)` | Auth patch complete, bring back online |
| 7 | `resume_service(app-server-01)` | DB & Auth online → apps can restart |
| 8 | `apply_patch(web-frontend-01, CVE-2024-4003)` | Patch web vuln |
| 9 | `apply_patch(web-frontend-02, CVE-2024-4003)` | Patch second web server |

**Key Insight**: Prioritize exploited CVEs to prevent spreading. The auth server exploit would have infected app-server-02 by turn 5 if left unpatched.

---

## ⚫ Zero-Day Cascade: Adaptive Planning

**Scenario**: Manageable initial state, but zero-days emerge at turns 5 and 15

```
Initial State: 10 nodes, 2 vulnerabilities (CVE-2024-5001 HIGH, CVE-2024-5002 MEDIUM)

Turn 5 SURPRISE: CVE-2024-5099 (CVSS 9.9, CRITICAL, EXPLOITED!)
                  Affects: auth-server-01, db-primary-01

Turn 15 SURPRISE: CVE-2024-5100 (CVSS 8.4, HIGH)
                   Affects: web-frontend-01, web-frontend-02, api-gateway-01
```

### Adaptive Strategy

| Phase | Turns | Strategy |
|-------|-------|----------|
| **Phase 1** | 1-4 | Patch initial vulns (CVE-2024-5001 on app servers) |
| **Phase 2** | 5-10 | 🚨 PIVOT! Drop everything, handle zero-day CVE-2024-5099 on Tier 1 nodes |
| **Phase 3** | 10-14 | Clean up CVE-2024-5002 on cache, resume normal operations |
| **Phase 4** | 15-22 | Handle second zero-day CVE-2024-5100 on web layer |

**Key Insight**: The agent that scores highest is one that **immediately reprioritizes** when the zero-day alert appears at turn 5, rather than continuing its original plan.

---

## 🔑 Universal Strategy Principles

1. **Always check dependencies before acting** — one wrong suspend cascades the entire stack
2. **Exploited CVEs are ticking time bombs** — they accrue 2x penalty AND can spread
3. **Tier 1 nodes require the suspend-patch-resume dance** — never skip the suspend step
4. **Parallel patching is efficient** — while one node patches, work on independent branches
5. **Recovery comes before patching** — a CRASHED node must be RESUMED before it can be patched
6. **Stay alert for dynamic events** — new CVEs can appear mid-episode in advanced modes
