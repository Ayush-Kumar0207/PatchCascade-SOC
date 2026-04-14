# PatchCascade SOC — Executive Summary

> **One-line pitch:** A research-grade RL training environment that teaches AI agents to autonomously manage vulnerability patches across interdependent enterprise networks — with real-world mechanics like exploit spreading, zero-day injection, and cascade failure propagation.

---

## 🎯 The Problem

Security Operations Centers (SOCs) face a **$4.88M average breach cost** (IBM 2024) while managing thousands of CVEs across complex infrastructure. Human operators must decide:
- **What** to patch first (CVSS 9.8 vs 7.5?)
- **When** to patch (downtime during business hours?)
- **How** to patch (suspend services? risk cascade failures?)

These decisions are sequential, stochastic, and inter-dependent — a **perfect RL problem**.

## 🛡️ Our Solution

PatchCascade SOC is a fully-functional RL environment built on the **OpenEnv framework** that simulates realistic SOC decision-making:

| Feature | Description |
|---------|-------------|
| **5 Task Levels** | `easy` → `medium` → `hard` → `incident_response` → `zero_day` |
| **Dynamic Events** | Exploit spreading, zero-day injection, stochastic node degradation |
| **Multi-Dimensional Grading** | Completion × Efficiency × Safety × Strategy |
| **Gymnasium Compatible** | Standard `gym.Env` interface for SB3/RLlib/CleanRL |
| **Rich Observations** | 201-dimensional encoded state vector or JSON for LLMs |
| **Dependency Graphs** | Hard/soft service dependencies with cascade failure modeling |

## 🏗️ Architecture (30-second overview)

```
Agent → action(type, target, cve_id) → Environment → observation + reward
                                            ↕
                                    6-Phase Step Pipeline:
                                    1. Validate action
                                    2. Execute state change
                                    3. Process patches
                                    4. Simulate dynamic events
                                    5. Calculate shaped reward
                                    6. Check termination
```

## 📊 Key Innovation: Potential-Based Reward Shaping

We implement **Ng et al. (1999) potential-based shaping** to provide dense learning signals:

```
R_shaped = R_base + γ·Φ(s') - Φ(s)
```

Where `Φ(s)` encodes vulnerability risk, downtime cost, and node health — guaranteeing **optimal policy invariance** while enabling faster convergence.

## 🔧 Quick Start (3 commands)

```bash
pip install -e .
python -c "from gym_wrapper import PatchCascadeGymEnv; env = PatchCascadeGymEnv('medium'); print(env.reset()[0].shape)"
python train_rl.py --task easy --steps 10000  # Train a PPO agent
```

## 💡 Why PatchCascade Stands Out

1. **Not a toy environment** — Models real CVEs, CVSS scoring, service dependencies, and exploit dynamics
2. **Gymnasium-native** — `pip install` and train with any RL library in 3 lines
3. **5-level curriculum** — Progressive difficulty enables curriculum learning research
4. **Multi-agent ready** — Architecture supports attacker-defender extensions
5. **Production-grade code** — Pydantic v2 models, FastAPI server, comprehensive test suite

---

*Built for the Meta × PyTorch OpenEnv Hackathon 2026 by Ayush Kumar & Ravi Prashant*
