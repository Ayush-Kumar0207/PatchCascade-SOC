---
title: PatchCascade SOC
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

<div align="center">

# 🛡️ PatchCascade SOC

### *Autonomous Cyber-Resilience Through Reinforcement Learning*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=for-the-badge)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-teal?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-orange?style=for-the-badge)](LICENSE)

**Train AI agents to manage vulnerability patches across enterprise networks—without crashing production.**

[Quick Start](#-quick-start) • [The Challenge](#-the-challenge) • [Architecture](#-architecture) • [API Reference](#-api-reference) • [Contributors](#-contributors)

---

🏆 **Meta PyTorch OpenEnv Hackathon 2026 Submission**

</div>

---

## 🎯 The Problem: The Patching Paradox

> *"Every unpatched CVE is a ticking time bomb. But every patch is a potential outage."*

Security teams face an impossible tradeoff:

| **Patch Immediately** | **Delay Patching** |
|----------------------|-------------------|
| ✅ Fixes vulnerability | ❌ Leaves exploit window open |
| ❌ Causes service downtime | ✅ Maintains uptime |
| ❌ Risk of cascade failures | ❌ Risk of breach |
| ❌ Angry customers | ❌ Angry regulators |

**The real nightmare?** Modern infrastructure has *dependencies*. Patch your database, and suddenly your web servers crash. Take down authentication, and your entire stack follows. One wrong move triggers a **cascade failure** that costs millions.

**PatchCascade SOC** is an OpenEnv-compatible reinforcement learning environment that trains AI agents to navigate this paradox—learning to patch vulnerabilities *in the optimal order* while minimizing downtime and avoiding catastrophic cascades.

---

## 💡 Why PatchCascade?

| Feature | Traditional RL Envs | PatchCascade SOC |
|---------|--------------------:|:-----------------|
| **Observation Space** | Numeric arrays | Rich JSON with semantic meaning |
| **Action Space** | Discrete indices | Named actions with parameters |
| **Reward Signal** | Sparse (win/lose) | Dense (continuous feedback) |
| **Real-World Mapping** | Abstract | Direct SOC workflow simulation |
| **LLM Compatibility** | Requires embedding | Native JSON schema descriptions |
| **Dependency Modeling** | None | Full cascade simulation |

---

## ✨ Feature Highlights

### 🏢 **Tiered Asset Management**
Not all servers are equal. Our environment models real-world criticality:

| Tier | Type | Examples | Downtime Penalty | Patch Rules |
|------|------|----------|------------------|-------------|
| **1** | 🔴 CRITICAL | Databases, Auth Servers | **3x** multiplier | Must SUSPEND before patching |
| **2** | 🟡 IMPORTANT | Web Servers, APIs | **2x** multiplier | Can patch while ONLINE |
| **3** | 🟢 STANDARD | Dev Servers, Monitoring | **1x** multiplier | Can patch while ONLINE |

### 🔗 **Dynamic Dependency Graph**
Real-time cascade failure simulation with hard and soft dependencies:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Web-01    │─────▶│   App-01    │─────▶│  DB-Primary │
│  (Tier 2)   │      │  (Tier 2)   │      │   (Tier 1)  │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                    │
       │                    │                    ▼
       │                    │             ┌─────────────┐
       │                    └────────────▶│  Auth-01    │
       │                                  │  (Tier 1)   │
       ▼                                  └─────────────┘
┌─────────────┐
│   Web-02    │
│  (Tier 2)   │
└─────────────┘

⚠️  If DB-Primary goes OFFLINE → App-01 CRASHES → Web-01 & Web-02 CASCADE CRASH
```

### 📊 **CVSS-Driven Risk Scoring**
Vulnerabilities are modeled with real-world severity metrics:

- **CVSS 9.0-10.0** (CRITICAL): Remote code execution, zero-click exploits
- **CVSS 7.0-8.9** (HIGH): Privilege escalation, data exfiltration  
- **CVSS 4.0-6.9** (MEDIUM): DoS, information disclosure
- **Exploit in Wild**: **2x penalty multiplier** for actively exploited CVEs

### 🎮 **Dense Reward Shaping**
Unlike sparse-reward environments, PatchCascade provides continuous feedback:

```
Reward = (Previous Penalty) - (Current Penalty)

Where Penalty = Risk_Penalty + Downtime_Penalty
```

| Event | Reward Impact |
|-------|--------------|
| Patch a CRITICAL vuln | **+9.0 to +19.6** (doubled if exploit active) |
| Cause a cascade crash | **-6.0 to -12.0** per crashed node |
| Invalid action | **-0.5** penalty |
| Victory (all patched) | **+50.0** bonus |
| Catastrophic failure | **-100.0** penalty |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     PatchCascade SOC Stack                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  inference.py │───▶│   client.py  │───▶│   server.py  │     │
│  │  (LLM Agent)  │    │ (HTTP Client)│    │  (FastAPI)   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                 │              │
│                                                 ▼              │
│                                          ┌──────────────┐     │
│                                          │environment.py│     │
│                                          │ (Core Logic) │     │
│                                          └──────────────┘     │
│                                                 │              │
│                                                 ▼              │
│                                          ┌──────────────┐     │
│                                          │  models.py   │     │
│                                          │  (Pydantic)  │     │
│                                          └──────────────┘     │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  🐳 Docker │ 🚀 Uvicorn │ 📡 OpenEnv Protocol │ 🤗 HF Spaces  │
└────────────────────────────────────────────────────────────────┘
```

| Component | Purpose |
|-----------|---------|
| `models.py` | Pydantic schemas with rich `Field()` descriptions for LLM comprehension |
| `environment.py` | Core state machine: reset, step, cascade logic, reward calculation |
| `server.py` | FastAPI wrapper exposing `/reset`, `/step`, `/observation` endpoints |
| `client.py` | Async HTTP client with type-safe request/response handling |
| `inference.py` | Baseline LLM agent using OpenAI-compatible API |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the container
docker build -t patchcascade-soc .

# Run the server
docker run -p 8000:8000 patchcascade-soc

# Test the endpoint
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "medium"}'
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Run the baseline agent
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Validate Your Submission

```bash
bash validate.sh https://your-space.hf.space
```

### 🎬 Sample Interaction

```python
from client import PatchCascadeLocalClient, PatchCascadeAction

# Initialize local client (no server needed)
client = PatchCascadeLocalClient()
obs = client.reset(task_level="easy")

print(f"Nodes: {[n.hostname for n in obs.nodes]}")
print(f"Vulns: {[v.cve_id for v in obs.vulnerabilities]}")

# Apply a patch
action = PatchCascadeAction(
    action_type="apply_patch",
    target="web-frontend-01",
    cve_id="CVE-2024-1234"
)
obs, reward, done, truncated, info = client.step(action)
print(f"Reward: {reward:.2f}, Done: {done}")
```

---

## 🎮 The Challenge

PatchCascade offers three difficulty levels, each testing different agent capabilities:

### 🟢 Easy Mode
> *"Learn the basics"*

| Parameter | Value |
|-----------|-------|
| Nodes | 3-5 servers |
| Dependencies | None |
| Vulnerabilities | 1 (Medium/High) |
| Max Turns | 30 |
| Key Skill | Basic patch sequencing |

### 🟡 Medium Mode
> *"Handle dependencies"*

| Parameter | Value |
|-----------|-------|
| Nodes | 5-8 servers |
| Dependencies | Linear chain (Web → App → DB) |
| Vulnerabilities | 2 (including 1 on Tier 1) |
| Max Turns | 50 |
| Key Skill | Suspend-patch-resume workflow |

### 🔴 Hard Mode
> *"Survive the chaos"*

| Parameter | Value |
|-----------|-------|
| Nodes | 10-15 servers |
| Dependencies | Complex graph (LB → Web → App → DB + Auth) |
| Vulnerabilities | 5 (2 actively exploited CRITICAL) |
| Max Turns | 100 |
| Key Skill | Multi-objective optimization under pressure |

---

## 📡 API Reference

### `POST /reset`
Initialize a new episode.

```json
// Request
{ "task_level": "medium", "seed": 42 }

// Response
{ "observation": { "nodes": [...], "vulnerabilities": [...], ... } }
```

### `POST /step`
Execute an action.

```json
// Request
{
  "action_type": "apply_patch",
  "target": "web-frontend-01",
  "cve_id": "CVE-2024-1234"
}

// Response
{
  "observation": { ... },
  "reward": 7.5,
  "done": false,
  "truncated": false,
  "info": { "valid": true, "cascade_failures": 0 }
}
```

### Action Types

| Action | Target Required | CVE Required | Effect |
|--------|----------------|--------------|--------|
| `scan_host` | ✅ | ❌ | Inspect node details |
| `suspend_service` | ✅ | ❌ | Gracefully offline node |
| `apply_patch` | ✅ | ✅ | Fix vulnerability (1 turn) |
| `resume_service` | ✅ | ❌ | Bring node back online |
| `noop` | ❌ | ❌ | Skip turn |

---

## 📈 Evaluation Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **Success Rate** | % of episodes with all CVEs patched | Maximize |
| **Total Reward** | Cumulative reward across episode | Maximize |
| **Steps to Completion** | Turns needed to win | Minimize |
| **Cascade Failures** | Nodes crashed from dependencies | Minimize |

### 📊 Baseline Scores (Reference)

Based on initial evaluations using our `inference.py` baseline agent on the **Medium Task** (max 50 steps):

| Agent Type | Expected Score | Behavior Profile |
|------------|----------------|------------------|
| **Random Action** | `-150.0` to `-300.0` | Usually triggers cascade failures by suspending Tier 1 nodes without warning. |
| **Simple Heuristic** | `+10.0` to `+20.0` | Patches critical vulnerabilities but incurs heavy downtime penalties. |
| **Optimal Target (RL)** | `+50.0` (Max) | Patches all vulnerabilities in the correct dependency order with minimal downtime. |

---

## 🧠 Agent Strategy Tips

1. **Suspend dependents first**: Before patching a Tier 1 node, suspend all nodes that depend on it
2. **Prioritize exploited CVEs**: `exploit_in_wild=true` means 2x risk penalty per turn
3. **Batch patches efficiently**: While one node is PATCHING, work on independent branches
4. **Don't fear downtime**: A controlled SUSPENDED state is better than an uncontrolled CRASH

---

## 📄 License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Ayush-Kumar0207">
        <img src="https://github.com/Ayush-Kumar0207.png" width="100px;" alt="Ayush Kumar"/><br/>
        <sub><b>Ayush Kumar</b></sub>
      </a><br/>
      <sub>🚀 Team Lead | Core Builder</sub>
    </td>
    <td align="center">
      <a href="https://github.com/cypher00grd">
        <img src="https://github.com/cypher00grd.png" width="100px;" alt="Ravi Prashant"/><br/>
        <sub><b>Ravi Prashant</b></sub>
      </a><br/>
      <sub>🏗️ Architect and Developer</sub>
    </td>
  </tr>
</table>

---

## 🛠️ Built With

| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Core language |
| **Pydantic v2** | Data validation & serialization |
| **FastAPI** | High-performance async API |
| **Uvicorn** | ASGI server |
| **Docker** | Containerization |
| **OpenAI SDK** | LLM integration |

---

## 🙏 Acknowledgments

- **Meta AI** — For hosting the PyTorch OpenEnv Hackathon
- **Hugging Face** — For Spaces infrastructure
- **OpenEnv Community** — For the standardized RL environment protocol

---

<div align="center">

**Built for the Meta PyTorch OpenEnv Hackathon 2026**

**Created by Ayush Kumar & Ravi Prashant**

*Train smarter. Patch faster. Crash never.*

Made with ❤️ in India

</div>