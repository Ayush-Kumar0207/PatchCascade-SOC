# 🛡️ PatchCascade SOC

## The Complete Beginner's Guide — v2.0

### 🧭 Quick Navigation Grid
| Chapter | Topic | Highlights |
| :--- | :--- | :--- |
| [**1-2**](file:///c:/Users/kumar/Desktop/PatchCascade%20SOC/BEGINNERS_GUIDE.md#L48) | **The Story** | Hospital analogies & Why patching is hard |
| [**3-4**](file:///c:/Users/kumar/Desktop/PatchCascade%20SOC/BEGINNERS_GUIDE.md#L143) | **Key Concepts** | Servers, Hard/Soft Dependencies, CVSS |
| [**5-6**](file:///c:/Users/kumar/Desktop/PatchCascade%20SOC/BEGINNERS_GUIDE.md#L484) | **How it Works** | The 6-phase pipeline & File map |
| [**7-10**](file:///c:/Users/kumar/Desktop/PatchCascade%20SOC/BEGINNERS_GUIDE.md#L750) | **Deep Dive** | Tiers, States, Rewards, 5-Level Curriculum |
| [**11-15**](file:///c:/Users/kumar/Desktop/PatchCascade%20SOC/BEGINNERS_GUIDE.md#L1200) | **Advanced** | Exploit Spreading, Zero-Days, Multi-Dim Grading |
| [**16-19**](file:///c:/Users/kumar/Desktop/PatchCascade%20SOC/BEGINNERS_GUIDE.md#L2009) | **Ops & FAQ** | Live Dashboard, Running Docker, Design Philosophy |

---

## Understanding Our Project From Scratch

```
No coding experience required!
```

```
📖 Reading Time    ~60 minutes (grab a coffee ☕)
🎯 Goal            Understand every part of the project — deeply
👤 For             Anyone curious about the project
🏆 Context         Meta PyTorch OpenEnv Hackathon 2026 — Bangalore Finals
📝 Authors         Ayush Kumar & Ravi Prashant
🔖 Version         2.0.0 (Upgraded for Hackathon Finals)
```

---

## 🎬 Before We Begin: What Is This Document?

Hey there! 👋

Welcome to the **complete beginner's guide** to PatchCascade SOC — upgraded and expanded for the **Meta PyTorch OpenEnv Hackathon 2026 Finals in Bangalore**.

Since you're reading this, you're probably curious about one of these things:

- ✅ What problem we're solving (and why it matters in the real world)
- ✅ How our solution works (step by step, turn by turn)
- ✅ Why we made every technical decision (and what alternatives we rejected)
- ✅ What all the "scary" technical terms actually mean
- ✅ How our **5-level difficulty curriculum** trains AI agents from novice to expert
- ✅ How our **multi-dimensional grading system** evaluates agent intelligence
- ✅ How **dynamic events** (exploit spreading, zero-day injection, stochastic degradation) make our environment realistic
- ✅ The exact math behind every reward calculation
- ✅ How the **LLM-powered inference agent** actually works
- ✅ Our complete **testing and validation pipeline**
- ✅ How the **Live Command Center (Dashboard)** lets you visualize everything in real-time

**No prior knowledge required.** We'll explain everything from the ground up.

Let's start with a story...

---

## 📖 Chapter 1: The Story Behind PatchCascade

### 🏢 Imagine You Work at a Hospital

Picture this: You're the IT person at a large hospital. Your job is to keep all the computer systems running smoothly.

The hospital has:

🏥 **Patient Record Servers** — Store all patient data (VERY important)
💊 **Pharmacy Systems** — Track medications
🖥️ **Doctor Workstations** — Where doctors access patient info
📱 **Appointment Apps** — Patients book appointments here
🔐 **Login Servers** — Verify everyone's identity

Now, here's the scary part...

### 🚨 The Security Alert

One morning, you get an urgent notification:

> ⚠️ **SECURITY ALERT**
> A hacker has discovered a way to break into servers running "Ubuntu 22.04"
> Your Patient Record Servers are vulnerable!
> Patch immediately to prevent data breach!

A **"patch"** is like a software band-aid. It fixes the security hole.

**Sounds simple, right?** Just apply the patch!

### 😰 But Here's The Problem...

To install the patch, you need to **restart the server**. That means:

1. **The Patient Record Server goes offline** (stops working for a few minutes)
2. But wait... **Doctor Workstations need Patient Records**
3. If Patient Records are offline → **Doctor Workstations CRASH** ❌
4. And **Pharmacy Systems also need Patient Records**
5. So Pharmacy Systems CRASH too! ❌

**One restart caused THREE systems to fail!**

This is called a **Cascade Failure** — like dominoes falling.

### 🎯 The Patching Paradox

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  Option A: Patch Now               Option B: Don't Patch      │
│  ✅ Fixes security hole             ✅ Everything keeps working │
│  ❌ Causes cascade failure           ❌ Hackers can steal data  │
│  ❌ Doctors can't see records        ❌ Hospital could be sued  │
│  ❌ Pharmacy can't give meds         ❌ Patients' privacy lost  │
│                                                               │
│                 Both options are terrible!                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 💡 The Smart Solution

What if there was a **better way**?

What if, before patching the Patient Record Server, you:

1. First, **warn the Doctor Workstations** "Hey, I'm about to restart the database"
2. **Safely pause** the Doctor Workstations (so they don't crash)
3. **Safely pause** the Pharmacy Systems
4. NOW patch the Patient Record Server (no cascade!)
5. Bring everything back online in the **correct order**

**This requires planning.** You need to:
- Know which systems depend on which
- Patch in the right order
- Balance security vs. uptime

### 🤖 Enter: PatchCascade SOC

**This is exactly what our project does!**

We built a **training simulator** where AI agents (think: smart computer programs) can practice making these decisions. They learn:

- 🏆 Which servers are most important
- 💥 What happens if each server goes down
- 📋 The best order to patch things
- 🔄 How to recover from mistakes
- 🧠 How to adapt when new threats appear mid-operation
- ⚡ How to triage an active breach while systems are already burning

**It's like a flight simulator, but for IT security!**

> 🎯 **Why this matters:** Real SOC teams face these exact decisions every day. Training AI to handle this could save organizations millions of dollars and protect millions of people's data.

---

## 📖 Chapter 2: Understanding the Key Concepts

Before we dive deeper, let's learn some vocabulary. Don't worry — we'll explain everything simply!

### 🎮 What is "Reinforcement Learning"?

**Reinforcement Learning (RL)** is a way to teach computers by trial and error.

**Think of it like training a dog:**

```
┌───────────────────────┬───────────────────────────────────────┐
│     Dog Training      │          RL Training                  │
├───────────────────────┼───────────────────────────────────────┤
│ Dog sits              │ AI patches correctly                  │
│   → Gets a treat ✅    │   → Gets points ✅                    │
│ Dog jumps on furniture│ AI causes a crash                     │
│   → "No!" ❌           │   → Loses points ❌                   │
│ Dog learns what       │ AI learns what                        │
│   gets treats         │   gets points                         │
└───────────────────────┴───────────────────────────────────────┘
```

The AI tries random actions, sees what works, and learns over time.

**In our project:**
- Good action (patch a vulnerability) → **Positive reward** 📈
- Bad action (crash a server) → **Negative reward** 📉
- The AI learns to **maximize its total reward**

### 🖥️ What is a "Server"?

A **server** is just a computer that provides services to other computers.

- **Your laptop** = "client" (asks for things)
- **Netflix's computers** = "servers" (provide the movies)

When you watch Netflix:
1. Your laptop asks: "Can I watch Stranger Things?"
2. Netflix's server says: "Sure, here's the video data!"

In our hospital example:
- Patient Record Server **serves** patient data
- Doctor Workstations **ask for** that data

### 🔗 What is a "Dependency"?

A **dependency** is when one thing needs another to work.

**Real-life examples:**
- A car **depends on** gasoline (no gas = no driving)
- A plant **depends on** water (no water = dead plant)
- Your phone **depends on** a charged battery
- A YouTube video **depends on** your internet connection

**In our project:**
- Doctor Workstations **depend on** Patient Record Server
- If Patient Records goes down → Doctor Workstations fail

### 📚 How to Read Our Arrows (Important — Read This First!)

Before we draw any diagrams, let's learn **how to read the arrows**. This is the most important skill for understanding our project!

**The arrow always means "needs" or "depends on":**

```
  ┌─────────────┐       ┌─────────────┐
  │   Doctor    │       │   Patient   │
  │ Workstation │ ───►  │  Records   │
  └─────────────┘       └─────────────┘
      │                       │
  "I NEED you"        "I am NEEDED"

  Read this as: "Doctor Workstation DEPENDS ON Patient Records"
  Or simply:    "Doctor Workstation NEEDS Patient Records to work"
```

**Here's the key insight:** The arrow points FROM the thing that needs help TO the thing providing help. Think of it like this:

```
  The arrow is like a phone call:

  ┌─────────┐       ┌─────────┐
  │ Doctors │ ───►  │Database │
  └─────────┘       └─────────┘
  "Hello Database,         "Sure, here's
   I need patient           the data!"
   records please!"

  If the Database goes down? The Doctors get no answer.
  The phone rings and rings... and Doctors CRASH. 💥
```

**Simple Rule of Thumb:**
> If the thing on the RIGHT side of the arrow goes down,
> the thing on the LEFT side is in trouble.

Let's practice! Can you read this?

```
  ┌────────────┐       ┌────────────┐       ┌────────────┐
  │ Web Server │ ───►  │ App Server │ ───►  │  Database  │
  └────────────┘       └────────────┘       └────────────┘

  ✔ Answer: "Web Server needs App Server, and App Server needs Database"
  ✔ If Database goes down: App Server crashes, THEN Web Server crashes too!
  ✔ This is a CASCADE — one failure causing a chain reaction!
```

### 🆕 Hard vs. Soft Dependencies (New in v2.0!)

Our upgraded project has TWO types of arrows, and it's **critical** to understand the difference:

#### 🟥 HARD Dependency (thick solid arrow: ━━━►)

```
  Think of it like electricity to your house.
  No electricity? NOTHING works. Lights off. Fridge off. Everything stops.

  ┌────────────┐              ┌────────────┐
  │ App Server │ ━━━━━━━━►  │  Database  │
  └────────────┘   (HARD)    └────────────┘
       │                            │
  "I absolutely               "I'm the power.
   CANNOT work                 Without me,
   without you!"               you're dead."

  Database goes DOWN? ➡️ App Server CRASHES instantly! 💥
```

#### 🟨 SOFT Dependency (dotted arrow: ┄┄┄►)

```
  Think of it like wi-fi to your laptop.
  No wi-fi? Your laptop still works! You just can't browse the internet.
  It's slower/limited, but it doesn't DIE.

  ┌──────────────┐              ┌────────────┐
  │Load Balancer│ ┄┄┄┄┄┄┄┄►  │ Web Server │
  └──────────────┘   (SOFT)    └────────────┘
       │                            │
  "I work BETTER             "If I go down,
   with you, but              you'll slow down
   I'll survive              but won't crash."
   without you."

  Web Server goes DOWN? ➡️ Load Balancer is slower, but SURVIVES ✔️
```

**Why does this matter for the AI?**

The agent needs to know: *"If I take down Server X for patching, which other servers will CRASH (hard dependency) vs. which will just slow down (soft dependency)?"*

```
  Quick Reference Card:
  ━━━►  HARD = "Will crash if removed"    → MUST handle before patching!
  ┄┄┄►  SOFT = "Will survive if removed"  → Can often ignore safely
```

### 🕸️ What is a "Dependency Graph"?

Now let's put it all together! When you have MANY servers depending on each other, you draw a **map** of all the connections. Let's build one step by step:

**Step 1: Start with just 2 servers**

```
  ┌────────────┐       ┌────────────┐
  │ App Server │ ━━━►  │  Database  │
  └────────────┘       └────────────┘
  "I need the database to function."
```

**Step 2: Add a web server that needs the app server**

```
  ┌────────────┐       ┌────────────┐       ┌────────────┐
  │ Web Server │ ━━━►  │ App Server │ ━━━►  │  Database  │
  └────────────┘       └────────────┘       └────────────┘
  ✅ This is a CHAIN — a straight line of dependencies.
  ✔ Database down? → App crashes → Web crashes too! (3 dominoes!)
```

**Step 3: Now it gets interesting — MULTIPLE servers need the SAME database**

```
  ┌────────────┐
  │ App Svr 01 │ ━━━┐
  └────────────┘    │    ┌────────────┐
                    ┣━►  │  Database  │    Both app servers need
  ┌────────────┐    │    └────────────┘    the SAME database!
  │ App Svr 02 │ ━━━┘
  └────────────┘

  💥 If Database goes down, BOTH app servers crash at the same time!
     This is what makes it a "graph" and not just a chain.
```

**Step 4: The full picture — our Hard mode network**

Now let's see a real dependency graph from our project. Don't panic — just read it **one arrow at a time!**

```
                    ┌──────────────┐
                    │  Load        │
                    │  Balancers   │  Tier 2
                    └──────┬───────┘
                           │
                     ┄┄┄┄┄┄ (SOFT — won't crash if web servers go down)
                           │
                           ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ Web Server   │   │ Web Server   │   │ Web Server   │  Tier 2
    │  Frontend 01 │   │  Frontend 02 │   │  Frontend 03 │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                  │
     ━━━━━━ (HARD!)     ━━━━━━ (HARD!)    ━━━━━━ (HARD!)
           │                  │                  │
           ▼                  ▼                  ▼
    ┌──────────────┐   ┌──────────────┐
    │ App Server   │   │ App Server   │   Tier 2
    │     01       │   │     02       │
    └──┬────┬──────┘   └──┬────┬──────┘
       │    │             │    │
 ━━━━━━  ━━━━━━   ━━━━━━  ━━━━━━  (ALL HARD!)
       │    │             │    │
       ▼    ▼             ▼    ▼
  ┌────────────┐   ┌─────────────┐
  │ DB Primary │   │ Auth Server │   Tier 1 (CRITICAL!)
  │    01      │   │     01      │
  └─────┬──────┘   └─────────────┘
        │
  ━━━━━━ (HARD!)
        │
        ▼
  ┌────────────┐
  │ DB Replica │   Tier 1 (CRITICAL!)
  │    01      │
  └────────────┘
```

**🧠 Reading Practice! Try answering these:**

```
  Q1: What happens if "DB Primary 01" goes down?

  A1: Follow ALL the arrows that point TO db-primary-01:
      • App Server 01 has a HARD dependency on it → CRASHES 💥
      • App Server 02 has a HARD dependency on it → CRASHES 💥
      • DB Replica has a HARD dependency on it   → CRASHES 💥
      Then the cascade continues:
      • Web Frontend 01 depends on App Server 01 → CRASHES 💥
      • Web Frontend 02 depends on App Servers    → CRASHES 💥
      • Web Frontend 03 depends on App Server 02 → CRASHES 💥
      Result: 1 server goes down, 6 MORE crash! That's a CASCADE.

  Q2: What happens if "Load Balancers" go down?

  A2: Load Balancers have SOFT dependencies (dotted arrows).
      • Web Servers might slow down, but they DO NOT crash.
      Result: The system degrades but survives. Much less scary!

  Q3: What's the SAFEST way to patch DB Primary 01?

  A3: Suspend everything that depends on it FIRST (bottom-up):
      1. Suspend web servers (they depend on app servers)
      2. Suspend app servers (they depend on db-primary)
      3. Suspend db-primary (now safe — no one will crash!)
      4. Apply the patch
      5. Bring everything back (top-down, reverse order)
```

This map of connections is called a **Directed Acyclic Graph (DAG)**. That's a fancy name, but really it just means:

```
  Directed  = "the arrows point in one direction (who depends on whom)"
  Acyclic   = "no circles (A can't depend on B which depends on A)"
  Graph     = "a picture with boxes and connections"

  Don't worry about remembering "DAG" — just remember:
  "It's a map showing which servers need which other servers."
```

> 🆕 **Upgrade note:** The old version only had simple linear chains (A→B→C). Our v2.0 has **complex multi-path dependency graphs** with both hard and soft edges, multiple branches, and shared dependencies (like both App Servers depending on the same DB). This is what makes the problem HARD — and interesting!

### 🦠 What is a "Vulnerability"?

A **vulnerability** is a weakness in software that hackers can exploit.

**Think of it like:**
- A vulnerability is a **broken lock** on your front door
- A hacker is a **burglar** who knows about the broken lock
- A patch is **replacing the lock** with a working one

**Real example:**
`"CVE-2024-1234: A bug in the login page lets hackers bypass passwords"`

**CVE** stands for "Common Vulnerabilities and Exposures" — it's the official naming system for security bugs. Like how hurricanes get names (Hurricane Katrina), security bugs get CVE IDs.

### 📊 What is a "CVSS Score"?

When security researchers find a vulnerability, they rate how dangerous it is on a scale of 0 to 10.

```
┌──────────────┬──────────────┬───────────────────────────────────────────┐
│ CVSS Score   │ Severity     │ What It Means                             │
├──────────────┼──────────────┼───────────────────────────────────────────┤
│ 9.0 - 10.0   │ 🔴 CRITICAL  │ Hackers can take over your system remotely │
│ 7.0 - 8.9    │ 🟠 HIGH      │ Hackers can steal data or damage systems   │
│ 4.0 - 6.9    │ 🟡 MEDIUM    │ Hackers can cause some problems            │
│ 0.1 - 3.9    │ 🟢 LOW       │ Minor issues, low risk                    │
└──────────────┴──────────────┴───────────────────────────────────────────┘
```

In our project, we use CVSS scores to decide **which vulnerabilities to patch first**. A CVSS 9.8 bug is more urgent than a CVSS 4.0 bug!

### 🔥 What Does "Exploit in the Wild" Mean?

This is a critical concept in our project.

When a vulnerability is being **actively exploited in the wild**, it means hackers are **right now, at this very moment**, using that bug to break into systems.

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Normal CVE:           "We know this lock is broken"               │
│                         → Bad, but no one is picking it yet        │
│                         → Penalty: CVSS × affected servers         │
│                                                                    │
│  Exploit in Wild:      "Burglars are ACTIVELY picking this lock!"  │
│                         → Emergency! They're inside RIGHT NOW!     │
│                         → Penalty: CVSS × affected servers × 2     │
│                         → The risk penalty is DOUBLED!             │
│                                                                    │
│  🆕 In v2.0:          Exploited CVEs can also SPREAD to           │
│                         connected servers over time!                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📖 Chapter 3: How Our Project Works

Now that you understand the concepts, let's see how PatchCascade SOC actually works!

### 🎭 The Two Main Players

Our project has two parts that talk to each other:

```
┌─────────────────────┐                  ┌─────────────────────┐
│                     │   HTTP/Local     │                     │
│    THE AGENT        │ ◄──────────────► │  THE ENVIRONMENT    │
│  (Decision Maker)   │    Messages      │   (The Simulator)   │
│                     │                  │                     │
│  🤖 Powered by LLM  │                  │  🌍 5 Difficulty     │
│  (Qwen 2.5 72B)     │                  │     Levels          │
│                     │                  │  📊 Multi-Dim Grader │
│                     │                  │  🔥 Dynamic Events   │
└─────────────────────┘                  └─────────────────────┘
   "What should                             "Here's what
    I do next?"                              happened!"
```

**The Agent** 🤖
- This is the AI that makes decisions
- It looks at the current situation
- It chooses an action
- In our case, it's powered by a Large Language Model (**Qwen/Qwen2.5-72B-Instruct** via HuggingFace)
- It can also be any OpenEnv-compatible agent

**The Environment** 🌍
- This is the simulator (the "game world")
- It tracks all the servers, vulnerabilities, dependencies
- It processes the agent's actions through a **6-phase pipeline**
- It fires **dynamic events** (exploit spreading, zero-days, stochastic degradation)
- It calculates rewards using **potential-based reward shaping**
- It grades performance across **4 dimensions** (completion, efficiency, safety, strategy)

### 📁 Our Project Files — The Complete Map

Here's what **every single file** in our upgraded project does:

```
┌───────────────────────┬──────────────────────────────────┬──────────────────────────────────┐
│ File                  │ What It Does                     │ Simple Analogy                   │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ models.py (832 lines) │ Data structures & validation     │ The dictionary — defines all     │
│                       │ ServerNode, Vulnerability,       │ the words and their rules        │
│                       │ Dependency, Action, Observation  │                                  │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ environment.py        │ Core simulation engine           │ The game engine — makes          │
│ (1821 lines!)         │ 5 scenario generators            │ everything happen. The biggest   │
│                       │ 6-phase step pipeline            │ and most complex file.           │
│                       │ Dynamic event system             │                                  │
│                       │ ASCII art renderer               │                                  │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ grader.py (657 lines) │ Multi-dimensional scoring        │ The judge — scores your          │
│                       │ 5 task-specific graders           │ performance across 4 dimensions  │
│                       │ Completion, efficiency,          │ like Olympic figure skating      │
│                       │ safety, strategy                 │                                  │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ _server.py (488 lines)│ FastAPI REST server               │ The receptionist — handles       │
│                       │ 15+ API endpoints                │ all communication                │
│                       │ OpenEnv-compliant wrapper        │                                  │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ client.py (498 lines) │ HTTP + Local + Sync clients      │ The messenger — carries          │
│                       │ 3 client implementations         │ messages in 3 different ways     │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ inference.py          │ LLM-powered AI agent             │ The player — the brain that      │
│ (371 lines)           │ Async OpenAI-compatible          │ decides what to do each turn     │
│                       │ Retry logic & error handling     │                                  │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ tasks/ (5 files)      │ Task definitions & registry      │ The game levels — from           │
│                       │ Easy through Zero-Day            │ tutorial to nightmare mode       │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ tests/ (5 files)      │ Comprehensive test suite         │ The quality inspector —          │
│                       │ Environment, models, grader,     │ makes sure nothing is broken     │
│                       │ server tests                     │                                  │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ smoke_test.py         │ End-to-end validation            │ The final exam — runs all        │
│ (303 lines)           │ Heuristic agent runs all 5 tasks │ 5 levels without needing an LLM  │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ openenv.yaml (308!)   │ Full OpenEnv specification       │ The ID card — tells the          │
│                       │ Tasks, graders, schemas, events  │ hackathon exactly who we are     │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Dockerfile            │ Container build instructions     │ The recipe book — how to set     │
│                       │ With health checks               │ things up anywhere               │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ static/ (7 files!)     │ Live Command Center (Dashboard)  │ The control room — a beautiful   │
│                       │ Real-time visualization          │ visual interface to watch the    │
│                       │ D3.js topology & charts          │ AI play the game.                │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ ARCHITECTURE.md       │ Design deep-dive with diagrams   │ The blueprint — how everything   │
│                       │ Mermaid state diagrams           │ fits together architecturally    │
└───────────────────────┴──────────────────────────────────┴──────────────────────────────────┘
```

### 🎮 The Game Loop

Every "turn" in our simulation follows this pattern:

```
    ┌─────────────────────────────────┐
    │  1. Agent SEES the current       │
    │     state (observation)          │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │  2. Agent THINKS and chooses     │
    │     an action (LLM reasoning)    │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │  3. Environment PROCESSES the    │
    │     action (6-phase pipeline)    │
    │                                  │
    │     Phase 1: Validation ✅        │
    │     Phase 2: Apply Action 🎬     │
    │     Phase 3: Time Progression ⏰ │
    │     Phase 3.5: Dynamic Events 🔥│
    │     Phase 4: Cascade Check 🌊   │
    │     Phase 5: Reward Calc 📊     │
    │     Phase 6: Done Check 🏁      │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │  4. Agent gets REWARD            │
    │     (positive or negative)       │
    └──────────────┬──────────────────┘
                   │
                   ▼
            Repeat until done!
```

Let's trace through an example...

### 🎬 Example: One Complete Turn

**Scenario:** The agent is managing 13 servers in Hard mode. There's a critical, actively exploited vulnerability on the database server.

**Step 1: Agent SEES the current state**

The environment sends this JSON observation to the agent:

```
📊 Current Situation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERVERS:
• db-primary-01   [ONLINE]  Tier 1 (Critical)   🔥 EXPLOIT!
• db-replica-01   [ONLINE]  Tier 1 (Critical)   🔥 EXPLOIT!
• auth-server-01  [ONLINE]  Tier 1 (Critical)   🔥 EXPLOIT!
• lb-primary-01   [ONLINE]  Tier 2 (Important)
• lb-secondary-01 [ONLINE]  Tier 2 (Important)
• web-frontend-01 [ONLINE]  Tier 2 (Important)  ⚠️ Vuln
• web-frontend-02 [ONLINE]  Tier 2 (Important)  ⚠️ Vuln
• web-frontend-03 [ONLINE]  Tier 2 (Important)  ⚠️ Vuln
• app-server-01   [ONLINE]  Tier 2 (Important)  ⚠️ Vuln
• app-server-02   [ONLINE]  Tier 2 (Important)  ⚠️ Vuln
• cache-redis-01  [ONLINE]  Tier 3 (Standard)
• mq-rabbitmq-01  [ONLINE]  Tier 3 (Standard)   ⚠️ Vuln
• monitoring-01   [ONLINE]  Tier 3 (Standard)

VULNERABILITIES (5 total!):
🔴 CVE-2024-3001 (CVSS 9.8) CRITICAL 🔥 EXPLOITED
    "Remote code execution in PostgreSQL replication protocol"
    Affects: db-primary-01, db-replica-01
🔴 CVE-2024-3002 (CVSS 9.1) CRITICAL 🔥 EXPLOITED
    "Authentication bypass in Keycloak SAML parser"
    Affects: auth-server-01
🟠 CVE-2024-3003 (CVSS 8.2) HIGH
    "Server-side request forgery in Nginx"
    Affects: web-frontend-01, web-frontend-02, web-frontend-03
🟠 CVE-2024-3004 (CVSS 7.5) HIGH
    "Deserialization vulnerability in Django REST framework"
    Affects: app-server-01, app-server-02
🟡 CVE-2024-3005 (CVSS 5.3) MEDIUM
    "Information disclosure in RabbitMQ management API"
    Affects: mq-rabbitmq-01

DEPENDENCIES (12 edges):
(Remember: arrow means "I NEED this server". Thick = CRASH if missing. Dotted = survive.)

• web-frontend-01 ━━► app-server-01   (hard — web CRASHES if app goes down)
• web-frontend-02 ━━► app-server-01   (hard — web CRASHES if app goes down)
• web-frontend-02 ━━► app-server-02   (hard — web CRASHES if app goes down)
• web-frontend-03 ━━► app-server-02   (hard — web CRASHES if app goes down)
• app-server-01   ━━► db-primary-01   (hard — app CRASHES if DB goes down)
• app-server-02   ━━► db-primary-01   (hard — app CRASHES if DB goes down)
• app-server-01   ━━► auth-server-01  (hard — app CRASHES if auth goes down)
• app-server-02   ━━► auth-server-01  (hard — app CRASHES if auth goes down)
• lb-primary-01   ┄┄► web-frontend-01 (soft — LB slows down, but SURVIVES)
• lb-primary-01   ┄┄► web-frontend-02 (soft — LB slows down, but SURVIVES)
• lb-secondary-01 ┄┄► web-frontend-02 (soft — LB slows down, but SURVIVES)
• lb-secondary-01 ┄┄► web-frontend-03 (soft — LB slows down, but SURVIVES)
• db-replica-01   ━━► db-primary-01   (hard — replica CRASHES if primary goes down)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Step 2: Agent THINKS**

The LLM agent analyzes this complex state:

> "There are 2 CRITICAL, actively exploited CVEs on Tier 1 servers. These are the highest priority because:
> 1. They have the highest CVSS scores (9.8 and 9.1)
> 2. They're actively being exploited (2x penalty!)
> 3. 🔥 If I don't patch them in 4 turns, they'll SPREAD to connected servers!
>
> BUT — these are Tier 1 servers. I MUST suspend them before patching.
> AND — app-server-01 and app-server-02 both depend on db-primary-01 AND auth-server-01.
> If I just suspend db-primary-01, BOTH app servers crash, which cascades to web servers!
>
> **Plan:** First suspend the app servers (which will prevent cascade), then suspend db-primary-01, then patch."

**Step 3: Agent chooses ACTION**

```json
{
    "action_type": "suspend_service",
    "target": "app-server-01",
    "reason": "Must suspend dependent before suspending its dependency db-primary-01"
}
```

**Step 4: Environment PROCESSES** (all 6 phases run)

The environment:
1. ✅ Validates: Yes, app-server-01 exists and is ONLINE
2. 🎬 Applies: Changes app-server-01 from ONLINE → SUSPENDED
3. ⏰ Time progression: No pending patches to advance
4. 🔥 Dynamic events: Checks exploit spread timers (not yet at threshold)
5. 🌊 Cascade check: No cascades (we suspended safely!)
6. 📊 Calculates reward

**Step 5: Agent gets REWARD**

```
📊 After Action:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Action: suspend_service on app-server-01
  Result: SUCCESS ✅
  Reward: -2.1 (downtime penalty + time pressure)
  Cascade Failures: 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The reward is negative because we now have a server offline (downtime penalty = 2.0 for Tier 2, plus 0.1 time pressure). But this is a **smart trade-off** — we're setting up for a safe patch!

**The agent would then continue:**
```
Turn  2: Suspend web-frontend-01    (depends on app-server-01)
Turn  3: Suspend web-frontend-02    (depends on both app servers)
Turn  4: Suspend app-server-02      (other dependent of db-primary-01)
Turn  5: Suspend web-frontend-03    (depends on app-server-02)
Turn  6: Suspend db-primary-01      (now safe to suspend!)
Turn  7: Apply patch CVE-2024-3001 on db-primary-01
Turn  8: NOOP (wait for patch to complete)
Turn  9: Apply patch CVE-2024-3001 on db-replica-01 (can patch Tier 1 ONLINE? No — suspend first!)
Turn 10: ...and so on, methodically working through all 5 CVEs
```

---

## 📖 Chapter 4: The Technical Details (Made Simple)

### 🏗️ Server Tiers Explained

Not all servers are equally important. We categorize them into three "tiers":

#### 🔴 Tier 1: CRITICAL

```
Examples:       Databases, Authentication servers, Payment processors
Why critical:   If these fail, EVERYTHING fails. A database holds all the
                data — no database means no data for anyone.
Downtime Cost:  3.0 points per turn (highest)
Crash Cost:     6.0 points per turn (3.0 × 2 crash multiplier!)
Special Rule:   You MUST suspend these before patching. No shortcuts!
```

#### 🟡 Tier 2: IMPORTANT

```
Examples:       Web servers, API servers, Application servers, Load Balancers
Why important:  These are what users interact with directly. If they fail,
                users notice immediately.
Downtime Cost:  2.0 points per turn (medium)
Crash Cost:     4.0 points per turn (2.0 × 2 crash multiplier)
Rule:           Can patch while ONLINE (no need to suspend first)
```

#### 🟢 Tier 3: STANDARD

```
Examples:       Monitoring, Logging, Development servers, Message Queues
Why less:       Nice to have, but life goes on without them temporarily.
Downtime Cost:  1.0 points per turn (lowest)
Crash Cost:     2.0 points per turn (1.0 × 2 crash multiplier)
Rule:           Can patch while ONLINE
```

### 🎬 Server States — The Complete State Machine

A server can be in one of **five states**:

```
┌──────────┬──────┬──────────────────────────────┬────────────────────┐
│ State    │ Icon │ What It Means                │ Can It Be Patched? │
├──────────┼──────┼──────────────────────────────┼────────────────────┤
│ ONLINE   │ 🟢   │ Working normally              │ Yes (Tier 2-3)     │
│ SUSPENDED│ 🟡   │ Safely paused (controlled)    │ Yes (Tier 1)       │
│ PATCHING │ 🔵   │ Currently being patched       │ No (wait 1 turn)   │
│ CRASHED  │ 🔴   │ Failed due to cascade/stress  │ No (must resume)   │
│ OFFLINE  │ ⚪   │ Intentionally shut down       │ No                 │
└──────────┴──────┴──────────────────────────────┴────────────────────┘
```

**State Transitions (how states change):**

A "state transition" just means "one state turning into another." Think of it like a traffic light changing colors. Here are ALL the ways a server's state can change:

```
  HOW A SERVER'S STATE CHANGES — Read Each Arrow as "can become"
  ──────────────────────────────────────────────────────

     🟢 ONLINE (working normally)
        │
        ├─── "suspend_service" action ───► 🟡 SUSPENDED (safely paused)
        │                                    │
        ├─── "apply_patch" (Tier 2-3) ───► 🔵 PATCHING (being fixed)
        │                                    │
        │                                    └─── (1 turn later, automatically) ─► 🟢 ONLINE
        │
        └─── cascade failure / stress ────► 🔴 CRASHED (broken!)

     🟡 SUSPENDED (safely paused)
        │
        ├─── "apply_patch" (Tier 1) ────► 🔵 PATCHING (being fixed)
        │                                    │
        │                                    └─── (1 turn later, automatically) ─► 🟢 ONLINE
        │
        └─── "resume_service" action ───► 🟢 ONLINE (back to work!)

     🔴 CRASHED (broken!)
        │
        └─── "resume_service" action ───► 🟢 ONLINE (recovered!)


  READING EXAMPLE:
  ────────────────
  Start: Server is 🟢 ONLINE
  You run: "suspend_service"
  Now:    Server is 🟡 SUSPENDED
  You run: "apply_patch" (with a CVE ID)
  Now:    Server is 🔵 PATCHING
  Wait 1 turn...
  Now:    Server is 🟢 ONLINE again, and the vulnerability is FIXED! 🎉
```

### 🔢 The Actions Explained

The agent can do exactly **5 things** each turn:

#### 🔍 1. SCAN_HOST

```
What it does:   Get detailed information about a server
When to use:    When you want to know more before acting
Effect:         No state change, just information gathering
Cost:           Free (no penalty, but wastes a turn)
```

**Example:**
```json
{"action_type": "scan_host", "target": "db-primary-01"}
```

**Response message:** `"Scan of db-primary-01: state=online, tier=1, vulns=['CVE-2024-3001']"`

#### ⏸️ 2. SUSPEND_SERVICE

```
What it does:   Safely pause a server (like putting it to sleep)
When to use:    Before patching a Tier 1 server, or to prevent cascade
Effect:         Server changes from ONLINE → SUSPENDED
Cost:           Downtime penalty begins for this node
```

**Example:**
```json
{"action_type": "suspend_service", "target": "app-server-01"}
```

#### 🩹 3. APPLY_PATCH

```
What it does:   Fix a vulnerability on a server
When to use:    When you want to eliminate a security risk
Effect:         Server → PATCHING state, returns to ONLINE next turn
Requirement:    Tier 1 servers MUST be SUSPENDED first!
                Target must have the specified CVE
```

**Example:**
```json
{
    "action_type": "apply_patch",
    "target": "db-primary-01",
    "cve_id": "CVE-2024-3001"
}
```

#### ▶️ 4. RESUME_SERVICE

```
What it does:   Bring a paused or crashed server back online
When to use:    After patching, or to recover from a crash
Effect:         Server changes from SUSPENDED/CRASHED → ONLINE
```

**Example:**
```json
{"action_type": "resume_service", "target": "app-server-01"}
```

#### 💤 5. NOOP (No Operation)

```
What it does:   Skip this turn, do nothing
When to use:    When you need to wait for a patch to complete
Effect:         Time passes, no state change
Cost:           -0.1 time pressure penalty (every turn costs something!)
```

**Example:**
```json
{"action_type": "noop", "target": ""}
```

---

## 📖 Chapter 5: The 5-Level Difficulty Curriculum 🆕

This is one of the biggest upgrades from v1.0! Instead of 3 difficulty levels, we now have **5 levels** forming a complete training curriculum — like levels in a video game:

### 🟢 Level 1: Easy (Tutorial)

```
🎯 Purpose:       Learn basic patching without dependencies
📊 Complexity:     3-5 nodes, NO dependencies, 1 vulnerability
⏱️ Max Turns:      30
🎓 What It Tests:  Can the agent identify a vulnerability and patch it?
💡 Strategy:       Just find the CVE and apply_patch. Simple!

Example Scenario:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ web-svr-01  │     │ api-svr-01  │     │ dev-svr-01  │
│  ONLINE 🟢   │     │  ONLINE 🟢   │     │  ONLINE 🟢   │
│  Tier 2     │     │  Tier 2     │     │  Tier 3     │
│  ⚠️ CVE-1001│     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘

No arrows = No dependencies = No cascade risk!
Just patch CVE-1001 on web-svr-01 and you win!
```

### 🟡 Level 2: Medium (Dependency Awareness)

```
🎯 Purpose:       Learn the suspend-patch-resume workflow
📊 Complexity:     5-8 nodes, LINEAR dependency chain, 2 vulnerabilities
⏱️ Max Turns:      50
🎓 What It Tests:  Can the agent respect Tier 1 suspend rules?
💡 Strategy:       Suspend dependents → Suspend Tier 1 → Patch → Resume

Key Challenge: One vulnerability is on the db-primary-01 (Tier 1)!
You MUST suspend it before patching, which means suspending
everything that depends on it first.

Dependency Chain (read left to right: each one NEEDS the next):

  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
  │ web-frontend │ ━━━►  │  app-server  │ ━━━►  │ db-primary-01│
  │   Tier 2     │       │   Tier 2     │       │   Tier 1     │
  └──────────────┘       └──────────────┘       └──────────────┘
  "I need the"          "I need the"          "⚠️ HAS A VULN!"
  "app server"          "database"            "MUST suspend"
                                              "before patching!"

  The challenge: You can't just patch db-primary-01 directly!
  If you suspend it, app-server CRASHES (hard dependency).
  If app-server crashes, web-frontend CRASHES too (cascade!).
  Solution: Suspend web first, then app, THEN db. Reverse dominos!
```

### 🔴 Level 3: Hard (Multi-Objective Optimization)

```
🎯 Purpose:       Master complex graphs with cascading risks
📊 Complexity:     10-15 nodes, COMPLEX graph, 5 vulnerabilities
                  (2 actively exploited!)
⏱️ Max Turns:      100
🎓 What It Tests:  Can the agent prioritize correctly under pressure?
💡 Strategy:       Prioritize exploit_in_wild CVEs, manage multi-path
                  dependencies, balance risk vs downtime
```

**What makes Hard mode SO hard? Let's visualize it:**

```
  The network is no longer a simple chain. It's a WEB of connections:

  ┌────────────┐  ┌────────────┐
  │ lb-pri-01  │  │ lb-sec-01  │  ◄─ Load Balancers (Tier 2)
  └────┬───────┘  └────┬───────┘     These have SOFT arrows down.
       │┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│          If web servers go down,
       ▼              ▼          ▼          LBs survive.
  ┌───────────┐ ┌───────────┐ ┌───────────┐
  │web-fe-01 │ │web-fe-02 │ │web-fe-03 │  ◄─ Web Frontends (Tier 2)
  │ ⚠️ CVE-3  │ │ ⚠️ CVE-3  │ │ ⚠️ CVE-3  │     All 3 share the same vuln!
  └────┬──────┘ └────┬──────┘ └────┬──────┘     These have HARD arrows down.
       │━━━━━━━━━━━━━━━━━━━│━━━━━━━━━┘     If app goes down, webs CRASH.
       ▼                  ▼
  ┌───────────┐   ┌───────────┐
  │app-svr-01│   │app-svr-02│          ◄─ App Servers (Tier 2)
  │ ⚠️ CVE-4  │   │ ⚠️ CVE-4  │             Both share the same vuln!
  └──┬───┬────┘   └──┬───┬────┘             HARD arrows down to BOTH
     │     │           │     │               db-primary AND auth.
     │  ━━━┘━━━━━━━━━━━┘     │
     ▼     ▼                  ▼
  ┌──────────┐  ┌───────────┐
  │db-pri-01│  │auth-svr-01│           ◄─ Critical Tier 1 servers!
  │🔥 CVE-1  │  │🔥 CVE-2   │              BOTH are actively exploited!
  └────┬─────┘  └───────────┘              🔥 = exploit_in_wild = true
       │━━━━━━━━━━━━━━                    = 2x penalty!
       ▼                                    = will SPREAD to neighbors!
  ┌──────────┐
  │db-rep-01│                              ◄─ Also Tier 1
  │🔥 CVE-1  │                                 Shares CVE with primary
  └──────────┘

  The dilemma: You need to patch db-pri-01 and auth-svr-01 FIRST
  (they're exploited!), but they're at the BOTTOM of the graph.
  Everything above them depends on them!
  You must work TOP-DOWN to suspend, then BOTTOM-UP to patch.
```

### 🟣 Level 4: Incident Response (Active Breach!) 🆕

```
🎯 Purpose:       Triage an active breach under degraded conditions
📊 Complexity:     8 nodes (2 ALREADY CRASHED!), 3 vulnerabilities
                  (2 actively exploited), exploit spreading
⏱️ Max Turns:      60
🎓 What It Tests:  Can the agent recover AND patch simultaneously?
💡 Strategy:       Triage → Recover crashed nodes → Contain exploit →
                  Patch remaining CVEs → Stabilize
```

**You don't start clean — you start in the MIDDLE of a disaster!**

```
  The breach has already happened. Here's what the agent sees on Turn 0:

  ┌──────────────┐    ┌──────────────┐
  │ db-primary-01│    │ app-server-01│
  │  🔴 CRASHED   │    │  🔴 CRASHED   │    Two servers are already DOWN!
  │  (breach     │    │  (cascaded   │    The hackers got in through the DB,
  │   entry!)    │    │   from DB)   │    and the app server crashed because
  └──────────────┘    └──────────────┘    it NEEDS the DB (hard dependency).

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │auth-server-01│    │ app-server-02│    │web-frontend-01│
  │  🟢 ONLINE    │    │  🟢 ONLINE    │    │  🟢 ONLINE    │
  │  🔥 EXPLOITED!│    │              │    │              │
  └──────────────┘    └──────────────┘    └──────────────┘
     │                                        │
     │  The auth server is ONLINE but being     │
     │  actively exploited RIGHT NOW!            │
     │  • 3% chance per turn it randomly crashes  │
     │  • Exploit will SPREAD in 4 turns           │
     │                                             │
     └──── The agent's dilemma: ────────────────┘

     Do you RECOVER the crashed DB first?  (Fixes cascades but DB is
                                            still vulnerable!)
     Or PATCH the auth server first?       (Stops the active exploit
                                            but DB is still dead!)

     There's no perfect answer — THIS is what makes IR hard.
```

Special rules for this level:
- The 2 initial crashes don't count against your Safety score
- Exploit spreading is ACTIVE from Turn 0
- 3% stochastic degradation means any exploited node might randomly crash

### ⚫ Level 5: Zero-Day Cascade (Dynamic Threats!) 🆕

```
🎯 Purpose:       Adapt strategy when new threats appear mid-operation
📊 Complexity:     10 nodes, 2 initial + 2 DYNAMICALLY INJECTED vulns
⏱️ Max Turns:      80
🎓 What It Tests:  Can the agent reprioritize when the rules change?
💡 Strategy:       Start with a plan → Adapt at Turn 5 → Adapt again
                  at Turn 15. Stay flexible!

Dynamic Events:
  🚨 Turn 5:  CVE-2024-5099 APPEARS! (CVSS 9.9, CRITICAL, EXPLOITED!)
              Affects: auth-server-01 AND db-primary-01
              "ZERO-DAY: Critical authentication bypass!"
              → Agent MUST drop everything and deal with this!

  ⚠️ Turn 15: CVE-2024-5100 APPEARS! (CVSS 8.4, HIGH)
              Affects: web-frontend-01, web-frontend-02, api-gateway-01
              "HTTP request smuggling in reverse proxy"
              → Agent must integrate this into remaining plan

The agent starts with a plan for 2 manageable CVEs...
then gets hit with a CRITICAL zero-day at turn 5...
then gets hit again at turn 15!
This tests adaptability, not just planning.
```

---

## 📖 Chapter 6: How We Calculate Rewards — The Complete Math

This is the heart of how the AI learns! Let's understand the reward system step by step.

### 📊 The Basic Idea

Every turn, the environment calculates a **"penalty"** score. Think of it as "how bad is the current situation?"

```
Total Penalty = Risk Penalty + Downtime Penalty
```

- **Risk Penalty** = Danger from unpatched vulnerabilities on ONLINE servers
- **Downtime Penalty** = Cost of servers being offline

### 🎯 The Reward Formula

The agent's reward each turn is:

```
Reward = (Last Turn's Penalty) - (This Turn's Penalty) + Time Pressure

Where Time Pressure = -0.1 per turn (always negative)
```

**Translation:** "Did things get better or worse, with a small cost for each passing turn?"

```
┌────────────────────────────┬─────────────────┬──────────────┐
│ If...                      │ Penalty Change  │ Reward       │
├────────────────────────────┼─────────────────┼──────────────┤
│ You patched a vulnerability│ Decreased ↓     │ Positive! 📈 │
│ You caused a crash         │ Increased ↑     │ Negative! 📉 │
│ Nothing changed            │ Same            │ -0.1 (time)  │
│ Invalid action attempted   │ Same + penalty  │ -0.6 📉      │
└────────────────────────────┴─────────────────┴──────────────┘
```

> 💡 **Why -0.1 per turn?** This is called **time pressure**. It ensures the reward is NEVER exactly zero (dense signal). Without it, the AI might learn that doing nothing is "free" — but in reality, every turn wasted is a turn the exploit is active. This small constant creates urgency.

### 🧮 Calculating Risk Penalty (Detailed)

For each vulnerability that's still unpatched:

```
Risk = CVSS_Score × Number_of_Affected_ONLINE_Servers × Exploit_Multiplier

Where:
  Exploit_Multiplier = 2.0 if exploit_in_wild == True
  Exploit_Multiplier = 1.0 if exploit_in_wild == False
```

**Key insight:** Only ONLINE servers contribute to risk! If a server is SUSPENDED or CRASHED, hackers can't reach it, so it doesn't add risk. But it DOES add downtime penalty...

**Example (Hard mode):**

```
┌──────────────┬──────┬───────────────────┬──────────┬──────────────┐
│ Vulnerability│ CVSS │ Affected ONLINE   │ Exploited│ Risk         │
├──────────────┼──────┼───────────────────┼──────────┼──────────────┤
│ CVE-3001     │ 9.8  │ 2 (db-pri, db-rep)│ Yes (×2) │ 9.8×2×2=39.2│
│ CVE-3002     │ 9.1  │ 1 (auth-svr)      │ Yes (×2) │ 9.1×1×2=18.2│
│ CVE-3003     │ 8.2  │ 3 (web×3)         │ No  (×1) │ 8.2×3×1=24.6│
│ CVE-3004     │ 7.5  │ 2 (app×2)         │ No  (×1) │ 7.5×2×1=15.0│
│ CVE-3005     │ 5.3  │ 1 (mq)            │ No  (×1) │ 5.3×1×1= 5.3│
├──────────────┼──────┼───────────────────┼──────────┼──────────────┤
│              │      │                   │          │ TOTAL: 102.3 │
└──────────────┴──────┴───────────────────┴──────────┴──────────────┘
```

### 🧮 Calculating Downtime Penalty (Detailed)

For each server that's NOT online:

```
Downtime = Tier_Multiplier × Crash_Multiplier

Where:
  Tier_Multiplier:
    Tier 1 (Critical)  = 3.0
    Tier 2 (Important) = 2.0
    Tier 3 (Standard)  = 1.0

  Crash_Multiplier:
    CRASHED state    = 2.0 (it's uncontrolled, worse!)
    SUSPENDED state  = 1.0 (it's controlled, less bad)
    PATCHING state   = 1.0 (temporary, the patch will finish)
```

**Example:**

```
┌────────────────┬──────┬───────────┬────────────────┬─────────┐
│ Server         │ Tier │ State     │ Calculation    │ Penalty │
├────────────────┼──────┼───────────┼────────────────┼─────────┤
│ db-primary-01  │ 1    │ CRASHED   │ 3.0 × 2.0     │ 6.0     │
│ app-server-01  │ 2    │ SUSPENDED │ 2.0 × 1.0     │ 2.0     │
│ monitoring-01  │ 3    │ PATCHING  │ 1.0 × 1.0     │ 1.0     │
├────────────────┼──────┼───────────┼────────────────┼─────────┤
│                │      │           │ TOTAL:         │ 9.0     │
└────────────────┴──────┴───────────┴────────────────┴─────────┘
```

### 🎉 Terminal Bonuses and Penalties

```
┌──────────────────────────────────────────┬────────────┐
│ Event                                    │ Bonus      │
├──────────────────────────────────────────┼────────────┤
│ 🎉 VICTORY — All vulnerabilities patched │ +50.0      │
│ 💀 CATASTROPHE — All servers crashed     │ -100.0     │
│ ❌ Invalid action attempted               │ -0.5       │
│ ⏱️ Each turn that passes (time pressure)  │ -0.1       │
└──────────────────────────────────────────┴────────────┘
```

### 🎉 Putting It All Together — A Real Example

```
Turn 5 State:
  Risk Penalty:     46.7
  Downtime Penalty:  8.0
  TOTAL PENALTY:    54.7

Turn 6: Agent patches CVE-2024-3001 on both db servers
  Risk Penalty:      7.5 (the 39.2 from CVE-3001 is GONE!)
  Downtime Penalty:  8.0 (same servers still down)
  TOTAL PENALTY:    15.5

Turn 6 Reward = 54.7 - 15.5 - 0.1 = +39.1 🎉
```

The agent just got a BIG positive reward for making excellent progress!

> 💡 **Design Decision: Why "Potential-Based Reward Shaping"?**
> 
> This is a well-known technique in RL research. By giving `reward = previous_penalty - current_penalty`, we create a "dense" signal that tells the agent how much better or worse things got each turn. This is much better than "sparse" rewards (only giving +50 at the end), because the agent gets feedback EVERY turn, making learning much faster.
>
> The mathematical property that makes this safe is that potential-based shaping preserves the optimal policy — meaning the best strategy is the same whether you use shaped or unshaped rewards.

---

## 📖 Chapter 7: Dynamic Event System 🆕

One of the biggest upgrades in v2.0 is the **Dynamic Event System**. This makes our environment feel ALIVE instead of static.

### 🔥 1. Exploit Spreading

**What it is:** Actively exploited vulnerabilities can SPREAD to connected servers!

**How it works:**
1. If a CVE with `exploit_in_wild = True` remains unpatched on an ONLINE server...
2. A counter starts ticking for that CVE
3. After **4 consecutive turns** unpatched on an ONLINE node...
4. The exploit **spreads** to a randomly selected connected node (via the dependency graph)
5. The counter resets and starts again

**Let's watch it happen step by step:**

```
  Turn 1: CVE-3001 is on db-primary-01. Exploit timer starts.

  ┌──────────────┐       ┌──────────────┐
  │ db-primary-01│ ━━━━  │ app-server-01│
  │ 🔥 CVE-3001  │       │  (clean)     │
  │ timer: 1/4  │       │              │
  └──────────────┘       └──────────────┘
  (Dependency arrow shows app-server NEEDS db-primary)

  Turn 2: Still not patched... timer: 2/4
  Turn 3: Still not patched... timer: 3/4

  Turn 4: Timer hits 4! The exploit SPREADS along the dependency graph!

  ┌──────────────┐  🔥🔥🔥  ┌──────────────┐
  │ db-primary-01│ ───►  │ app-server-01│
  │ 🔥 CVE-3001  │ SPREAD │ 🔥 CVE-3001  │  ← NOW INFECTED!
  │ timer: RESET │       │ (just got it)│
  └──────────────┘       └──────────────┘

  🚨 The message appears:
  "EXPLOIT SPREAD: CVE-2024-3001 has spread to app-server-01!"

  Now the agent has to patch BOTH servers instead of just one!
  And the timer resets and starts counting again...
  If still not patched 4 more turns later, it spreads AGAIN!
```

**Which tasks use this:** Hard, Incident Response, Zero-Day

### 🧊 2. Zero-Day CVE Injection

**What it is:** Brand new vulnerabilities appear out of nowhere mid-episode!

**How it works:**
- At **Turn 5**: A CRITICAL zero-day (CVSS 9.9) appears on auth-server-01 and db-primary-01
- At **Turn 15**: A HIGH severity CVE (CVSS 8.4) appears on web frontends and API gateway

The agent planned for 2 CVEs. Suddenly there are 4. **The plan must change.**

**Which tasks use this:** Zero-Day only

### ⚡ 3. Stochastic Node Degradation

**What it is:** Compromised nodes can randomly crash from internal stress!

**How it works:**
- Each turn, for every ONLINE node that has an actively exploited vulnerability...
- There's a **3% chance** the node spontaneously crashes
- This simulates real-world behavior where compromised systems are unstable
- The agent can't predict WHEN it will happen, only that it MIGHT happen

```
⚠️ ALERT: db-primary-01 crashed due to exploit-induced stress!
```

**Which tasks use this:** Hard, Incident Response

> 💡 **Why these mechanics matter:**
> Static environments let agents **memorize** solutions. If the same CVE is always on the same server, the agent just learns a script. Dynamic events force **generalization** — the agent must actually understand the principles, not just memorize a sequence.

---

## 📖 Chapter 8: The Multi-Dimensional Grading System 🆕

This is one of our proudest achievements. Instead of a single score, we evaluate agents across **4 independent dimensions** — like Olympic figure skating!

### 🏅 The Four Scoring Dimensions

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  📊 COMPOSITE SCORE = w₁×Completion + w₂×Efficiency             │
│                     + w₃×Safety    + w₄×Strategy                │
│                                                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐│
│  │ COMPLETION  │ │ EFFICIENCY  │ │   SAFETY    │ │  STRATEGY  ││
│  │ "Did you    │ │ "How fast   │ │ "Did you    │ │ "Were your ││
│  │  finish?"   │ │  were you?" │ │  break it?" │ │  choices   ││
│  │             │ │             │ │             │ │  smart?"   ││
│  │ 0%──────100%│ │ 0%──────100%│ │ 0%──────100%│ │ 0%────100% ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### 📋 Dimension 1: Completion (default weight: 40%)

**"How many vulnerabilities did you patch?"**

- Full credit (1.0): All CVEs patched — complete victory
- Partial credit: Proportional to fraction of CVEs patched
- Formula: `1.0 - (remaining_vulns / initial_vulns)`

#### ⚡ Dimension 2: Efficiency (default weight: 20%)

**"How quickly did you complete the task?"**

- Full credit (1.0): Completed in optimal number of steps
- Linear decay: Score drops as steps increase beyond optimal
- Non-completing agents get base score of 0.1
- Each task has a **theoretical optimal step count:**
  - Easy: 3 steps, Medium: 8, Hard: 18, IR: 12, Zero-Day: 15

#### 🛡️ Dimension 3: Safety (default weight: 20%)

**"Did you cause any cascade failures?"**

- Full credit (1.0): Zero cascade failures
- Penalty proportional to cascades vs. total nodes
- Catastrophic failure (all crashed): 0.0
- For Incident Response: Initial 2 crashes don't count (they're part of the scenario)

#### 🧠 Dimension 4: Strategy (default weight: 20%)

**"Were your decisions intelligent?"**

- Did you prioritize `exploit_in_wild` CVEs first? (+0.25 bonus)
- Did you suspend dependents before dependencies? (+0.15 bonus)
- What percentage of your actions were valid? (+up to 0.10)
- Base score: 0.5

### 🎛️ Task-Specific Weight Profiles

Different tasks emphasize different dimensions:

```
┌─────────────────────┬────────────┬────────────┬────────┬──────────┐
│ Task                │ Completion │ Efficiency │ Safety │ Strategy │
├─────────────────────┼────────────┼────────────┼────────┼──────────┤
│ Easy                │    40%     │    20%     │  20%   │   20%    │
│ Medium              │    40%     │    20%     │  20%   │   20%    │
│ Hard                │    40%     │    20%     │  20%   │   20%    │
│ Incident Response   │    30%     │    15%     │  35%   │   20%    │
│ Zero-Day            │    35%     │    30%     │  15%   │   20%    │
└─────────────────────┴────────────┴────────────┴────────┴──────────┘
```

**Why different weights?**
- **Incident Response** uses **SAFETY-FOCUSED** weights (35% safety) because during an active breach, preventing further damage is more important than speed
- **Zero-Day** uses **EFFICIENCY-FOCUSED** weights (30% efficiency) because adapting quickly to new threats is the key skill being tested

### 📊 What Different Scores Mean

```
Score ~1.0:  Perfect agent — patches everything, fast, no crashes, smart
Score ~0.7:  Good agent — patches everything but slowly or with minor issues
Score ~0.5:  Okay agent — partial patching, some mistakes
Score ~0.3:  Poor agent — lots of crashes, incomplete patching
Score ~0.1:  Random agent — basically clicking buttons randomly
Score ~0.0:  Catastrophic — crashed everything
```

---

## 📖 Chapter 9: The Step Processing Pipeline — Deep Dive

Let's trace EXACTLY what happens inside `environment.step(action)`. This is the engine room.

### The 6-Phase Pipeline

Each call to `env.step(action)` executes these phases in **strict order**.

Think of it like an assembly line in a factory — the product (your action) moves through 6 stations, one at a time, and each station does its specific job:

```
  STATION 1                    STATION 2                    STATION 3
  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
  │  ✅ VALIDATION    │        │  🎬 APPLY ACTION │        │  ⏰ TIME          │
  │                  │        │                  │        │    PROGRESSION   │
  │ "Is this action  │        │ "Do the thing   │        │ "Advance the    │
  │  even legal?"    │  ──►   │  the agent      │  ──►   │  clock. Finish  │
  │                  │        │  asked for."    │        │  pending patches"│
  └─────────────────┘        └─────────────────┘        └─────────────────┘
                                                                  │
          ┌─────────────────────────────────────────────────────┘
          ▼
  STATION 3.5                  STATION 4                    STATION 5 & 6
  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
  │  🔥 DYNAMIC      │        │  🌊 CASCADE     │        │  📊 REWARD +    │
  │    EVENTS       │        │    CHECK       │        │  🏁 DONE CHECK  │
  │                  │        │                  │        │                  │
  │ "Did exploits   │        │ "Did anything  │        │ "Calculate      │
  │  spread? Any    │  ──►   │  break? Check  │  ──►   │  the score.     │
  │  zero-days?     │        │  ALL hard      │        │  Is the game    │
  │  Random crash?" │        │  dependencies."│        │  over?"         │
  └─────────────────┘        └─────────────────┘        └─────────────────┘
```

**What each station does in detail:**

```
  Phase 1: VALIDATION  ✅
  ────────────────────
  "Is the action even legal?"
    • Does the target server exist?
    • Is the CVE ID valid (for apply_patch)?
    • Is the server in the correct state for this action?
    • For Tier 1 apply_patch: Is it SUSPENDED?
    → Valid → continue to Phase 2
    → Invalid → apply -0.5 penalty, skip to Phase 3

  Phase 2: ACTION APPLICATION  🎬
  ───────────────────────────
  "Do the thing the agent asked for."
    • NOOP: Do nothing
    • SCAN_HOST: Gather info, add message
    • SUSPEND_SERVICE: node goes 🟢→🟡
    • RESUME_SERVICE: node goes 🟡/🔴→🟢
    • APPLY_PATCH: node goes 🟢/🟡→🔵, timer starts

  Phase 3: TIME PROGRESSION  ⏰
  ───────────────────────
  "Advance the clock. Finish pending patches."
    For each 🔵 PATCHING node:
      • Countdown timer decreases by 1
      • If timer hits 0:
          → node goes 🔵→🟢 (back online!)
          → CVE is removed from that server
          → If no servers have that CVE anymore, it's FULLY RESOLVED! 🎉

  Phase 3.5: DYNAMIC EVENTS  🔥 (New in v2.0!)
  ───────────────────────────
  "The world is alive — things happen whether you're ready or not!"
    • Exploit Spreading: unpatched exploited CVE for 4+ turns? → SPREADS!
    • Zero-Day Injection: On turns 5 and 15, new CVEs appear!
    • Stochastic Degradation: 3% chance per exploited 🟢 node to crash

  Phase 4: DEPENDENCY CASCADE  🌊
  ────────────────────────
  "Check ALL hard dependency arrows. Did anything break?"
    REPEAT until nothing changes:
      For each HARD dependency arrow (A ━━► B):
        If B is DOWN (crashed/suspended/offline)
        AND A is still 🟢 ONLINE
        → A CRASHES! 💥 (increment cascade counter)
    This loop catches MULTI-LEVEL cascades:
    DB crashes → App crashes → Web crashes (all in one phase!)

  Phase 5: HEALTH & REWARD  📊
  ──────────────────────
  "How much better or worse did things get?"
    reward = (last turn's penalty) - (this turn's penalty) - 0.1

  Phase 6: TERMINATION CHECK  🏁
  ─────────────────────────
  "Is the game over?"
    • All CVEs patched? → VICTORY! (+50 bonus) 🎉
    • All nodes crashed? → CATASTROPHE! (-100 penalty) 💠
    • Turn limit reached? → Timeout (truncated)
    • Otherwise → Continue to next turn
```

> 💡 **Why does the ORDER of stations matter?**
>
> Phase 3 (time progression) must happen BEFORE Phase 4 (cascade check) because a completed patch brings a server back 🟢 ONLINE, which changes which dependencies are satisfied.
>
> Phase 3.5 (dynamic events) happens before Phase 4 (cascade check) because a newly infected or crashed server affects the dependency graph.
>
> Think of it this way: first we update the world, THEN we check if anything broke. Not the other way around.

---

## 📖 Chapter 10: The AI Agent (inference.py) — Deep Dive 🆕

Let's understand how our LLM-powered agent actually works!

### 🧠 The Architecture

Here's what happens when the AI agent makes a decision. Follow the numbered arrows:

```
  inference.py — "The AI Brain"
  ─────────────────────────────

  ① The environment gives us the current situation (JSON observation)
     │
     ▼
  ┌───────────────┐
  │  Observation   │  "Here are all the servers, vulnerabilities,
  │  (JSON data)   │   dependencies, and health metrics."
  └───────┬───────┘
          │
  ② We combine it with instructions for the AI
          │
          ▼
  ┌───────────────┐
  │ System Prompt  │  "You are an expert SOC engineer.
  │ + Observation  │   Here's the situation. What do you do?
  │ (combined)     │   Respond with ONLY a JSON action."
  └───────┬───────┘
          │
  ③ We send this to the LLM (via internet API call)
          │
          ▼
  ┌───────────────┐
  │   LLM API      │  Qwen/Qwen2.5-72B-Instruct
  │  (HuggingFace) │  (a very smart AI model)
  └───────┬───────┘
          │
  ④ The LLM responds with a JSON action
          │
          ▼
  ┌───────────────┐
  │  Parse JSON    │  Extract the action from the LLM's response
  │  response      │  (with error handling if it's malformed)
  └───────┬───────┘
          │
  ⑤ Create a proper Action object and send it to the environment
          │
          ▼
  ┌───────────────┐
  │ PatchCascade  │
  │ Action object │ ───► environment.step()  ──► back to ①!
  └───────────────┘


  What if something goes wrong? We have 4 safety nets:
  ─────────────────────────────────────────
  • LLM gives bad JSON?     → Retry up to 3 times
  • API connection fails?    → Retry with longer wait
  • All retries exhausted?   → Fallback to NOOP (safe do-nothing)
  • Any unexpected error?    → Script NEVER crashes (exit code = 0)
     (This is critical for the hackathon validator!)
```

### 📝 The System Prompt

The agent gets a carefully crafted system prompt that tells it:

1. **Who it is:** "You are an expert SOC engineer"
2. **What to do:** Patch vulnerabilities while minimizing downtime
3. **Critical rules:** Tier 1 must be suspended, dependency ordering matters
4. **Available actions:** All 5 action types with descriptions
5. **Output format:** Must respond with ONLY a valid JSON object

### 🔄 The Inference Loop

```python
# Pseudocode of the inference loop

for each task_level in ["easy", "medium", "hard", "incident_response", "zero_day"]:
    print("[START] task={level}")

    observation = env.reset(task_level)

    while not done and step < MAX_STEPS:
        # 1. Send observation to LLM
        action = await get_llm_action(observation)

        # 2. Execute in environment
        result = env.step(action)

        # 3. Track everything
        observation = result.observation
        reward = result.reward
        done = result.done

        print("[STEP] step={n} action={json} reward={r} done={done}")

    # 4. Compute normalized score
    score = normalize(sum(rewards))  # Map to (0, 1)

    print("[END] success={bool} steps={n} score={s}")
```

### 📊 Output Format (Hackathon Standard)

The inference script produces output in a **strict format** required by the hackathon validator:

```
[START] task=easy env=patchcascade model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"scan_host","target":"web-server-01"} reward=-0.10 done=false error=null
[STEP] step=2 action={"action_type":"apply_patch","target":"web-server-01","cve_id":"CVE-2024-1001"} reward=5.50 done=false error=null
[STEP] step=3 action={"action_type":"noop","target":""} reward=55.10 done=true error=null
[END] success=true steps=3 score=0.872 rewards=-0.10,5.50,55.10
```

---

## 📖 Chapter 11: The Technology Stack

Let's understand every tool and technology we used!

### 🐍 Python 3.11

**What it is:** A programming language known for being easy to read and write.
**Why we use it:** #1 language for AI/ML projects, beginner-friendly, and our target runtime.

```python
# Python code looks almost like English!
if server.state == NodeState.CRASHED:
    print("Oh no, the server crashed!")
    server.state = NodeState.ONLINE  # Resume it
```

### 📦 Pydantic v2

**What it is:** A Python library for defining data structures with built-in validation.
**Why it's critical for us:** It validates ALL data flowing through the system.

```python
class ServerNode(BaseModel):
    hostname: str = Field(..., min_length=1, max_length=64)
    tier: CriticalityTier  # Must be 1, 2, or 3
    state: NodeState  # Must be one of 5 allowed states
    # If someone tries tier="banana" → Pydantic says "No!"
```

**Bonus:** Pydantic models include `Field(description=...)` annotations. These descriptions are embedded in the JSON schema, which means LLM agents can **read the schema and understand what each field means** — enabling zero-shot performance!

### ⚡ FastAPI

**What it is:** A modern Python framework for building web APIs.
**Simple analogy:** FastAPI is like a restaurant waiter.

Our endpoints (the full menu):

```
┌────────────────────────┬──────────────────────────────────────────┐
│ Endpoint               │ What It Does                             │
├────────────────────────┼──────────────────────────────────────────┤
│ GET  /                 │ Root info with all endpoint URLs         │
│ GET  /health           │ Health check (for Docker/k8s)            │
│ POST /reset            │ Start a new episode                      │
│ POST /step             │ Take one action                          │
│ GET  /observation      │ See current state without advancing      │
│ GET  /state            │ Internal state (for debugging)           │
│ GET  /render           │ ASCII art visualization                  │
│ GET  /schema/action    │ JSON Schema for actions                  │
│ GET  /schema/observation│ JSON Schema for observations            │
│ GET  /schema           │ Combined schemas                         │
│ GET  /tasks            │ List all 5 tasks with graders            │
│ GET  /tasks/{id}       │ Get specific task details                │
│ GET  /graders          │ List all 5 graders                       │
│ POST /grade/{task_id}  │ Grade an episode                         │
│ GET  /metadata         │ Full OpenEnv metadata                    │
│ GET  /docs             │ Auto-generated API documentation         │
│ GET  /redoc            │ Alternative API documentation            │
└────────────────────────┴──────────────────────────────────────────┘
```

### 🚀 Uvicorn

**What it is:** A lightning-fast ASGI server that runs FastAPI.
**Simple analogy:** If FastAPI is the waiter, Uvicorn is the restaurant manager.

### 🐳 Docker

**What it is:** A way to package your application so it runs the same everywhere.

Our Dockerfile includes:
- Python 3.11 slim base image
- All project files and dependencies
- **Health check** built in (`curl /health` every 30 seconds)
- Environment variables for consistent behavior

### 🤖 OpenAI-Compatible LLM API

**What it is:** We use the OpenAI client library to talk to any compatible LLM.
**Our default model:** `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router
**Why this model:** Open-source, powerful, fast, and free via HF Inference API

### 🤗 Hugging Face Spaces

**What it is:** A platform to host and share AI projects.
**Our Space URL:** https://ayush-kumar0207-patchcascade-soc.hf.space

### 📦 httpx

**What it is:** A modern async HTTP client for Python.
**Why we use it:** For our 3-tier client architecture (async, sync, and local).

### 🧪 pytest

**What it is:** Python testing framework.
**Our test suite:** 5 test files covering environment, models, grader, and server.

---

## 📖 Chapter 12: The Complete Code Walkthrough

Let's trace EXACTLY what happens when the agent takes an action, from request to response!

### Step 1: Agent Sends Request

The agent (our AI via inference.py) decides to apply a patch. It sends:

```
POST /step
Content-Type: application/json

{
    "action_type": "apply_patch",
    "target": "db-primary-01",
    "cve_id": "CVE-2024-3001",
    "reason": "Patching critical exploited CVE on primary database"
}
```

### Step 2: FastAPI Receives and Parses

`_server.py` receives the request:

```python
@app.post("/step")
async def step_environment(request: StepRequest):
    # Pydantic validates the request body automatically
    action_type = ActionType(request.action_type)  # Validates enum

    action = PatchCascadeAction(
        action_type=action_type,
        target=request.target,        # "db-primary-01"
        cve_id=request.cve_id,        # "CVE-2024-3001"
        reason=request.reason,        # "Patching critical..."
    )

    result = env.step(action)
    return StepResponse(observation=..., reward=..., done=..., info=...)
```

### Step 3: Environment Processes (The 6 Phases in Action)

```python
# Phase 1: Validation ✅
is_valid, error = validate_action_for_observation(action, obs)
# Checks: Does "db-primary-01" exist? ✓
# Is "CVE-2024-3001" a real CVE? ✓
# Is it a Tier 1 node? ✓ → Is it SUSPENDED? ✓ (we suspended it earlier)
# Result: VALID

# Phase 2: Apply Action 🎬
node = get_node("db-primary-01")
node.state = NodeState.PATCHING
node.patch_turns_remaining = 1
pending_patches["db-primary-01"] = "CVE-2024-3001"

# Phase 3: Time Progression ⏰
for node in all_nodes:
    if node.state == PATCHING and node.patch_turns_remaining > 0:
        node.patch_turns_remaining -= 1
        if node.patch_turns_remaining == 0:
            node.state = ONLINE  # Patch complete!
            remove "db-primary-01" from CVE-2024-3001.affected_hosts

# Phase 3.5: Dynamic Events 🔥
# Check exploit spread timers (CVE-3002 on auth-server is ticking...)
# Check zero-day injection triggers (not zero_day mode)
# Check stochastic degradation (3% chance per exploited node)

# Phase 4: Cascade Check 🌊
# db-primary-01 just came back ONLINE from patching
# No dependencies are violated → 0 cascades

# Phase 5: Reward Calculation 📊
current_penalty = risk_penalty + downtime_penalty
reward = last_penalty - current_penalty - 0.1

# Phase 6: Termination Check 🏁
# Are all CVEs patched? Not yet (4 more to go)
# All nodes crashed? No
# Turn limit? No → Continue
```

### Step 4: Response Sent Back

```json
{
    "observation": {
        "nodes": [
            {"hostname": "db-primary-01", "state": "online", "tier": 1, ...},
            ...
        ],
        "vulnerabilities": [
            {"cve_id": "CVE-2024-3001", "affected_hosts": ["db-replica-01"], ...},
            ...
        ],
        "dependencies": [...],
        "health": {
            "total_nodes": 13,
            "nodes_online": 11,
            "nodes_crashed": 0,
            "active_critical_vulns": 1,
            ...
        },
        "last_action_result": "success",
        "messages": [
            "Started patching CVE-2024-3001 on db-primary-01. Will complete next turn.",
            "Patch completed: CVE-2024-3001 on db-primary-01."
        ]
    },
    "reward": 19.5,
    "done": false,
    "truncated": false,
    "info": {
        "valid": true,
        "cascade_failures": 0,
        "total_cascade_failures": 0,
        "invalid_actions": 0
    }
}
```

---

## 📖 Chapter 13: Testing & Validation Pipeline 🆕

We take quality seriously. Here's our complete testing story:

### 🧪 Test Suite (4 Test Files)

```
tests/
├── conftest.py           # Shared test fixtures (pre-built environments & actions)
├── test_environment.py   # 18KB — Tests all 5 scenarios, step pipeline, cascades
├── test_grader.py        # 11KB — Tests all 5 graders, score ranges, edge cases
├── test_models.py        # 10KB — Tests all Pydantic models, validation, enums
└── test_server.py        # 11KB — Tests all API endpoints, error handling
```

### 🔥 Smoke Test (smoke_test.py)

A **self-contained validation script** that runs WITHOUT an LLM:

```
$ python smoke_test.py

======================================================================
  PatchCascade SOC — End-to-End Smoke Test
======================================================================

📋 Check 1: Task Registry
   Total tasks: 5
   Tasks with graders: 5
   ✅ PASS — 5 tasks with 5 graders

📊 Check 2: Grader Registry
   Total graders: 5
   - easy: EasyGrader (threshold=0.5)
   - medium: MediumGrader (threshold=0.6)
   - hard: HardGrader (threshold=0.7)
   - incident_response: IncidentResponseGrader (threshold=0.5)
   - zero_day: ZeroDayGrader (threshold=0.6)
   ✅ PASS — 5 graders registered

🎮 Check 3: Running all 5 tasks with heuristic agent
                easy: score=0.872 steps=  4 reward=    56.1 ✅ PASS (12ms)
              medium: score=0.756 steps= 15 reward=    38.2 ✅ PASS (23ms)
                hard: score=0.612 steps= 42 reward=   -12.4 ⚠️ PARTIAL (45ms)
   incident_response: score=0.534 steps= 28 reward=   -34.1 ⚠️ PARTIAL (31ms)
            zero_day: score=0.489 steps= 55 reward=   -52.3 ⚠️ PARTIAL (67ms)

🔍 Check 4: Score Validation
   ✅ PASS — All scores in valid range (0, 1)

📦 Check 5: Import Validation
   ✅ PASS — All modules import successfully

======================================================================
  ✅ ALL CHECKS PASSED — Submission is ready!
======================================================================
```

The heuristic agent used in the smoke test implements a simple priority-based strategy:
1. Resume crashed nodes first
2. Then patch highest-CVSS vulnerabilities
3. Handle Tier 1 suspend-patch-resume correctly
4. Resume suspended nodes after patching
5. NOOP if nothing to do

---

## 📖 Chapter 14: The Client Architecture 🆕

We provide **3 different ways** to connect to the environment:

### 1. PatchCascadeLocalClient (No network needed)

```python
# Directly wraps the environment — no HTTP, no server
client = PatchCascadeLocalClient(seed=42)
obs = client.reset(task_level="medium")
result = client.step(action)
# Perfect for: local testing, inference.py, smoke tests
```

### 2. PatchCascadeClient (Async HTTP)

```python
# Async HTTP client for production use
async with PatchCascadeClient("http://localhost:8000") as client:
    obs = await client.reset(task_level="hard")
    result = await client.step(action)
# Perfect for: async agents, high-performance scenarios
```

### 3. PatchCascadeClientSync (Sync HTTP)

```python
# Synchronous HTTP client for simple scripts
with PatchCascadeClientSync("http://localhost:8000") as client:
    obs = client.reset(task_level="easy")
    result = client.step(action)
# Perfect for: simple testing, scripts, Jupyter notebooks
```

All three share the same API surface — you can swap between them without changing your agent code!

---

## 📖 Chapter 15: ASCII Art Visualization 🆕

Our environment includes a beautiful ASCII art renderer for debugging:

```
╔════════════════════════════════════════════════════════════════════╗
║      🛡️ PatchCascade SOC — Turn 3/30 (Easy)                       ║
╠════════════════════════════════════════════════════════════════════╣
║  NETWORK TOPOLOGY                                                  ║
║                                                                    ║
║  🟢 web-server-0  [ ONLINE ] T2 ⚠️    🟢 api-server-0  [ ONLINE ] T2║
║  🟢 dev-server-0  [ ONLINE ] T3       🟢 monitoring-0  [ ONLINE ] T3║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║  VULNS: 1 active (1 HIGH)                                         ║
║  HEALTH: 4/4 online | Risk: 15.0 | Downtime: 0.0                  ║
║  REWARD: -0.30 (last: -0.10)                                      ║
╚════════════════════════════════════════════════════════════════════╝
```

Visual indicators (remember from Chapter 2!):
- 🟢 Online | 🟡 Suspended | 🔵 Patching | 🔴 Crashed | ⚪ Offline
- ⚠️ Has vulnerabilities | 🔥 Has ACTIVELY EXPLOITED vulnerabilities (2x penalty!)
- `━━►` Hard dependency ("I CRASH without you") | `┄┄►` Soft dependency ("I survive without you")

---

## 📖 Chapter 16: Common Questions Answered

### ❓ "Why not just patch everything at once?"

Because of **dependencies**! If you patch the database, everything that depends on it could crash. You need to:
1. Identify the dependency tree
2. Suspend dependents first (bottom-up)
3. Then suspend and patch the target
4. Bring everything back online (top-down)

### ❓ "Why does the AI need to learn this? Can't we just program the rules?"

The optimal strategy depends on MANY interacting factors:
- Which vulnerabilities are most urgent?
- Which are actively exploited (and might spread)?
- Which servers have the most dependencies?
- How much downtime is acceptable?
- What if a zero-day appears mid-operation?
- What if a node spontaneously crashes from stress?

These trade-offs are complex. It's easier to let the AI learn through experience than to hard-code every possible scenario.

### ❓ "What's 'OpenEnv' and why do we care?"

OpenEnv is a **standard protocol** created for the Meta PyTorch Hackathon. It defines:
- How agents and environments communicate
- What format the messages should be in
- What endpoints to expose
- How tasks and graders are registered

Following the standard means our project works with **any OpenEnv-compatible agent** — not just ours!

### ❓ "Why 5 difficulty levels instead of 3?"

```
┌─────────────────────┬──────────────────────────────────────────────────┐
│ Level               │ What It Tests                                    │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Easy                │ Basic patching mechanics                         │
│ Medium              │ Dependency awareness, suspend-patch-resume       │
│ Hard                │ Multi-objective optimization, exploit priority    │
│ Incident Response   │ Triage under pressure, recovery + patching       │
│ Zero-Day            │ Adaptive planning, strategy revision mid-flight  │
└─────────────────────┴──────────────────────────────────────────────────┘
```

This forms a **complete training curriculum.** The agent graduates from basic skills to mastering complex, dynamic, unpredictable scenarios — exactly like training real SOC analysts.

### ❓ "What makes this project special compared to others?"

1. **Not a demo — real RL infrastructure** designed for actual training
2. **5 progressive difficulty levels** forming a curriculum
3. **Multi-dimensional grading** (not just pass/fail)
4. **Dynamic events** (exploit spreading, zero-days, stochastic degradation)
5. **Dense reward shaping** with proper potential-based design
6. **LLM-native design** — JSON observations with semantic field descriptions
7. **Reproducible** via seeded random generation
8. **Comprehensive testing** — unit tests, integration tests, smoke tests
9. **3 client implementations** — local, async HTTP, sync HTTP
10. **Beautiful ASCII visualization** for debugging

### ❓ "What if the AI sends garbage?"

We handle it gracefully through **3 layers of defense:**
1. **Pydantic validates** — Invalid JSON is rejected (HTTP 400)
2. **Validation function checks** — Invalid targets/CVEs/states caught (returns detailed error)
3. **LLM retry logic** — Parse errors trigger retries with exponential backoff
4. **Graceful fallback** — If all retries fail, treat as NOOP
5. **Never crashes** — The inference script always exits with code 0

The agent gets a -0.5 penalty for invalid actions, teaching it to send valid requests.

---

## 📖 Chapter 17: Running the Project

### Option 1: Docker (Recommended)

```bash
# Build the container
docker build -t patchcascade-soc .

# Run it
docker run -p 8000:8000 patchcascade-soc

# Test it
curl http://localhost:8000/health
# Should return: {"status":"healthy","environment":"patchcascade","version":"2.0.0"}

# See all tasks
curl http://localhost:8000/tasks

# See API docs
# Open http://localhost:8000/docs in your browser
```

### Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8000

# In another terminal, run smoke test
python smoke_test.py

# Or run inference with an LLM
export HF_TOKEN=your_token_here
python inference.py
```

### Option 3: Hugging Face Space

Just visit: **https://ayush-kumar0207-patchcascade-soc.hf.space**

The environment is already running in the cloud!

### Option 4: Run Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term
```

---

## 📖 Chapter 18: Design Philosophy

### Why These Design Choices?

#### Dense Rewards Over Sparse
Sparse rewards (win/lose at episode end) make credit assignment extremely difficult. Our potential-based shaping provides feedback every turn while maintaining the same optimal policy.

#### Multi-Dimensional Grading Over Single Metric
A single normalized reward doesn't capture *how* an agent succeeded. An agent that causes 10 cascades but recovers is fundamentally different from one that avoids cascades entirely. Our 4-dimension system distinguishes these cases.

#### Dynamic Events Over Static Scenarios
Static environments allow agents to memorize solutions. Dynamic exploit spreading, stochastic degradation, and zero-day injection force generalization and adaptive planning — skills critical for real-world deployment.

#### JSON Observations Over Numeric Arrays
LLM agents natively understand JSON with semantic field names. Our Pydantic models include rich `Field(description=...)` annotations that serve as documentation directly in the schema, enabling zero-shot agent performance.

#### Reproducibility Via Seeded Randomness
Every scenario is generated using a seeded random number generator. Same seed = same scenario, every time. This is critical for fair benchmarking and debugging.

---

## 📖 Chapter 5: The Live Command Center (Dashboard)

Wait, did we mention we built a **world-class dashboard** too? 🚀

### 🖥️ Why a Dashboard?

In the old days of cybersecurity, everything happened in a black-and-white terminal window. It was hard to see the "big picture."

Our **Live Command Center** changes that. It's a premium, dark-themed web interface that lets you:
1. **WATCH the network** — See the servers move around and connect in a live "force-directed" map.
2. **SEE the damage** — Nodes pulse red when they're exploited and turn gray when they crash.
3. **FOLLOW the agent** — Watch the "Action Feed" as the AI explains its reasoning in real-time.
4. **TRACK the rewards** — See a live graph of points going up and down.

### 🎮 How to Use It

1. Start the server (using Docker or Python).
2. Open your browser to `http://localhost:8000`.
3. Pick a difficulty level (like "Hard").
4. Click **Initialize** to build the network.
5. Click **Auto-Run** and lean back.

You'll see the AI methodically patching servers, managing downtime, and avoiding cascades—all visualized with beautiful animations and neon glow effects.

> 🏆 **Hackathon Tip:** This dashboard is our secret weapon. It proves that our project isn't just "vibe coding"—it's a working, professional-grade solution that anyone can understand at a glance.

---

## 📖 Chapter 19: Glossary (Expanded)

```
┌────────────────────────┬──────────────────────────────────────────────────┐
│ Term                   │ Simple Definition                                │
├────────────────────────┼──────────────────────────────────────────────────┤
│ Agent                  │ The AI that makes decisions                      │
│ API                    │ A way for programs to talk to each other         │
│ ASCII Art              │ Pictures drawn with text characters               │
│ ASGI                   │ Standard for Python async web servers             │
│ Cascade Failure        │ When one failure causes others (like dominoes)    │
│ CORS                   │ Security rule that lets browsers make cross-site  │
│                        │   requests (we allow all origins)                 │
│ CVE                    │ Official ID for a security vulnerability          │
│ CVSS                   │ Score (0-10) showing how dangerous a vuln is      │
│ DAG                    │ A graph where arrows go one way with no loops     │
│ Dense Reward           │ Feedback every turn (not just at the end)         │
│ Dependency (Hard)      │ "I MUST have this to function" → crash if missing │
│ Dependency (Soft)      │ "I work better with this" → degrade if missing    │
│ Docker                 │ Tool to package apps so they run anywhere          │
│ Dynamic Event          │ Something that happens during the episode         │
│                        │   (exploit spread, zero-day, degradation)         │
│ Endpoint               │ A specific URL that accepts requests              │
│ Environment            │ The simulated world where the agent acts           │
│ Exploit in Wild        │ Hackers are ACTIVELY using this bug right now     │
│ Exploit Spreading      │ An active exploit infects connected servers        │
│ FastAPI                │ Python framework for building web APIs             │
│ Heuristic Agent        │ A rule-based agent that doesn't use AI            │
│ httpx                  │ Modern async HTTP client for Python               │
│ JSON                   │ A text format for sending structured data          │
│ LLM                    │ Large Language Model (like ChatGPT, Qwen)         │
│ Multi-Dimensional      │ Scoring across multiple independent dimensions     │
│ NOOP                   │ "No operation" — do nothing this turn              │
│ OpenEnv                │ The standard protocol for this hackathon           │
│ Patch                  │ A software fix for a vulnerability                 │
│ Penalty                │ Bad points (we want to minimize this)              │
│ Potential-Based Shaping│ Reward = improvement in penalty each turn          │
│ Pydantic               │ Library for validating data with Python types      │
│ Reinforcement Learning │ Teaching AI by rewards and penalties               │
│ REST API               │ A style of API using HTTP methods                  │
│ Reward                 │ Points the agent earns (positive = good)           │
│ Server                 │ A computer that provides services                  │
│ Smoke Test             │ Quick end-to-end validation without an LLM        │
│ Sparse Reward          │ Feedback only at the very end (win/lose)           │
│ State                  │ The current condition of a server                  │
│ State Machine          │ Rules for how states can change                    │
│ Stochastic Degradation │ Random chance of crash (3%) per compromised node  │
│ Tier                   │ Importance level (1 = most critical)               │
│ Time Pressure          │ -0.1 penalty per turn to incentivize speed         │
│ Truncated              │ Episode ended due to time limit (not terminal)     │
│ Uvicorn                │ Server that runs FastAPI applications              │
│ Vulnerability          │ A security weakness hackers can exploit            │
│ Zero-Day               │ A brand new vulnerability with no known fix        │
│ Zero-Day Injection     │ New CVEs appearing mid-episode dynamically         │
└────────────────────────┴──────────────────────────────────────────────────┘
```

---

## 🎉 Congratulations!

You've made it to the end of this **epic guide!** You now understand:

✅ **The Problem** — The Patching Paradox and cascade failures in real SOCs
✅ **The Solution** — A 5-level RL environment with dynamic events and multi-dimensional grading
✅ **The Concepts** — Servers, dependencies (hard vs soft), vulnerabilities, tiers, CVSS scores
✅ **The Math** — Potential-based reward shaping, risk penalty, downtime penalty, time pressure
✅ **The Architecture** — 6-phase step pipeline, 3 client implementations, ASCII renderer
✅ **The AI Agent** — LLM-powered inference with retry logic and graceful error handling
✅ **The Grading** — 4-dimension scoring (completion, efficiency, safety, strategy) with task-specific weights
✅ **The Dynamic Events** — Exploit spreading, zero-day injection, stochastic degradation
✅ **The Technology** — Python 3.11, FastAPI, Pydantic v2, Docker, OpenAI API, pytest
✅ **The Testing** — Unit tests, integration tests, smoke tests, heuristic agent validation
✅ **The Dashboard** — Real-time Force-Directed Topology, Live Action Feed, and Strategic Charts
✅ **The Code** — Every file, every function, every design decision explained

### 💡 Key Takeaways

```
 1. Patching is hard because of dependencies between systems
 2. Cascade failures happen when you patch without planning
 3. Reinforcement learning teaches AI through trial and error
 4. Dense rewards give feedback every turn (not just win/lose)
 5. The agent learns to balance security vs. uptime vs. speed
 6. Dynamic events prevent memorization and force generalization
 7. Multi-dimensional grading captures HOW an agent succeeds, not just IF
 8. 5 difficulty levels form a complete training curriculum
 9. LLM-native design (JSON + semantic descriptions) enables zero-shot play
10. OpenEnv compliance means our project works with the hackathon ecosystem
```

### 🏆 Why We're Ready for Bangalore

Our project isn't just a demo — it's **production-grade RL infrastructure**:
- **1,821 lines** of environment logic
- **832 lines** of Pydantic models with semantic descriptions
- **657 lines** of multi-dimensional grading
- **5 tasks**, **5 graders**, **3 client implementations**
- **3 dynamic event systems** (exploit spreading, zero-day, degradation)
- **Comprehensive test suite** with heuristic smoke testing
- **Premium Live Dashboard** with D3.js topology and real-time metrics
- **Full OpenEnv compliance** for seamless hackathon validation

---

### Questions?

```
Feel free to ask us anything!

Created by Ayush Kumar & Ravi Prashant
PatchCascade SOC Team
Meta PyTorch OpenEnv Hackathon 2026 — Bangalore Finals

"Train smarter. Patch faster. Crash never." 🛡️
```
