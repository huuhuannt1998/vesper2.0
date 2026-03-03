# VESPER — Virtual Environment for Smart-home Platform Evaluation & Research

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![802.11 WiFi](https://img.shields.io/badge/WiFi-mac80211__hwsim-informational.svg)](#emulated-80211-wifi-network)
[![Habitat 3.0](https://img.shields.io/badge/Habitat-3.0-orange.svg)](https://aihabitat.org/)
[![Attacks](https://img.shields.io/badge/attacks-36_unique-red.svg)](#rqsec-security-campaign)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](docker/)

> **Paper:** *VESPER: Measured IoT Network Security Through Full-Stack Smart Home Emulation with 802.11 WiFi*
>
> Submitted to ACM MobiCom 2026.

VESPER is a full-stack IoT simulation platform that bridges **virtual smart-home devices** to **real cloud platforms** (Samsung SmartThings). Each virtual device runs compiled ARM firmware inside QEMU, communicating over a **real 802.11 WiFi stack** emulated by the Linux kernel's `mac80211_hwsim` subsystem with `hostapd` and `wpa_supplicant`. The platform integrates:

- **Emulated 802.11 WiFi** — `mac80211_hwsim` + `hostapd` + `wpa_supplicant` in Linux network namespaces; supports bridge mode and full 802.11 mode with WPA2/WPA3-SAE, PMF, and AP isolation
- **Firmware-in-the-loop emulation** — real ARM Cortex-M3 firmware in QEMU Docker containers
- **LLM-driven activity generation** — GPT-OSS 20B generates daily schedules from 10 diverse personas
- **3D embodied simulation** — Habitat 3.0 with HSSD scenes and humanoid navigation
- **Bi-directional cloud sync** — Samsung SmartThings Schema Connector
- **Five-suite security framework** — 36 attacks with tshark-captured 802.11 frame evidence

```
SmartThings App (Phone)
        │
        ▼
  SmartThings Cloud ◄──── HTTPS (ngrok) ────► VESPER Schema Connector
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                 ▼                 ▼
                              ┌──────────┐     ┌──────────┐     ┌──────────┐
                              │ Docker   │     │ Docker   │     │ Docker   │
                              │ QEMU ARM │     │ QEMU ARM │     │ QEMU ARM │
                              │ Firmware │     │ Firmware │     │ Firmware │
                              └──────────┘     └──────────┘     └──────────┘
                              Kitchen Light    Living Room      Bedroom Light
                                    │                 │                 │
                              ┌─────┴─────────────────┴─────────────────┘
                              ▼
              ┌──────────── mac80211_hwsim (Linux kernel) ─────────────┐
              │                                                        │
    Bridge Mode (veth + brctl)              802.11 Mode (4 radios)
    ┌────────────────────────┐     ┌──────────────────────────────────┐
    │ brX bridge             │     │ phy0/wlan0 ─ AP (hostapd)       │
    │ ├── veth-sta0          │     │ phy1/wlan1 ─ sta0 (ns-sta0)    │
    │ └── veth-sta1          │     │ phy2/wlan2 ─ sta1 (ns-sta1)    │
    └────────────────────────┘     │ phy3/wlan3 ─ attacker          │
                                   └──────────────────────────────────┘
              │                                    │
    ┌─────────┼────────────────────────────────────┘
    ▼
tshark capture (per-namespace, per-attack)
    + Mosquitto MQTT broker
```

### Citation

If you use VESPER in your research, please cite:

```bibtex
@inproceedings{vesper2026,
  title     = {{VESPER}: Measured {IoT} Network Security Through Full-Stack
               Smart Home Emulation with 802.11 {WiFi}},
  author    = {Bui, Huan and [co-authors]},
  booktitle = {Proc.\ ACM MobiCom},
  year      = {2026},
}
```

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Emulated 802.11 WiFi Network](#emulated-80211-wifi-network)
- [Reproducing the Paper Experiments](#reproducing-the-paper-experiments)
  - [Prerequisites](#prerequisites-for-experiments)
  - [Step 1: Environment Setup](#step-1-environment-setup)
  - [Step 2: Compile Firmware & Build Docker Image](#step-2-compile-firmware--build-docker-image)
  - [Step 3: Download Datasets](#step-3-download-datasets)
  - [Step 4: Start the LLM Server](#step-4-start-the-llm-server)
  - [Step 5: Run the Autonomous Evaluation (RQ-S, RQ-H)](#step-5-run-the-autonomous-evaluation-rq-s-rq-h)
  - [Step 6: Run WiFi Network Experiments (RQ-N1, RQ-N2)](#step-6-run-wifi-network-experiments-rq-n1-rq-n2)
  - [Step 7: Run the Security Assessment (RQ-Sec)](#step-7-run-the-security-assessment-rqsec)
  - [Step 8: Generate Paper Figures](#step-8-generate-paper-figures)
- [Modes of Operation](#modes-of-operation)
- [SmartThings Setup](#smartthings-setup-optional)
- [Artifact & Reproducibility](#artifact--reproducibility)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Features

- **Emulated 802.11 WiFi** — Linux kernel `mac80211_hwsim` with `hostapd` AP and `wpa_supplicant` stations in network namespaces; supports bridge mode (veth + brctl) and full 802.11 mode (WPA2-PSK, WPA3-SAE, PMF, AP isolation); every frame traverses the real mac80211 stack
- **Real Firmware Emulation** — ARM Cortex-M3 firmware compiled with `arm-none-eabi-gcc`, running in QEMU
- **Docker-per-Device** — Each virtual IoT device is an isolated Docker container
- **6 Device Types** — Smart light, motion sensor, temperature sensor, humidity sensor, door sensor, smart plug
- **SmartThings Bi-Directional Sync** — Devices appear in the Samsung SmartThings app with real-time sync
- **LLM-Driven Activity Generation** — GPT-OSS 20B generates realistic daily schedules from 10 diverse personas
- **3D Habitat Integration** — Habitat 3.0 with HSSD scenes, humanoid navigation, and proximity-based automation
- **Five-Suite Security Framework** — 36 unique attacks (18 firmware + 14 network + 3 phantom-delay + 1 SmartApp + 1 ESP32 overflow) with self-assessed CVSS 3.1 scoring and MITRE ATT&CK mapping
- **tshark Frame Capture** — Per-attack and per-namespace captures on emulated 802.11 interfaces; all `.pcap` files independently verifiable with Wireshark
- **Automated Evaluation Pipeline** — Reproducible experiments with LaTeX table/figure generation
- **Event-Driven Architecture** — Pub/sub event bus with sub-millisecond dispatch (P99 = 7 μs)

---

## Quick Start

### Prerequisites

| Tool | Version | Install (macOS) | Install (Linux) |
|------|---------|-----------------|-----------------|
| Python | 3.9+ | `brew install python@3.9` | `sudo apt install python3.9` |
| Docker | 20+ | [docker.com](https://www.docker.com/products/docker-desktop/) | [docker.com](https://docs.docker.com/engine/install/) |
| ngrok | 3+ | `brew install ngrok` | `snap install ngrok` |
| ARM GCC | 13+ | `brew install arm-none-eabi-gcc` | `sudo apt install gcc-arm-none-eabi` |
| QEMU | 8+ | `brew install qemu` | `sudo apt install qemu-system-arm` |
| tshark | 4.0+ | `brew install wireshark` | `sudo apt install tshark` |
| Conda | — | [miniforge](https://github.com/conda-forge/miniforge) | [miniforge](https://github.com/conda-forge/miniforge) |

### 1. Clone & Install

```bash
git clone https://github.com/huuhuannt1998/vesper2.0.git
cd vesper

# Create conda environment (recommended for Habitat support)
conda create -n vesper python=3.9 cmake=3.22 -y
conda activate vesper

# Install Habitat-Sim with Bullet physics
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Install VESPER and all dependencies
pip install -e ".[all]"
pip install aiohttp
```

### 2. Compile the Firmware

```bash
cd vesper/firmware/samples
make
# → Produces 6 firmware .elf files (one per device type)
cd ../../..
```

### 3. Build the Docker Image

```bash
docker build -f docker/Dockerfile.device -t vesper-qemu-arm:latest .
```

### 4. Run a Quick Demo

```bash
# Firmware-only demo (no cloud, no 3D)
python scripts/firmware_demo.py

# 3D environment with humanoid navigation
python scripts/vesper_objectnav_camera_humanoid.py

# Full stack with SmartThings (requires ngrok + credentials)
python scripts/vesper_smartthings.py
```

---

## Emulated 802.11 WiFi Network

VESPER's networking layer uses the Linux kernel's **`mac80211_hwsim`** subsystem to provide real 802.11 protocol processing without physical radio hardware. Unlike bridge-only testbeds, every frame traverses the full `mac80211` → `cfg80211` → `nl80211` kernel path, so WiFi-layer attacks (deauthentication, evil twin, PMKID capture) produce genuine 802.11 management frames observable with `tshark`.

### Why mac80211_hwsim?

| Approach | Protocol Stack | RF Propagation | Kernel Path | Attacks |
|----------|---------------|----------------|-------------|---------|
| Linux bridge (veth) | ❌ L2 only | ❌ | Ethernet | Network only |
| Mininet-WiFi + wmediumd | ✅ | ✅ (modeled) | mac80211 | Full |
| **mac80211_hwsim (VESPER)** | **✅** | **❌ (zero-loss)** | **mac80211** | **Full** |
| Physical APs | ✅ | ✅ (real) | mac80211 | Full |

VESPER uses `mac80211_hwsim` directly: the full 802.11 authentication / association / 4-way handshake runs in the kernel, but frames are delivered with zero loss and zero propagation delay. This is sufficient for security evaluation (where protocol correctness matters) but means that absolute throughput and retransmission counts are not representative of physical deployments.

### Network Topology (4 radios)

```
modprobe mac80211_hwsim radios=4

  phy0 / wlan0 ── AP (hostapd, root namespace)
                    │ SSID: VesperNet, WPA2-PSK or WPA3-SAE
                    │
  phy1 / wlan1 ── Station 0 (wpa_supplicant, namespace ns-sta0)
  phy2 / wlan2 ── Station 1 (wpa_supplicant, namespace ns-sta1)
  phy3 / wlan3 ── Attacker   (monitor mode,   namespace ns-atk)
```

**Critical: network namespaces are required.** Without `iw phy phyN set netns name <ns>`, all interfaces share the root namespace and the kernel short-circuits routing via the loopback path — bypassing the 802.11 stack entirely. We verified this empirically: RTT drops from 0.354 ms (correct, via mac80211) to <0.01 ms (incorrect, loopback bypass) when namespaces are omitted.

### Two Operating Modes

| Mode | How It Works | When to Use |
|------|-------------|-------------|
| **Bridge** | `brctl addbr brX` + veth pairs between namespaces; frames cross a Linux bridge, not the 802.11 stack | Baseline comparison (RQ-N1), fast iteration |
| **802.11** | `hostapd` runs on phy0; stations associate via `wpa_supplicant`; all traffic traverses the full mac80211 path | WiFi-layer attack evaluation (RQ-N1, RQ-N2), hardening experiments |

### Supported WiFi Configurations (RQ-N2)

The hardening experiment (RQ-N2) sweeps 8 configurations by toggling:

| Parameter | Off | On |
|-----------|-----|-----|
| Encryption | WPA2-PSK (CCMP) | WPA3-SAE |
| PMF | Disabled | Required (`ieee80211w=2`) |
| AP Isolation | Disabled | Enabled (`ap_isolate=1`) |
| MQTT Auth | Anonymous | Username/password + TLS 1.3 |

### Tools & Versions

All WiFi experiments run on a **Linux VM** (we use [Multipass](https://multipass.run/) on macOS):

| Tool | Version | Role |
|------|---------|------|
| Linux kernel | 5.15.0-171-generic | `mac80211_hwsim` host |
| hostapd | 2.10 | Software AP |
| wpa_supplicant | 2.10 | Station authentication |
| Mosquitto | 2.0.11 | MQTT broker (with optional TLS) |
| tshark | 3.6.2 | Per-namespace frame capture |
| iperf3 | 3.9 | Throughput measurement |
| hping3 / Scapy | — | Attack injection |

---

## Reproducing the Paper Experiments

This section provides step-by-step instructions to reproduce all five research questions and the large-scale autonomous evaluation from the paper.

### Prerequisites for Experiments

- **Hardware:** Apple M2 Pro or equivalent (32 GB RAM recommended)
- **OS:** macOS 14+ (host) + Ubuntu 22.04 VM for WiFi experiments (see below)
- **Disk:** ~20 GB free (datasets + Docker images)
- **Docker:** Must be running with at least 8 GB RAM allocated
- **LMStudio:** Required for LLM-based schedule generation
- **Linux VM:** Required for RQ-N1 and RQ-N2 (WiFi experiments). We use [Multipass](https://multipass.run/):

```bash
# Create the VM (one-time setup)
brew install multipass
multipass launch 22.04 --name vesper-vm --cpus 4 --memory 8G --disk 40G

# Install WiFi dependencies inside the VM
multipass shell vesper-vm
sudo apt update && sudo apt install -y \
    hostapd wpa-supplicant mosquitto mosquitto-clients \
    tshark iperf3 iw net-tools bridge-utils hping3 \
    python3-pip python3-scapy
```

### Step 1: Environment Setup

```bash
# Create and activate conda environment
conda create -n vesper python=3.9 cmake=3.22 -y
conda activate vesper

# Install Habitat-Sim with Bullet physics
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Install VESPER with all dependencies
pip install -e ".[all]"
pip install aiohttp scipy matplotlib seaborn
```

### Step 2: Compile Firmware & Build Docker Image

```bash
# Compile all 6 firmware variants (smart_light, motion_sensor, etc.)
cd vesper/firmware/samples
make
cd ../../..

# Build the QEMU ARM Docker image
docker build -f docker/Dockerfile.device -t vesper-qemu-arm:latest .

# Verify the image
docker run --rm vesper-qemu-arm:latest ls /firmware/
# Should list: smart_light.elf  motion_sensor.elf  temperature_sensor.elf
#              humidity_sensor.elf  door_sensor.elf  smart_plug.elf
```

### Step 3: Download Datasets

```bash
# Download HSSD-Hab scenes, humanoid assets, and test scenes (~12 GB)
python -m habitat_sim.utils.datasets_download --uids \
    hssd-hab habitat_humanoids hab_fetch \
    hab3_bench_assets replica_cad_dataset habitat_test_scenes \
    --data-path data/
```

For the activity realism evaluation (RQ-H), you also need CASAS and ARAS datasets:
```bash
# Download CASAS and ARAS datasets
python vesper/evaluation/download_datasets.py
# → Downloads to data/datasets/casas/ and data/datasets/aras/
```

### Step 4: Start the LLM Server

The evaluation uses **GPT-OSS 20B** for generating activity schedules. You need a local LLM server:

1. Download and install [LMStudio](https://lmstudio.ai/)
2. Download a model (GPT-OSS 20B recommended, GGUF format, 4-bit quantization)
3. Load the model in LMStudio
4. Start the local server → it runs at `http://localhost:1234` by default

Verify it's running:
```bash
curl http://localhost:1234/v1/models
```

### Step 5: Run the Autonomous Evaluation (RQ-S, RQ-H)

This is the **large-scale evaluation** from the paper: 30 HSSD scenes × 3 simulated days with full SmartThings cloud integration.

#### Without SmartThings (simpler, no cloud credentials needed)

```bash
conda activate vesper
python scripts/run_autonomous_eval.py \
    --num-scenes 30 \
    --num-days 3 \
    --time-acceleration 60 \
    --headless
```

#### With SmartThings Cloud Sync (full paper configuration)

```bash
# Terminal 1: Start ngrok tunnel
ngrok http 8443

# Terminal 2: Set credentials and run
export SMARTTHINGS_CLIENT_ID="your-oauth-client-id"
export SMARTTHINGS_CLIENT_SECRET="your-oauth-client-secret"
export ST_APP_CLIENT_SECRET="your-app-client-secret"

conda activate vesper
python scripts/run_autonomous_eval.py \
    --num-scenes 30 \
    --num-days 3 \
    --with-smartthings \
    --time-acceleration 60 \
    --headless
```

**Parameters:**

| Flag | Description | Paper Value |
|------|-------------|-------------|
| `--num-scenes N` | Number of HSSD scenes to evaluate | 30 |
| `--num-days D` | Simulated days per scene | 3 |
| `--time-acceleration X` | Speedup factor (60× = 1 sim-day per 24 min) | 60 |
| `--with-smartthings` | Enable SmartThings cloud sync | Yes |
| `--headless` | No 3D visualization (faster) | Yes |
| `--allow-fallback-tasks` | Use emergency schedule on LLM failure | Yes |

**Expected runtime:** ~70 hours on Apple M2 Pro for 30 scenes × 3 days.

**Monitor progress:**
```bash
# Follow live output
tail -f logs/vesper_objectnav_*.log

# Check navigation success
grep "Navigation trials" logs/vesper_objectnav_*.log

# Check SmartThings sync
grep "proximity_toggles" logs/vesper_objectnav_*.log
```

**Results location:**
```
results/vesper_autonomous_eval/
├── <scene_id>/              # Per-scene results (28 folders)
│   ├── eval_results.json    # Detailed per-scene data
│   └── *.log                # Scene-specific logs
├── eval_summary.txt         # Aggregate summary across all scenes
└── eval_metadata.json       # Configuration and timestamps
```

### Step 6: Run WiFi Network Experiments (RQ-N1, RQ-N2)

These are the **flagship networking experiments**. They require a Linux VM with `mac80211_hwsim` (see [Prerequisites](#prerequisites-for-experiments)).

#### Set Up the VM Environment

```bash
# SSH into the VM
multipass shell vesper-vm

# Copy experiment scripts to the VM
# (from the host, run:)
multipass transfer scripts/run_rqn1_native.py vesper-vm:/home/ubuntu/
multipass transfer scripts/run_rqn2_native.py vesper-vm:/home/ubuntu/
```

#### RQ-N1: Bridge vs. 802.11 Divergence

Measures how security and performance results differ between a Linux bridge network and the emulated 802.11 WiFi stack.

```bash
# Inside the VM:
sudo python3 /home/ubuntu/run_rqn1_native.py

# Runs:
#   3 trials × bridge mode (veth + brctl)
#   3 trials × 802.11 mode (hostapd + wpa_supplicant + mac80211_hwsim)
# Each trial: 19 attacks + RTT measurement (ping -c 200) + iperf3
```

**Expected runtime:** ~16 minutes (6 trials total).

**What it measures:**
- Per-attack exploit success/failure in both modes
- ICMP RTT (mean, P50, P99) via `ping -c 200`
- TCP reconnection latency after deauthentication
- WiFi-specific attack effectiveness (deauth, evil twin, PMKID)

**Key results (from our runs):**

| Metric | Bridge | 802.11 | Interpretation |
|--------|--------|--------|----------------|
| Firmware attacks | 55.6% | 55.6% | Identical — firmware layer is mode-independent |
| Network attacks | 60.0% | 20.0% | 802.11 namespaces block broadcast-dependent attacks |
| WiFi attacks | 0.0% | 53.3% | Only possible with real 802.11 stack |
| Mean RTT | 0.106 ms | 0.354 ms | 802.11 adds mac80211 processing overhead |
| RTT jitter (P99/P50) | — | 20.2× | WiFi tail latency is protocol-realistic |

#### RQ-N2: Measured Hardening Tradeoffs

Sweeps 8 WiFi/MQTT configurations (3 binary toggles) and measures security–availability tradeoffs.

```bash
# Inside the VM:
sudo python3 /home/ubuntu/run_rqn2_native.py

# Runs: 8 configurations × 3 trials = 24 trials
# Each trial: full 19-attack suite + iperf3 throughput + reconnection latency
```

**Expected runtime:** ~33 minutes (24 trials total).

**What it measures:**
- Exploit rate (fraction of 19 attacks that succeed) per configuration
- TCP throughput (iperf3, 10 seconds) under each configuration
- ICMP reconnection latency after WiFi-layer disruption
- Per-attack breakdown showing which hardening toggle blocks which attack

**Key results (from our runs):**

| Configuration | Exploit Rate | Δ vs. Baseline |
|--------------|-------------|----------------|
| C0: WPA2 baseline | 78.9% | — |
| C3: + MQTT auth | 52.6% | −26.3 pp |
| C5: + AP isolation | 68.4% | −10.5 pp |
| C7: Full hardening | 42.1% | **−36.8 pp** |

#### Results Location

```
results/rqn1/
├── rqn1_bridge_trial0.json    # Per-trial attack results (bridge)
├── rqn1_80211_trial0.json     # Per-trial attack results (802.11)
├── rqn1_summary.json          # Aggregate comparison
└── rqn1_rtt_*.csv             # Raw RTT samples

results/rqn2/
├── rqn2_config0_trial0.json   # Per-config per-trial results
├── rqn2_summary.json          # 8-config aggregate
└── rqn2_throughput.csv        # iperf3 results per config
```

### Step 7: Run the Security Assessment (RQ-Sec)

The security assessment runs the full five-suite attack campaign: 18 firmware attacks × 6 devices, 14 network attacks, 3 phantom-delay variants, plus standalone SmartApp and ESP32 overflow demonstrations.

```bash
conda activate vesper

# Run the full per-scene attack suites (firmware + network + phantom-delay)
python scripts/run_attack_demo.py

# Run firmware attacks only
python scripts/run_attack_demo.py --firmware-only

# Run network attacks only
python scripts/run_attack_demo.py --network-only

# Run phantom-delay attacks only
python scripts/run_attack_demo.py --phantom-delay-only

# Target a specific device type
python scripts/run_attack_demo.py --device-type smart_light

# Use Docker containers (recommended for full fidelity)
python scripts/run_attack_demo.py --use-docker
```

**Parameters:**

| Flag | Description | Default |
|------|-------------|---------|
| `--firmware-only` | Run only the 18 firmware attacks per device | Off |
| `--network-only` | Run only the 14 network attacks | Off |
| `--phantom-delay-only` | Run only the 3 phantom-delay variants | Off |
| `--device-type TYPE` | Target one device (e.g., `smart_light`) | All 6 |
| `--use-docker` | Use Docker containers for QEMU | Off |
| `--base-port PORT` | Base TCP port for QEMU instances | 15020 |
| `--mqtt-port PORT` | MQTT broker port for network attacks | 11883 |
| `--output-dir DIR` | Output directory for attack results | `results/security` |

**Expected runtime:** ~15–20 minutes for all per-scene attacks.

**Results location:**
```
results/security/
├── firmware_attacks_smart_light_*.json
├── firmware_attacks_motion_sensor_*.json
├── firmware_attacks_temperature_sensor_*.json
├── firmware_attacks_humidity_sensor_*.json
├── firmware_attacks_door_sensor_*.json
├── firmware_attacks_smart_plug_*.json
├── network_attacks_*.json
├── phantom_delay_attacks_*.json
└── security_summary_*.json
```

### Step 8: Generate Paper Figures

After running all experiments, regenerate the publication-quality PDF figures:

```bash
conda activate vesper

# Generate all PDF figures for the paper
python scripts/generate_paper_figures.py
# → Outputs to paper-latex/figures/

# Analyze the autonomous evaluation data
python scripts/analyze_eval.py
# → Outputs aggregate statistics to results/analysis_output.txt
```

### Expected Results Summary

If everything runs correctly, you should see results comparable to:

| Metric | Expected Value |
|--------|---------------|
| **RQ-N1: Bridge vs. 802.11 (6 trials, ~16 min)** | |
| Firmware exploit rate | 55.6% (both modes — identical) |
| Network exploit rate (bridge) | 60.0% |
| Network exploit rate (802.11) | 20.0% (namespace isolation) |
| WiFi exploit rate (bridge) | 0.0% (no 802.11 stack) |
| WiFi exploit rate (802.11) | 53.3% (deauth, evil twin succeed) |
| Mean RTT (bridge) | 0.106 ms |
| Mean RTT (802.11) | 0.354 ms |
| RTT jitter ratio (P99/P50) | 20.2× (802.11) |
| **RQ-N2: Hardening Tradeoffs (24 trials, ~33 min)** | |
| Baseline exploit rate (C0: WPA2) | 78.9% |
| MQTT auth reduction (C3) | −26.3 pp |
| AP isolation reduction (C5) | −10.5 pp |
| Full hardening reduction (C7) | −36.8 pp (42.1% final) |
| Reconnection latency (C7) | 107.3 ms |
| **Autonomous Evaluation (30 scenes × 3 days, ~70 h)** | |
| Navigation success rate | 98.7% (1,489/1,508 trials) |
| SmartThings cloud updates | Zero data loss |
| **RQ-Sec: Security Campaign** | |
| Total unique attacks | 36 (5 suites) |
| Self-assessed CVSS 3.1 (mean) | 8.1 |
| MITRE ATT&CK coverage | 83% (10/12 tactics) |

> **Note:** Exact numbers may vary slightly due to kernel scheduling jitter and MQTT broker timing. WiFi experiment results are deterministic at the protocol level (same attacks always succeed/fail) but RTT values will differ by ±0.05 ms across runs.

---

## Modes of Operation

### 1. Full Stack — SmartThings + Docker + Firmware *(recommended)*

The primary mode. Real compiled firmware in Docker containers, synced to the SmartThings cloud.

```bash
python scripts/unified_smartthings_firmware.py
```

### 2. Firmware Demo — No Cloud

Test QEMU firmware devices locally with interactive serial communication:

```bash
python scripts/firmware_demo.py
```

Type commands (`ON`, `OFF`, `GET_TEMP`, `STATUS`) directly in the terminal.

### 3. Simulated Sensors — No Docker, No QEMU

Pure-Python sensor simulation for rapid prototyping:

```bash
python scripts/simulated_sensors_demo.py
python scripts/simulated_sensors_demo.py --interactive
python scripts/simulated_sensors_demo.py --room kitchen
```

Supports motion, temperature, humidity, door/window, light, smoke, CO2, water leak, thermostat, and smart plug sensors.

### 4. Docker Compose

Spin up the full device fleet without the Python server:

```bash
cd docker
docker compose up -d
docker compose ps
docker compose down
```

### 5. 3D Habitat Environment *(optional)*

Requires Habitat-Sim via conda. See [Step 1: Environment Setup](#step-1-environment-setup) and [Step 3: Download Datasets](#step-3-download-datasets) in the Reproducing section.

```bash
python scripts/vesper_objectnav_camera_humanoid.py
```

| Key | Action |
|-----|--------|
| W / A / S / D | Move / Turn |
| G | Random navigation goal |
| I | Toggle IoT device panel |
| V | Toggle 1st / 3rd person view |
| ESC | Quit |

---

## SmartThings Setup (Optional)

For bi-directional cloud sync with the SmartThings app:

1. Go to the [SmartThings Developer Portal](https://developer.smartthings.com)
2. Create a new project → **Device Integration** → **SmartThings Schema Connector**
3. Fill in (use your ngrok URL):

| Field | Value |
|-------|-------|
| App Name | VESPER Smart Home |
| Target URL | `https://<NGROK_URL>/schema` |
| OAuth Authorization URI | `https://<NGROK_URL>/oauth/authorize` |
| Token URI | `https://<NGROK_URL>/oauth/token` |

4. Save and note **two sets** of credentials:
   - **Device Cloud Credentials** → Client ID and Client Secret
   - **App Credentials** (top of page) → Click **Regenerate** if hidden

5. Start ngrok and VESPER:

```bash
# Terminal 1
ngrok http 8443

# Terminal 2
export SMARTTHINGS_CLIENT_ID="your-oauth-client-id"
export SMARTTHINGS_CLIENT_SECRET="your-oauth-client-secret"
export ST_APP_CLIENT_SECRET="your-app-client-secret"
python scripts/vesper_smartthings.py
```

6. Link in the SmartThings app: **+** → **Add device** → **Partner devices** → **VESPER Smart Home**

---

## Artifact & Reproducibility

This repository is the **complete artifact** accompanying the VESPER paper. It contains:

| Artifact | Location | Description |
|----------|----------|-------------|
| Platform source | `vesper/` | ~15,000 lines Python + ~1,600 lines C firmware |
| Attack framework | `vesper/attacks/` | 36 unique attacks across 5 suites (~3,200 lines) |
| WiFi experiment scripts | `scripts/run_rqn1_native.py`, `scripts/run_rqn2_native.py` | RQ-N1 (bridge vs. 802.11) and RQ-N2 (hardening tradeoffs) |
| Evaluation pipeline | `vesper/evaluation/` | Automated RQ-S, RQ-H, RQ-Sec experiments |
| WiFi results | `results/rqn1/`, `results/rqn2/` | Per-trial JSON + RTT CSVs from real Linux experiments |
| Autonomous eval | `results/vesper_autonomous_eval/` | 30-scene × 3-day evaluation outputs |
| Paper source | `paper-latex/` | Full LaTeX source (ACM sigconf, 8 sections, 14 tables) |
| Configs | `configs/`, `vesper/evaluation/configs/` | All experiment YAML configurations |
| Docker | `docker/` | Dockerfile + compose for QEMU ARM containers |

All experiments are reproducible from a single clone:
```bash
git clone https://github.com/huuhuannt1998/vesper2.0.git && cd vesper
pip install -e ".[all]"
# See "Reproducing the Paper Experiments" above for full instructions
```

---

## Project Structure

```
vesper/
├── vesper/                          # Main package (~15,000 lines Python)
│   ├── core/                        # Event bus, environment engine
│   ├── devices/                     # IoT device models
│   ├── agents/                      # LLM-controlled agents (10 personas)
│   ├── firmware/                    # Firmware emulation layer
│   │   ├── samples/                 # ARM Cortex-M3 firmware source (~1,800 lines C)
│   │   │   ├── smart_light.c        # Smart light firmware
│   │   │   ├── motion_sensor.c      # Motion sensor firmware
│   │   │   ├── temperature_sensor.c # Temperature sensor firmware
│   │   │   ├── humidity_sensor.c    # Humidity sensor firmware
│   │   │   ├── door_sensor.c        # Door sensor firmware
│   │   │   ├── smart_plug.c         # Smart plug firmware
│   │   │   ├── linker.ld            # LM3S6965 memory layout (64KB flash / 20KB SRAM)
│   │   │   └── Makefile             # Cross-compilation build
│   │   ├── device_firmware_manager.py  # Docker container lifecycle
│   │   └── qemu_runner.py           # QEMU process management
│   ├── attacks/                     # Security testing framework (~3,200 lines Python)
│   │   ├── firmware_attacks.py      # Suite 1: 18 firmware attacks (9 categories)
│   │   ├── network_attacks.py       # Suite 2: 14 network attacks (5 categories)
│   │   └── phantom_delay_attack.py  # Suite 3: 3 phantom-delay variants
│   ├── network/                     # 802.11 WiFi emulation (mac80211_hwsim)
│   │   └── home_network.py          # Bridge + 802.11 mode, namespace mgmt, tshark capture
│   ├── integrations/                # Cloud platform connectors
│   │   ├── schema_connector.py      # SmartThings Schema Protocol
│   │   └── sync_bridge.py           # Bi-directional state sync
│   ├── evaluation/                  # Automated evaluation pipeline
│   │   ├── experiment_runner.py     # RQ-S, RQ-H experiment orchestrator
│   │   ├── security_eval.py         # RQ-Sec security evaluation
│   │   ├── activity_comparison.py   # Activity realism metrics
│   │   ├── scalability_bench.py     # Scalability benchmarks
│   │   ├── latency_profiler.py      # Latency profiling
│   │   ├── llm_ablation.py          # LLM model comparison
│   │   ├── report_generator.py      # LaTeX table/figure generation
│   │   └── configs/                 # Experiment YAML configs
│   ├── habitat/                     # Habitat 3.0 integration
│   ├── protocol/                    # IoT protocol implementations
│   └── simulation/                  # Simulation engine
├── scripts/
│   ├── vesper_smartthings.py        # Full stack: 3D + firmware + SmartThings
│   ├── run_rqn1_native.py           # RQ-N1: bridge vs. 802.11 (runs on Linux VM)
│   ├── run_rqn2_native.py           # RQ-N2: 8-config hardening sweep (runs on Linux VM)
│   ├── run_autonomous_eval.py       # Autonomous 30-scene evaluation
│   ├── run_attack_demo.py           # Per-scene security assessment (Suites 1–3)
│   ├── analyze_eval.py              # Aggregate evaluation analysis
│   ├── generate_paper_figures.py    # Publication-quality PDF figure generation
│   ├── firmware_demo.py             # Standalone QEMU demo
│   ├── vesper_objectnav_camera_humanoid.py  # 3D navigation demo
│   ├── simulated_sensors_demo.py    # Pure-Python sensor demo
│   └── attacks/                     # Standalone attack scripts
│       ├── smartapp.py              # Suite 4: Malicious SmartApp
│       ├── esp32_overflow.py        # Suite 5: ESP32 Buffer Overflow
│       ├── relay.py                 # Relay attack utility
│       ├── firmware.py              # Firmware attack CLI
│       └── network.py              # Network attack CLI
├── docker/
│   ├── Dockerfile.device            # QEMU ARM device image
│   ├── docker-compose.yml           # Multi-device orchestration
│   └── entrypoint.sh               # Container startup
├── tests/                           # Unit tests (10 test modules)
├── results/                         # Experiment outputs
├── configs/                         # Default configuration
│   └── default.yaml                 # Network, device, and evaluation defaults
└── paper-latex/                     # Paper source (ACM sigconf)
    ├── main.tex                     # Main document
    ├── sections/                    # 8 section files
    ├── tables/                      # 14 LaTeX tables
    └── figures/                     # TikZ sources + PDF figures
```

---

## Configuration

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|-------------|
| `SMARTTHINGS_CLIENT_ID` | OAuth Client ID (from Device Cloud Credentials) | SmartThings sync |
| `SMARTTHINGS_CLIENT_SECRET` | OAuth Client Secret (from Device Cloud Credentials) | SmartThings sync |
| `ST_APP_CLIENT_SECRET` | SmartThings App Client Secret (from App Credentials) | Bi-directional sync (3D→Phone) |

 **Critical:** `ST_APP_CLIENT_SECRET` is **required** for 3D→SmartThings proactive state updates. Find it at the top of your SmartThings Developer Portal project page under "App Credentials" (click Regenerate if hidden).

### Adding More Devices

Edit the device list in `scripts/unified_smartthings_firmware.py`:

```python
device_configs = [
    FirmwareDeviceConfig(
        device_id="vesper-fw-garage",
        name="Garage Door Sensor",
        room="Garage",
        device_type=DeviceHandlerType.DIMMER,
        host_port=15004,  # unique port per device
    ),
    # ...
]
```

Then unlink and re-link in the SmartThings app to trigger a new device discovery.

### Programmatic Usage

```python
from vesper.integrations import (
    SmartThingsSchemaConnector,
    SchemaConnectorConfig,
    VirtualDeviceDefinition,
    DeviceHandlerType,
)

config = SchemaConnectorConfig(
    host="0.0.0.0",
    port=8443,
    webhook_path="/schema",
    smartthings_client_id="your-client-id",
    smartthings_client_secret="your-secret",
)
connector = SmartThingsSchemaConnector(config)

device = VirtualDeviceDefinition(
    external_device_id="my-device",
    friendly_name="My Custom Light",
    device_handler_type=DeviceHandlerType.DIMMER,
    manufacturer_name="VESPER",
    model_name="Virtual Light",
)
connector.register_device(device)

async def handle_command(device_id, capability, command, args):
    print(f"Command: {device_id} → {capability}.{command}")
    return True

connector.on_command(handle_command)
await connector.start()
```

---

## Troubleshooting

### Port 8443 already in use

```bash
lsof -ti:8443 | xargs kill -9
```

### ngrok URL changed

Free-tier ngrok assigns a new URL on every restart. Update all three URLs (Target, OAuth, Token) in the [SmartThings Developer Portal](https://developer.smartthings.com).

### Devices not appearing in SmartThings

- Ensure the VESPER server is running **and** ngrok is forwarding
- Check server logs for incoming `discoveryRequest` — if absent, the URL is wrong
- Unlink and re-link in the SmartThings app

### 3D → SmartThings sync not working

- Verify `ST_APP_CLIENT_SECRET` is set (check startup banner)
- SmartThings only sends `grantCallbackAccess` during initial linking — **fully remove** the integration, then re-add it
- If still failing with `INVALID-CLIENT-SECRET`, regenerate App Credentials in Developer Portal

### Docker containers won't start

```bash
docker info                    # Check Docker daemon is running
docker rm -f $(docker ps -aq --filter "name=vesper")   # Remove stale containers
docker build -f docker/Dockerfile.device -t vesper-qemu-arm:latest .  # Rebuild
```

### Firmware won't compile

```bash
arm-none-eabi-gcc --version    # Verify toolchain is installed

# Install if missing:
brew install arm-none-eabi-gcc          # macOS
sudo apt install gcc-arm-none-eabi      # Linux
```

### LLM generation fails

```bash
# Check LMStudio is running
curl http://localhost:1234/v1/models

# Increase timeout in vesper/agents/llm_client.py
# Default: timeout=180 → increase to 300
```

### Out of memory during evaluation

```bash
# Use headless mode (saves ~4GB GPU memory)
python scripts/run_autonomous_eval.py --num-scenes 10 --num-days 3 --headless

# Clean up old Docker containers
docker rm -f $(docker ps -aq --filter "name=vesper-fw")
```

### Navmesh failures

Some HSSD scenes may fail to generate valid navmeshes (preventing humanoid navigation). The evaluation handles this gracefully — security attacks still run in these scenes, but no navigation trials are produced. In our 30-scene evaluation, 2 scenes exhibited this behavior.

---

## Tests

```bash
conda activate vesper
python -m pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud Platform | Samsung SmartThings (Schema Connector) |
| Webhook Server | aiohttp (async Python) |
| HTTPS Tunnel | ngrok |
| WiFi Emulation | `mac80211_hwsim` (Linux 5.15) + `hostapd` 2.10 + `wpa_supplicant` 2.10 |
| Containerization | Docker (QEMU ARM firmware containers) |
| Firmware Emulation | QEMU 10.2 — ARM Cortex-M3 (LM3S6965EVB) |
| Firmware Toolchain | arm-none-eabi-gcc |
| Firmware Language | C (bare-metal, no stdlib, ~1,800 lines across 6 devices) |
| MQTT Broker | Mosquitto 2.0.11 (optional TLS 1.3 + username/password ACLs) |
| 3D Environment | Habitat 3.0 / Habitat-Sim, HSSD-Hab scenes |
| LLM Engine | GPT-OSS 20B via LMStudio (OpenAI-compatible API) |
| Security Framework | 5 suites, 36 unique attacks, self-assessed CVSS 3.1, MITRE ATT&CK for IoT |
| Network Analysis | tshark 3.6.2 per-namespace 802.11 frame capture |
| Evaluation Framework | Custom Python + LaTeX/PDF auto-generation |

## License

MIT License — See [LICENSE](LICENSE) for details.
