# VESPER — Virtual Environment for Smart-home Platform Evaluation & Research

A full-stack IoT simulation platform that bridges **virtual smart-home devices** to **real cloud platforms** (Samsung SmartThings). Each virtual device runs compiled ARM firmware inside QEMU, packaged in its own Docker container, and is controllable from your phone. VESPER also includes a comprehensive **security testing framework** with 32 unique attacks and an **LLM-driven activity generation** pipeline for autonomous evaluation.

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
```

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Reproducing the Paper Experiments](#reproducing-the-paper-experiments)
  - [Prerequisites](#prerequisites-for-experiments)
  - [Step 1: Environment Setup](#step-1-environment-setup)
  - [Step 2: Compile Firmware & Build Docker Image](#step-2-compile-firmware--build-docker-image)
  - [Step 3: Download Datasets](#step-3-download-datasets)
  - [Step 4: Start the LLM Server](#step-4-start-the-llm-server)
  - [Step 5: Run the Autonomous Evaluation (RQ1–RQ4)](#step-5-run-the-autonomous-evaluation-rq1rq4)
  - [Step 6: Run the RQ Experiments (RQ1–RQ4)](#step-6-run-the-rq-experiments-rq1rq4)
  - [Step 7: Run the Security Assessment (RQ5)](#step-7-run-the-security-assessment-rq5)
  - [Step 8: Generate the Security Evaluation Report](#step-8-generate-the-security-evaluation-report)
- [Modes of Operation](#modes-of-operation)
- [SmartThings Setup](#smartthings-setup-optional)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Features

- **Real Firmware Emulation** — ARM Cortex-M3 firmware compiled with `arm-none-eabi-gcc`, running in QEMU
- **Docker-per-Device** — Each virtual IoT device is an isolated Docker container
- **6 Device Types** — Smart light, motion sensor, temperature sensor, humidity sensor, door sensor, smart plug
- **SmartThings Bi-Directional Sync** — Devices appear in the Samsung SmartThings app with real-time sync
- **LLM-Driven Activity Generation** — GPT-OSS 20B generates realistic daily schedules from 10 diverse personas
- **3D Habitat Integration** — Habitat 3.0 with HSSD scenes, humanoid navigation, and proximity-based automation
- **Security Testing Framework** — 32 unique attacks (18 firmware + 14 network) with CVSS 3.1 scoring and MITRE ATT&CK mapping
- **Automated Evaluation Pipeline** — Reproducible experiments with LaTeX table/figure generation
- **Event-Driven Architecture** — Pub/sub event bus with sub-millisecond dispatch

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

## Reproducing the Paper Experiments

This section provides step-by-step instructions to reproduce all five research questions (RQ1–RQ5) and the large-scale autonomous evaluation from the paper.

### Prerequisites for Experiments

- **Hardware:** Apple M2 Pro or equivalent (32 GB RAM recommended)
- **OS:** macOS 14+ or Ubuntu 22.04+
- **Disk:** ~20 GB free (datasets + Docker images)
- **Docker:** Must be running with at least 8 GB RAM allocated
- **LMStudio:** Required for LLM-based schedule generation

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

For the Sim2Real evaluation (RQ4), you also need CASAS and ARAS datasets:
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

### Step 5: Run the Autonomous Evaluation (RQ1–RQ4)

This is the **main experiment** from the paper: 30 HSSD scenes × 5 simulated days with full SmartThings cloud integration.

#### Without SmartThings (simpler, no cloud credentials needed)

```bash
conda activate vesper
python scripts/run_autonomous_eval.py \
    --num-scenes 30 \
    --num-days 5 \
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
    --num-days 5 \
    --with-smartthings \
    --time-acceleration 60 \
    --headless
```

**Parameters:**

| Flag | Description | Paper Value |
|------|-------------|-------------|
| `--num-scenes N` | Number of HSSD scenes to evaluate | 30 |
| `--num-days D` | Simulated days per scene | 5 |
| `--time-acceleration X` | Speedup factor (60× = 1 sim-day per 24 min) | 60 |
| `--with-smartthings` | Enable SmartThings cloud sync | Yes |
| `--headless` | No 3D visualization (faster) | Yes |
| `--allow-fallback-tasks` | Use emergency schedule on LLM failure | Yes |

**Expected runtime:** ~23.5 hours on Apple M2 Pro for 30 scenes × 5 days.

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
├── eval_results.json      # Per-scene detailed results
├── eval_summary.txt       # Human-readable summary
└── eval_metadata.json     # Configuration and timestamps
```

### Step 6: Run the RQ Experiments (RQ1–RQ4)

The evaluation framework provides individual RQ experiments via the `ExperimentRunner`:

```bash
conda activate vesper

# Run ALL RQ experiments (RQ1–RQ4)
python -m vesper.evaluation.experiment_runner \
    --config vesper/evaluation/configs/full_evaluation.yaml \
    --output results/full_evaluation

# Or run individual experiments:

# RQ1: Activity Realism (JS divergence against CASAS/ARAS)
python -m vesper.evaluation.experiment_runner \
    --experiment activity \
    --output results/rq1_activity

# RQ2: Scalability (5→200 devices, throughput, CPU, memory)
python -m vesper.evaluation.experiment_runner \
    --experiment scalability \
    --output results/rq2_scalability

# RQ3: Latency (event bus, database, LLM profiling)
python -m vesper.evaluation.experiment_runner \
    --experiment latency \
    --output results/rq3_latency

# RQ4: LLM Ablation (6 models × 50 attempts)
# Requires all 6 models loaded in LMStudio
python -m vesper.evaluation.experiment_runner \
    --experiment llm \
    --output results/rq4_llm_ablation
```

**Generate a default config:**
```bash
python -m vesper.evaluation.experiment_runner --generate-config
# → Creates configs/evaluation.yaml that you can customize
```

**Key configuration** (`vesper/evaluation/configs/full_evaluation.yaml`):
```yaml
seed: 42
num_trials: 5
confidence_level: 0.95
device_counts: [5, 10, 25, 50, 100, 200]     # RQ2
latency_iterations: 1000                       # RQ3
comparison_days: 30                            # RQ1
```

### Step 7: Run the Security Assessment (RQ5)

The security assessment runs 122 attacks (108 firmware + 14 network) against all 6 device types:

```bash
conda activate vesper

# Run the full attack suite (firmware + network)
python scripts/run_attack_demo.py

# Run firmware attacks only
python scripts/run_attack_demo.py --firmware-only

# Run network attacks only
python scripts/run_attack_demo.py --network-only

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
| `--device-type TYPE` | Target one device (e.g., `smart_light`) | All 6 |
| `--use-docker` | Use Docker containers for QEMU | Off |
| `--base-port PORT` | Base TCP port for QEMU instances | 15020 |
| `--mqtt-port PORT` | MQTT broker port for network attacks | 11883 |
| `--output-dir DIR` | Output directory for attack results | `results/security` |

**Expected runtime:** ~5–10 minutes for all 122 attacks.

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
└── security_summary_*.json
```

### Step 8: Generate the Security Evaluation Report

After running the attacks, generate the full evaluation report with CVSS scoring, MITRE ATT&CK mapping, and publication-ready tables/figures:

```bash
conda activate vesper

# Generate the full security evaluation report
python -m vesper.evaluation.security_eval \
    --results-dir results/security \
    --output-dir results/report

# Skip figure generation (faster, text-only)
python -m vesper.evaluation.security_eval \
    --results-dir results/security \
    --output-dir results/report \
    --no-figures
```

**Output:**
```
results/report/
├── security_evaluation_<timestamp>.json   # Full evaluation data (118 KB)
├── tables/
│   ├── tab_security_summary.tex           # Aggregate results
│   ├── tab_cvss_distribution.tex          # CVSS severity breakdown
│   ├── tab_device_comparison.tex          # Per-device analysis
│   ├── tab_mitre_coverage.tex             # MITRE ATT&CK mapping
│   ├── tab_kill_chain.tex                 # IoT Cyber Kill Chain
│   └── tab_statistical_tests.tex          # Statistical significance
└── figures/
    ├── fig_device_heatmap.pdf             # Per-device exploit heatmap
    ├── fig_cvss_distribution.pdf          # CVSS score distribution
    ├── fig_kill_chain.pdf                 # Kill chain coverage
    ├── fig_attack_surface.pdf             # Attack surface analysis
    ├── fig_tte_boxplot.pdf                # Time-to-exploit by severity
    └── fig_mitre_tactics.pdf              # MITRE tactic coverage
```

The generated LaTeX tables can be directly included in the paper with `\input{tables/tab_security_summary}`.

### Expected Results Summary

If everything runs correctly, you should see results comparable to:

| Metric | Expected Value |
|--------|---------------|
| **Autonomous Evaluation** | |
| Navigation success rate | ~99.5% (1,748 trials) |
| SmartThings cloud updates | ~20,685 (zero data loss) |
| LLM generation success | ~98.0% (197 attempts) |
| Unique activity types | ~432 |
| **RQ1: Activity Realism** | |
| Mean JS divergence | ~0.146 (95% CI: [0.10, 0.19]) |
| Best match | ARAS-House B (JS = 0.093) |
| **RQ2: Scalability** | |
| Max throughput (200 devices) | ~10,901 events/s |
| CPU at 200 devices | <36% |
| Memory at 200 devices | <106 MB |
| **RQ3: Latency** | |
| Event bus P99 | ~7 μs |
| Database write P99 | ~2.84 ms |
| **RQ5: Security Assessment** | |
| Total attacks | 122 (108 firmware + 14 network) |
| Exploit rate | ~53.3% (65/122) |
| Mean CVSS | ~7.8 |
| MITRE ATT&CK coverage | 83% (10/12 tactics) |
| Kill chain completeness | 100% (7/7 stages) |

> **Note:** Exact numbers may vary slightly due to LLM non-determinism, system load, and network conditions. Confidence intervals account for this variation.

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

## Project Structure

```
vesper/
├── vesper/                          # Main package (~12,000 lines Python)
│   ├── core/                        # Event bus, environment engine
│   ├── devices/                     # IoT device models
│   ├── agents/                      # LLM-controlled agents
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
│   ├── attacks/                     # Security testing framework
│   │   ├── firmware_attacks.py      # 18 firmware attacks (9 categories)
│   │   └── network_attacks.py       # 14 network attacks (5 suites)
│   ├── network/                     # Simulated home network
│   │   └── home_network.py          # Docker bridge, MQTT broker, protocol simulators
│   ├── integrations/                # Cloud platform connectors
│   │   ├── schema_connector.py      # SmartThings Schema Protocol
│   │   └── sync_bridge.py           # Bi-directional state sync
│   ├── evaluation/                  # Automated evaluation pipeline
│   │   ├── experiment_runner.py     # RQ1-RQ4 experiment orchestrator
│   │   ├── security_eval.py         # RQ5 security evaluation (900 lines)
│   │   ├── activity_comparison.py   # Activity realism metrics
│   │   ├── scalability_bench.py     # Scalability benchmarks
│   │   ├── latency_profiler.py      # Latency profiling
│   │   ├── llm_ablation.py          # LLM model comparison
│   │   ├── report_generator.py      # LaTeX table/figure generation
│   │   └── configs/                 # Experiment YAML configs
│   ├── habitat/                     # Habitat 3.0 integration
│   └── simulation/                  # Simulation engine
├── scripts/
│   ├── vesper_smartthings.py        # Full stack: 3D + firmware + SmartThings
│   ├── run_autonomous_eval.py       # Autonomous 30-scene evaluation
│   ├── run_attack_demo.py           # Security assessment (122 attacks)
│   ├── firmware_demo.py             # Standalone QEMU demo
│   ├── vesper_objectnav_camera_humanoid.py  # 3D navigation demo
│   └── simulated_sensors_demo.py    # Pure-Python sensor demo
├── docker/
│   ├── Dockerfile.device            # QEMU ARM device image
│   ├── docker-compose.yml           # Multi-device orchestration
│   └── entrypoint.sh               # Container startup
├── tests/                           # Unit tests
├── results/                         # Experiment outputs
├── configs/                         # Default configuration
└── paper-latex/                     # Paper source (ACM sigconf)
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
python scripts/run_autonomous_eval.py --num-scenes 10 --num-days 5 --headless

# Clean up old Docker containers
docker rm -f $(docker ps -aq --filter "name=vesper-fw")
```

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
| Containerization | Docker |
| Firmware Emulation | QEMU 10.2 — ARM Cortex-M3 (LM3S6965EVB) |
| Firmware Toolchain | arm-none-eabi-gcc |
| Firmware Language | C (bare-metal, no stdlib, ~1,800 lines across 6 devices) |
| 3D Environment | Habitat 3.0 / Habitat-Sim, HSSD-Hab scenes |
| LLM Engine | GPT-OSS 20B via LMStudio (OpenAI-compatible API) |
| Security Evaluation | CVSS 3.1, MITRE ATT&CK for IoT, IoT Cyber Kill Chain |
| Evaluation Framework | Custom Python + LaTeX/PDF auto-generation |

## License

MIT License — See [LICENSE](LICENSE) for details.
