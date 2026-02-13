# VESPER — Virtual Environment for Smart-home Protocol Emulation & Research

A full-stack IoT simulation platform that bridges **virtual smart-home devices** to **real cloud platforms** (Samsung SmartThings). Each virtual device runs compiled ARM firmware inside QEMU, packaged in its own Docker container, and is controllable from your phone.

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

## Features

- **Real Firmware Emulation** — ARM Cortex-M3 firmware compiled with `arm-none-eabi-gcc`, running in QEMU
- **Docker-per-Device** — Each virtual IoT device is an isolated Docker container
- **SmartThings Bi-Directional Sync** — Devices appear in the Samsung SmartThings app with real-time sync:
  - **Phone → 3D:** Toggle in SmartThings app → updates 3D environment instantly
  - **3D → Phone:** Press F key or humanoid proximity → SmartThings app updates in real-time
- **SmartThings Schema Protocol** — Full cloud-to-cloud integration (discovery, state refresh, commands, proactive callbacks)
- **Pure-Python Sensor Simulation** — No hardware required; simulated motion, temperature, humidity, smoke, and more
- **3D Habitat Integration** — Optional Habitat 3.0 support for 3D smart-home environments with humanoid agents
- **Event-Driven Architecture** — Pub/sub event bus with MQTT support for real IoT bridging

---

## Quick Start

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.9+ | `brew install python` |
| Docker | 20+ | [docker.com](https://www.docker.com/products/docker-desktop/) |
| ngrok | 3+ | `brew install ngrok` |
| ARM GCC | 13+ | `brew install arm-none-eabi-gcc` |
| QEMU | 8+ | `brew install qemu` |

> **Linux:** Replace `brew install` with `sudo apt install` (packages: `qemu-system-arm`, `gcc-arm-none-eabi`).

### 1. Clone & Install

```bash
git clone https://github.com/huuhuannt1998/vesper2.0.git
cd vesper

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install VESPER and dependencies
pip install -e ".[all]"
pip install aiohttp
```

### 2. Compile the Firmware

```bash
cd vesper/firmware/samples
make
# → sensor_firmware.elf  (ARM Cortex-M3, ~1.6 KB)
cd ../../..
```

### 3. Build the Docker Image

```bash
docker build -f docker/Dockerfile.device -t vesper-qemu-arm:latest .
```

### 4. Register a SmartThings Schema App

1. Go to the [SmartThings Developer Portal](https://developer.smartthings.com)
2. Create a new project → **Device Integration** → **SmartThings Schema Connector**
3. Fill in (use your ngrok URL from step 5):

| Field | Value |
|-------|-------|
| App Name | VESPER Smart Home |
| Target URL | `https://<NGROK_URL>/schema` |
| OAuth Authorization URI | `https://<NGROK_URL>/oauth/authorize` |
| Token URI | `https://<NGROK_URL>/oauth/token` |

4. **Important:** Save and note **two sets** of credentials:
   - **Device Cloud Credentials** → Client ID and Client Secret (for OAuth account linking)
   - **App Credentials** (top of page) → Click **Regenerate** if hidden → SmartThings Client ID and Client Secret (for proactive state updates)

### 5. Start Everything

Open **three terminals**:

**Terminal 1 — ngrok tunnel:**

```bash
ngrok http 8443
```

> Copy the HTTPS forwarding URL (e.g. `https://abcd-1234.ngrok-free.app`).
> Update all three URLs in the SmartThings Developer Portal if the URL changed.

**Terminal 2 — VESPER server:**

```bash
source .venv/bin/activate

# OAuth credentials (from Device Cloud Credentials)
export SMARTTHINGS_CLIENT_ID="your-oauth-client-id"
export SMARTTHINGS_CLIENT_SECRET="your-oauth-client-secret"

# App credentials (from App Credentials at top of portal page)
export ST_APP_CLIENT_SECRET="your-app-client-secret"

python scripts/vesper_smartthings.py
```

> **Note:** `ST_APP_CLIENT_SECRET` enables **bi-directional sync** (3D → SmartThings proactive state updates). Without it, only polling-based sync works.

You should see:

```
✅ Kitchen Light (Firmware)   (port 15001)
✅ Living Room (Firmware)     (port 15002)
✅ Bedroom Light (Firmware)   (port 15003)

SERVER RUNNING
Webhook URL:  http://localhost:8443/schema
```

**Terminal 3 — verify Docker containers:**

```bash
docker ps --filter "name=vesper"
```

```
CONTAINER ID   IMAGE              STATUS    PORTS                     NAMES
e02e65a9a139   vesper-qemu-arm    Up        0.0.0.0:15001->5555/tcp   vesper-vesper-fw-kitchen
6ab99663f1de   vesper-qemu-arm    Up        0.0.0.0:15002->5555/tcp   vesper-vesper-fw-living
bacfbed406f4   vesper-qemu-arm    Up        0.0.0.0:15003->5555/tcp   vesper-vesper-fw-bedroom
```

### 6. Link in the SmartThings App

1. Open the **SmartThings** app on your phone
2. Tap **+** → **Add device** → **Partner devices**
3. Find **VESPER Smart Home** and tap to link
4. Authorize the connection
5. Your 3 firmware devices appear — toggle on/off from the app!

---

## ArchitectureBi-Directional)

**Phone → 3D (ST→3D):**
1. You tap **Off → On** in the SmartThings app
2. SmartThings cloud sends a `commandRequest` to the ngrok URL
3. VESPER Schema Connector receives the webhook, extracts the command
4. Connector opens a TCP connection to the device's Docker container
5. Sends `ON\n` over the QEMU serial port
6. ARM firmware processes the command, sets GPIO, responds `SWITCH:on\nACK\n`
7. Connector reads the response, updates device state
8. 3D environment reflects the change (visual indicator updates)

**3D → Phone (3D→ST):**
1. Humanoid enters room (proximity) or user presses F key
2. `check_proximity_interaction()` or `toggle_device_in_room()` calls `fw.handle_command('ON')`
3. Firmware updates state via Docker serial TCP
4. Bridge calls `connector.update_device_state(device_id, 'switch', 'on')`
5. Connector sends proactive `stateCallback` POST to `https://c2c-us.smartthings.com/device/events`
6. SmartThings app refreshes **instantly** without polling `SWITCH:on\nACK\n`
7. Connector reads the response, updates device state
8. Returns updated state to SmartThings → app UI refreshes

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SmartThings Cloud                                    │
│  discoveryRequest · stateRefreshRequest · commandRequest · grantCallback   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ HTTPS (ngrok tunnel)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VESPER Schema Connector          (vesper/integrations/schema_connector.py) │
│  ├─ Webhook server (aiohttp, port 8443)                                    │
│  ├─ OAuth2 endpoints (/oauth/authorize, /oauth/token)                      │
│  ├─ Device registry (VirtualDeviceDefinition)                              │
│  └─ Command routing → Docker firmware devices                              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ TCP socket (per-device port)
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
             ┌────────────┐┌────────────┐┌────────────┐
             │  Docker    ││  Docker    ││  Docker    │
             │  Container ││  Container ││  Container │
             │            ││            ││            │
             │  QEMU ARM  ││  QEMU ARM  ││  QEMU ARM │
             │  Cortex-M3 ││  Cortex-M3 ││  Cortex-M3 │
             │  Firmware  ││  Firmware  ││  Firmware  │
             │  (1.6 KB)  ││  (1.6 KB)  ││  (1.6 KB)  │
             └────────────┘└────────────┘└────────────┘
              :15001        :15002        :15003
```

### Firmware Protocol

The ARM Cortex-M3 firmware communicates over UART serial (exposed as TCP by QEMU):

| Command | Response | Description |
|---------|----------|-------------|
| `ON` | `SWITCH:on\nACK` | Turn device on |
| `OFF` | `SWITCH:off\nACK` | Turn device off |
| `GET_SWITCH` | `SWITCH:on` or `SWITCH:off` | Query switch state |
| `GET_TEMP` | `TEMP:22.5` | Read temperature (fixed-point) |
| `GET_HUMIDITY` | `HUMIDITY:45.0` | Read humidity |
| `GET_ALL` | Multi-line state dump | All sensor readings |
| `STATUS` | `STATUS:OK` | Health check |
| `IDENTIFY` | Device info block | Firmware version & capabilities |

---

## Project Structure

```
vesper/
├── vesper/                          # Main package
│   ├── core/                        # Event bus, environment engine
│   ├── devices/                     # IoT device models
│   ├── protocol/                    # Message types & codec
│   ├── network/                     # Transport, router, broker
│   ├── agents/                      # LLM-controlled agents
│   ├── firmware/                    # Firmware emulation layer
│   │   ├── samples/                 # ARM Cortex-M3 firmware source
│   │   │   ├── sensor_firmware.c    # Firmware (integer-only, no stdlib)
│   │   │   ├── linker.ld           # LM3S6965 memory layout
│   │   │   └── Makefile            # Cross-compilation build
│   │   ├── qemu_runner.py          # QEMU process management
│   │   ├── emulator.py             # Firmware emulator abstraction
│   │   └── sensor_templates.py     # Pure-Python sensor simulation
│   ├── integrations/                # Cloud platform connectors
│   │   ├── schema_connector.py     # SmartThings Schema Protocol
│   │   ├── device_registry.py      # SQLite device persistence
│   │   ├── docker_device_manager.py # Docker container lifecycle
│   │   └── sync_bridge.py          # Bi-directional state sync
│   ├── habitat/                     # Habitat 3.0 integration (optional)
│   └── simulation/                  # Simulation engine
├── docker/
│   ├── Dockerfile.device           # QEMU ARM device image (Ubuntu + QEMU)
│   ├── docker-compose.yml          # Multi-device orchestration
│   └── entrypoint.sh              # Container startup (QEMU + TCP serial)
├── scripts/
│   ├── unified_smartthings_firmware.py  # ★ Main entry point
│   ├── firmware_demo.py            # Standalone QEMU demo (no cloud)
│   ├── smartthings_server.py       # SmartThings-only (no firmware)
│   └── simulated_sensors_demo.py   # Pure-Python sensor demo
├── tests/                          # Test 3D Habitat *(recommended)*

The primary mode. Real compiled firmware in Docker containers, 3D Habitat environment with humanoid navigation, fully synced to the SmartThings cloud with bi-directional real-time updates.

```bash
python scripts/vesper_smartthings.py
```

Features:
- ✅ 3D visualization with humanoid agent
- ✅ Bi-directional SmartThings sync (3D ↔ Phone)
- ✅ Docker QEMU firmware devices
- ✅ Proximity-based automation (humanoid triggers lights)
- ✅ Manual control (F key to toggle lights)
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

Requires Habitat-Sim via conda. See [Habitat Setup](#3d-habitat-setup-optional).

```bash
python scripts/vesper_objectnav_camera_humanoid.py
```

---

## Configuration

### Environment Variables

| Variable | Description | DOAuth Client ID (from Device Cloud Credentials) | — |
| `SMARTTHINGS_CLIENT_SECRET` | OAuth Client Secret (from Device Cloud Credentials) | — |
| `ST_APP_CLIENT_SECRET` | SmartThings App Client Secret (from App Credentials) | — |

> **Critical:** `ST_APP_CLIENT_SECRET` is **required** for 3D→SmartThings proactive state updates. Find it at the top of your SmartThings Developer Portal project page under "App Credentials" (click Regenerate if hidden).
| `SMARTTHINGS_CLIENT_ID` | SmartThings app client ID | — |
| `SMARTTHINGS_CLIENT_SECRET` | SmartThings app client secret | — |

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

## 3D Habitat Setup (Optional)

VESPER optionally integrates with Meta's Habitat 3.0 for 3D smart-home simulation with humanoid agents. This is **not required** for the SmartThings/firmware features.

### Install Habitat-Sim

```bash
conda create -n vesper python=3.9 cmake=3.22 -y
conda activate vesper

conda install habitat-sim withbullet -c conda-forge -c aihabitat  # macOS / Linux
pip install -e ".[all]"
```

### Download Datasets (~12 GB)

```bash
python -m habitat_sim.utils.datasets_download --uids \
    habitat_test_scenes replica_cad_dataset hab3_bench_assets \
    habitat_humanoids hab_fetch \
    --data-path data/
```

### Run the 3D Demo

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


### 3D → SmartThings sync not working

- Verify `ST_APP_CLIENT_SECRET` is set (check startup banner for credential status)
- SmartThings only sends `grantCallbackAccess` during initial linking — **fully remove** the VESPER integration from SmartThings app, then re-add it
- Check logs for `✅ Stored callback credentials` after re-linking
- If still failing with `INVALID-CLIENT-SECRET`, regenerate App Credentials in Developer Portal and update `ST_APP_CLIENT_SECRET`

---

## Large-Scale Autonomous Evaluation

VESPER includes a comprehensive autonomous evaluation framework that validates end-to-end system reliability across multiple HSSD scenes with LLM-driven activity generation and full SmartThings cloud integration.

### Setup for Evaluation

#### 1. Install Dependencies

```bash
conda create -n vesper python=3.9 cmake=3.22 -y
conda activate vesper

# Install Habitat-Sim with Bullet physics
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Install VESPER with all dependencies
cd /path/to/vesper
pip install -e ".[all]"
```

#### 2. Download HSSD Dataset

The evaluation uses HSSD-Hab articulated scenes (161 scenes available):

```bash
python -m habitat_sim.utils.datasets_download \
    --uids hssd-hab habitat_humanoids hab_fetch \
    --data-path data/
```

This downloads ~5GB of 3D scene data to `data/scene_datasets/hssd-hab/`.

#### 3. Setup LLM Server

The evaluation uses **GPT-OSS 20B** (or any OpenAI-compatible LLM) for generating daily activity schedules.

**Option A: LMStudio (Recommended for local execution)**

1. Download [LMStudio](https://lmstudio.ai/)
2. Download GPT-OSS 20B model (GGUF format, 4-bit quantization)
3. Load the model in LMStudio
4. Start the local server (default: `http://localhost:1234`)

**Option B: OpenAI API**

```bash
export OPENAI_API_KEY="your-api-key"
```

Then edit `vesper/agents/llm_client.py` to use OpenAI endpoint.

#### 4. Setup SmartThings (Optional)

For full Sim2Real validation with SmartThings cloud sync:

```bash
# OAuth credentials
export SMARTTHINGS_CLIENT_ID="your-oauth-client-id"
export SMARTTHINGS_CLIENT_SECRET="your-oauth-client-secret"

# App credentials for proactive state updates
export ST_APP_CLIENT_SECRET="your-app-client-secret"

# Start ngrok in a separate terminal
ngrok http 8443
```

### Running the Evaluation

The main evaluation script is `scripts/run_autonomous_eval.py`. It supports various configurations:

#### Basic Evaluation (1 scene, 2 days, no cloud)

```bash
conda activate vesper
python scripts/run_autonomous_eval.py \
    --num-scenes 1 \
    --num-days 2 \
    --headless
```

#### Full Evaluation with SmartThings (30 scenes, 5 days)

This is the configuration used in the paper:

```bash
conda activate vesper
python scripts/run_autonomous_eval.py \
    --num-scenes 30 \
    --num-days 5 \
    --with-smartthings \
    --time-acceleration 60 \
    --headless
```

**Parameters:**
- `--num-scenes N`: Number of HSSD scenes to evaluate (randomly sampled)
- `--num-days D`: Number of simulated days per scene (5 days = 120 simulated hours)
- `--with-smartthings`: Enable SmartThings cloud sync (requires ngrok + credentials)
- `--time-acceleration X`: Simulation speedup (60× means 1 sim-day = 24 real minutes)
- `--headless`: Run without 3D visualization (faster, lower resource usage)
- `--allow-fallback-tasks`: Continue even if LLM generation fails (uses emergency schedule)

#### Monitor Progress

The evaluation logs to both console and file:

```bash
# Follow live progress
tail -f logs/batch_30scenes_5days_visual.log

# Check navigation success
grep "Navigation trials" logs/batch_30scenes_5days_visual.log

# Check LLM generation
grep "LLM generated" logs/batch_30scenes_5days_visual.log

# Check for errors
grep -i "error\|failed" logs/batch_30scenes_5days_visual.log
```

### Evaluation Results

Results are saved to `results/vesper_autonomous_eval/`:

```
results/vesper_autonomous_eval/
├── eval_results.json          # Per-scene detailed results
├── eval_summary.txt           # Human-readable summary
└── eval_metadata.json         # Configuration and timestamps
```

#### View Results

```bash
# Human-readable summary
cat results/vesper_autonomous_eval/eval_summary.txt

# Quick stats
python -c "
import json
with open('results/vesper_autonomous_eval/eval_results.json') as f:
    data = json.load(f)
print(f'Scenes evaluated: {len(data)}')
print(f'Total nav trials: {sum(len(s[\"nav_trials\"]) for s in data)}')
print(f'Total tasks: {sum(s[\"tasks_scheduled\"] for s in data)}')
print(f'Total toggles: {sum(s[\"st_proximity_toggles\"] for s in data)}')
"
```

### Example Output

A successful 30-scene evaluation produces:

```
VESPER Autonomous Evaluation — Summary
============================================================
Scenes evaluated: 30
Simulated days (total): 145
Scenes fully complete (5/5 days): 28 / 30 (93.3%)
Wall-clock runtime: 23.5 h

Navigation:
  Total trials: 1,748
  Success rate: 99.5%
  Mean SPL: 1.000

LLM Activity Generation:
  Model: GPT-OSS 20B
  Schedules generated: 193
  Success rate: 98.0% (4 timeouts)
  Tasks scheduled: 1,701
  Unique task types: 432
  Avg tasks per schedule: 12.2

SmartThings Cloud Sync:
  Proximity toggles: 20,685
  Cloud state pushes: 20,685
  Data loss: 0
  Scenes with active sync: 30/30
```

### Performance Benchmarks

From the 30-scene evaluation (Apple M2 Pro, 32GB RAM):

| Metric | Value |
|--------|-------|
| Navigation success rate | 99.5% |
| Navigation SPL | 1.000 |
| LLM generation success | 98.0% |
| Event-bus P99 latency | 7 μs |
| Database write P99 | 2.84 ms |
| LLM generation P50 / P95 | 29.5s / 87.1s |
| SmartThings cloud updates | 20,685 (zero loss) |
| Articulated object interactions | 4,603 |
| Average room coverage | 51.2% |

### Troubleshooting Evaluation

#### LLM generation fails

```bash
# Check LMStudio is running
curl http://localhost:1234/v1/models

# Increase timeout in vesper/agents/llm_client.py
# Default: timeout=180 → increase to 300
```

#### Navigation failures

The evaluation automatically filters disconnected rooms (upper floors, isolated areas). If navigation still fails:

```bash
# Check reachability stats in log
grep "reachable rooms" logs/batch_30scenes_5days_visual.log

# Reduce scene complexity
python scripts/run_autonomous_eval.py --num-scenes 10  # Use fewer scenes
```

#### Out of memory

```bash
# Enable headless mode (saves ~4GB GPU memory)
--headless

# Reduce parallel scenes (default: 1 at a time)
# Split evaluation into batches:
python scripts/run_autonomous_eval.py --num-scenes 10 --num-days 5
python scripts/run_autonomous_eval.py --num-scenes 10 --num-days 5 --seed 42
python scripts/run_autonomous_eval.py --num-scenes 10 --num-days 5 --seed 84
```

#### Docker container limit

The evaluation launches 6 firmware containers per scene. If you hit Docker limits:

```bash
# Increase Docker resource limits (Docker Desktop → Settings → Resources)
# Or reduce containers per scene in run_autonomous_eval.py

# Clean up old containers
docker rm -f $(docker ps -aq --filter "name=vesper-fw")
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

---

## Tests

```bash
source .venv/bin/activate
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
| Firmware Emulation | QEMU — ARM Cortex-M3 (LM3S6965EVB) |
| Firmware Toolchain | arm-none-eabi-gcc |
| Firmware Language | C (no stdlib, integer-only math, ~1.6 KB) |
| Sensor Simulation | Pure Python |
| 3D Environment | Habitat 3.0 / Habitat-Sim (optional) |
| LLM Agents | OpenAI-compatible API (optional) |

## License

MIT License — See [LICENSE](LICENSE) for details.
