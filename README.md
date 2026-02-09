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
- **SmartThings Bi-Directional Sync** — Devices appear in the Samsung SmartThings app; toggle on/off from your phone and the firmware responds
- **SmartThings Schema Protocol** — Full cloud-to-cloud integration (discovery, state refresh, commands, callbacks)
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

4. Save and note your **Client ID** and **Client Secret**

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

export SMARTTHINGS_CLIENT_ID="your-client-id"
export SMARTTHINGS_CLIENT_SECRET="your-client-secret"

python scripts/unified_smartthings_firmware.py
```

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

## Architecture

### Command Flow (Phone → Firmware)

1. You tap **Off → On** in the SmartThings app
2. SmartThings cloud sends a `commandRequest` to the ngrok URL
3. VESPER Schema Connector receives the webhook, extracts the command
4. Connector opens a TCP connection to the device's Docker container
5. Sends `ON\n` over the QEMU serial port
6. ARM firmware processes the command, sets GPIO, responds `SWITCH:on\nACK\n`
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
├── tests/                          # Test suite
├── configs/                        # YAML configurations
└── data/                           # Habitat 3.0 datasets (optional)
```

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

Requires Habitat-Sim via conda. See [Habitat Setup](#3d-habitat-setup-optional).

```bash
python scripts/vesper_objectnav_camera_humanoid.py
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
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
