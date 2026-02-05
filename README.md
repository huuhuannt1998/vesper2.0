# Vesper: Habitat 3.0 + IoT Interactive Simulation Testbed

A simulation platform extending Habitat 3.0 with IoT device interaction, LLM-controlled agents, and real-time event streaming.

## Platform Support

| Platform | 3D Simulation (Habitat) | IoT Simulation | Status |
|----------|-------------------------|----------------|--------|
| 🍎 macOS (Apple Silicon) | ✅ Full | ✅ Full | **Recommended** |
| 🐧 Linux (x64) | ✅ Full + CUDA | ✅ Full | **Production** |
| 🪟 Windows (WSL2) | ✅ Full | ✅ Full | Recommended for Windows |
| 🪟 Windows (Native) | ❌ Not available | ✅ IoT-only | Limited |

> **Note:** habitat-sim conda packages are only available for Linux and macOS Apple Silicon.
> Windows users should use WSL2 for full functionality.

## Features

- **3D Smart Home Simulation**: Interactive ReplicaCAD apartments with 6 different layouts
- **Humanoid Agent Control**: 12 diverse avatar models (male/female/neutral) controlled by LLMs
- **Smart IoT Device Placement**: Room-aware motion sensors, door contacts, light sensors
- **Real-time LLM Task Generation**: Continuous daily task assignment with context awareness
- **Event-Driven Architecture**: Pub/sub event bus with MQTT support for real IoT integration
- **Multi-Dataset Support**: HSSD and ReplicaCAD environments with official Habitat 3.0
- **Cross-Platform**: Develop on Mac/Windows, deploy on GPU workstations

## Quick Start

### Prerequisites

- Python 3.9+
- Conda (recommended) or pip
- Git

### Installation

**macOS (Apple Silicon)** - See [MACOS_SETUP.md](MACOS_SETUP.md) for detailed instructions.

```bash
# Clone the repository
cd /path/to/vesper

# Create conda environment (Python 3.9 required for macOS ARM)
conda create -n vesper python=3.9 cmake=3.22 -y
conda activate vesper

# Install Habitat-Sim (with Bullet physics)
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Install project dependencies
pip install -e .
pip install pygame pybullet
```

**Windows** - See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for WSL2 setup or IoT-only mode.

**Linux** - Similar to macOS, but add `headless` flag for servers:
```bash
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```

### Verify Installation

```bash
python -c "import vesper; print(f'Vesper v{vesper.__version__}')"
python -c "import habitat_sim; print(f'Habitat-Sim v{habitat_sim.__version__}')"
```

## 🎮 3D Environment Setup (Habitat 3.0)

VESPER uses the official **Habitat 3.0** framework for 3D simulation with humanoid agents and robots.

### Step 1: Install Habitat-Lab

```bash
# Clone habitat-lab (the high-level library)
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git habitat-lab-official
cd habitat-lab-official

# Install habitat-lab
pip install -e habitat-lab

# Install interactive dependencies
pip install pygame pybullet
conda install -c conda-forge pybullet  # If pip fails on macOS

# Symlink data folder
ln -sf ../data data
cd ..
```

### Step 2: Download Datasets

Download all required datasets for Habitat 3.0 (~12 GB total):

```bash
# Essential datasets
python -m habitat_sim.utils.datasets_download --uids \
    habitat_test_scenes \
    replica_cad_dataset \
    hab3_bench_assets \
    habitat_humanoids \
    hab_fetch \
    --data-path data/

# Rearrangement task episodes
python -m habitat_sim.utils.datasets_download --uids \
    rearrange_pick_dataset_v0 \
    rearrange_dataset_v2 \
    --data-path data/

# (Optional) HSSD scenes - large download (~8 GB)
python -m habitat_sim.utils.datasets_download --uids hssd-hab --data-path data/
```

**Dataset Overview:**

| Dataset | Size | Purpose |
|---------|------|---------|
| `habitat_test_scenes` | ~150 MB | Basic test scenes (castle, apartment) |
| `replica_cad_dataset` | ~2 GB | Interactive ReplicaCAD apartments |
| `hab3_bench_assets` | ~500 MB | Habitat 3.0 benchmark assets |
| `habitat_humanoids` | ~300 MB | Humanoid avatar models |
| `hab_fetch` | ~50 MB | Fetch robot model |
| `hssd-hab` | ~8 GB | HSSD photorealistic scenes |

### Step 3: Run Interactive Demo

```bash
# Start the official Habitat 3.0 interactive viewer
cd habitat-lab-official
python examples/interactive_play.py --never-end
```

**Controls:**

| Keys | Action |
|------|--------|
| **I/J/K/L** | Move robot base forward/left/backward/right |
| **W/A/S/D** | Move arm end-effector |
| **E/Q** | Move arm up/down |
| **Mouse** | Look around |

### Step 4: Quick Scene Viewer

For a quick view of any 3D scene without the full rearrangement task:

```bash
# View a scene with matplotlib
python scripts/simple_viewer.py --scene data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
```

## Project Structure

```
vesper/
├── vesper/                  # Main VESPER package
│   ├── core/                # Event bus and environment
│   ├── devices/             # IoT device models
│   ├── protocol/            # Message types and codec
│   ├── network/             # Transport, router, broker
│   ├── agents/              # LLM-controlled agents
│   ├── habitat/             # Habitat-Sim integration
│   │   ├── iot_overlay.py   # IoT device visualization
│   │   ├── iot_bridge.py    # MQTT-based IoT communication
│   │   ├── iot_config_menu.py # Interactive config menu UI
│   │   ├── humanoid.py      # Humanoid avatar controller
│   │   └── vesper_integration.py # Main integration module
│   └── utils/               # Utilities
├── habitat-lab-official/    # Official Habitat 3.0 (cloned)
│   ├── examples/            # Interactive demos
│   ├── habitat-lab/         # Core library
│   └── habitat-baselines/   # RL training baselines
├── scripts/
│   ├── vesper_objectnav.py  # Main ObjectNav demo with IoT
│   └── download_datasets.py # Dataset download utility
├── data/                    # Downloaded datasets
│   ├── scene_datasets/      # 3D scenes (HSSD, test scenes)
│   ├── replica_cad/         # ReplicaCAD apartments
│   ├── robots/              # Robot models (Fetch, Spot)
│   └── humanoids/           # Humanoid avatar models
├── configs/                 # YAML configurations
└── tests/                   # Test suite (152 tests)
```

## Usage

```python
from vesper.simulation import Simulation
from vesper.agents import SmartAgent, SmartAgentConfig

# Create simulation
with Simulation() as sim:
    # Spawn LLM-controlled agent
    config = SmartAgentConfig(
        name="HomeAssistant",
        model="openai/gpt-oss-120b",
        use_llm=True,
    )
    agent = sim.agent_controller.spawn(SmartAgent, config=config)
    
    # Assign a task
    agent.set_task("Monitor the house and lock doors when no motion detected")
    
    # Run simulation
    sim.run(duration=60.0)
```

## 🖥️ Windows GPU Testing Checklist

When testing on the Windows PC with NVIDIA GPU, run these steps:

### 1. Setup (First Time Only)

```powershell
# Clone/copy the project
cd C:\Projects\vesper

# Create conda environment with CUDA support
conda create -n vesper python=3.10 cmake=3.22 -y
conda activate vesper

# Install Habitat-Sim with CUDA + Bullet physics (Linux only - headless for servers)
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat

# Install Vesper
pip install -e .
pip install httpx python-dotenv
```

### 2. Run All Tests

```powershell
# Verify all 131 tests pass
python -m pytest tests/ -v
```

### 3. Test LLM Agent (OpenWebUI)

```powershell
# Create .env file with your API key
copy .env.example .env
# Edit .env and set OPENWEBUI_API_KEY

# Test LLM connection
python -c "from vesper.agents import LLMClient; c = LLMClient(); print(c.complete('Hello').content)"
```

### 4. Download Dataset & Test Habitat-Sim

```powershell
# Download HSSD dataset (~5GB)
python scripts/download_datasets.py --dataset hssd-hab

# Test with real 3D scene
python -c "
from vesper.habitat import HabitatSimulator, SimulatorConfig
sim = HabitatSimulator(SimulatorConfig(scene_path='data/hssd-hab/scenes/102816036.glb'))
if sim.initialize():
    print('Habitat-Sim working!')
    agent_id = sim.add_agent()
    obs = sim.step()
    print(f'Got observations: {list(obs[agent_id].keys())}')
    sim.close()
"
```

### 5. Full Integration Test

```powershell
# Run end-to-end simulation with LLM agent
python -c "
from vesper.simulation import Simulation
from vesper.agents import SmartAgent, SmartAgentConfig

with Simulation() as sim:
    agent = sim.agent_controller.spawn(
        SmartAgent,
        SmartAgentConfig(name='Test', use_llm=True)
    )
    agent.set_task('Check all doors are locked')
    sim.run(duration=10.0)
    print(f'Completed {sim.stats.ticks} ticks')
"
```

### Expected Results

| Test | Expected |
|------|----------|
| pytest | 131 passed |
| LLM connection | Response from OpenWebUI |
| Habitat-Sim | RGB/depth observations |
| Full simulation | 100+ ticks completed |

---

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=vesper
```

## License

MIT License - See LICENSE for details.

## Roadmap

- [x] **Phase 1: Habitat 3.0 Official Integration** ✅ Complete
  - Official Habitat-Lab integration with ReplicaCAD apartments
  - Dataset downloads and 3D scene loading
- [x] **Phase 2: Smart IoT Foundation** ✅ Complete  
  - Device models (motion, contact, light sensors, smart doors)
  - Event bus architecture with real-time pub/sub
  - Device manager with room-aware placement
  - MQTT transport for real IoT integration
- [x] **Phase 3: LLM Agent Framework** ✅ Complete
  - Humanoid avatar control (12 models: male/female/neutral)
  - LLM task generation with environmental context
  - Continuous task assignment and completion tracking
  - Multiple apartment layouts (apt_0 through apt_5)
- [x] **Phase 4: ObjectNav + IoT Integration** ✅ Complete
  - HSSD scene navigation with GreedyGeodesicFollower
  - Real-time IoT device overlay in 3D environment
  - Interactive config menu for adding devices/automation rules
  - Motion sensor detection based on room entry
  - Automation rules (motion → lights)
  - MQTT-based pub/sub communication
- [ ] **Phase 5: Realistic IoT Sensors & Humanoid Embodiment** 🔄 In Progress
  - **Realistic Motion Sensors**: PIR-style detection with cone/range, angle, and cooldown
  - **Security Cameras**: Track humanoid position, field of view visualization
  - **Virtual Humanoid Avatar**: First-person (eye-level) and third-person views
  - **SmartThings Integration**: Bridge virtual devices to real IoT platform
- [ ] **Phase 6: Autonomous Daily Life Simulation** 🎯 Ultimate Goal
  - **Synchronized Time**: Environment time matches real-world time
  - **Realistic Task Duration**: Cooking takes real cooking time, etc.
  - **LLM Daily Task Generation**: AI generates contextual daily activities
  - **Task History Database**: Store completed tasks for analysis
  - **Long-term Autonomy**: Humanoids maintain daily routines for months
  - **Event Stream Generation**: Continuous realistic sensor data output
- [x] **Phase 7: Real-World Integration** ✅ In Progress
  - Hardware IoT device bridging
  - [x] QEMU firmware simulation (ARM Cortex-M, ESP32, RISC-V)
  - Production deployment patterns

---

## 🔧 Sensor Simulation (No Hardware Required!)

VESPER includes a complete **pure Python sensor simulation** system. No QEMU, no ARM toolchain, no ESP32, no physical devices needed!

### Quick Start - Simulated Sensors

```bash
# Run a whole-house sensor simulation
python scripts/simulated_sensors_demo.py

# Run specific room sensors
python scripts/simulated_sensors_demo.py --room kitchen

# Run a single sensor type
python scripts/simulated_sensors_demo.py --single thermostat

# Interactive command mode
python scripts/simulated_sensors_demo.py --interactive
```

### Supported Sensor Types

| Sensor Type | Simulated Behavior | Events Generated |
|-------------|-------------------|------------------|
| Motion (PIR) | Random detection with cooldown | `motion: true/false` |
| Temperature | Slow drift with noise | `temperature: 22.5` |
| Humidity | Gradual changes | `humidity: 45.0` |
| Door/Window | Contact sensor state | `open: true/false`, `state_change` |
| Light (Lux) | Day/night cycle simulation | `lux: 350`, `light_level: normal` |
| Smoke | Rare smoke events with alarm | `smoke_level`, `alarm` |
| CO2 | Indoor air quality simulation | `co2_ppm`, `air_quality` |
| Water Leak | Rare leak detection | `leak`, `moisture`, `alert` |
| Thermostat | HVAC simulation | `current_temp`, `target_temp`, `hvac_state` |
| Smart Plug | Power monitoring | `on`, `power_watts`, `energy_kwh` |
| Multi-Sensor | Combined motion+temp+humidity+light | All of the above |

### Programmatic Usage

```python
from vesper.firmware import (
    SensorNetwork,
    SensorConfig,
    SensorType,
    create_whole_house_sensors,
)

# Create a complete house sensor network
network = create_whole_house_sensors()

# Or create custom sensors
network = SensorNetwork()
network.add_sensor(SensorConfig(
    sensor_type=SensorType.MOTION,
    device_id="living_room_motion",
    location="living_room",
    motion_probability=0.15,
))

# Handle sensor events
network.on_sensor_data(lambda device, key, value: 
    print(f"{device}: {key}={value}")
)

# Run the simulation
await network.start()
await asyncio.sleep(60)  # Run for 60 seconds
await network.stop()

# Send commands to sensors
response = network.send_command("living_room_motion", "GET_MOTION")
```

### Room Presets

```python
from vesper.firmware import (
    create_living_room_sensors,  # Multi-sensor, smart plug, light
    create_bedroom_sensors,      # Motion, temperature, window
    create_kitchen_sensors,      # Motion, smoke, temp, water leak
    create_bathroom_sensors,     # Motion, humidity, water leak
)
```

---

## 🔧 QEMU Firmware Emulation (Optional Advanced)

For advanced users who want to run **actual compiled firmware**, VESPER also supports QEMU emulation. This is **optional** - the simulated sensors above work without any of this.

### Supported Architectures

| Architecture | QEMU Binary | Example Boards |
|--------------|-------------|----------------|
| ARM Cortex-M0/M3/M4/M7 | `qemu-system-arm` | STM32F4 Discovery, nRF52840, LM3S6965 |
| ESP32 | `qemu-system-xtensa` | ESP32 DevKit (requires espressif/qemu) |
| ESP32-C3 (RISC-V) | `qemu-system-riscv32` | ESP32-C3 DevKit |
| RISC-V | `qemu-system-riscv32/64` | SiFive HiFive1 |

### QEMU Setup (Only if you need real firmware)

```bash
# Install QEMU (if not already installed)
brew install qemu                    # macOS
sudo apt install qemu-system-arm     # Ubuntu/Debian

# Build sample firmware (requires ARM GCC toolchain)
cd vesper/firmware/samples
make
cd ../../..

# Run with real QEMU emulation
python scripts/qemu_firmware_demo.py --firmware vesper/firmware/samples/sensor_firmware.elf
```

### QEMU Programmatic Usage

```python
from vesper.firmware import QEMURunner, QEMUConfig, BoardType, VesperFirmwareBridge

# Create QEMU runner for STM32
config = QEMUConfig(
    board=BoardType.STM32F4_DISCOVERY,
    firmware_path="path/to/firmware.elf",
    enable_serial=True,
)
runner = QEMURunner(config)

# Create bridge to VESPER event bus
bridge = VesperFirmwareBridge(runner, event_bus)
await bridge.start()

# Send commands to firmware
response = await bridge.send_command("get_temperature")
print(f"Temperature: {response.value}°C")

# Firmware events appear on event bus
# event_bus.subscribe("firmware.*", handle_event)
```

### Sample Firmware Protocol

The sample firmware uses a simple text protocol:

| Command | Response | Description |
|---------|----------|-------------|
| `GET_TEMP` | `TEMP:22.5` | Get temperature |
| `GET_HUMIDITY` | `HUMIDITY:45.0` | Get humidity |
| `GET_ALL` | Multiple lines | Get all sensor data |
| `SET_LED:1` | `ACK:SET_LED` | Turn on LED |
| `STATUS` | `STATUS:OK` | Check firmware status |
| `IDENTIFY` | Device info | Get device identification |

---

## 🎮 VESPER ObjectNav Demo (Recommended)

Run the interactive 3D ObjectNav demo with IoT integration:

```bash
python scripts/vesper_objectnav_camera_humanoid.py
```

**Features:**
- ✅ First-person navigation in photorealistic 3D houses (HSSD dataset)
- ✅ Articulated humanoid avatar with walking animations
- ✅ Bird's-eye view camera (5m above, looking straight down)
- ✅ **Automatic IoT sensor placement** (motion sensors + cameras in every room)
- ✅ **Simulated firmware sensors** - No hardware required!
  - Motion sensors trigger when humanoid enters rooms
  - Temperature rises by +1.5°C when room is occupied
  - Humidity increases by +5% when room is occupied
  - All events flow to VESPER event bus
- ✅ LLM-powered task generation (optional)
- ✅ Real-time sensor data visualization

**What you'll see:**
```
[BRIDGE] Firmware sensors active: 60
[BRIDGE] Rooms with environmental sensors: 17

Room: living_room
├── Motion Sensor (3D) → Triggers firmware sensor
├── Security Camera (3D) → Monitors room
├── Temperature Sensor (firmware) → 22.0°C → 23.5°C (occupied)
└── Humidity Sensor (firmware) → 45% → 50% (occupied)
```

**Verify integration:**
```bash
python scripts/verify_sensor_integration.py
```

```bash
# Start the ObjectNav demo
python scripts/vesper_objectnav.py
```

### ObjectNav Controls

| Key | Action |
|-----|--------|
| **W/↑** | Move forward |
| **S/↓** | Move backward |
| **A/←** | Turn left |
| **D/→** | Turn right |
| **Q/E** | Look up/down |
| **G** | Set random navigation goal |
| **T** | Generate LLM task |
| **N** | Auto-navigate to goal |
| **I** | Toggle IoT device panel |
| **C** | Toggle config menu (add devices/rules) |
| **L** | Print event log to terminal |
| **V** | Toggle 1st/3rd person view |
| **H** | Toggle help |
| **ESC** | Quit |

### ObjectNav Features

- **3D Navigation**: Explore HSSD scenes with first/third-person views
- **IoT Device Panel (I key)**: Real-time device states by room
  - Motion sensors trigger when entering rooms
  - Shows current room and recent events
- **Config Menu (C key)**: Interactive UI with mouse click support
  - Add devices: select type + room from dropdowns
  - Create automation rules: trigger device → action device
  - View existing devices and rules
- **Automation**: Motion sensors automatically trigger lights
- **Event Logging**: Press L to see IoT events in terminal

---

## 🎮 VESPER Interactive Demo (Legacy)

Run the full VESPER demo with IoT devices and LLM-controlled agents:

```bash
# Interactive demo with pygame UI
python scripts/vesper_demo.py

# Headless mode (no graphics)
python scripts/vesper_demo.py --headless --duration 30

# Without LLM (faster, no external dependencies)
python scripts/vesper_demo.py --no-llm
```

### Demo Features

- **12 IoT Devices**: Motion sensors, smart doors, contact sensors, light sensors
- **2 LLM Agents**: HomeBot (home assistant), SecureBot (security monitor)
- **Real-time Event Bus**: Pub/sub communication between devices
- **Pygame Visualization**: Live status of all devices and agents

### Demo Controls

| Key | Action |
|-----|--------|
| **SPACE** | Trigger motion event |
| **D** | Toggle a door |
| **ESC** | Quit |

---

## 🤖 LLM Configuration

VESPER supports multiple LLM backends:

### LM Studio (Recommended for Development)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load any model (e.g., Llama, Mistral, Qwen)
3. Go to "Local Server" tab → Click "Start Server"
4. Configure `.env`:

```bash
OPENWEBUI_URL=http://localhost:1234/v1/chat/completions
OPENWEBUI_API_KEY=lm-studio
```

### OpenWebUI (School Network)

```bash
OPENWEBUI_URL=http://cci-siscluster1.charlotte.edu:8080/api/chat/completions
OPENWEBUI_API_KEY=your-api-key
```

### Testing LLM Connection

```bash
python -c "
from vesper.agents import LLMClient, LLMConfig
client = LLMClient(LLMConfig(max_tokens=50))
print(client.complete('Hello!').content)
"
```


**All 152 tests passing ✅**

---

## Troubleshooting

### Pygame window doesn't appear
Make sure you have SDL2 installed:
```bash
# macOS
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
pip install pygame
```

### Dataset download prompts for confirmation
Use `yes |` to auto-confirm:
```bash
yes | python -m habitat_sim.utils.datasets_download --uids <dataset> --data-path data/
```

### "wget not found" error
```bash
brew install wget  # macOS
sudo apt install wget  # Linux
```

### Navmesh warnings
These are non-critical warnings about navigation meshes for staging scenes. The simulation will still run.
