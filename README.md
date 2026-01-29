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
│   └── utils/               # Utilities
├── habitat-lab-official/    # Official Habitat 3.0 (cloned)
│   ├── examples/            # Interactive demos
│   ├── habitat-lab/         # Core library
│   └── habitat-baselines/   # RL training baselines
├── data/                    # Downloaded datasets
│   ├── scene_datasets/      # 3D scenes (HSSD, test scenes)
│   ├── replica_cad/         # ReplicaCAD apartments
│   ├── robots/              # Robot models (Fetch, Spot)
│   └── humanoids/           # Humanoid avatar models
├── configs/                 # YAML configurations
├── scripts/                 # Utility scripts
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
- [ ] **Phase 4: Advanced Embodiment** 🔄 In Progress
  - Multi-agent scenarios with social interactions
  - Advanced manipulation and object interaction
  - Semantic understanding and spatial reasoning
- [ ] **Phase 5: Real-World Integration** 
  - Hardware IoT device bridging
  - QEMU firmware simulation
  - Production deployment patterns

---

## 🎮 VESPER Interactive Demo

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
