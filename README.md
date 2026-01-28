# Vesper: Habitat 3.0 + IoT Interactive Simulation Testbed

A simulation platform extending Habitat 3.0 with IoT device interaction, LLM-controlled agents, and real-time event streaming.

## Features

- **IoT Device Simulation**: Motion sensors, contact sensors, smart doors, and light sensors
- **Event-Driven Architecture**: Pub/sub event bus for device communication
- **LLM Agent Control**: Humanoid agents controlled by language models
- **Multi-Dataset Support**: HSSD and ReplicaCAD environments
- **Cross-Platform**: Develop on Mac/Windows, deploy on GPU workstations

## Quick Start

### Prerequisites

- Python 3.9+
- Conda (recommended) or pip
- Git

### Installation

```bash
# Clone the repository
cd /path/to/vesper

# Create conda environment
conda create -n vesper python=3.10 -y
conda activate vesper

# Install Habitat (with Bullet physics)
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Install project dependencies
pip install -e .
```

### Verify Installation

```bash
python -c "import vesper; print(f'Vesper v{vesper.__version__}')"
```

### Download Datasets

```bash
python scripts/download_datasets.py --dataset hssd-hab
```

## Project Structure

```
vesper/
├── vesper/              # Main package
│   ├── core/            # Event bus and environment
│   ├── devices/         # IoT device models
│   ├── protocol/        # Message types and codec
│   ├── network/         # Transport, router, broker
│   ├── agents/          # LLM-controlled agents
│   ├── habitat/         # Habitat-Sim integration
│   └── utils/           # Utilities
├── configs/             # YAML configurations
├── scripts/             # Utility scripts
└── tests/               # Test suite (131 tests)
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
conda create -n vesper python=3.10 -y
conda activate vesper

# Install Habitat-Sim with CUDA + Bullet physics
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

- [x] Phase 1: Foundation (device models, event bus)
- [x] Phase 2: Protocol & Network layer
- [x] Phase 3: LLM Agent Framework
- [x] Phase 4: Habitat-Sim Integration
- [x] Phase 5: QEMU Firmware Integration

**All 152 tests passing ✅**
