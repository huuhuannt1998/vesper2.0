# Windows Setup Guide

This guide is for setting up Vesper on a Windows PC.

> ⚠️ **CRITICAL PLATFORM LIMITATION:**
> 
> **habitat-sim does NOT have conda packages for Windows!**
> 
> As of January 2025, habitat-sim only provides conda packages for:
> - ✅ Linux (linux-64)
> - ✅ macOS Apple Silicon (macOS-arm64)
> 
> **Windows Options:**
> 1. **Recommended**: Use WSL2 with Ubuntu (see below)
> 2. **Advanced**: Build habitat-sim from source (complex, not recommended)
> 3. **Limited**: Run Vesper without 3D simulation (IoT-only mode)

---

## Option 1: WSL2 Setup (Recommended for Windows)

WSL2 provides a full Linux environment on Windows with GPU support.

### Prerequisites

- Windows 10 (version 2004+) or Windows 11
- NVIDIA GPU with updated drivers (for CUDA in WSL2)

### Step 1.1: Install WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu-22.04

# Restart your computer when prompted
```

### Step 1.2: Set up NVIDIA CUDA in WSL2

```powershell
# Verify WSL2 sees your GPU (run inside WSL2 Ubuntu terminal)
nvidia-smi
```

If nvidia-smi doesn't work, install the [NVIDIA CUDA on WSL driver](https://developer.nvidia.com/cuda/wsl).

### Step 1.3: Install Miniconda in WSL2

```bash
# Inside WSL2 Ubuntu terminal
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Step 1.4: Follow Linux Setup

After WSL2 is configured, follow the standard Linux setup:

```bash
# Create conda environment
conda create -n vesper python=3.10 cmake=3.22 -y
conda activate vesper

# Install Habitat-Sim with CUDA support
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat

# Clone and install Vesper
git clone https://github.com/YOUR_USERNAME/vesper.git
cd vesper
pip install -e .

# Run tests
python -m pytest tests/ -v
```

---

## Option 2: Native Windows (IoT-Only Mode)

Run Vesper without the 3D Habitat simulator. Useful for testing IoT device simulation, protocols, and agents without 3D environment.

### Prerequisites

- Windows 10/11
- [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### Step 1: Clone the Repository

```powershell
git clone https://github.com/YOUR_USERNAME/vesper.git
cd vesper
```

## Step 2: Create Conda Environment

```powershell
# Create environment (NO habitat-sim on native Windows)
conda create -n vesper python=3.10 -y
conda activate vesper

# Install PyTorch (optional, for LLM features)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Step 3: Install Vesper (IoT-Only Mode)

```powershell
# Install project in editable mode (skip habitat dependencies)
pip install -e ".[no-habitat]"

# Or install minimal dependencies manually
pip install pyyaml pydantic aiohttp websockets
```

> **Note:** In IoT-only mode, the 3D simulation features are disabled.
> You can still test: IoT devices, protocols, network layer, event bus, and agents.

## Step 4: Run Tests (Partial)

```powershell
# Run tests that don't require habitat-sim
python -m pytest tests/test_devices.py tests/test_event_bus.py tests/test_protocol.py tests/test_network.py -v

# Skip habitat-related tests
python -m pytest tests/ -v --ignore=tests/test_habitat.py
```

## Step 5: Verify GPU is Working (for PyTorch)

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Troubleshooting

### CUDA not detected in WSL2
```bash
# Check NVIDIA driver in Windows first
# Then in WSL2:
nvidia-smi

# If not working, reinstall WSL CUDA driver from NVIDIA
```

### PyTorch CUDA issues on native Windows
```powershell
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
- Reduce `max_agents` in `configs/default.yaml`
- Enable headless mode (for WSL2): `simulation.headless: true`

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `conda activate vesper` | Activate environment |
| `python -m pytest tests/ -v --ignore=tests/test_habitat.py` | Run non-habitat tests |
| `nvidia-smi` | Check GPU status |
| `wsl` | Enter WSL2 Linux environment |

---

## Platform Comparison

| Feature | WSL2 (Recommended) | Native Windows |
|---------|-------------------|----------------|
| Habitat-Sim | ✅ Full support | ❌ Not available |
| 3D Simulation | ✅ Yes | ❌ No |
| IoT Devices | ✅ Yes | ✅ Yes |
| Event Bus | ✅ Yes | ✅ Yes |
| LLM Agents | ✅ Yes | ✅ Yes |
| CUDA/GPU | ✅ Yes (with setup) | ✅ Yes |

