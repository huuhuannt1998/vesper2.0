# Windows GPU Setup Guide

This guide is for setting up Vesper on a Windows PC with an NVIDIA GPU.

## Prerequisites

- Windows 10/11
- NVIDIA GPU with CUDA support
- [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [CUDA Toolkit 11.7+](https://developer.nvidia.com/cuda-downloads)
- Git

## Step 1: Clone the Repository

```powershell
git clone https://github.com/YOUR_USERNAME/vesper.git
cd vesper
```

## Step 2: Create Conda Environment

```powershell
# Create environment
conda create -n vesper python=3.10 -y
conda activate vesper

# Install PyTorch with CUDA (check your CUDA version first)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Step 3: Install Habitat-Sim with CUDA

```powershell
# Install Habitat-Sim with Bullet physics (CUDA enabled)
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat

# Verify installation
python -c "import habitat_sim; print(habitat_sim.__version__)"
```

## Step 4: Install Vesper

```powershell
# Install project in editable mode
pip install -e .

# Or install all optional dependencies
pip install -e ".[all]"
```

## Step 5: Download Datasets

```powershell
# Download HSSD dataset (~8GB)
python scripts/download_datasets.py --dataset hssd-hab

# Verify download
python scripts/download_datasets.py --verify hssd-hab
```

## Step 6: Run Tests

```powershell
# Run all tests
python -m pytest tests/ -v

# Expected: 39 passed
```

## Step 7: Verify GPU is Working

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX XXXX
```

## Troubleshooting

### CUDA not detected
```powershell
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Habitat-Sim import error
```powershell
# Try reinstalling with specific build
conda install habitat-sim=0.3.0 withbullet -c conda-forge -c aihabitat
```

### Out of Memory
- Reduce `max_agents` in `configs/default.yaml`
- Enable headless mode: `simulation.headless: true`

## Running the Full Simulation

```powershell
# After Phase 5 is complete, run:
python -m vesper.run --config configs/default.yaml
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `conda activate vesper` | Activate environment |
| `python -m pytest tests/ -v` | Run tests |
| `python scripts/download_datasets.py --list` | List datasets |
| `nvidia-smi` | Check GPU status |
