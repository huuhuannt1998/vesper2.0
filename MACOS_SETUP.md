# macOS Setup Guide (Apple Silicon)

This guide is for setting up Vesper on **macOS with Apple Silicon (M1/M2/M3/M4)**.

> ⚠️ **Important Platform Notes:**
> - **habitat-sim** conda packages are **only available** for:
>   - ✅ Linux (linux-64)
>   - ✅ macOS Apple Silicon (macOS-arm64)
> - **NOT supported** via conda:
>   - ❌ Windows (must build from source - complex)
>   - ❌ macOS Intel (macOS-64) - last supported version was 0.3.1

## Prerequisites

- macOS with Apple Silicon (M1, M2, M3, M4)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- Git
- Xcode Command Line Tools

## Step 1: Install Miniconda (if not installed)

```bash
# Download Miniconda for Apple Silicon
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install Miniconda
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3

# Initialize conda
$HOME/miniconda3/bin/conda init zsh

# Restart terminal or run:
source ~/.zshrc

# Verify installation
conda --version
```

## Step 2: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/vesper.git
cd vesper
```

## Step 3: Create Conda Environment

```bash
# Create environment with Python 3.9 (required for habitat-sim on macOS arm64)
conda create -n vesper python=3.9 cmake=3.22 -y
conda activate vesper
```

## Step 4: Install Habitat-Sim

```bash
# Install Habitat-Sim with Bullet physics
# Note: 'headless' flag is NOT supported on macOS
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Verify installation
python -c "import habitat_sim; print(f'Habitat-Sim v{habitat_sim.__version__}')"
```

## Step 5: Install Vesper

```bash
# Install project dependencies
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

## Step 6: Download Datasets

```bash
# Download HSSD dataset (~8GB)
python scripts/download_datasets.py --dataset hssd-hab

# Verify download
python scripts/download_datasets.py --verify hssd-hab
```

## Step 7: Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Expected: All tests pass
```

## Step 8: Verify Installation

```bash
# Check Vesper version
python -c "import vesper; print(f'Vesper v{vesper.__version__}')"

# Check Habitat-Sim
python -c "import habitat_sim; print(f'Habitat-Sim v{habitat_sim.__version__}')"
```

## Troubleshooting

### Habitat-Sim import error

```bash
# Try reinstalling with specific version
conda uninstall habitat-sim
conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat
```

### CMake version issues

```bash
# Update cmake
conda install cmake=3.27 -c conda-forge
```

### Python version conflicts

habitat-sim on macOS arm64 requires Python 3.9. Other platforms may support Python 3.10-3.12.

```bash
# Check Python version
python --version

# If needed, recreate environment with correct version
conda deactivate
conda remove -n vesper --all
conda create -n vesper python=3.9 cmake=3.22 -y
```

### OpenGL/Display issues

On macOS, the `headless` flag is NOT supported. If you encounter display issues:

```bash
# Make sure you're not using headless mode in config
# In configs/default.yaml, set:
# simulation:
#   headless: false
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `conda activate vesper` | Activate environment |
| `python -m pytest tests/ -v` | Run tests |
| `python scripts/download_datasets.py --list` | List datasets |
| `python -c "import habitat_sim"` | Verify habitat-sim |

## Running the Full Simulation

```bash
# After setup is complete:
python -m vesper.run --config configs/default.yaml
```

---

## Development Notes

### Cross-Platform Development Strategy

Since habitat-sim has limited platform support, consider this workflow:

1. **Development on Mac (Apple Silicon)**: Full 3D simulation with habitat-sim
2. **Windows**: Limited to IoT simulation without 3D (or use WSL2 with Linux)
3. **Linux GPU Servers**: Full simulation with CUDA support

### Windows Alternative: WSL2

For Windows users, consider using WSL2 (Windows Subsystem for Linux):

```powershell
# In PowerShell (Admin)
wsl --install -d Ubuntu-22.04

# Then follow Linux setup instructions inside WSL2
```

This provides a Linux environment on Windows with full habitat-sim support.
