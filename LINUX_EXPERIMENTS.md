# Running VESPER's Flagship Experiments on Linux

This guide explains how to set up a Linux environment and run the three
flagship MobiCom experiments (RQ-N1, RQ-N2, Trace Validation) that require
`mac80211_hwsim` — the Linux kernel module for virtual 802.11 radios.

**macOS cannot run these experiments** because `mac80211_hwsim` is a Linux
kernel module. You need a Linux host or VM.

---

## Step 1: Choose a Linux Environment

### Option A: UTM VM on macOS (Free, Recommended for Local Development)

1. Download [UTM](https://mac.getutm.app/) (free on GitHub, or paid on App Store)
2. Download [Ubuntu 22.04 Server ARM64 ISO](https://releases.ubuntu.com/jammy/)
3. Create a new VM:
   - **Type:** Virtualize → Linux
   - **RAM:** 8 GB minimum (16 GB recommended)
   - **CPU:** 4 cores minimum
   - **Disk:** 40 GB
   - **Network:** Shared Network (NAT) — important for internet access
4. Install Ubuntu from the ISO
5. After install, set up SSH for file transfer:
   ```bash
   # In the VM:
   sudo apt install openssh-server
   # On your Mac, find the VM's IP:
   # UTM → VM Settings → Network → note the IP
   ```
6. Transfer the VESPER codebase:
   ```bash
   # On macOS:
   rsync -avz --exclude='.venv' --exclude='node_modules' --exclude='data/' \
       ~/Desktop/vesper/ user@<VM_IP>:~/vesper/
   ```

### Option B: AWS EC2 (Fast, ~$0.17/hr)

```bash
# Launch Ubuntu 22.04 instance
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type t3.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3"}}]'

# SSH in
ssh -i your-key.pem ubuntu@<PUBLIC_IP>

# Transfer code
rsync -avz -e "ssh -i your-key.pem" \
    --exclude='.venv' --exclude='data/' \
    ~/Desktop/vesper/ ubuntu@<PUBLIC_IP>:~/vesper/
```

**Cost estimate:** t3.xlarge ($0.1664/hr) × 4 hours = ~$0.67 total

### Option C: GCP Compute Engine (~$0.13/hr)

```bash
gcloud compute instances create vesper-experiments \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=40GB

gcloud compute scp --recurse ~/Desktop/vesper user@vesper-experiments:~/
```

### Option D: Hetzner Cloud (Cheapest, ~€0.01/hr)

```bash
# Create CX31 server (4 vCPU, 8 GB RAM)
hcloud server create --name vesper --type cx31 --image ubuntu-22.04
```

---

## Step 2: Set Up the Linux Environment

SSH into your Linux machine, then:

```bash
cd ~/vesper

# Run the automated setup (installs everything)
sudo bash scripts/setup_linux_vm.sh

# After system setup completes, run user-level setup:
bash scripts/setup_linux_vm.sh --user

# Verify everything works:
bash scripts/setup_linux_vm.sh --smoke
```

The setup script installs:
- `mac80211_hwsim` kernel module (12 virtual WiFi radios)
- Docker Engine + Docker Compose v2
- Mininet-WiFi (with `wmediumd` path-loss propagation)
- `tshark` / `tcpdump` for packet capture
- `hostapd`, `wpa_supplicant`, `dnsmasq`
- `iperf3` for throughput testing
- Python 3.10+ venv with all VESPER dependencies
- Built Docker images (`vesper-router`, `vesper-esp32`)

**Time: ~15-20 minutes** (mostly Docker image builds)

---

## Step 3: Run All Experiments

```bash
cd ~/vesper
source .venv/bin/activate

# Run everything (RQ-N1 + RQ-N2 + Trace Validation)
bash scripts/run_all_flagship_experiments.sh

# Or run specific experiments:
bash scripts/run_all_flagship_experiments.sh --rqn1-only    # ~30-45 min
bash scripts/run_all_flagship_experiments.sh --rqn2-only    # ~2-3 hours
bash scripts/run_all_flagship_experiments.sh --trace-only   # ~5-10 min
```

### What Each Experiment Does

#### RQ-N1: Bridge vs. 802.11 Divergence (~30-45 min)

Runs the **same** 37 attacks over two network backends:
1. **Bridge mode:** 6 ESP32 containers on Docker bridge (`docker0`)
2. **802.11 mode:** Same 6 containers on Mininet-WiFi (`mac80211_hwsim` + `hostapd` WPA2-PSK)

Per trial (5 trials × 2 modes = 10 runs):
- Pre-attack RTT baseline (ICMP + TCP handshake, 50 probes each)
- 18 firmware attacks via QEMU serial
- 14 network/MQTT attacks
- 11 WiFi-layer attacks (802.11 mode only: deauth, evil twin, ARP spoof, etc.)
- Post-attack RTT measurement
- 802.11 retransmission count (`tshark wlan.fc.retry`)
- Deauth → reconnection latency measurement
- Full pcap capture per trial

**Outputs:** `tab_bridge_vs_80211.tex`, `fig_rtt_bridge_vs_80211.pdf`

#### RQ-N2: WiFi Hardening Sweep (~2-3 hours)

Tests 8 WiFi configurations (all combinations of 4 binary factors):

| Config | Encryption | PMF | AP Isolation | MQTT |
|--------|-----------|-----|-------------|------|
| 0 (baseline) | WPA2-PSK | off | off | anonymous |
| 1 | WPA2-PSK | off | off | Auth+TLS |
| 2 | WPA2-PSK | off | on | anonymous |
| 3 | WPA2-PSK | off | on | Auth+TLS |
| 4 | WPA2-PSK | required | off | anonymous |
| 5 | WPA2-PSK | required | off | Auth+TLS |
| 6 | WPA3-SAE | required | off | anonymous |
| 7 (hardened) | WPA3-SAE | required | on | Auth+TLS |

Per config per trial (8 × 5 = 40 runs):
- Start Mininet-WiFi with specific `hostapd` + firewall config
- Pre-attack availability baseline
- All 37 attacks (firmware + WiFi + network)
- Post-attack availability
- `iperf3` throughput measurement
- Reconnection latency after AP restart

**Outputs:** `tab_hardening_measured.tex`, `fig_hardening_pareto.pdf`

#### Trace Validation (~5-10 min)

Analyzes the **real pcap files** from the 30-scene autonomous evaluation
(90 pcaps, 8.3 MB total) using `tshark`:
- Flow count per hour (unique TCP/UDP conversations)
- Packet-size distribution (CDF percentiles)
- Per-minute burstiness (coefficient of variation)
- Diurnal pattern (hourly distribution, Pearson r vs UNSW IoT reference)
- Keepalive periodicity detection
- TCP flag breakdown

**Outputs:** `tab_trace_validation.tex`, `fig_pkt_size_cdf.pdf`, `fig_diurnal.pdf`

---

## Step 4: Copy Results Back to macOS

```bash
# On macOS:
rsync -avz user@<VM_IP>:~/vesper/results/flagship_* ~/Desktop/vesper/results/
rsync -avz user@<VM_IP>:~/vesper/paper-latex/tables/ ~/Desktop/vesper/paper-latex/tables/
rsync -avz user@<VM_IP>:~/vesper/paper-latex/figures/ ~/Desktop/vesper/paper-latex/figures/
```

The experiment scripts automatically copy generated tables and figures to
`paper-latex/tables/` and `paper-latex/figures/`, so after syncing back
to your Mac, the paper is ready to compile.

---

## Step 5: Verify and Compile Paper

```bash
cd ~/Desktop/vesper/paper-latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or upload to Overleaf if that's your workflow.

---

## Troubleshooting

### `mac80211_hwsim: module not found`
```bash
# Install kernel modules package
sudo apt install linux-modules-extra-$(uname -r)
sudo modprobe mac80211_hwsim radios=12
```

### Docker permission denied
```bash
sudo usermod -aG docker $USER
# Log out and back in, or:
newgrp docker
```

### Mininet-WiFi topology fails to start
```bash
# Clean up stale state
sudo mn -c
# Reload hwsim
sudo modprobe -r mac80211_hwsim
sudo modprobe mac80211_hwsim radios=12
```

### ESP32 QEMU containers fail to start
```bash
# Rebuild images
docker build -f docker/Dockerfile.esp32 -t vesper-esp32:latest .
# Check logs
docker logs vesper-dev-kitchen
```

### Experiments hang or timeout
```bash
# Kill all Docker containers
docker compose -f docker/docker-compose.yml down --timeout 5
docker compose -f docker/docker-compose-bridge.yml down --timeout 5
# Kill stale Mininet processes
sudo killall -9 hostapd wpa_supplicant dnsmasq 2>/dev/null
sudo mn -c
```

---

## Time and Cost Summary

| Experiment | Duration | Cloud Cost (t3.xlarge) |
|-----------|----------|----------------------|
| Setup | ~20 min | $0.06 |
| RQ-N1 | ~45 min | $0.12 |
| RQ-N2 | ~3 hours | $0.50 |
| Trace validation | ~10 min | $0.03 |
| **Total** | **~4 hours** | **~$0.71** |

For budget-conscious runs, use `--trials=3` instead of the default 5:
```bash
bash scripts/run_all_flagship_experiments.sh --trials=3   # ~2.5 hours
```
