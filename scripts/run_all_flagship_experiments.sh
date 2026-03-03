#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# VESPER — Run All Flagship Experiments
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the three flagship experiments for the MobiCom paper:
#   1. RQ-N1: Bridge vs. 802.11 Divergence  (~30-45 min)
#   2. RQ-N2: WiFi Hardening Sweep           (~2-3 hours)
#   3. Trace Validation: Real pcap analysis   (~5-10 min)
#
# Then updates the paper LaTeX with measured results.
#
# Prerequisites:
#   - Linux with mac80211_hwsim (run setup_linux_vm.sh first)
#   - Docker with built images (vesper-router, vesper-esp32)
#   - VESPER Python venv activated
#
# Usage:
#   cd /path/to/vesper
#   source .venv/bin/activate
#   bash scripts/run_all_flagship_experiments.sh
#
#   # Run only specific experiments:
#   bash scripts/run_all_flagship_experiments.sh --rqn1-only
#   bash scripts/run_all_flagship_experiments.sh --rqn2-only
#   bash scripts/run_all_flagship_experiments.sh --trace-only
#
# Estimated total time: 3-4 hours
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

VESPER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRIALS=${VESPER_TRIALS:-5}
LOG_DIR="$VESPER_DIR/results/flagship_${TIMESTAMP}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)]${NC} $*" >&2; }
header() { echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"; echo -e "${CYAN}║${NC}  $*"; echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"; }

# Parse args
RUN_RQN1=true
RUN_RQN2=true
RUN_TRACE=true
RUN_UPDATE=true
BUILD_DOCKER=false

for arg in "$@"; do
    case $arg in
        --rqn1-only)  RUN_RQN2=false; RUN_TRACE=false ;;
        --rqn2-only)  RUN_RQN1=false; RUN_TRACE=false ;;
        --trace-only) RUN_RQN1=false; RUN_RQN2=false ;;
        --no-update)  RUN_UPDATE=false ;;
        --build)      BUILD_DOCKER=true ;;
        --trials=*)   TRIALS="${arg#*=}" ;;
        --help)
            echo "Usage: $0 [--rqn1-only|--rqn2-only|--trace-only] [--no-update] [--build] [--trials=N]"
            exit 0 ;;
    esac
done

mkdir -p "$LOG_DIR"
cd "$VESPER_DIR"

# ── Preflight checks ────────────────────────────────────────────────────────
header "Preflight checks"

# Check Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    err "FATAL: Must run on Linux (current: $(uname -s))"
    err "See: scripts/setup_linux_vm.sh and LINUX_EXPERIMENTS.md"
    exit 1
fi

# Check mac80211_hwsim
if ! lsmod | grep -q mac80211_hwsim; then
    log "Loading mac80211_hwsim..."
    sudo modprobe mac80211_hwsim radios=12
fi
log "✓ mac80211_hwsim loaded ($(iw dev | grep -c Interface) interfaces)"

# Check Docker
if ! docker info >/dev/null 2>&1; then
    err "FATAL: Docker not running"
    exit 1
fi
log "✓ Docker running"

# Check Docker images
if ! docker images | grep -q vesper-router; then
    warn "vesper-router image not found — building..."
    BUILD_DOCKER=true
fi

if [[ "$BUILD_DOCKER" == "true" ]]; then
    log "Building Docker images..."
    docker build -f docker/Dockerfile.router -t vesper-router:latest . 2>&1 | tail -3
    docker build -f docker/Dockerfile.esp32 -t vesper-esp32:latest . 2>&1 | tail -3
    log "✓ Docker images built"
else
    log "✓ Docker images exist"
fi

# Check Python
if ! python3 -c "import vesper" 2>/dev/null; then
    err "FATAL: VESPER Python package not importable"
    err "Run: source .venv/bin/activate && pip install -e ."
    exit 1
fi
log "✓ VESPER Python package importable"

# Check tshark
if ! command -v tshark &>/dev/null; then
    err "FATAL: tshark not found. Install: sudo apt install tshark"
    exit 1
fi
log "✓ tshark available"

TOTAL_START=$(date +%s)

# ═════════════════════════════════════════════════════════════════════════════
# Experiment 1: RQ-N1 — Bridge vs. 802.11
# ═════════════════════════════════════════════════════════════════════════════
if [[ "$RUN_RQN1" == "true" ]]; then
    header "RQ-N1: Bridge vs. 802.11 Divergence ($TRIALS trials × 2 modes)"
    RQN1_DIR="$LOG_DIR/rqn1"
    RQN1_START=$(date +%s)

    log "Running full bridge vs 802.11 comparison..."
    python3 scripts/run_rqn1_bridge_vs_wifi.py \
        --full \
        --trials "$TRIALS" \
        --output "$RQN1_DIR" \
        2>&1 | tee "$RQN1_DIR.log"

    RQN1_END=$(date +%s)
    RQN1_ELAPSED=$((RQN1_END - RQN1_START))
    log "RQ-N1 completed in $((RQN1_ELAPSED / 60))m $((RQN1_ELAPSED % 60))s"

    # Copy paper artifacts
    if [[ -f "$RQN1_DIR/tab_bridge_vs_80211.tex" ]]; then
        cp "$RQN1_DIR/tab_bridge_vs_80211.tex" paper-latex/tables/
        log "✓ tab_bridge_vs_80211.tex → paper-latex/tables/"
    fi
    if [[ -f "$RQN1_DIR/fig_rtt_bridge_vs_80211.pdf" ]]; then
        cp "$RQN1_DIR/fig_rtt_bridge_vs_80211.pdf" paper-latex/figures/
        log "✓ fig_rtt_bridge_vs_80211.pdf → paper-latex/figures/"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# Experiment 2: RQ-N2 — WiFi Hardening Sweep
# ═════════════════════════════════════════════════════════════════════════════
if [[ "$RUN_RQN2" == "true" ]]; then
    header "RQ-N2: WiFi Hardening Sweep (8 configs × $TRIALS trials)"
    RQN2_DIR="$LOG_DIR/rqn2"
    RQN2_START=$(date +%s)

    log "Running hardening sweep..."
    python3 scripts/run_rqn2_hardening_sweep.py \
        --full \
        --trials "$TRIALS" \
        --output "$RQN2_DIR" \
        2>&1 | tee "$RQN2_DIR.log"

    RQN2_END=$(date +%s)
    RQN2_ELAPSED=$((RQN2_END - RQN2_START))
    log "RQ-N2 completed in $((RQN2_ELAPSED / 60))m $((RQN2_ELAPSED % 60))s"

    # Copy paper artifacts
    if [[ -f "$RQN2_DIR/tab_hardening_measured.tex" ]]; then
        cp "$RQN2_DIR/tab_hardening_measured.tex" paper-latex/tables/
        log "✓ tab_hardening_measured.tex → paper-latex/tables/"
    fi
    if [[ -f "$RQN2_DIR/fig_hardening_pareto.pdf" ]]; then
        cp "$RQN2_DIR/fig_hardening_pareto.pdf" paper-latex/figures/
        log "✓ fig_hardening_pareto.pdf → paper-latex/figures/"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# Experiment 3: Trace Validation (real pcap analysis)
# ═════════════════════════════════════════════════════════════════════════════
if [[ "$RUN_TRACE" == "true" ]]; then
    header "Trace Validation: Real pcap analysis"
    TRACE_DIR="$LOG_DIR/trace_validation"
    TRACE_START=$(date +%s)

    # Analyze existing pcaps from the 30-scene autonomous eval
    PCAP_SOURCE="results/vesper_autonomous_eval"
    if [[ ! -d "$PCAP_SOURCE" ]]; then
        warn "No autonomous eval pcaps found at $PCAP_SOURCE"
        warn "Checking for RQ-N1 pcaps..."
        PCAP_SOURCE="$LOG_DIR/rqn1"
    fi

    if [[ -d "$PCAP_SOURCE" ]]; then
        log "Analyzing pcaps from: $PCAP_SOURCE"
        python3 scripts/analyze_real_pcaps.py \
            --pcap-dir "$PCAP_SOURCE" \
            --output "$TRACE_DIR" \
            --generate-paper \
            --paper-dir paper-latex \
            2>&1 | tee "$TRACE_DIR.log"
    else
        warn "No pcap source found — running trace simulation as fallback"
        python3 scripts/run_trace_validation.py \
            --simulate --duration 24 \
            2>&1 | tee "$TRACE_DIR.log"
    fi

    TRACE_END=$(date +%s)
    TRACE_ELAPSED=$((TRACE_END - TRACE_START))
    log "Trace validation completed in $((TRACE_ELAPSED / 60))m $((TRACE_ELAPSED % 60))s"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Update paper LaTeX with measured results
# ═════════════════════════════════════════════════════════════════════════════
if [[ "$RUN_UPDATE" == "true" ]]; then
    header "Updating paper LaTeX with measured results"

    python3 scripts/update_paper_from_results.py \
        --results-dir "$LOG_DIR" \
        --paper-dir paper-latex \
        2>&1 | tee "$LOG_DIR/update_paper.log"

    log "Paper updated ✓"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Final summary
# ═════════════════════════════════════════════════════════════════════════════
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

header "EXPERIMENT SUMMARY"
echo ""
log "Total time:    $((TOTAL_ELAPSED / 3600))h $((TOTAL_ELAPSED % 3600 / 60))m $((TOTAL_ELAPSED % 60))s"
log "Results dir:   $LOG_DIR"
log "Paper tables:  paper-latex/tables/tab_bridge_vs_80211.tex"
log "               paper-latex/tables/tab_hardening_measured.tex"
log "               paper-latex/tables/tab_trace_validation.tex"
log "Paper figures: paper-latex/figures/fig_rtt_bridge_vs_80211.pdf"
log "               paper-latex/figures/fig_hardening_pareto.pdf"
log "               paper-latex/figures/fig_pkt_size_cdf.pdf"
echo ""
log "All results are from REAL experiments, not projections."
log "Ready to compile paper: cd paper-latex && pdflatex main.tex"
echo ""
