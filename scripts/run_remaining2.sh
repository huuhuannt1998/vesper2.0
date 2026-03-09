#!/bin/bash
# Finish remaining experiments
# RQ-N1 baseline was interrupted during WiFi trial 2
# Need: RQ-N1 baseline (restart), RQ-N2 wmediumd, RQ-N2 baseline
set -x
OUTDIR=/home/hbui11/results/wmediumd_20260309_000453
cd /home/hbui11

# Clean up any leftover network state
modprobe -r mac80211_hwsim 2>/dev/null || true
ip link del vesper-br0 2>/dev/null || true
pkill -9 hostapd 2>/dev/null || true
pkill -9 wpa_supplicant 2>/dev/null || true
pkill -9 iperf3 2>/dev/null || true
pkill -9 wmediumd 2>/dev/null || true
sleep 2

echo "=== RQ-N1 Baseline (restart) ==="
rm -rf $OUTDIR/rqn1_baseline
python3 scripts/run_rqn1_native.py --full --trials 3 --output $OUTDIR/rqn1_baseline 2>&1 | tee $OUTDIR/rqn1_baseline.log
echo "RQ-N1 baseline exit code: $?"

echo "=== RQ-N2 with wmediumd ==="
python3 scripts/run_rqn2_native.py --full --trials 3 --wmediumd --wmediumd-scenario typical_home --output $OUTDIR/rqn2_wmediumd 2>&1 | tee $OUTDIR/rqn2_wmediumd.log
echo "RQ-N2 wmediumd exit code: $?"

echo "=== RQ-N2 Baseline ==="
python3 scripts/run_rqn2_native.py --full --trials 3 --output $OUTDIR/rqn2_baseline 2>&1 | tee $OUTDIR/rqn2_baseline.log
echo "RQ-N2 baseline exit code: $?"

echo "=== ALL EXPERIMENTS COMPLETE ==="
date
ls -la $OUTDIR/
