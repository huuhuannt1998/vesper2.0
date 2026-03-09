#!/bin/bash
# Run remaining 3 experiments after RQ-N1 wmediumd completed
OUTDIR=/home/hbui11/results/wmediumd_20260309_000453
cd /home/hbui11

echo "=== RQ-N1 Baseline (no wmediumd) ==="
python3 scripts/run_rqn1_native.py --full --trials 3 --output $OUTDIR/rqn1_baseline 2>&1 | tee $OUTDIR/rqn1_baseline.log

echo "=== RQ-N2 with wmediumd ==="
python3 scripts/run_rqn2_native.py --full --trials 3 --wmediumd --wmediumd-scenario typical_home --output $OUTDIR/rqn2_wmediumd 2>&1 | tee $OUTDIR/rqn2_wmediumd.log

echo "=== RQ-N2 Baseline (no wmediumd) ==="
python3 scripts/run_rqn2_native.py --full --trials 3 --output $OUTDIR/rqn2_baseline 2>&1 | tee $OUTDIR/rqn2_baseline.log

echo "=== ALL EXPERIMENTS COMPLETE ==="
ls -la $OUTDIR/
