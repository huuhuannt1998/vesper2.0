#!/usr/bin/env bash
# Regenerate coupled network captures for the gemma episodes that lack them, by
# REPLAYING each episode's recorded activity to the VM live and capturing fresh.
# Net must already be up (scripts/vm_wifi_net.sh). Reads home ids from $HOMES_FILE.
# Arg 1 = max episodes to do (default all) — use 1 for a single-episode test.
set -uo pipefail
VM=192.168.2.2
MODEL=gemma-2-9b-it
MAX="${1:-99}"
HOMES_FILE="${HOMES_FILE:-/tmp/regen_homes.txt}"
RAW_ROOT=results/vesper_sh_raw
OUT_ROOT=results/vesper_sh_regen
PY=/Users/huanbui/miniconda3/envs/vesper/bin/python
mkdir -p "$OUT_ROOT"
i=0
while read -r HOME_ID <&3; do
  [ -z "$HOME_ID" ] && continue
  i=$((i+1)); [ $i -gt "$MAX" ] && break
  RAW="$RAW_ROOT/${HOME_ID}__${MODEL}__1"
  OUT="$OUT_ROOT/${HOME_ID}__${MODEL}__1"
  if [ ! -f "$RAW/events.jsonl" ]; then echo "[$i] SKIP $HOME_ID (no raw events)"; continue; fi
  mkdir -p "$OUT"
  SPAN=$($PY -c "import json;ts=[json.loads(l)['ts'] for l in open('$RAW/events.jsonl') if l.strip() and 'ts' in json.loads(l)];print(int(max(ts)-min(ts)) if ts else 0)")
  DUR=$(( SPAN + 45 )); [ $DUR -lt 285 ] && DUR=285
  echo "[$i] $HOME_ID span=${SPAN}s dur=${DUR}s start=$(date +%T)"
  # fresh agent (per-episode vm sync log) + fresh capture, backgrounded on the VM
  multipass exec vesper-vm -- sudo bash -c "fuser -k 6000/tcp 2>/dev/null; killall tshark 2>/dev/null; sleep 1; rm -rf /tmp/ep; mkdir -p /tmp/ep; nohup python3 /tmp/vm_device_agent.py --sync-log /tmp/ep/bridge_sync_vm.jsonl >/tmp/agent.log 2>&1 & sleep 1; nohup bash /tmp/vm_dataset_gen.sh /tmp/ep $DUR >/tmp/gen.log 2>&1 &" </dev/null
  sleep 8   # let capture start + benign warm-up begin
  $PY scripts/dataset/replay_episode.py "$RAW" "$OUT" "$VM" || echo "  replay FAILED $HOME_ID"
  sleep $(( DUR - SPAN + 6 ))   # wait out the remaining capture window
  multipass exec vesper-vm -- sudo bash -c "chmod 644 /tmp/ep/*.pcap /tmp/ep/*.jsonl 2>/dev/null" 2>/dev/null
  for f in ap.pcap rf.pcap attack_schedule.jsonl bridge_sync_vm.jsonl; do
    multipass transfer "vesper-vm:/tmp/ep/$f" "$OUT/$f" 2>/dev/null || echo "  transfer miss: $f"
  done
  RF=$(wc -c < "$OUT/rf.pcap" 2>/dev/null || echo 0); AP=$(wc -c < "$OUT/ap.pcap" 2>/dev/null || echo 0)
  SCH=$(wc -l < "$OUT/attack_schedule.jsonl" 2>/dev/null || echo 0)
  echo "  done $HOME_ID: rf=${RF}B ap=${AP}B sched_rounds=${SCH} end=$(date +%T)"
done 3< "$HOMES_FILE"
echo "REGEN CAPTURE DONE ($i episodes attempted)"
