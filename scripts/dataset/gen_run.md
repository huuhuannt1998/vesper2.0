# VESPER-SH generation run — runbook

How one dataset **episode** is produced by running the Mac-native Habitat eval and
the VM-side capture/attack runner together, coupled over the TCP bridge.

## Roles
- **Mac (native, `vesper` conda env):** runs `run_autonomous_eval.py` with
  `VESPER_WIFI_VM=<vm_ip>` (couples 3D activity → VM as 802.11) and
  `VESPER_DATASET_OUT=<episode_dir>` (writes `events.jsonl` + `bridge_sync_mac.jsonl`).
- **VM (`vesper-vm`, Multipass):** attackable Wi-Fi (`vm_wifi_net.sh`) + the device
  agent (`vm_device_agent.py --sync-log <episode_dir>/bridge_sync_vm.jsonl`) +
  `vm_dataset_gen.sh <episode_dir> <dur>` (continuous dual-vantage capture +
  scheduled 5-attack suite → `ap.pcap`, `rf.pcap`, `attack_schedule.jsonl`).

## Per-episode files (the raw material the exporter consumes)
| File | Where written | By |
|------|---------------|----|
| `events.jsonl`          | Mac episode dir | eval logger (Task 1) |
| `bridge_sync_mac.jsonl` | Mac episode dir | eval forwarder (Task 1) |
| `ap.pcap`, `rf.pcap`    | VM episode dir  | `vm_dataset_gen.sh` |
| `attack_schedule.jsonl` | VM episode dir  | `vm_dataset_gen.sh` |
| `bridge_sync_vm.jsonl`  | VM episode dir  | agent `--sync-log` (Task 2) |

After a run, the VM files are transferred next to the Mac files so one
`<home>__<model>__<run>/` dir holds all six; `build_dataset.py` (Task 11) then
exports each into `windows.parquet` + `labels.csv` + `meta.json`.

## ⚠️ pkill-self-kill pitfall (learned the hard way)
Do **NOT** run `pkill -f vm_device_agent` inside a `bash -c` block that also contains
the agent's start command — the pattern matches the block's *own* command line and
kills it before the generator runs. To free the agent's port for a fresh agent, kill
the **port holder** instead (no pattern self-match): `fuser -k 6000/tcp 2>/dev/null`.

## Smoke test (one short episode — capture machinery only, no agent needed)
The smoke test validates dual-vantage capture + the 5-attack schedule; it does NOT
need the device agent (there's no Mac eval driving events), so it avoids the pkill
pitfall entirely.
```bash
multipass transfer scripts/vm_dataset_gen.sh scripts/dataset/check_episode.sh vesper-vm:/tmp/
# net must already be up (scripts/vm_wifi_net.sh /tmp/vsh_net); then:
multipass exec vesper-vm -- sudo bash -c 'killall tshark 2>/dev/null; sleep 1; \
  mkdir -p /tmp/vsh_ep; rm -f /tmp/vsh_ep/*; \
  bash /tmp/vm_dataset_gen.sh /tmp/vsh_ep 280; \
  bash /tmp/check_episode.sh /tmp/vsh_ep'
```
`280` s covers ≥1 full 5-attack cycle (20 s warm-up + ~6×45 s). Expect `EPISODE OK`
(rf deauth>0, ap arp>0, scheduled rounds≥5).

## Full run (Task 12) — agent IS needed for coupling
Start the agent with `fuser -k 6000/tcp` (not pkill) to free the port, then run the
Mac eval (`VESPER_WIFI_VM` + `VESPER_DATASET_OUT`) concurrently with
`vm_dataset_gen.sh` so `events.jsonl` + both sync logs are also populated per episode.
