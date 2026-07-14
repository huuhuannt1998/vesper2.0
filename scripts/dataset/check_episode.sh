#!/usr/bin/env bash
# Assert an episode capture is well-formed. Usage: check_episode.sh <episode_dir>
set -uo pipefail
D="$1"; ok=1
for f in ap.pcap rf.pcap attack_schedule.jsonl; do
  [ -s "$D/$f" ] || { echo "MISSING/empty: $f"; ok=0; }
done
rf_deauth=$(tshark -r "$D/rf.pcap" -Y "wlan.fc.type_subtype==0x000c" 2>/dev/null | wc -l|tr -d ' ')
ap_arp=$(tshark -r "$D/ap.pcap" -Y "arp" 2>/dev/null | wc -l|tr -d ' ')
sched=$(wc -l < "$D/attack_schedule.jsonl" 2>/dev/null|tr -d ' ')
echo "rf deauth=$rf_deauth  ap arp=$ap_arp  scheduled rounds=$sched"
[ "$rf_deauth" -gt 0 ] && [ "$ap_arp" -gt 0 ] && [ "$sched" -ge 5 ] || ok=0
[ $ok -eq 1 ] && echo "EPISODE OK" || { echo "EPISODE INCOMPLETE"; exit 1; }
