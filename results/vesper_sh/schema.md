# VESPER-SH — Schema Reference

Precise, field-by-field documentation of every file the dataset ships.
Column names, types, and semantics below mirror the actual exporter
(`scripts/dataset/export_episode.py`, `scripts/dataset/event_features.py`,
`scripts/dataset/net_features.py`, `scripts/dataset/labeling.py`) — not an
aspirational design, the literal behavior of the pipeline that writes these
files. See [`README.md`](README.md) for the directory layout and loader API,
and [`DATASHEET.md`](DATASHEET.md) for provenance/collection/limitations.

## Episode directory naming

```
episodes/<home>__<model>__<run>/
```

Double-underscore-separated, exactly 3 tokens:

| Token | Meaning | Type (as parsed by the loader) |
|---|---|---|
| `home` | Habitat scene identifier the episode was generated in. | `str` |
| `model` | Name of the resident LLM policy that drove the episode's embodied agent. | `str` |
| `run` | Run index, for regenerating multiple runs of the same `(home, model)` pair. | `str` in the directory name / `discover()`'s parsed dict; cast to `int` in `meta.json`'s `run` field (see below). |

`scripts/dataset/vesper_sh.py:discover()` parses this from the directory
name via `name.split("__")` and skips any directory whose name doesn't
split into exactly 3 tokens, or that has no `windows.parquet` inside it.

## `windows.parquet`

One row per **1-second, non-overlapping, half-open window** `[t0, t0+1)`.
The window grid for an episode spans `floor(min timestamp)` to
`ceil(max timestamp)` across every event and network frame observed in that
episode (canonical clock, after offset correction — see `meta.json.offset`
below); `window_idx` is the window's 0-based position in that grid, so
`ts == window_idx + floor(min timestamp)` for every row and there are no
gaps in the sequence.

Columns, in the order the exporter emits them:

| Column | Type | Block | Description |
|---|---|---|---|
| `window_idx` | int | — | 0-based sequential window index within the episode. |
| `ts` | float | — | Canonical-clock window start time (epoch seconds; integer-valued — window boundaries fall on whole seconds). |
| `act_motion` | int | activity | Count of `motion_detected` events with `ts` in `[t0, t1)`. |
| `act_rooms` | int | activity | Number of **distinct** rooms with a motion or room-transition event in the window (a set-cardinality count, not a sum of events — e.g. two motion events in the same room in the same second still count as 1). |
| `act_transitions` | int | activity | Count of `agent_entered_room` / `agent_left_room` events. |
| `act_doors` | int | activity | Count of `door_opened` events. |
| `act_any` | int (0/1) | activity | 1 if any event of any type fell in the window, else 0. |
| `dev_state_changes` | int | device | Count of `device_state_changed` / `state_change` events. |
| `dev_tap_firings` | int | device | Count of `tap_fired` events (SmartThings/automation-rule firings). |
| `dev_firmware_updates` | int | device | Count of `firmware_state_update` events. |
| `net_total` | int | network | `len(rf frames in window) + len(ap frames in window)`. |
| `net_mgmt` | int | network | Count of **`rf.pcap`-only** frames whose `wlan.fc.type` field (as emitted by `tshark`) equals the string `"0"` (802.11 management-frame type). Not computed from `ap.pcap`. |
| `net_data` | int | network | `len(ap frames in window)` — i.e. the AP-vantage frame count. **Naming note:** despite the name, this is *not* an 802.11 data-subtype filter; `ap.pcap` is a router/LAN-side capture (not an 802.11 monitor capture), so no per-frame type filtering is applied — it is simply "how many AP-vantage frames were observed in this window." |
| `net_beacon` | int | network | Count of `rf.pcap` frames with `wlan.fc.type_subtype == 0x0008` (beacon). |
| `net_deauth` | int | network | Count of `rf.pcap` frames with `wlan.fc.type_subtype == 0x000c` (deauthentication). |
| `net_probe` | int | network | Count of `rf.pcap` frames with `wlan.fc.type_subtype == 0x0004` (probe request). |
| `net_disassoc` | int | network | Count of `rf.pcap` frames with `wlan.fc.type_subtype == 0x000a` (disassociation). |
| `net_arp` | int | network | Count of `ap.pcap` frames carrying an ARP header (`arp` field truthy). |
| `net_dhcp` | int | network | Count of `ap.pcap` frames with a non-empty `bootp.type` field (DHCP). |
| `net_unique_src` | int | network | Number of distinct source addresses in the window: the union of `rf.pcap`'s `wlan.sa` values and `ap.pcap`'s `eth.src` values, excluding empty/`None`. |
| `net_bytes` | int | network | Sum of `frame.len` across all `rf.pcap` + `ap.pcap` frames in the window. |
| `net_syn` | int | network | Count of `ap.pcap` frames with `tcp.flags.syn` set (field value `"1"` or `"True"`). |
| `net_unique_dports` | int | network | Number of distinct `tcp.dstport` values seen across `ap.pcap` frames in the window, excluding empty. |

Notes:
- `net_arp` / `net_dhcp` / `net_syn` / `net_unique_dports` are LAN-layer
  fields and are derived from `ap.pcap` only (there is no ARP/DHCP/TCP
  concept on the RF-monitor vantage).
- `net_unique_src` and `net_bytes` are the only two network fields that mix
  both vantages in one number.
- All `net_*` counts are computed on frames whose (offset-corrected)
  timestamp falls in `[t0, t1)`; frames whose `frame.time_epoch` field
  fails to parse as a float are dropped upstream during pcap parsing and
  never reach any window.

## `labels.csv`

One row per window, in the same order/count as `windows.parquet` (joins on
`window_idx`).

| Column | Type | Description |
|---|---|---|
| `window_idx` | int | Joins to `windows.parquet.window_idx`. |
| `ts` | float | Same canonical window start time as the corresponding `windows.parquet` row. |
| `label` | str | One of `benign`, `deauth`, `evil_twin`, `beacon_flood`, `arp_spoof`, `lan_scan`. |
| `binary` | int (0/1) | `0` if `label == "benign"`, else `1`. |

**Labeling rule:** a window `[t0, t1)` is assigned the class of any
attack-schedule entry whose canonical-clock interval `[start, end)`
overlaps it, i.e. `start < t1 and end > t0`; if none overlap, `benign`.
Schedule entries are sorted by `start` and the first (earliest-starting)
match is returned, as a deterministic tie-break — in practice this doesn't
fire, because injection rounds are scheduled not to overlap each other.

## `meta.json`

One JSON object per episode.

| Key | Type | Description |
|---|---|---|
| `home` | str | Same as the episode directory's `home` token. |
| `model` | str | Same as the episode directory's `model` token. |
| `run` | int | Run index, **cast via `int(run)`** at export time. Caveat: if you zero-pad `run` in directory names (e.g. `"01"`), this field drops the leading zero (`1`) even though the directory name and `discover()`'s parsed `run` string retain the literal token. |
| `offset` | float | Seconds. The VM-clock → canonical-clock offset, `median(vm_ts - mac_ts)` over sequence-matched bridge events (see `scripts/dataset/clock_sync.py`). `canonical_ts = vm_ts - offset`. If neither `bridge_sync_mac.jsonl` nor `bridge_sync_vm.jsonl` exists for the episode, `offset` defaults to `0.0` (the VM clock is treated as canonical) — a distinct code path from a *broken* sync (both files present but no sequence numbers match), which raises at export time instead of silently defaulting. |
| `n_windows` | int | Number of rows in this episode's `windows.parquet` / `labels.csv`. |
| `class_counts` | dict[str, int] | `labels.csv["label"].value_counts()` for this episode — one entry per label value actually present (`benign` plus whichever attack classes occurred), not necessarily all 6 possible values. |

## `splits/`

All three artifacts are written by `scripts/dataset/vesper_sh.py:make_splits(root, seed=0, test_frac=0.34, k=5)`,
derived from `discover(root)`'s episode list.

**`splits/by_home/train.txt`, `splits/by_home/test.txt`**
Newline-separated home ids, disjoint. Generation: the sorted list of
distinct home ids is shuffled with `random.Random(seed)`; the first
`round(test_frac * n_homes)` (minimum 1) become the test set, the remainder
the train set. With the default `test_frac=0.34`, this yields roughly 1/3
test homes and 2/3 train homes.

**`splits/by_resident/train.txt`, `splits/by_resident/test.txt`**
Newline-separated resident-model names. All but the last of the *sorted*
distinct model names go to train; the last goes to test. Only written if at
least 2 distinct models are present across the discovered episodes.

**`splits/folds.json`**
A JSON list of `k` (default 5) objects:
```json
{"fold": 0, "test_homes": ["..."], "train_homes": ["..."]}
```
Generation: the sorted home-id list is shuffled once with `random.Random(seed)`
(same `rng` instance/state as the `by_home` split, applied after it), then
partitioned round-robin: fold `i`'s `test_homes` is `shuffled_homes[i::k]`,
and `train_homes` is every home not in that fold's `test_homes`. By
construction, no home appears in both `test_homes` and `train_homes` within
the same fold.

## Loader API (`scripts/dataset/vesper_sh.py`)

```python
def discover(root) -> list[dict]:
    """One {"home": str, "model": str, "run": str, "path": str} entry per
    valid `<root>/episodes/<home>__<model>__<run>/` directory (must split
    into exactly 3 `__`-separated tokens AND contain windows.parquet;
    otherwise silently skipped). Sorted by directory name."""

def make_splits(root, seed=0, test_frac=0.34, k=5) -> None:
    """Writes splits/by_home/{train,test}.txt, splits/by_resident/{train,test}.txt,
    and splits/folds.json under root, per the generation rules above."""

def load_xy(root, homes) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Concatenates windows.parquet (merged with labels.csv's `label` column
    on `window_idx`) for every episode discovered under root whose `home`
    is in `homes`. Returns (X, y, groups):
      X      -- feature columns only (window_idx/ts/label dropped)
      y      -- the `label` column (multiclass string)
      groups -- each row's `home` id, repeated per window (for group-aware CV)
    Raises ValueError if `homes` selects zero episodes."""
```

`X`'s exact column set is whatever `windows.parquet` contains minus
`window_idx`/`ts`/`label` — i.e. the full `act_*` / `dev_*` / `net_*`
feature block documented above, in file order.
