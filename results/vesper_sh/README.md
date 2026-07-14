# VESPER-SH

A labeled, timestamp-aligned, multi-modal (activity + device + network)
smart-home security benchmark dataset with benign traffic and 5 attack
classes, built for **cross-environment generalization** research: does a
detector trained on some homes and resident behaviors generalize to homes
and residents it has never seen?

Authored by the VESPER-SH authors (anonymized for review). Full details on
motivation, composition, collection, and known limitations are in
[`DATASHEET.md`](DATASHEET.md); precise column/field documentation is in
[`schema.md`](schema.md). This file covers what the dataset is, how it's
laid out on disk, how to load it, the splits, and the attack taxonomy.

## What this is

SWaT, WaDI, and HAI are the canonical ICS/CPS security benchmark datasets:
each is a long, continuous capture of a *single* physical environment.
VESPER-SH takes the same idea — a labeled, reusable, benchmark-able
security capture — and applies it to a structurally different, complementary
setting: **many diverse homes and resident behaviors, reproducibly
generated, with precise attack labels** (attacks are injected on a recorded
schedule, so ground truth is exact, not inferred).

Each episode is produced by coupling:
- an LLM-driven embodied resident acting in a simulated Habitat 3.0 home,
- a firmware-in-the-loop simulated smart-home device network (with
  SmartThings/Matter/Home-Assistant-style automation logic driving device
  state and "tap" rule firing),
- an emulated 802.11 Wi-Fi network (`mac80211_hwsim` + `wmediumd`), captured
  continuously from two vantage points, and
- a scheduled 5-class attack suite injected against that network.

See [`DATASHEET.md`](DATASHEET.md) §8 for the honestly-stated limitations of
this setup (simulated firmware, emulated RF, single-station LAN topology,
RF-vantage capture scope, episode duration) before drawing conclusions that
depend on real-hardware or real-RF fidelity.

## Directory layout

```
results/vesper_sh/
  DATASHEET.md          # Gebru et al. datasheet (motivation/composition/.../limitations)
  README.md             # this file
  schema.md             # precise column/field documentation

  episodes/<home>__<model>__<run>/
    windows.parquet      # 1-second ML-ready feature table (the primary artifact)
    labels.csv           # per-window label: window_idx, ts, label, binary
    meta.json            # home, model, run, clock offset, n_windows, class_counts
    ap.pcap              # raw: continuous AP/router-vantage capture (wlan0)
    rf.pcap               # raw: continuous RF-monitor-vantage capture (wlan2)
    events.jsonl           # raw: activity/device events, canonical clock
    attack_schedule.jsonl  # raw: attack-injection ground-truth intervals

  splits/
    by_home/{train,test}.txt        # cross-home generalization split
    by_resident/{train,test}.txt    # cross-resident-model generalization split
    folds.json                      # k=5 grouped-by-home cross-validation folds
```

`home` is a Habitat scene identifier, `model` names the resident LLM policy
that drove that episode, and `run` is an integer run index; together they
form the episode's directory name `<home>__<model>__<run>`. See
[`schema.md`](schema.md) for the exact meaning of every field.

Note on `raw/` material: the `ap.pcap`/`rf.pcap`/`events.jsonl`/
`attack_schedule.jsonl` files are the pre-windowing raw capture/log that
`windows.parquet`/`labels.csv`/`meta.json` are derived from — kept per
episode so features can be re-derived differently if needed (see
`DATASHEET.md` §4). Consumers who only want the ML-ready tables can ignore
them entirely.

## Loading the dataset

The loader (`discover`, `make_splits`, `load_xy`) ships in the VESPER
repository at `scripts/dataset/vesper_sh.py` and operates directly on a
dataset root directory (this directory, `results/vesper_sh`, when working
inside the repo; the extracted root of the anonymized release archive
otherwise — see `scripts/dataset/package_release.py`).

```python
import sys
sys.path.insert(0, "scripts")                  # repo layout: scripts/dataset/vesper_sh.py
from dataset.vesper_sh import discover, make_splits, load_xy

root = "results/vesper_sh"

# 1. Enumerate episodes (skips any directory that isn't a well-formed
#    <home>__<model>__<run> dir with a windows.parquet inside it).
episodes = discover(root)          # -> [{"home", "model", "run", "path"}, ...]

# 2. (Re)generate the splits/ directory (deterministic given `seed`).
make_splits(root, seed=0)          # writes splits/by_home, splits/by_resident, splits/folds.json

# 3. Load a feature matrix / label vector / group vector for a set of homes.
train_homes = open(f"{root}/splits/by_home/train.txt").read().split()
test_homes  = open(f"{root}/splits/by_home/test.txt").read().split()

X_train, y_train, groups_train = load_xy(root, train_homes)
X_test,  y_test,  groups_test  = load_xy(root, test_homes)
```

**Loader API**
- `discover(root) -> list[dict]` — one `{"home", "model", "run", "path"}`
  entry per valid episode directory under `<root>/episodes/`.
- `make_splits(root, seed=0, test_frac=0.34, k=5) -> None` — writes
  `splits/by_home/{train,test}.txt`, `splits/by_resident/{train,test}.txt`,
  and `splits/folds.json` under `root` (see Splits below).
- `load_xy(root, homes) -> (X, y, groups)` — concatenates `windows.parquet`
  joined with `labels.csv` for every discovered episode whose `home` is in
  `homes`. `X` is a `pandas.DataFrame` of feature columns only
  (`window_idx`/`ts`/`label` are dropped); `y` is the `label` `Series`
  (multiclass string); `groups` is a `Series` giving each row's `home` id,
  for group-aware cross-validation. Raises `ValueError` if `homes` selects
  zero episodes.

## Splits

Three complementary split artifacts, all computed by `make_splits` from the
set of discovered episodes (see `schema.md` for exact generation
semantics):

| Split | Files | Measures |
|---|---|---|
| **by_home** (primary) | `splits/by_home/{train,test}.txt` | Cross-environment (layout/behavior) generalization: home ids are disjoint between train and test, ~2/3 train / ~1/3 test homes, seeded shuffle. |
| **by_resident** (secondary) | `splits/by_resident/{train,test}.txt` | Cross-resident generalization: trains on 2 of the 3 resident LLM policies, tests on the held-out third. |
| **folds** | `splits/folds.json` | k=5 grouped cross-validation, grouped by home — no home appears in both the train and test side of any fold. |

All splits are **group-aware at the home level**: an episode's windows
never straddle a train/test boundary, because splitting happens on whole
homes, not on individual windows or episodes independently of their home.

## Attack taxonomy

Five attack classes, each injected at a realistic defender vantage point
(the corresponding real-world detector for that class would observe it at
that vantage — see `DATASHEET.md` §8 point 4 for why the RF vantage's
benign baseline specifically comes from the AP-side capture):

| Class | Layer | Vantage (capture) | Mechanism |
|---|---|---|---|
| `deauth` | RF / management | `rf.pcap` (RF monitor) | Spoofed 802.11 deauthentication frames, AP → station. |
| `evil_twin` | RF / management | `rf.pcap` | Rogue beacon frames cloning the legitimate SSID. |
| `beacon_flood` | RF / management | `rf.pcap` | Flood of beacon frames advertising random BSSIDs. |
| `arp_spoof` | LAN | `ap.pcap` (AP/router) | Gratuitous ARP frames with a real Ethernet source but a spoofed ARP payload. |
| `lan_scan` | LAN | `ap.pcap` | SYN/port scan and flood from the station's real MAC address, propagating through the AP. Replaces a DHCP-starvation attack, which requires multiple distinct clients and would not propagate faithfully in this testbed's single-station LAN emulation (`DATASHEET.md` §8 point 3). |

`deauth`/`evil_twin`/`beacon_flood` are captured on the RF-monitor vantage
because they are 802.11 management-frame attacks; `arp_spoof`/`lan_scan`
are captured on the AP/router vantage because they are LAN-layer attacks.
Every injection's class and exact `[start, end)` interval is logged to
`attack_schedule.jsonl` at generation time, so `labels.csv` labels are
derived from ground truth, not inferred (`DATASHEET.md` §4).

## Baselines

Two detectors are reported on the `by_home` (cross-home) test split, to
demonstrate that the labels are learnable and that cross-home
generalization is a real, measurable axis rather than a trivial one:

- **IsolationForest** (unsupervised) — trained on **benign windows of the
  training homes only** (a supervised attack signal is not assumed
  available at training time, which is the realistic anomaly-detection
  setting). Windows are standardized on training statistics; the anomaly
  threshold is set at a target benign false-positive rate on the training
  distribution. Reported: per-attack-class recall and the realized benign
  false-positive rate on the test homes.
- **RandomForest** (supervised) — trained on all labeled windows of the
  training homes. Reported: per-class precision/recall/F1, macro-F1, and
  the confusion matrix on the test homes.

Both baselines are group-aware by construction (they consume the `by_home`
split, so no home's windows appear in both train and test). See
`scripts/dataset/baselines.py` for the metric implementations and
`scripts/dataset/run_baselines.py` for the end-to-end driver that writes
`baseline_results.json` and the `tables/composition.tex` /
`tables/baselines.tex` LaTeX tables consumed by the paper. Numeric results
are reported in the accompanying paper and in `baseline_results.json` once
the dataset has been generated; they are intentionally not restated here
(see `DATASHEET.md` §5).

## Regenerating / extending the dataset

The dataset is produced by a reproducible pipeline, not collected by hand,
so it can be regenerated or extended (more homes, more resident models,
more runs, or new attack classes) by re-running the same steps:

1. **Structured logging** — an event logger on the simulation side
   (`scripts/dataset/event_log.py`) and a device-agent logger on the
   network-emulation side write timestamped JSONL events plus
   sequence-matched clock-sync records.
2. **Generation run** — coupled activity + scheduled attack injection +
   continuous dual-vantage capture; see `scripts/dataset/gen_run.md` for
   the runbook and `scripts/dataset/split_run.py` for slicing one combined
   capture into per-episode raw directories.
3. **Export** — `scripts/dataset/build_dataset.py` walks raw episode
   directories and calls `scripts/dataset/export_episode.py` on each,
   producing `windows.parquet` / `labels.csv` / `meta.json`.
4. **Splits + baselines** — `scripts/dataset/run_baselines.py` calls
   `vesper_sh.make_splits` and `scripts/dataset/baselines.py`, and writes
   the results/tables described above.
5. **Anonymized packaging** — `scripts/dataset/package_release.py` asserts
   no author-identifying strings are present under a dataset root, then
   zips it for double-blind distribution.

## Citation

This dataset is currently under double-blind review. Please do not cite it
by author name during the review period; refer to it as "the VESPER-SH
dataset accompanying the submission." A full citation (with persistent
identifier) will be added here upon acceptance / de-anonymization.

## License

Not yet finalized — see `DATASHEET.md` §6 (Distribution).
