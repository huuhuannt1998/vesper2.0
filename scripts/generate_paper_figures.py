#!/usr/bin/env python3
"""Generate key figures for the Vesper paper from real experiment results.

Produces:
  1. fig_rtt_bridge_vs_80211.pdf  — CDF of RTT: Bridge vs 802.11 (wmediumd)
  2. fig_hardening_pareto.pdf     — Pareto frontier: security vs availability

Usage:
    python scripts/generate_paper_figures.py
"""

import json
import glob
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
RQN1_DIR = ROOT / "results" / "wmediumd_real" / "rqn1_wmediumd"
RQN2_JSON = ROOT / "results" / "wmediumd_real" / "rqn2_wmediumd" / "rqn2_summary.json"
OUT_DIR = ROOT / "paper-latex" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: RTT CDF – Bridge vs 802.11
# ═════════════════════════════════════════════════════════════════════════════
def load_rtt_samples(mode: str) -> np.ndarray:
    """Load all RTT samples across trials for a given mode (bridge | wifi).

    Each raw_rtt.json maps IP addresses to lists of RTT values (ms).
    We pool every value across all IPs and all trials.
    """
    pattern = str(RQN1_DIR / mode / "trial_*" / "raw_rtt.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No raw_rtt.json found for mode={mode!r} at {pattern}"
        )
    samples: list[float] = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for _ip, rtts in data.items():
            samples.extend(rtts)
    return np.asarray(samples, dtype=float)


def mean_jitter(mode: str) -> float:
    """Compute mean jitter matching rqn1_comparison.json methodology.

    Reads per-IP stdev_ms from each trial_result.json, then averages
    across all (IP, trial) pairs that have valid measurements.
    """
    pattern = str(RQN1_DIR / mode / "trial_*" / "trial_result.json")
    files = sorted(glob.glob(pattern))
    stdevs: list[float] = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        icmp = data.get("icmp_rtt", data)  # top-level if no icmp_rtt key
        for _ip, stats in icmp.items():
            if isinstance(stats, dict) and stats.get("count", 0) > 0:
                stdevs.append(stats["stdev_ms"])
    return float(np.mean(stdevs)) if stdevs else 0.0


def _plot_cdf(ax, values, **kwargs):
    """Plot an empirical CDF on *ax*."""
    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ys, where="post", **kwargs)


def fig_rtt_cdf():
    """Generate fig_rtt_bridge_vs_80211.pdf — CDF comparing RTT distributions."""
    bridge = load_rtt_samples("bridge")
    wifi = load_rtt_samples("wifi")

    bridge_mean = float(np.mean(bridge))
    wifi_mean = float(np.mean(wifi))

    # Jitter: mean-of-per-trial std (matches rqn1_comparison.json methodology)
    bridge_jitter = mean_jitter("bridge")
    wifi_jitter = mean_jitter("wifi")
    jitter_ratio = wifi_jitter / bridge_jitter if bridge_jitter > 0 else float("inf")

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    _plot_cdf(ax, bridge, color="#2060c0", label="Bridge")
    _plot_cdf(ax, wifi, color="#d04020", label="802.11")

    # Vertical dashed lines at means
    ax.axvline(bridge_mean, color="#2060c0", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(wifi_mean, color="#d04020", ls="--", lw=0.8, alpha=0.7)

    # Annotate mean values
    y_annot = 0.45
    ax.annotate(
        f"$\\mu$={bridge_mean:.3f} ms",
        xy=(bridge_mean, y_annot),
        xytext=(bridge_mean + 0.25, y_annot - 0.10),
        fontsize=7,
        color="#2060c0",
        arrowprops=dict(arrowstyle="-", color="#2060c0", lw=0.6),
    )
    ax.annotate(
        f"$\\mu$={wifi_mean:.3f} ms",
        xy=(wifi_mean, y_annot),
        xytext=(wifi_mean + 0.35, y_annot - 0.10),
        fontsize=7,
        color="#d04020",
        arrowprops=dict(arrowstyle="-", color="#d04020", lw=0.6),
    )

    # Jitter ratio box
    ax.text(
        0.97, 0.12,
        f"Jitter ratio: {jitter_ratio:.2f}\u00d7",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5),
    )

    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.7")

    out = OUT_DIR / "fig_rtt_bridge_vs_80211.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  \u2713 {out.name}")
    print(f"    Bridge: n={len(bridge)}, mean={bridge_mean:.3f} ms, jitter={bridge_jitter:.3f}")
    print(f"    WiFi:   n={len(wifi)},  mean={wifi_mean:.3f} ms, jitter={wifi_jitter:.3f}")
    print(f"    Jitter ratio: {jitter_ratio:.2f}\u00d7")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Pareto frontier – Security vs Availability
# ═════════════════════════════════════════════════════════════════════════════
def _pareto_front_indices(xs, ys):
    """Return indices of Pareto-optimal points (minimising both x and y).

    A point p dominates q if p.x <= q.x and p.y <= q.y with at least
    one strict inequality.
    """
    pts = np.column_stack([xs, ys])
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1]
                    and (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1])):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def fig_hardening_pareto():
    """Generate fig_hardening_pareto.pdf — scatter with Pareto frontier."""
    with open(RQN2_JSON) as fh:
        data = json.load(fh)

    names: list[str] = []
    atk_pct: list[float] = []
    reconn_ms: list[float] = []
    for cfg in data["configs"]:
        names.append(cfg["name"])
        atk_pct.append(cfg["mean_success_rate"])
        reconn_ms.append(cfg["mean_reconnection_ms"])

    atk = np.asarray(atk_pct)
    rec = np.asarray(reconn_ms)

    # Short labels for the plot (C0 … C7)
    short_labels = [
        "C0 Baseline",
        "C1 +Auth",
        "C2 +AP-iso",
        "C3 +AP-iso+auth",
        "C4 +PMF",
        "C5 +PMF+auth",
        "C6 WPA3-SAE",
        "C7 Full",
    ]

    pareto_idx = _pareto_front_indices(rec, atk)

    # Colour palette
    base_color = "#5580bb"
    pareto_color = "#22884a"
    star_color = "#c43030"

    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    # Non-Pareto points (circles)
    non_pareto = np.setdiff1d(np.arange(len(names)), pareto_idx)
    ax.scatter(
        rec[non_pareto], atk[non_pareto],
        s=40, c=base_color, edgecolors="white", linewidths=0.4,
        zorder=3, label="Configuration",
    )

    # Pareto-optimal points (diamonds), except C3 which gets a star
    pareto_no_star = [i for i in pareto_idx if i != 3]
    ax.scatter(
        rec[pareto_no_star], atk[pareto_no_star],
        s=55, c=pareto_color, marker="D", edgecolors="white", linewidths=0.4,
        zorder=4, label="Pareto-optimal",
    )

    # C3 (+AP-iso+auth) as a prominent star — recommended config
    if 3 in pareto_idx:
        ax.scatter(
            [rec[3]], [atk[3]],
            s=120, c=star_color, marker="*", edgecolors="white", linewidths=0.3,
            zorder=5, label="C3 +AP-iso+auth (rec.)",
        )

    # Pareto frontier line
    if len(pareto_idx) > 1:
        order = np.argsort(rec[pareto_idx])
        px = rec[pareto_idx][order]
        py = atk[pareto_idx][order]
        ax.plot(px, py, color=pareto_color, ls="--", lw=1.0, alpha=0.6, zorder=2)

    # Per-point labels — offset in *points* for predictable placement.
    # Three horizontal bands (y≈79, y≈53, y≈42) require staggered labels.
    #                 (dx_pt, dy_pt, ha)
    nudges = {
        0: (8, 8, "left"),           # C0 Baseline (110,79) → right-above
        1: (8, -14, "left"),         # C1 +Auth (112,53) → right-below
        2: (8, -4, "left"),          # C2 +AP-iso (114,68) → right
        3: (8, -14, "left"),         # C3 star (110,42) → right-below
        4: (0, 12, "center"),        # C4 +PMF (108,79) → above
        5: (-8, -14, "right"),       # C5 +PMF+auth (109,53) → left-below
        6: (-8, 10, "right"),        # C6 WPA3-SAE (106,79) → left-above
        7: (-8, 10, "right"),        # C7 Full (108,42) → left-above
    }
    for i, label in enumerate(short_labels):
        dx, dy, ha = nudges.get(i, (8, 0, "left"))
        ax.annotate(
            label,
            xy=(rec[i], atk[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6,
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4),
        )

    ax.set_xlabel("Reconnection latency (ms)")
    ax.set_ylabel("Attack success rate (%)")
    ax.legend(
        loc="upper right", fontsize=6.5, frameon=True,
        fancybox=False, edgecolor="0.7", handletextpad=0.4,
    )

    # Axis limits with breathing room
    ax.set_xlim(rec.min() - 3, rec.max() + 4)
    ax.set_ylim(atk.min() - 10, atk.max() + 10)

    out = OUT_DIR / "fig_hardening_pareto.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  \u2713 {out.name}")
    print(f"    Configs: {len(names)}")
    print(f"    Pareto-optimal: {[short_labels[i] for i in pareto_idx]}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating paper figures …\n")
    fig_rtt_cdf()
    print()
    fig_hardening_pareto()
    print(f"\nAll figures saved to {OUT_DIR.resolve()}")
