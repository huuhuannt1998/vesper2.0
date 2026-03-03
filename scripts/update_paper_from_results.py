#!/usr/bin/env python3
"""
VESPER — Update Paper LaTeX from Experiment Results

Reads real experiment results (RQ-N1, RQ-N2, Trace Validation) and
updates the paper LaTeX sections with measured values, replacing the
protocol-derived placeholders with actual data.

Usage:
    python3 scripts/update_paper_from_results.py \\
        --results-dir results/flagship_20260303_120000 \\
        --paper-dir paper-latex
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("vesper.update_paper")


def load_json(path: str) -> Optional[Dict]:
    """Load a JSON file, return None if not found."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def find_result_file(results_dir: str, *patterns: str) -> Optional[str]:
    """Find a result file matching any of the patterns."""
    for pattern in patterns:
        for root, dirs, files in os.walk(results_dir):
            for fname in files:
                if fname == pattern or fname.endswith(pattern):
                    return os.path.join(root, fname)
    return None


def update_abstract(paper_dir: str, rqn1: Dict, rqn2: Dict, trace: Dict):
    """Update abstract numbers in main.tex."""
    main_path = f"{paper_dir}/main.tex"
    if not os.path.exists(main_path):
        logger.warning(f"main.tex not found at {main_path}")
        return

    with open(main_path) as f:
        content = f.read()

    changes = 0

    # RQ-N1: WiFi attack classes missed
    if rqn1:
        summary = rqn1.get("summary", {})
        wifi_missed = int(summary.get("wifi_layer_attacks_missed_by_bridge", 0))
        jitter_ratio = summary.get("rtt_jitter_ratio", 0)

        if wifi_missed > 0:
            content = re.sub(
                r'\\textbf\{(\d+)\}\s*WiFi-layer attack classes',
                f'\\\\textbf{{{wifi_missed}}} WiFi-layer attack classes',
                content
            )
            changes += 1

        if jitter_ratio:
            content = re.sub(
                r'under\\-estimate jitter by \\textbf\{[\d.]+\}',
                f'under\\\\-estimate jitter by \\\\textbf{{{jitter_ratio}}}',
                content
            )
            changes += 1

    # RQ-N2: config count, attack reduction, reconnection
    if rqn2:
        configs = rqn2.get("configs_tested", 8)
        best = rqn2.get("fully_hardened", {})
        atk_reduction = best.get("delta_atk_pct", 0)
        reconn_ms = best.get("reconnection_ms", 0)

        if atk_reduction:
            content = re.sub(
                r'\\textbf\{[\d.]+\}\\% attack\s*\n?\s*reduction',
                f'\\\\textbf{{{abs(atk_reduction)}}}\\\\% attack\nreduction',
                content
            )
            changes += 1

        if reconn_ms:
            content = re.sub(
                r'\\textbf\{[\d.]+\}\\,ms mean reconnection',
                f'\\\\textbf{{{int(reconn_ms)}}}\\\\,ms mean reconnection',
                content
            )
            changes += 1

    if changes > 0:
        with open(main_path, "w") as f:
            f.write(content)
        logger.info(f"Updated abstract with {changes} measured values")
    else:
        logger.info("Abstract: no changes needed (values already correct or no data)")


def update_intro(paper_dir: str, rqn1: Dict, rqn2: Dict, trace: Dict):
    """Update intro contribution bullet points."""
    intro_path = f"{paper_dir}/sections/01_intro.tex"
    if not os.path.exists(intro_path):
        return

    with open(intro_path) as f:
        content = f.read()

    changes = 0

    if rqn1:
        summary = rqn1.get("summary", {})
        wifi_missed = int(summary.get("wifi_layer_attacks_missed_by_bridge", 0))
        jitter_ratio = summary.get("rtt_jitter_ratio", 0)

        if wifi_missed > 0:
            old = re.search(r'\\textbf\{(\d+)\}\s*WiFi-layer attack classes', content)
            if old and int(old.group(1)) != wifi_missed:
                content = content.replace(old.group(0),
                    f'\\textbf{{{wifi_missed}}} WiFi-layer attack classes')
                changes += 1

        if jitter_ratio:
            old = re.search(r'underestimate latency jitter by \\textbf\{[\d.]+\}', content)
            if old:
                content = content.replace(old.group(0),
                    f'underestimate latency jitter by \\textbf{{{jitter_ratio}}}')
                changes += 1

    if rqn2:
        best = rqn2.get("fully_hardened", {})
        atk_reduction = best.get("delta_atk_pct", 0)
        reconn_ms = best.get("reconnection_ms", 0)

        if atk_reduction:
            old = re.search(r'reduces exploitable attacks by \\textbf\{[\d.]+\}', content)
            if old:
                content = content.replace(old.group(0),
                    f'reduces exploitable attacks by \\textbf{{{abs(atk_reduction)}}}')
                changes += 1

        if reconn_ms:
            old = re.search(r'reconnection cost of \\textbf\{[\d.]+\}', content)
            if old:
                content = content.replace(old.group(0),
                    f'reconnection cost of \\textbf{{{int(reconn_ms)}}}')
                changes += 1

    if changes > 0:
        with open(intro_path, "w") as f:
            f.write(content)
        logger.info(f"Updated intro with {changes} measured values")


def update_conclusion(paper_dir: str, rqn1: Dict, rqn2: Dict, trace: Dict):
    """Update conclusion summaries."""
    concl_path = f"{paper_dir}/sections/08_conclusion.tex"
    if not os.path.exists(concl_path):
        return

    with open(concl_path) as f:
        content = f.read()

    # The conclusion was already updated with protocol-derived numbers.
    # If the real measured numbers differ significantly, update them.
    changes = 0

    if rqn1:
        summary = rqn1.get("summary", {})
        wifi_missed = int(summary.get("wifi_layer_attacks_missed_by_bridge", 0))
        jitter_ratio = summary.get("rtt_jitter_ratio", 0)

        rtt = rqn1.get("rtt", {})
        bridge_jitter = rtt.get("bridge", {}).get("mean_jitter_ms")
        wifi_jitter = rtt.get("wifi", {}).get("mean_jitter_ms")
        reconn = rqn1.get("reconnection", {}).get("wifi", {}).get("mean_ms")

        if wifi_missed > 0:
            old = re.search(r'Bridge networks miss (\d+) of (\d+)', content)
            if old:
                total = int(summary.get("wifi_layer_attacks_missed_by_bridge", 0)) + 14  # bridge has 14
                content = content.replace(old.group(0),
                    f'Bridge networks miss {wifi_missed} of {total}')
                changes += 1

        if jitter_ratio and bridge_jitter and wifi_jitter:
            old = re.search(r'underestimate RTT jitter by\s*\$[\d.]+\\times\$', content)
            if old:
                content = content.replace(old.group(0),
                    f'underestimate RTT jitter by\n    ${jitter_ratio}\\times$')
                changes += 1

    if changes > 0:
        with open(concl_path, "w") as f:
            f.write(content)
        logger.info(f"Updated conclusion with {changes} measured values")


def print_summary(rqn1: Dict, rqn2: Dict, trace: Dict):
    """Print a summary of what was updated."""
    print("\n" + "═" * 60)
    print("  PAPER UPDATE SUMMARY")
    print("═" * 60)

    if rqn1:
        s = rqn1.get("summary", {})
        rtt = rqn1.get("rtt", {})
        print(f"\n  RQ-N1 (Bridge vs 802.11):")
        print(f"    WiFi attacks missed by bridge: {s.get('wifi_layer_attacks_missed_by_bridge', 'N/A')}")
        print(f"    Jitter ratio:                  {s.get('rtt_jitter_ratio', 'N/A')}×")
        print(f"    Bridge RTT:                    {rtt.get('bridge', {}).get('mean_rtt_ms', 'N/A')} ms")
        print(f"    WiFi RTT:                      {rtt.get('wifi', {}).get('mean_rtt_ms', 'N/A')} ms")
        print(f"    Retransmissions (wifi):         {s.get('wifi_retransmissions', 'N/A')}")
    else:
        print("\n  RQ-N1: No results found")

    if rqn2:
        best = rqn2.get("fully_hardened", {})
        print(f"\n  RQ-N2 (Hardening Sweep):")
        print(f"    Configs tested:     {rqn2.get('configs_tested', 'N/A')}")
        print(f"    Baseline atk %:     {rqn2.get('baseline', {}).get('atk_pct', 'N/A')}")
        print(f"    Hardened atk %:     {best.get('atk_pct', 'N/A')}")
        print(f"    Δ attack:           {best.get('delta_atk_pct', 'N/A')}%")
        print(f"    Reconn (hardened):  {best.get('reconnection_ms', 'N/A')} ms")
    else:
        print("\n  RQ-N2: No results found")

    if trace:
        agg = trace.get("aggregate", {})
        comp = trace.get("comparison", {}).get("metrics", {})
        print(f"\n  Trace Validation:")
        print(f"    Pcap files:    {agg.get('pcap_count', 'N/A')}")
        print(f"    Total frames:  {agg.get('total_frames', 'N/A'):,}")
        print(f"    Flows/hour:    {agg.get('flows_per_hour', 'N/A')}")
        print(f"    Burstiness:    {agg.get('burstiness', {}).get('cov', 'N/A')}")
        print(f"    Diurnal r:     {agg.get('diurnal_correlation_r', 'N/A')}")
    else:
        print("\n  Trace Validation: No results found")

    print("\n" + "═" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Update paper LaTeX with real experiment results"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--paper-dir", default="paper-latex",
                        help="Paper LaTeX directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load results
    rqn1_file = find_result_file(args.results_dir, "rqn1_comparison.json")
    rqn2_file = find_result_file(args.results_dir, "rqn2_summary.json")
    trace_file = find_result_file(args.results_dir, "pcap_analysis.json")

    rqn1 = load_json(rqn1_file) if rqn1_file else None
    rqn2 = load_json(rqn2_file) if rqn2_file else None
    trace = load_json(trace_file) if trace_file else None

    if rqn1:
        logger.info(f"Loaded RQ-N1 results from {rqn1_file}")
    if rqn2:
        logger.info(f"Loaded RQ-N2 results from {rqn2_file}")
    if trace:
        logger.info(f"Loaded trace results from {trace_file}")

    if not any([rqn1, rqn2, trace]):
        logger.warning("No experiment results found in %s", args.results_dir)
        logger.info("Expected files: rqn1_comparison.json, rqn2_summary.json, pcap_analysis.json")
        return

    # Update paper sections
    update_abstract(args.paper_dir, rqn1, rqn2, trace)
    update_intro(args.paper_dir, rqn1, rqn2, trace)
    update_conclusion(args.paper_dir, rqn1, rqn2, trace)

    # Print summary
    print_summary(rqn1, rqn2, trace)

    logger.info("Paper update complete. Re-compile to see changes.")


if __name__ == "__main__":
    main()
