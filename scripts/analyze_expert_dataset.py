#!/usr/bin/env python3
"""LaTeX-style diagnostics for extracted expert dataset."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze expert dataset metadata")
    parser.add_argument("--metadata", required=True, help="Path to expert_metadata.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for dashboard")
    parser.add_argument("--title", default="Expert Dataset Diagnostics", help="Dashboard title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    episodes = meta.get("episodes", [])
    if not episodes:
        raise SystemExit("No expert episodes found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths = np.array([e["length"] for e in episodes], dtype=np.float64)
    turn_rates = np.array([e.get("turn_rate", 0.0) for e in episodes], dtype=np.float64)
    layouts = sorted({e["layout_name"] for e in episodes})
    goals = sorted({e.get("goal_region_id", 0) for e in episodes})
    behaviors = sorted({e.get("planner_type", "unknown") for e in episodes})

    # Styling
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(args.title, fontsize=14, y=0.98)
    grid = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    def style(ax):
        ax.grid(alpha=0.2, linewidth=0.6)

    # Panel A: Summary
    ax_a = fig.add_subplot(grid[0, 0])
    ax_a.axis("off")
    sel = meta.get("selection", {})
    summary_text = "\n".join(
        [
            f"Episodes: {len(episodes)}",
            f"Transitions: {meta.get('transitions', 'n/a')}",
            f"Success rate: {meta.get('success_rate', 0.0):.2%}",
            f"Top pct: {sel.get('top_pct', 'n/a')}",
            f"Turn cutoff: {sel.get('turn_rate_cutoff', 'n/a')}",
        ]
    )
    ax_a.text(0.02, 0.98, "Expert Summary", fontsize=12, fontweight="bold", va="top")
    ax_a.text(0.02, 0.78, summary_text, fontsize=10, va="top")

    # Panel B: Length distribution
    ax_b = fig.add_subplot(grid[0, 1])
    ax_b.hist(lengths, bins=25, color="#718096", alpha=0.85)
    ax_b.set_title("Episode Lengths")
    ax_b.set_xlabel("Steps")
    ax_b.set_ylabel("Count")
    style(ax_b)

    # Panel C: Turn-rate distribution
    ax_c = fig.add_subplot(grid[0, 2])
    ax_c.hist(turn_rates, bins=25, color="#2b6cb0", alpha=0.85)
    ax_c.set_title("Turn Rate")
    ax_c.set_xlabel("Turns / Steps")
    style(ax_c)

    # Panel D: Coverage by layout
    ax_d = fig.add_subplot(grid[1, 0])
    layout_counts = [sum(e["layout_name"] == l for e in episodes) for l in layouts]
    ax_d.bar(layouts, layout_counts, color="#2f855a", alpha=0.85)
    ax_d.set_title("Layout Coverage")
    ax_d.tick_params(axis="x", rotation=25)
    style(ax_d)

    # Panel E: Coverage by goal region
    ax_e = fig.add_subplot(grid[1, 1])
    goal_counts = [sum(e.get("goal_region_id", 0) == g for e in episodes) for g in goals]
    ax_e.bar([str(g) for g in goals], goal_counts, color="#c05621", alpha=0.85)
    ax_e.set_title("Goal Region Coverage")
    ax_e.set_xlabel("Goal Region")
    style(ax_e)

    # Panel F: Coverage by behavior
    ax_f = fig.add_subplot(grid[1, 2])
    beh_counts = [sum(e.get("planner_type", "unknown") == b for e in episodes) for b in behaviors]
    ax_f.bar(behaviors, beh_counts, color="#6b46c1", alpha=0.85)
    ax_f.set_title("Behavior Source")
    ax_f.tick_params(axis="x", rotation=25)
    style(ax_f)

    # Panel G: Length by layout (boxplot)
    ax_g = fig.add_subplot(grid[2, :])
    length_by_layout = [[e["length"] for e in episodes if e["layout_name"] == l] for l in layouts]
    ax_g.boxplot(length_by_layout, labels=layouts, showfliers=False)
    ax_g.set_title("Episode Length by Layout")
    ax_g.set_ylabel("Steps")
    ax_g.tick_params(axis="x", rotation=25)
    style(ax_g)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = out_dir / "expert_dashboard.png"
    fig_pdf = out_dir / "expert_dashboard.pdf"
    fig.savefig(fig_path, dpi=200)
    fig.savefig(fig_pdf, dpi=300)
    plt.close(fig)
    print(f"[analyze_expert_dataset] wrote {fig_path} and {fig_pdf}")

    # Write a compact CSV summary
    summary_rows = []
    for l in layouts:
        vals = [e for e in episodes if e["layout_name"] == l]
        summary_rows.append(
            {
                "layout": l,
                "count": len(vals),
                "length_mean": float(np.mean([v["length"] for v in vals])),
                "turn_rate_mean": float(np.mean([v.get("turn_rate", 0.0) for v in vals])),
            }
        )
    with open(out_dir / "layout_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
