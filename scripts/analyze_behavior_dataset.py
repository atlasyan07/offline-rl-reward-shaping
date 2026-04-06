#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior dataset diagnostics dashboard")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--title", default="Behavior Dataset Diagnostics", help="Figure title")
    return parser.parse_args()


def _safe_rate(successes: List[bool]) -> float:
    return float(np.mean(successes)) if successes else 0.0


def summarize(episodes: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    by_behavior = defaultdict(list)
    by_layout = defaultdict(list)
    by_goal = defaultdict(list)
    for ep in episodes:
        by_behavior[ep["planner_type"]].append(ep)
        by_layout[ep["layout_name"]].append(ep)
        by_goal[ep.get("goal_region_id", -1)].append(ep)
    return by_behavior, by_layout, by_goal


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    episodes = metadata.get("episodes", [])
    if not episodes:
        raise SystemExit("No episodes found in metadata")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_behavior, by_layout, by_goal = summarize(episodes)
    behaviors = sorted(by_behavior.keys())
    layouts = sorted(by_layout.keys())
    goals = sorted(by_goal.keys())

    # Global stats
    success_all = [ep["success"] for ep in episodes]
    lengths_all = [ep["length"] for ep in episodes]

    # Styling (LaTeX-like without requiring a TeX install)
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

    def _style_axis(ax):
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)

    # Panel A: Summary box
    ax_a = fig.add_subplot(grid[0, 0])
    ax_a.axis("off")
    summary_text = "\n".join(
        [
            f"Episodes: {len(episodes):,}",
            f"Transitions: {metadata.get('transitions', 'n/a')}",
            f"Success rate: {_safe_rate(success_all):.2%}",
            f"Length (mean / p50 / p90): "
            f"{np.mean(lengths_all):.1f} / {np.median(lengths_all):.1f} / {np.percentile(lengths_all, 90):.1f}",
        ]
    )
    ax_a.text(0.02, 0.98, "Dataset Summary", fontsize=12, fontweight="bold", va="top")
    ax_a.text(0.02, 0.78, summary_text, fontsize=10, va="top")

    # Panel B: Success by behavior
    ax_b = fig.add_subplot(grid[0, 1])
    succ_b = [_safe_rate([ep["success"] for ep in by_behavior[b]]) for b in behaviors]
    counts_b = [len(by_behavior[b]) for b in behaviors]
    ax_b.bar(behaviors, succ_b, color="#2b6cb0", alpha=0.85)
    ax_b.set_ylim(0, 1.08)
    ax_b.set_title("Success Rate by Behavior")
    ax_b.set_ylabel("Success Rate")
    ax_b.tick_params(axis="x", rotation=25)
    for i, c in enumerate(counts_b):
        y = succ_b[i] - 0.06 if succ_b[i] > 0.92 else succ_b[i] + 0.03
        ax_b.text(i, y, f"n={c}", ha="center", fontsize=8, color="#1a202c")
    _style_axis(ax_b)

    # Panel C: Success by layout
    ax_c = fig.add_subplot(grid[0, 2])
    succ_l = [_safe_rate([ep["success"] for ep in by_layout[l]]) for l in layouts]
    counts_l = [len(by_layout[l]) for l in layouts]
    ax_c.bar(layouts, succ_l, color="#2f855a", alpha=0.85)
    ax_c.set_ylim(0, 1.08)
    ax_c.set_title("Success Rate by Layout")
    ax_c.tick_params(axis="x", rotation=25)
    for i, c in enumerate(counts_l):
        y = succ_l[i] - 0.06 if succ_l[i] > 0.92 else succ_l[i] + 0.03
        ax_c.text(i, y, f"n={c}", ha="center", fontsize=8, color="#1a202c")
    _style_axis(ax_c)

    # Panel D: Success heatmap (behavior x layout)
    ax_d = fig.add_subplot(grid[1, 0])
    heat = np.zeros((len(behaviors), len(layouts)), dtype=np.float32)
    for i, b in enumerate(behaviors):
        for j, l in enumerate(layouts):
            vals = [ep["success"] for ep in episodes if ep["planner_type"] == b and ep["layout_name"] == l]
            heat[i, j] = _safe_rate(vals)
    im = ax_d.imshow(heat, vmin=0, vmax=1, cmap="viridis")
    ax_d.set_title("Success (Behavior x Layout)")
    ax_d.set_xticks(range(len(layouts)))
    ax_d.set_xticklabels(layouts, rotation=30, ha="right")
    ax_d.set_yticks(range(len(behaviors)))
    ax_d.set_yticklabels(behaviors)
    fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)

    # Panel E: Length by behavior (boxplot)
    ax_e = fig.add_subplot(grid[1, 1])
    length_b = [[ep["length"] for ep in by_behavior[b]] for b in behaviors]
    ax_e.boxplot(length_b, labels=behaviors, showfliers=False)
    ax_e.set_title("Episode Length by Behavior")
    ax_e.set_ylabel("Steps")
    ax_e.tick_params(axis="x", rotation=25)
    _style_axis(ax_e)

    # Panel F: Length by layout (boxplot)
    ax_f = fig.add_subplot(grid[1, 2])
    length_l = [[ep["length"] for ep in by_layout[l]] for l in layouts]
    ax_f.boxplot(length_l, labels=layouts, showfliers=False)
    ax_f.set_title("Episode Length by Layout")
    ax_f.tick_params(axis="x", rotation=25)
    _style_axis(ax_f)

    # Panel G: Success by goal region
    ax_g = fig.add_subplot(grid[2, 0])
    goal_labels = [str(g) for g in goals]
    succ_g = [_safe_rate([ep["success"] for ep in by_goal[g]]) for g in goals]
    counts_g = [len(by_goal[g]) for g in goals]
    ax_g.bar(goal_labels, succ_g, color="#c05621", alpha=0.85)
    ax_g.set_ylim(0, 1.08)
    ax_g.set_title("Success Rate by Goal Region")
    ax_g.set_xlabel("Goal Region")
    for i, c in enumerate(counts_g):
        y = succ_g[i] - 0.06 if succ_g[i] > 0.92 else succ_g[i] + 0.03
        ax_g.text(i, y, f"n={c}", ha="center", fontsize=8, color="#1a202c")
    _style_axis(ax_g)

    # Panel H: Length histogram
    ax_h = fig.add_subplot(grid[2, 1])
    ax_h.hist(lengths_all, bins=30, color="#718096", alpha=0.85)
    ax_h.set_title("Episode Length Distribution")
    ax_h.set_xlabel("Steps")
    ax_h.set_ylabel("Count")
    _style_axis(ax_h)

    # Panel I: Success by split
    ax_i = fig.add_subplot(grid[2, 2])
    by_split = defaultdict(list)
    for ep in episodes:
        by_split[ep.get("split", "unknown")].append(ep)
    splits = sorted(by_split.keys())
    succ_s = [_safe_rate([ep["success"] for ep in by_split[s]]) for s in splits]
    counts_s = [len(by_split[s]) for s in splits]
    ax_i.bar(splits, succ_s, color="#6b46c1", alpha=0.85)
    ax_i.set_ylim(0, 1.08)
    ax_i.set_title("Success Rate by Split")
    ax_i.tick_params(axis="x", rotation=25)
    for i, c in enumerate(counts_s):
        y = succ_s[i] - 0.06 if succ_s[i] > 0.92 else succ_s[i] + 0.03
        ax_i.text(i, y, f"n={c}", ha="center", fontsize=8, color="#1a202c")
    _style_axis(ax_i)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = out_dir / "dashboard.png"
    fig_pdf = out_dir / "dashboard.pdf"
    fig.savefig(fig_path, dpi=200)
    fig.savefig(fig_pdf, dpi=300)
    plt.close(fig)

    # Summary tables
    behavior_rows = []
    for b in behaviors:
        lengths = [ep["length"] for ep in by_behavior[b]]
        behavior_rows.append(
            {
                "behavior": b,
                "count": len(lengths),
                "success_rate": _safe_rate([ep["success"] for ep in by_behavior[b]]),
                "length_mean": float(np.mean(lengths)) if lengths else 0.0,
                "length_p50": float(np.median(lengths)) if lengths else 0.0,
                "length_p90": float(np.percentile(lengths, 90)) if lengths else 0.0,
            }
        )
    layout_rows = []
    for l in layouts:
        lengths = [ep["length"] for ep in by_layout[l]]
        layout_rows.append(
            {
                "layout": l,
                "count": len(lengths),
                "success_rate": _safe_rate([ep["success"] for ep in by_layout[l]]),
                "length_mean": float(np.mean(lengths)) if lengths else 0.0,
                "length_p50": float(np.median(lengths)) if lengths else 0.0,
                "length_p90": float(np.percentile(lengths, 90)) if lengths else 0.0,
            }
        )

    write_summary_csv(str(out_dir / "behavior_summary.csv"), behavior_rows)
    write_summary_csv(str(out_dir / "layout_summary.csv"), layout_rows)

    # Small text summary
    summary = {
        "episodes": len(episodes),
        "success_rate": _safe_rate(success_all),
        "length_mean": float(np.mean(lengths_all)),
        "length_p50": float(np.median(lengths_all)),
        "length_p90": float(np.percentile(lengths_all, 90)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[analyze_behavior_dataset] wrote", fig_path)


if __name__ == "__main__":
    main()
