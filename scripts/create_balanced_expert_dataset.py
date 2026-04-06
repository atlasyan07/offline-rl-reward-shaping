#!/usr/bin/env python3
"""Create a balanced expert dataset by upsampling goal transitions.

Takes an existing expert dataset and duplicates goal transitions to achieve
a target goal ratio. This ensures AIRL sees enough goal states to learn
that they're special.

Outputs:
  - expert_dataset.npz: balanced dataset
  - expert_metadata.json: updated metadata
  - balanced_dashboard.png/pdf: diagnostic plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create balanced expert dataset")
    parser.add_argument("--input", required=True, help="Path to input expert_dataset.npz")
    parser.add_argument("--input-metadata", required=True, help="Path to input expert_metadata.json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--target-goal-ratio", type=float, default=0.20,
                        help="Target fraction of goal transitions (default: 0.20 = 20%%)")
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip upsampling and only generate the dashboard")
    parser.add_argument("--title", default="Balanced Expert Dataset", help="Dashboard title")
    return parser.parse_args()


def generate_dashboard(
    out_dir: Path,
    title: str,
    original_total: int,
    original_goal: int,
    new_total: int,
    new_goal: int,
    multiplier: int,
    target_ratio: float,
    balanced_data: dict,
    metadata: dict,
) -> None:
    """Generate diagnostic dashboard for the balanced dataset."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "CMU Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14, y=0.98)
    grid = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    def style(ax):
        ax.grid(alpha=0.2, linewidth=0.6)

    # Extract data
    r = balanced_data["r"]
    r_env_goal = r[r > 0]
    episodes = metadata.get("episodes", [])

    # Panel A: Summary text (top-left)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_a.axis("off")
    original_ratio = original_goal / original_total
    new_ratio = new_goal / new_total
    summary = "\n".join([
        f"Original Dataset:",
        f"  Transitions: {original_total:,}",
        f"  Goal: {original_goal:,} ({100*original_ratio:.1f}%)",
        f"",
        f"Balanced Dataset:",
        f"  Transitions: {new_total:,}",
        f"  Goal: {new_goal:,} ({100*new_ratio:.1f}%)",
        f"",
        f"Target ratio: {100*target_ratio:.0f}%",
        f"Goal multiplier: {multiplier}x",
        f"Episodes: {len(episodes)}",
    ])
    ax_a.text(0.05, 0.95, summary, fontsize=10, va="top", family="monospace",
              transform=ax_a.transAxes)
    ax_a.set_title("Summary")

    # Panel B: Before/After bar comparison (top-middle)
    ax_b = fig.add_subplot(grid[0, 1])
    labels = ["Original", "Balanced"]
    goal_counts = [original_goal, new_goal]
    non_goal_counts = [original_total - original_goal, new_total - new_goal]
    x = np.arange(len(labels))
    width = 0.35
    ax_b.bar(x - width/2, goal_counts, width, label="Goal", color="#2f855a", alpha=0.85)
    ax_b.bar(x + width/2, non_goal_counts, width, label="Non-goal", color="#718096", alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Transitions")
    ax_b.set_title("Goal vs Non-Goal Transitions")
    ax_b.legend()
    style(ax_b)

    # Panel C: Goal ratio pie chart (top-right)
    ax_c = fig.add_subplot(grid[0, 2])
    sizes = [new_goal, new_total - new_goal]
    labels_pie = [f"Goal\n({100*new_ratio:.1f}%)", f"Non-goal\n({100*(1-new_ratio):.1f}%)"]
    colors = ["#2f855a", "#718096"]
    ax_c.pie(sizes, labels=labels_pie, colors=colors, autopct="", startangle=90)
    ax_c.set_title("Balanced Dataset Composition")

    # Panel D: Goal reward distribution (middle-left)
    ax_d = fig.add_subplot(grid[1, 0])
    ax_d.hist(r_env_goal, bins=30, color="#c05621", alpha=0.85, edgecolor="white")
    ax_d.set_xlabel("Environment Reward")
    ax_d.set_ylabel("Count")
    ax_d.set_title("Goal Reward Distribution")
    ax_d.axvline(r_env_goal.mean(), color="red", linestyle="--",
                 label=f"Mean: {r_env_goal.mean():.3f}")
    ax_d.axvline(np.median(r_env_goal), color="blue", linestyle=":",
                 label=f"Median: {np.median(r_env_goal):.3f}")
    ax_d.legend(fontsize=8)
    style(ax_d)

    # Panel E: Episode length distribution (middle-middle)
    ax_e = fig.add_subplot(grid[1, 1])
    if episodes:
        lengths = [e["length"] for e in episodes]
        ax_e.hist(lengths, bins=30, color="#2b6cb0", alpha=0.85, edgecolor="white")
        ax_e.axvline(np.mean(lengths), color="red", linestyle="--",
                     label=f"Mean: {np.mean(lengths):.1f}")
        ax_e.legend(fontsize=8)
    ax_e.set_xlabel("Episode Length (steps)")
    ax_e.set_ylabel("Count")
    ax_e.set_title("Episode Length Distribution")
    style(ax_e)

    # Panel F: Success rate by layout (middle-right)
    ax_f = fig.add_subplot(grid[1, 2])
    if episodes:
        layouts = sorted(set(e["layout_name"] for e in episodes))
        success_rates = []
        for layout in layouts:
            layout_eps = [e for e in episodes if e["layout_name"] == layout]
            success_rate = sum(e.get("success", True) for e in layout_eps) / len(layout_eps)
            success_rates.append(success_rate * 100)
        ax_f.bar(range(len(layouts)), success_rates, color="#6b46c1", alpha=0.85)
        ax_f.set_xticks(range(len(layouts)))
        ax_f.set_xticklabels(layouts, rotation=30, ha="right")
        ax_f.set_ylim(0, 105)
    ax_f.set_ylabel("Success Rate (%)")
    ax_f.set_title("Success Rate by Layout")
    style(ax_f)

    # Panel G: Episodes per layout (bottom-left)
    ax_g = fig.add_subplot(grid[2, 0])
    if episodes:
        layouts = sorted(set(e["layout_name"] for e in episodes))
        counts = [sum(1 for e in episodes if e["layout_name"] == l) for l in layouts]
        ax_g.bar(range(len(layouts)), counts, color="#dd6b20", alpha=0.85)
        ax_g.set_xticks(range(len(layouts)))
        ax_g.set_xticklabels(layouts, rotation=30, ha="right")
    ax_g.set_ylabel("Episode Count")
    ax_g.set_title("Episodes per Layout")
    style(ax_g)

    # Panel H: Behavior source distribution (bottom-middle)
    ax_h = fig.add_subplot(grid[2, 1])
    if episodes:
        behaviors = sorted(set(e.get("planner_type", "unknown") for e in episodes))
        beh_counts = [sum(1 for e in episodes if e.get("planner_type", "unknown") == b) for b in behaviors]
        ax_h.bar(range(len(behaviors)), beh_counts, color="#319795", alpha=0.85)
        ax_h.set_xticks(range(len(behaviors)))
        ax_h.set_xticklabels(behaviors, rotation=30, ha="right")
    ax_h.set_ylabel("Episode Count")
    ax_h.set_title("Behavior Source Distribution")
    style(ax_h)

    # Panel I: Reward statistics box (bottom-right)
    ax_i = fig.add_subplot(grid[2, 2])
    ax_i.axis("off")
    stats_text = "\n".join([
        "Goal Reward Statistics:",
        f"  Min:    {r_env_goal.min():.4f}",
        f"  Max:    {r_env_goal.max():.4f}",
        f"  Mean:   {r_env_goal.mean():.4f}",
        f"  Median: {np.median(r_env_goal):.4f}",
        f"  Std:    {r_env_goal.std():.4f}",
        "",
        "Episode Statistics:",
    ])
    if episodes:
        lengths = [e["length"] for e in episodes]
        stats_text += "\n".join([
            f"  Min length:  {min(lengths)}",
            f"  Max length:  {max(lengths)}",
            f"  Mean length: {np.mean(lengths):.1f}",
        ])
    ax_i.text(0.05, 0.95, stats_text, fontsize=10, va="top", family="monospace",
              transform=ax_i.transAxes)
    ax_i.set_title("Statistics")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "balanced_dashboard.png", dpi=200)
    fig.savefig(out_dir / "balanced_dashboard.pdf", dpi=300)
    plt.close(fig)
    print(f"[balanced] Saved dashboard to {out_dir / 'balanced_dashboard.png'}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load input dataset
    data = np.load(args.input, allow_pickle=True)
    expert = {k: data[k] for k in data.files}

    with open(args.input_metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Find goal and non-goal indices
    r = expert["r"]
    goal_idx = np.where(r > 0)[0]
    non_goal_idx = np.where(r == 0)[0]

    n_goal = len(goal_idx)
    n_non_goal = len(non_goal_idx)
    n_total = len(r)
    current_ratio = n_goal / n_total

    print(f"Input dataset:")
    print(f"  Total transitions: {n_total}")
    print(f"  Goal transitions: {n_goal} ({100*current_ratio:.1f}%)")
    print(f"  Non-goal transitions: {n_non_goal}")
    print()

    # Calculate how many times to duplicate goal transitions
    # target_ratio = (n_goal * multiplier) / (n_non_goal + n_goal * multiplier)
    # Solving for multiplier:
    # target_ratio * n_non_goal + target_ratio * n_goal * multiplier = n_goal * multiplier
    # target_ratio * n_non_goal = n_goal * multiplier * (1 - target_ratio)
    # multiplier = (target_ratio * n_non_goal) / (n_goal * (1 - target_ratio))

    if args.no_balance:
        target_ratio = current_ratio
        multiplier = 1
        balanced = expert
    else:
        target_ratio = args.target_goal_ratio
        if current_ratio >= target_ratio:
            print(f"Current ratio ({100*current_ratio:.1f}%) >= target ({100*target_ratio:.1f}%)")
            print("No upsampling needed.")
            multiplier = 1
        else:
            multiplier = int(np.ceil((target_ratio * n_non_goal) / (n_goal * (1 - target_ratio))))
            print(f"Target ratio: {100*target_ratio:.1f}%")
            print(f"Duplicating goal transitions {multiplier}x")

        # Build new indices: all non-goal + duplicated goal
        new_indices = list(non_goal_idx)
        for _ in range(multiplier):
            new_indices.extend(goal_idx)
        new_indices = np.array(new_indices)

        # Shuffle
        np.random.seed(42)
        np.random.shuffle(new_indices)

        # Create new dataset
        balanced = {}
        for k, v in expert.items():
            balanced[k] = v[new_indices]

    # Verify new ratio
    new_r = balanced["r"]
    new_goal = (new_r > 0).sum()
    new_total = len(new_r)
    new_ratio = new_goal / new_total

    print()
    print(f"Output dataset:")
    print(f"  Total transitions: {new_total}")
    print(f"  Goal transitions: {new_goal} ({100*new_ratio:.1f}%)")
    print(f"  Non-goal transitions: {new_total - new_goal}")

    # Save dataset
    np.savez_compressed(out_dir / "expert_dataset.npz", **balanced)

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata["balanced"] = {
        "source": str(args.input),
        "target_goal_ratio": target_ratio,
        "goal_multiplier": multiplier,
        "original_transitions": n_total,
        "original_goal_ratio": current_ratio,
        "no_balance": args.no_balance,
    }
    new_metadata["transitions"] = new_total

    with open(out_dir / "expert_metadata.json", "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, indent=2)

    # Generate dashboard
    generate_dashboard(
        out_dir=out_dir,
        title=args.title,
        original_total=n_total,
        original_goal=n_goal,
        new_total=new_total,
        new_goal=new_goal,
        multiplier=multiplier,
        target_ratio=target_ratio,
        balanced_data=balanced,
        metadata=new_metadata,
    )

    print()
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
