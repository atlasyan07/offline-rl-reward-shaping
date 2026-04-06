#!/usr/bin/env python3
"""Generate DPO training dashboard in the same LaTeX-style as other dashboards.

Usage:
    python scripts/section4/analyze_dpo_metrics.py \
        --metrics outputs/section4_dpo_beta01/train_metrics.csv \
        --output-dir outputs/section4_dpo_beta01/training_diagnostics \
        --title "DPO Training Dashboard — β = 0.1"

    # Side-by-side comparison of two beta runs:
    python scripts/section4/analyze_dpo_metrics.py \
        --metrics outputs/section4_dpo_beta01/train_metrics.csv \
        --metrics2 outputs/section4_dpo_beta05/train_metrics.csv \
        --label1 "β = 0.1" --label2 "β = 0.5" \
        --output-dir outputs/section4_dpo_comparison \
        --title "DPO: KL Penalty Comparison"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style: match existing dashboards exactly ──────────────────────────────────
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

# Color palette (consistent with other dashboards)
C_LOSS = "#2b6cb0"       # Blue
C_MARGIN = "#2f855a"     # Green
C_CHOSEN = "#2b6cb0"     # Blue
C_REJECTED = "#c05621"   # Orange
C_ACC = "#805ad5"        # Purple
C_DRIFT = "#c53030"      # Red
C_SECONDARY = "#dd6b20"  # Orange (secondary axis)

C_RUN1 = "#1a4173"       # Dark blue (run 1 / low beta)
C_RUN2 = "#b33030"       # Dark red (run 2 / high beta)


def load_metrics(path: str) -> dict[str, np.ndarray]:
    """Load train_metrics.csv into arrays."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No data in {path}")
    result = {}
    for key in rows[0]:
        result[key] = np.array([float(r[key]) for r in rows])
    return result


def rolling(arr: np.ndarray, w: int = 20) -> np.ndarray:
    """Simple rolling mean."""
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training dashboard")
    parser.add_argument("--metrics", required=True, help="Path to train_metrics.csv")
    parser.add_argument("--metrics2", default=None, help="Optional second run for comparison")
    parser.add_argument("--label1", default="Run 1")
    parser.add_argument("--label2", default="Run 2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="DPO Training Dashboard")
    return parser.parse_args()


def build_single_run_dashboard(
    m: dict[str, np.ndarray],
    out_dir: Path,
    title: str,
) -> None:
    """6-panel dashboard for a single DPO run."""
    steps = m["step"]
    w = max(1, len(steps) // 50)  # adaptive smoothing window

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    fig.suptitle(title, fontsize=14, y=0.98)

    # ── Panel 1: DPO Loss ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, m["loss"], color=C_LOSS, alpha=0.25, linewidth=0.8)
    ax.plot(steps, rolling(m["loss"], w), color=C_LOSS, linewidth=1.5, label="loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("DPO Loss")
    ax.set_title("DPO Loss")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8)

    # ── Panel 2: Reward Margin ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, m["reward_margin"], color=C_MARGIN, alpha=0.25, linewidth=0.8)
    ax.plot(steps, rolling(m["reward_margin"], w), color=C_MARGIN, linewidth=1.5,
            label="margin")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\beta \cdot (\log\pi_{w} - \log\pi_{l})$")
    ax.set_title("Reward Margin (chosen − rejected)")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8)

    # ── Panel 3: Preference Accuracy ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(steps, m["accuracy"], color=C_ACC, alpha=0.25, linewidth=0.8)
    ax.plot(steps, rolling(m["accuracy"], w), color=C_ACC, linewidth=1.5,
            label="accuracy")
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Preference Accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8)

    # ── Panel 4: Chosen vs Rejected Rewards ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, rolling(m["chosen_reward_mean"], w), color=C_CHOSEN,
            linewidth=1.5, label="chosen")
    ax.plot(steps, rolling(m["rejected_reward_mean"], w), color=C_REJECTED,
            linewidth=1.5, label="rejected")
    ax.fill_between(steps,
                    rolling(m["chosen_reward_mean"], w),
                    rolling(m["rejected_reward_mean"], w),
                    alpha=0.08, color=C_MARGIN)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Implicit Reward")
    ax.set_title("Implicit Rewards: Chosen vs Rejected")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8)

    # ── Panel 5: Policy Drift (chosen reward magnitude) ───────────────────────
    ax = fig.add_subplot(gs[1, 1])
    # Policy drift = how far the implicit reward has moved from 0
    # (at init, log pi / pi_ref = 0, so reward = 0)
    drift = np.abs(m["chosen_reward_mean"]) + np.abs(m["rejected_reward_mean"])
    ax.plot(steps, drift, color=C_DRIFT, alpha=0.25, linewidth=0.8)
    ax.plot(steps, rolling(drift, w), color=C_DRIFT, linewidth=1.5,
            label="total drift")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$|r_w| + |r_l|$")
    ax.set_title("Policy Drift from Reference")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8)

    # ── Panel 6: Reward Margin Distribution (histogram of final window) ───────
    ax = fig.add_subplot(gs[1, 2])
    # Use last 20% of training for the histogram
    cutoff = max(1, int(len(steps) * 0.8))
    final_margins = m["reward_margin"][cutoff:]
    ax.hist(final_margins, bins=30, color=C_MARGIN, alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axvline(x=np.mean(final_margins), color=C_MARGIN, linestyle="--",
               linewidth=1.2, label=f"mean={np.mean(final_margins):.3f}")
    ax.set_xlabel("Reward Margin")
    ax.set_ylabel("Count")
    ax.set_title("Margin Distribution (final 20%)")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "dpo_dashboard.png", dpi=200)
    fig.savefig(out_dir / "dpo_dashboard.pdf", dpi=300)
    print(f"Saved {out_dir / 'dpo_dashboard.png'}")
    plt.close()


def build_comparison_dashboard(
    m1: dict[str, np.ndarray],
    m2: dict[str, np.ndarray],
    label1: str,
    label2: str,
    out_dir: Path,
    title: str,
) -> None:
    """6-panel comparison dashboard for two DPO runs (different betas)."""
    s1, s2 = m1["step"], m2["step"]
    w1 = max(1, len(s1) // 50)
    w2 = max(1, len(s2) // 50)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    fig.suptitle(title, fontsize=14, y=0.98)

    def _dual(ax, key, ylabel, panel_title, ylim=None):
        ax.plot(s1, rolling(m1[key], w1), color=C_RUN1, linewidth=1.5, label=label1)
        ax.plot(s2, rolling(m2[key], w2), color=C_RUN2, linewidth=1.5, label=label2)
        ax.plot(s1, m1[key], color=C_RUN1, alpha=0.15, linewidth=0.6)
        ax.plot(s2, m2[key], color=C_RUN2, alpha=0.15, linewidth=0.6)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
        if ylim:
            ax.set_ylim(ylim)

    # Panel 1: Loss
    _dual(fig.add_subplot(gs[0, 0]), "loss", "DPO Loss", "DPO Loss")

    # Panel 2: Reward Margin
    ax = fig.add_subplot(gs[0, 1])
    _dual(ax, "reward_margin", r"$\beta \cdot (\log\pi_w - \log\pi_l)$",
          "Reward Margin")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

    # Panel 3: Accuracy
    ax = fig.add_subplot(gs[0, 2])
    _dual(ax, "accuracy", "Accuracy", "Preference Accuracy", ylim=(0.3, 1.02))
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

    # Panel 4: Chosen rewards
    ax = fig.add_subplot(gs[1, 0])
    _dual(ax, "chosen_reward_mean", "Implicit Reward (chosen)",
          "Chosen Reward")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

    # Panel 5: Rejected rewards
    ax = fig.add_subplot(gs[1, 1])
    _dual(ax, "rejected_reward_mean", "Implicit Reward (rejected)",
          "Rejected Reward")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

    # Panel 6: Policy drift comparison
    ax = fig.add_subplot(gs[1, 2])
    drift1 = np.abs(m1["chosen_reward_mean"]) + np.abs(m1["rejected_reward_mean"])
    drift2 = np.abs(m2["chosen_reward_mean"]) + np.abs(m2["rejected_reward_mean"])
    ax.plot(s1, rolling(drift1, w1), color=C_RUN1, linewidth=1.5, label=label1)
    ax.plot(s2, rolling(drift2, w2), color=C_RUN2, linewidth=1.5, label=label2)
    ax.plot(s1, drift1, color=C_RUN1, alpha=0.15, linewidth=0.6)
    ax.plot(s2, drift2, color=C_RUN2, alpha=0.15, linewidth=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$|r_w| + |r_l|$")
    ax.set_title("Policy Drift from Reference")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "dpo_comparison.png", dpi=200)
    fig.savefig(out_dir / "dpo_comparison.pdf", dpi=300)
    print(f"Saved {out_dir / 'dpo_comparison.png'}")
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m1 = load_metrics(args.metrics)

    if args.metrics2:
        m2 = load_metrics(args.metrics2)
        build_comparison_dashboard(m1, m2, args.label1, args.label2, out_dir, args.title)
    else:
        build_single_run_dashboard(m1, out_dir, args.title)

    # Print summary stats
    print(f"\n{'='*50}")
    print(f"Run: {args.metrics}")
    print(f"  Steps: {int(m1['step'][-1])}")
    print(f"  Final loss: {m1['loss'][-10:].mean():.4f}")
    print(f"  Final margin: {m1['reward_margin'][-10:].mean():.4f}")
    print(f"  Final accuracy: {m1['accuracy'][-10:].mean():.3f}")
    if args.metrics2:
        print(f"\nRun: {args.metrics2}")
        print(f"  Steps: {int(m2['step'][-1])}")
        print(f"  Final loss: {m2['loss'][-10:].mean():.4f}")
        print(f"  Final margin: {m2['reward_margin'][-10:].mean():.4f}")
        print(f"  Final accuracy: {m2['accuracy'][-10:].mean():.3f}")


if __name__ == "__main__":
    main()
