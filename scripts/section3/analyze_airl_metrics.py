#!/usr/bin/env python3
"""LaTeX-style diagnostics for AIRL training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AIRL training metrics")
    parser.add_argument("--metrics-csv", required=True, help="Path to airl_metrics.csv")
    parser.add_argument("--heatmap", default=None, help="Path to a heatmap image (optional)")
    parser.add_argument("--heatmap-dir", default=None, help="Directory containing heatmaps")
    parser.add_argument("--iteration", type=int, default=None, help="Iteration to visualize heatmaps")
    parser.add_argument("--alpha-json", default=None, help="Path to alpha suggestion JSON (optional)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--title", default="AIRL Diagnostics", help="Figure title")
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(r.get(key, "nan")) for r in rows], dtype=np.float64)


def select_heatmaps(heatmap_dir: Path, iteration: int | None) -> List[Path]:
    if not heatmap_dir.exists():
        return []
    files = sorted(heatmap_dir.glob("iter*_*.png"))
    if not files:
        return []
    if iteration is None:
        # pick latest iteration from filenames
        def iter_num(p: Path) -> int:
            name = p.stem
            if name.startswith("iter"):
                try:
                    return int(name.split("_")[0].replace("iter", ""))
                except ValueError:
                    return -1
            return -1
        latest = max(iter_num(p) for p in files)
        iteration = latest if latest >= 0 else None
    if iteration is None:
        return []
    return sorted([p for p in files if p.stem.startswith(f"iter{iteration}_")])


def tile_images(paths: List[Path], cols: int = 2) -> np.ndarray | None:
    if not paths:
        return None
    images = [mpimg.imread(str(p)) for p in paths]
    h = min(img.shape[0] for img in images)
    w = min(img.shape[1] for img in images)
    images = [img[:h, :w] for img in images]
    rows = int(np.ceil(len(images) / cols))
    pad = 4
    bg = np.ones((h * rows + pad * (rows - 1), w * cols + pad * (cols - 1), 4), dtype=images[0].dtype)
    bg[:] = 1.0
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        bg[y0 : y0 + h, x0 : x0 + w] = img
    return bg

def keep_latest_run(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return rows
    iters = [int(r["iteration"]) for r in rows if "iteration" in r]
    start_idx = 0
    for i in range(1, len(iters)):
        if iters[i] < iters[i - 1]:
            start_idx = i
    return rows[start_idx:]


def main() -> None:
    args = parse_args()
    rows = keep_latest_run(read_csv(args.metrics_csv))
    if not rows:
        raise SystemExit("No AIRL metrics rows found")

    iters = np.array([int(r["iteration"]) for r in rows], dtype=np.int64)

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

    if args.alpha_json and Path(args.alpha_json).exists():
        with open(args.alpha_json, "r", encoding="utf-8") as f:
            alpha = json.load(f)
        fig.text(
            0.01,
            0.965,
            f"alpha_suggested={alpha.get('alpha_suggested', 'n/a'):.4f} "
            f"(mean_abs={alpha.get('r_airl_mean_abs', 'n/a'):.4f})",
            fontsize=9,
        )

    def style(ax):
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("Iteration")

    # Panel A: Discriminator (dual-axis)
    ax_a = fig.add_subplot(grid[0, 0])
    disc_loss = series(rows, "disc_loss")
    disc_acc = series(rows, "disc_acc")
    line1 = ax_a.plot(iters, disc_loss, label="discriminator_loss", color="#2b6cb0")
    ax_a.set_ylabel("Loss")
    ax_a_t = ax_a.twinx()
    line2 = ax_a_t.plot(iters, disc_acc, label="discriminator_accuracy", color="#2f855a")
    ax_a_t.set_ylabel("Accuracy")
    ax_a.set_title("Discriminator")
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_a.legend(lines, labels, fontsize=8, loc="upper left")
    style(ax_a)

    # Panel B: g(s,a) stats
    ax_b = fig.add_subplot(grid[0, 1])
    ax_b.plot(iters, series(rows, "g_expert_mean"), label="g_expert_mean", color="#6b46c1")
    ax_b.plot(iters, series(rows, "g_policy_mean"), label="g_policy_mean", color="#c05621")
    ax_b.set_title("Task Reward Head g(s,a)")
    ax_b.legend(fontsize=8)
    style(ax_b)

    # Panel C: g gap + std
    ax_c = fig.add_subplot(grid[0, 2])
    ax_c.plot(iters, series(rows, "g_gap"), label="g_gap", color="#dd6b20")
    ax_c.plot(iters, series(rows, "g_expert_std"), label="g_expert_std", color="#2c7a7b")
    ax_c.plot(iters, series(rows, "g_policy_std"), label="g_policy_std", color="#805ad5")
    ax_c.set_title("g Gap / Variability")
    ax_c.legend(fontsize=8)
    style(ax_c)

    # Panel D: h(s) stats (expert vs policy)
    ax_d = fig.add_subplot(grid[1, 0])
    ax_d.plot(iters, series(rows, "h_expert_mean"), label="h_expert_mean", color="#1f77b4")
    ax_d.plot(iters, series(rows, "h_policy_mean"), label="h_policy_mean", color="#ff7f0e")
    ax_d.plot(iters, series(rows, "h_expert_std"), label="h_expert_std", color="#2ca02c")
    ax_d.plot(iters, series(rows, "h_policy_std"), label="h_policy_std", color="#d62728")
    ax_d.set_title("Potential Head h(s)")
    lines = ax_d.get_lines()
    labels = [l.get_label() for l in lines]
    ax_d.legend(lines, labels, fontsize=8, loc="upper left")
    style(ax_d)

    # Panel E: policy entropy
    ax_e = fig.add_subplot(grid[1, 1])
    ax_e.plot(iters, series(rows, "policy_entropy"), color="#38a169")
    ax_e.set_title("Policy Entropy")
    style(ax_e)

    # Panel F: heatmap preview
    ax_f = fig.add_subplot(grid[1:, 2])
    heat_img = None
    if args.heatmap_dir:
        heatmaps = select_heatmaps(Path(args.heatmap_dir), args.iteration)
        heat_img = tile_images(heatmaps, cols=2)
    if heat_img is None and args.heatmap and Path(args.heatmap).exists():
        heat_img = mpimg.imread(args.heatmap)
    if heat_img is not None:
        ax_f.imshow(heat_img)
        ax_f.axis("off")
        ax_f.set_title("h(s) Heatmaps")
    else:
        ax_f.axis("off")
        ax_f.text(0.5, 0.5, "No heatmaps found", ha="center", va="center", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "airl_dashboard.png"
    fig_pdf = out_dir / "airl_dashboard.pdf"
    fig.savefig(fig_path, dpi=200)
    fig.savefig(fig_pdf, dpi=300)
    plt.close(fig)
    print(f"[analyze_airl_metrics] wrote {fig_path} and {fig_pdf}")


if __name__ == "__main__":
    main()
