#!/usr/bin/env python3
"""Render representation diagnostics dashboard."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze representation metrics")
    parser.add_argument("--rep-csv", required=True, help="Path to rep_metrics.csv")
    parser.add_argument("--rep-dir", required=True, help="Directory with PCA npz files")
    parser.add_argument("--title", default="Representation Diagnostics", help="Dashboard title")
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_by_epoch(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    buckets: Dict[int, Dict[str, List[float]]] = {}
    for row in rows:
        if "epoch" not in row:
            continue
        try:
            epoch = int(row["epoch"])
        except (TypeError, ValueError):
            continue
        if epoch not in buckets:
            buckets[epoch] = {}
        for k, v in row.items():
            if k == "epoch":
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            buckets[epoch].setdefault(k, []).append(val)

    out: List[Dict[str, str]] = []
    for epoch in sorted(buckets.keys()):
        row = {"epoch": str(epoch)}
        for k, vals in buckets[epoch].items():
            row[k] = str(float(np.nanmean(vals)))
        out.append(row)
    return out


def to_series(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(r.get(key, "nan")) for r in rows], dtype=np.float64)


def plot_series(ax, x: np.ndarray, y: np.ndarray, label: str, color: str) -> list:
    lines = []
    if len(x) == 0:
        return lines
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    mask = ~np.isnan(y_sorted)
    x_sorted = x_sorted[mask]
    y_sorted = y_sorted[mask]
    if len(x_sorted) <= 2:
        line = ax.plot(x_sorted, y_sorted, label=label, color=color, marker="o", linestyle="-")
        lines += line
        return lines
    start = 0
    for i in range(1, len(x_sorted)):
        if x_sorted[i] - x_sorted[i - 1] > 2:
            line = ax.plot(
                x_sorted[start:i],
                y_sorted[start:i],
                label=label if start == 0 else None,
                color=color,
                marker="o",
                markersize=3,
            )
            lines += line
            start = i
    line = ax.plot(
        x_sorted[start:],
        y_sorted[start:],
        label=label if start == 0 else None,
        color=color,
        marker="o",
        markersize=3,
    )
    lines += line
    return lines


def set_ylim_from_data(ax, y: np.ndarray, pad: float = 0.1, min_pad: float = 1e-3) -> None:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if y_min == y_max:
        delta = max(abs(y_min) * 0.05, min_pad)
    else:
        delta = (y_max - y_min) * pad
    ax.set_ylim(y_min - delta, y_max + delta)


def main() -> None:
    args = parse_args()
    rows = aggregate_by_epoch(read_csv(args.rep_csv))
    if not rows:
        raise SystemExit("No representation rows found")

    epochs = np.array([int(r["epoch"]) for r in rows], dtype=np.int64)
    out_dir = Path(args.rep_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(args.title, fontsize=14, y=0.98)
    grid = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)

    def style(ax):
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("Epoch")

    # Panel A: Rank metrics (dual-axis)
    ax_a = fig.add_subplot(grid[0, 0])
    lines = []
    labels = []
    eff_rank = to_series(rows, "rep_effective_rank")
    l1 = plot_series(ax_a, epochs, eff_rank, "effective_rank", "#2b6cb0")
    lines += l1
    labels += [l.get_label() for l in l1]
    ax_a.set_ylabel("Effective Rank")
    ax_a_t = ax_a.twinx()
    top_ratio = to_series(rows, "rep_top_eig_ratio")
    l2 = plot_series(ax_a_t, epochs, top_ratio, "top_eig_ratio", "#c05621")
    lines += l2
    labels += [l.get_label() for l in l2]
    ax_a_t.set_ylabel("Top Eig Ratio")
    ax_a.set_title("Rank / Spectrum")
    lines_labels = [(l, lab) for l, lab in zip(lines, labels) if not lab.startswith("_")]
    if lines_labels:
        ax_a.legend([l for l, _ in lines_labels], [lab for _, lab in lines_labels], fontsize=8, loc="upper left")
    set_ylim_from_data(ax_a, eff_rank)
    set_ylim_from_data(ax_a_t, top_ratio)
    style(ax_a)

    # Panel B: Feature magnitudes
    ax_b = fig.add_subplot(grid[0, 1])
    feat_std = to_series(rows, "rep_feat_std_mean")
    plot_series(ax_b, epochs, feat_std, "feat_std_mean", "#4a5568")
    ax_b.set_title("Feature Scale")
    ax_b.legend(fontsize=8)
    set_ylim_from_data(ax_b, feat_std, min_pad=1e-4)
    style(ax_b)

    # Panel C: Redundancy
    ax_c = fig.add_subplot(grid[0, 2])
    offdiag = to_series(rows, "rep_mean_offdiag_corr")
    plot_series(ax_c, epochs, offdiag, "mean_offdiag_corr", "#805ad5")
    ax_c.set_title("Redundancy")
    ax_c.legend(fontsize=8)
    set_ylim_from_data(ax_c, offdiag, min_pad=1e-4)
    style(ax_c)

    # Panel D: Temporal smoothness
    ax_d = fig.add_subplot(grid[1, 0])
    temporal = to_series(rows, "rep_temporal_cos")
    plot_series(ax_d, epochs, temporal, "temporal_cos", "#2f855a")
    ax_d.set_title("Temporal Smoothness")
    ax_d.legend(fontsize=8)
    set_ylim_from_data(ax_d, temporal, min_pad=1e-4)
    style(ax_d)

    # Panel E: PCA scatter
    ax_e = fig.add_subplot(grid[1, 1:])
    latest_epoch = epochs[-1]
    pca_path = out_dir / f"rep_pca_epoch_{latest_epoch}.npz"
    if pca_path.exists():
        data = np.load(pca_path)
        coords = data["coords"]
        labels = data["labels"]
        ax_e.scatter(coords[:, 0], coords[:, 1], c=labels, s=10, cmap="coolwarm", alpha=0.8)
        ax_e.set_title(f"PCA (epoch {latest_epoch})")
    else:
        ax_e.axis("off")
        ax_e.text(0.5, 0.5, "No PCA snapshot found", ha="center", va="center", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / "rep_dashboard.png"
    out_pdf = out_dir / "rep_dashboard.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[analyze_representation] wrote {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
