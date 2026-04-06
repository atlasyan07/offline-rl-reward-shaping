#!/usr/bin/env python3
"""Render LaTeX-style dashboard for epoch-level training metrics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze epoch-level training metrics")
    parser.add_argument("--train-csv", required=True, help="Path to train_metrics.csv")
    parser.add_argument("--eval-csv", default=None, help="Path to eval_metrics.csv (optional)")
    parser.add_argument("--out-dir", required=True, help="Output directory for dashboard")
    parser.add_argument("--title", default="Training Diagnostics (Epoch-Level)", help="Dashboard title")
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def keep_latest_run(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return rows
    epochs = [_safe_int(r.get("epoch")) for r in rows]
    valid_pairs = [(i, e) for i, e in enumerate(epochs) if e is not None]
    if not valid_pairs:
        return []
    indices, epochs_clean = zip(*valid_pairs)
    start_idx = 0
    for i in range(1, len(epochs_clean)):
        if epochs_clean[i] < epochs_clean[i - 1]:
            start_idx = indices[i]
    return rows[start_idx:]


def to_series(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    def parse(val: str) -> float:
        if val is None:
            return float("nan")
        val = str(val).strip()
        if not val:
            return float("nan")
        try:
            return float(val)
        except ValueError:
            return float("nan")

    return np.array([parse(r.get(key)) for r in rows], dtype=np.float64)


def series_if_present(rows: List[Dict[str, str]], key: str) -> np.ndarray | None:
    if not rows or key not in rows[0]:
        return None
    series = to_series(rows, key)
    if np.all(np.isnan(series)):
        return None
    return series


def plot_series(ax, x: np.ndarray, y: np.ndarray, label: str, color: str) -> list:
    """Plot with gaps handled; if very few points, use markers."""
    lines = []
    if len(x) == 0:
        return lines
    # Sort by x and drop NaNs
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    mask = ~np.isnan(y_sorted)
    x_sorted = x_sorted[mask]
    y_sorted = y_sorted[mask]
    if len(x_sorted) <= 2:
        line = ax.plot(x_sorted, y_sorted, label=label, color=color, marker="o")
        lines += line
        return lines
    # Split on gaps to avoid connecting unrelated segments
    start = 0
    for i in range(1, len(x_sorted)):
        if x_sorted[i] - x_sorted[i - 1] > 1:
            line = ax.plot(x_sorted[start:i], y_sorted[start:i], label=label if start == 0 else None, color=color)
            lines += line
            start = i
    line = ax.plot(x_sorted[start:], y_sorted[start:], label=label if start == 0 else None, color=color)
    lines += line
    return lines


def main() -> None:
    args = parse_args()
    train_rows = keep_latest_run(read_csv(args.train_csv))
    if not train_rows:
        raise SystemExit("No training rows found")

    eval_rows = keep_latest_run(read_csv(args.eval_csv)) if args.eval_csv and Path(args.eval_csv).exists() else []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs_clean = []
    train_rows_clean = []
    for row in train_rows:
        epoch_val = _safe_int(row.get("epoch"))
        if epoch_val is None:
            continue
        epochs_clean.append(epoch_val)
        train_rows_clean.append(row)
    if not train_rows_clean:
        raise SystemExit("No valid epoch rows found in train CSV")
    train_rows = train_rows_clean
    epochs = np.array(epochs_clean, dtype=np.int64)

    # Styling (LaTeX-like without requiring TeX)
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

    fig = plt.figure(figsize=(15, 18))
    fig.suptitle(args.title, fontsize=14, y=0.98)
    grid = fig.add_gridspec(6, 3, hspace=0.5, wspace=0.35)

    def style(ax):
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_xlabel("Epoch")

    # Panel A: Summary stats
    ax_a = fig.add_subplot(grid[0, 0])
    ax_a.axis("off")
    summary_lines = [
        f"Epochs: {epochs[-1]}",
        f"v_loss (last): {to_series(train_rows, 'v_loss')[-1]:.4f}",
        f"q_loss (last): {to_series(train_rows, 'q_loss')[-1]:.4f}",
        f"q_var (last): {to_series(train_rows, 'q_var')[-1]:.4f}",
        f"adv_abs_mean (last): {to_series(train_rows, 'adv_abs_mean')[-1]:.4f}",
    ]
    reward_alpha = series_if_present(train_rows, "reward_alpha")
    if reward_alpha is not None:
        summary_lines.append(f"reward_alpha (last): {reward_alpha[-1]:.4f}")
    summary_text = "\n".join(summary_lines)
    ax_a.text(0.02, 0.98, "Run Summary", fontsize=12, fontweight="bold", va="top")
    ax_a.text(0.02, 0.78, summary_text, fontsize=10, va="top")

    # Panel B: Value losses (dual-axis)
    ax_b = fig.add_subplot(grid[0, 1])
    v_loss = series_if_present(train_rows, "v_loss")
    q_loss = series_if_present(train_rows, "q_loss")
    lines = []
    labels = []
    if v_loss is not None:
        l1 = plot_series(ax_b, epochs, v_loss, "v_loss", "#2b6cb0")
        lines += l1
        labels += [l.get_label() for l in l1]
        ax_b.set_ylabel("V Loss")
    ax_b_t = ax_b.twinx()
    if q_loss is not None:
        l2 = plot_series(ax_b_t, epochs, q_loss, "q_loss", "#2f855a")
        lines += l2
        labels += [l.get_label() for l in l2]
        ax_b_t.set_ylabel("Q Loss")
    ax_b.set_title("Losses (V / Q)")
    if lines:
        ax_b.legend(lines, labels, fontsize=8, loc="upper left")
    style(ax_b)

    # Panel C: Q/V means
    ax_c = fig.add_subplot(grid[0, 2])
    v_mean = series_if_present(train_rows, "v_mean")
    q_mean = series_if_present(train_rows, "q_mean")
    if v_mean is not None:
        plot_series(ax_c, epochs, v_mean, "v_mean", "#4a5568")
    if q_mean is not None:
        plot_series(ax_c, epochs, q_mean, "q_mean", "#6b46c1")
    ax_c.set_title("Value Means")
    ax_c.legend(fontsize=8)
    style(ax_c)

    # Panel D: Q variance
    ax_d = fig.add_subplot(grid[1, 0])
    q_var = series_if_present(train_rows, "q_var")
    if q_var is not None:
        plot_series(ax_d, epochs, q_var, "q_var", "#dd6b20")
    ax_d.set_title("Intra-state Q Variance")
    style(ax_d)

    # Panel E: Advantage diagnostics (dual-axis)
    ax_e = fig.add_subplot(grid[1, 1])
    adv_abs_mean = series_if_present(train_rows, "adv_abs_mean")
    adv_mean = series_if_present(train_rows, "adv_mean")
    lines = []
    labels = []
    if adv_abs_mean is not None:
        l1 = plot_series(ax_e, epochs, adv_abs_mean, "adv_abs_mean", "#2c7a7b")
        lines += l1
        labels += [l.get_label() for l in l1]
        ax_e.set_ylabel("Adv Abs Mean")
    ax_e_t = ax_e.twinx()
    if adv_mean is not None:
        l2 = plot_series(ax_e_t, epochs, adv_mean, "adv_mean", "#805ad5")
        lines += l2
        labels += [l.get_label() for l in l2]
        ax_e_t.set_ylabel("Adv Mean")
    ax_e.set_title("Advantage Diagnostics")
    if lines:
        ax_e.legend(lines, labels, fontsize=8, loc="upper left")
    style(ax_e)

    # Panel F: Actor loss
    ax_f = fig.add_subplot(grid[1, 2])
    actor_loss = series_if_present(train_rows, "actor_loss")
    if actor_loss is not None:
        plot_series(ax_f, epochs, actor_loss, "actor_loss", "#c05621")
    ax_f.set_title("Actor Loss")
    style(ax_f)

    # Panel G: TD error diagnostics
    ax_g = fig.add_subplot(grid[2, 0])
    td_abs_mean = series_if_present(train_rows, "td_abs_mean")
    td_abs_p90 = series_if_present(train_rows, "td_abs_p90")
    if td_abs_mean is not None:
        plot_series(ax_g, epochs, td_abs_mean, "td_abs_mean", "#3182ce")
    if td_abs_p90 is not None:
        plot_series(ax_g, epochs, td_abs_p90, "td_abs_p90", "#63b3ed")
    ax_g.set_title("TD Error (Abs)")
    ax_g.legend(fontsize=8)
    style(ax_g)

    # Panel H: Policy entropy / action diversity (dual-axis)
    ax_h = fig.add_subplot(grid[2, 1])
    policy_entropy = series_if_present(train_rows, "policy_entropy")
    action_diversity = series_if_present(train_rows, "action_diversity")
    lines = []
    labels = []
    if policy_entropy is not None:
        l1 = plot_series(ax_h, epochs, policy_entropy, "policy_entropy", "#805ad5")
        ax_h.set_ylabel("Entropy")
        lines += l1
        labels += [l.get_label() for l in l1]
    ax_h_t = ax_h.twinx()
    if action_diversity is not None:
        l2 = plot_series(ax_h_t, epochs, action_diversity, "action_diversity", "#38a169")
        ax_h_t.set_ylabel("Diversity")
        lines += l2
        labels += [l.get_label() for l in l2]
    ax_h.set_title("Policy Entropy / Action Diversity")
    if lines:
        ax_h.legend(lines, labels, fontsize=8, loc="upper left")
    style(ax_h)

    # Panel I: Advantage spread + positive mass (dual-axis)
    ax_i = fig.add_subplot(grid[2, 2])
    adv_spread = series_if_present(train_rows, "adv_spread")
    pos_adv_mass = series_if_present(train_rows, "pos_adv_mass")
    lines = []
    labels = []
    if adv_spread is not None:
        l1 = plot_series(ax_i, epochs, adv_spread, "adv_spread", "#dd6b20")
        ax_i.set_ylabel("Spread")
        lines += l1
        labels += [l.get_label() for l in l1]
    ax_i_t = ax_i.twinx()
    if pos_adv_mass is not None:
        l2 = plot_series(ax_i_t, epochs, pos_adv_mass, "pos_adv_mass", "#c05621")
        ax_i_t.set_ylabel("Positive Mass")
        lines += l2
        labels += [l.get_label() for l in l2]
    ax_i.set_title("Advantage Spread / Positive Mass")
    if lines:
        ax_i.legend(lines, labels, fontsize=8, loc="upper left")
    style(ax_i)

    # Panel J: DR3 rank/top eigenvalue + feature std (dual-axis)
    ax_j = fig.add_subplot(grid[3, 0])
    dr3_rank = series_if_present(train_rows, "dr3_rank")
    dr3_top_eig = series_if_present(train_rows, "dr3_top_eigenvalue")
    dr3_feat_std = series_if_present(train_rows, "dr3_feature_std")
    if any(x is not None for x in [dr3_rank, dr3_top_eig, dr3_feat_std]):
        lines = []
        labels = []
        if dr3_rank is not None:
            l1 = plot_series(ax_j, epochs, dr3_rank, "dr3_rank", "#805ad5")
            lines += l1
            labels += [l.get_label() for l in l1]
            ax_j.set_ylabel("Rank / Top Eig")
        if dr3_top_eig is not None:
            l2 = plot_series(ax_j, epochs, dr3_top_eig, "dr3_top_eigenvalue", "#2f855a")
            lines += l2
            labels += [l.get_label() for l in l2]
        ax_j_t = ax_j.twinx()
        if dr3_feat_std is not None:
            l3 = plot_series(ax_j_t, epochs, dr3_feat_std, "dr3_feature_std", "#4a5568")
            lines += l3
            labels += [l.get_label() for l in l3]
            ax_j_t.set_ylabel("Feature Std")
        ax_j.set_title("DR3 Rank / Top Eig")
        if lines:
            ax_j.legend(lines, labels, fontsize=8, loc="upper left")
        style(ax_j)
    else:
        ax_j.axis("off")
        ax_j.text(0.5, 0.5, "No DR3 metrics found", ha="center", va="center", fontsize=10)

    # Panel K: DR3 eigenvalue ratio + loss (dual-axis)
    ax_k = fig.add_subplot(grid[3, 1])
    dr3_eig_ratio = series_if_present(train_rows, "dr3_eigenvalue_ratio")
    dr3_loss = series_if_present(train_rows, "dr3_loss")
    lines = []
    labels = []
    if dr3_eig_ratio is not None:
        l1 = plot_series(ax_k, epochs, dr3_eig_ratio, "dr3_eigenvalue_ratio", "#c05621")
        lines += l1
        labels += [l.get_label() for l in l1]
        ax_k.set_ylabel("Eig Ratio")
    ax_k_t = ax_k.twinx()
    if dr3_loss is not None:
        l2 = plot_series(ax_k_t, epochs, dr3_loss, "dr3_loss", "#2b6cb0")
        lines += l2
        labels += [l.get_label() for l in l2]
        ax_k_t.set_ylabel("DR3 Loss")
    ax_k.set_title("DR3 Eigenvalue Ratio / Loss")
    if lines:
        ax_k.legend(lines, labels, fontsize=8, loc="upper left")
        style(ax_k)
    else:
        ax_k.axis("off")
        ax_k.text(0.5, 0.5, "No DR3 eigenvalue ratio", ha="center", va="center", fontsize=10)

    # Panel L: Eval success (if available)
    ax_l = fig.add_subplot(grid[3, 2])
    if eval_rows:
        eval_epochs = np.array([int(r["epoch"]) for r in eval_rows], dtype=np.int64)
        for key in eval_rows[0].keys():
            if key == "epoch":
                continue
            series = np.array([float(r[key]) for r in eval_rows], dtype=np.float64)
            plot_series(ax_l, eval_epochs, series, key, None if None else None)
        ax_l.set_title("Eval Success Rates")
        ax_l.set_ylim(0.0, 1.0)
        ax_l.legend(fontsize=8)
        style(ax_l)
    else:
        ax_l.axis("off")
        ax_l.text(0.5, 0.5, "No eval_metrics.csv found", ha="center", va="center", fontsize=10)

    # Panel M: Reward scaling (dual-axis)
    ax_m = fig.add_subplot(grid[4, 0])
    scale_ratio_mean = series_if_present(train_rows, "reward_scale_ratio_mean")
    scale_ratio_median = series_if_present(train_rows, "reward_scale_ratio_median")
    reward_variance_ratio = series_if_present(train_rows, "reward_variance_ratio")
    lines = []
    labels = []
    if scale_ratio_mean is not None:
        l1 = plot_series(ax_m, epochs, scale_ratio_mean, "scale_ratio_mean", "#2b6cb0")
        lines += l1
        labels += [l.get_label() for l in l1]
        ax_m.set_ylabel("Scale Ratio")
    if scale_ratio_median is not None:
        l2 = plot_series(ax_m, epochs, scale_ratio_median, "scale_ratio_median", "#63b3ed")
        lines += l2
        labels += [l.get_label() for l in l2]
    ax_m_t = ax_m.twinx()
    if reward_variance_ratio is not None:
        l3 = plot_series(ax_m_t, epochs, reward_variance_ratio, "variance_ratio", "#4a5568")
        lines += l3
        labels += [l.get_label() for l in l3]
        ax_m_t.set_ylabel("Variance Ratio")
    ax_m.set_title("Reward Scaling")
    if lines:
        ax_m.legend(lines, labels, fontsize=8, loc="upper left")
        style(ax_m)
    else:
        ax_m.axis("off")
        ax_m.text(0.5, 0.5, "No reward scaling metrics", ha="center", va="center", fontsize=10)

    # Panel N: Reward dominance / agreement (dual-axis)
    ax_n = fig.add_subplot(grid[4, 1])
    dominance_rate = series_if_present(train_rows, "reward_dominance_rate")
    reward_corr = series_if_present(train_rows, "reward_corr_near_terminal")
    sign_agree = series_if_present(train_rows, "reward_sign_agree_terminal")
    lines = []
    labels = []
    if dominance_rate is not None:
        l1 = plot_series(ax_n, epochs, dominance_rate, "dominance_rate", "#c05621")
        lines += l1
        labels += [l.get_label() for l in l1]
        ax_n.set_ylabel("Dominance")
    ax_n_t = ax_n.twinx()
    if reward_corr is not None:
        l2 = plot_series(ax_n_t, epochs, reward_corr, "corr_near_terminal", "#2f855a")
        lines += l2
        labels += [l.get_label() for l in l2]
    if sign_agree is not None:
        l3 = plot_series(ax_n_t, epochs, sign_agree, "sign_agree_terminal", "#805ad5")
        lines += l3
        labels += [l.get_label() for l in l3]
    if reward_corr is not None or sign_agree is not None:
        ax_n_t.set_ylabel("Correlation / Sign")
    ax_n.set_title("Reward Dominance / Agreement")
    if lines:
        ax_n.legend(lines, labels, fontsize=8, loc="upper left")
        style(ax_n)
    else:
        ax_n.axis("off")
        ax_n.text(0.5, 0.5, "No reward dominance metrics", ha="center", va="center", fontsize=10)

    # Panel O: Policy impact (dual-axis)
    ax_o = fig.add_subplot(grid[4, 2])
    adv_shift_abs = series_if_present(train_rows, "adv_shift_abs")
    adv_shift_mean = series_if_present(train_rows, "adv_shift_mean")
    action_kl_mean = series_if_present(train_rows, "action_kl_mean")
    lines = []
    labels = []
    if adv_shift_abs is not None:
        l1 = plot_series(ax_o, epochs, adv_shift_abs, "adv_shift_abs", "#dd6b20")
        lines += l1
        labels += [l.get_label() for l in l1]
        ax_o.set_ylabel("Adv Shift")
    if adv_shift_mean is not None:
        l2 = plot_series(ax_o, epochs, adv_shift_mean, "adv_shift_mean", "#c05621")
        lines += l2
        labels += [l.get_label() for l in l2]
    ax_o_t = ax_o.twinx()
    if action_kl_mean is not None:
        l3 = plot_series(ax_o_t, epochs, action_kl_mean, "action_kl_mean", "#2f855a")
        lines += l3
        labels += [l.get_label() for l in l3]
        ax_o_t.set_ylabel("Action KL")
    ax_o.set_title("Policy Impact")
    if lines:
        ax_o.legend(lines, labels, fontsize=8, loc="upper left")
        style(ax_o)
    else:
        ax_o.axis("off")
        ax_o.text(0.5, 0.5, "No policy impact metrics", ha="center", va="center", fontsize=10)

    # Panel P: Reward totals (dual-axis)
    ax_p = fig.add_subplot(grid[5, 0])
    reward_env_sum = series_if_present(train_rows, "reward_env_sum_total")
    if reward_env_sum is None:
        reward_env_sum = series_if_present(train_rows, "reward_env_sum")
    reward_airl_sum = series_if_present(train_rows, "reward_airl_sum_total")
    if reward_airl_sum is None:
        reward_airl_sum = series_if_present(train_rows, "reward_airl_sum")
    reward_sum_ratio = series_if_present(train_rows, "reward_sum_ratio_total")
    if reward_sum_ratio is None:
        reward_sum_ratio = series_if_present(train_rows, "reward_sum_ratio")
    lines = []
    labels = []
    if reward_env_sum is not None:
        l1 = plot_series(ax_p, epochs, reward_env_sum, "env_sum", "#2b6cb0")
        lines += l1
        labels += [l.get_label() for l in l1]
        ax_p.set_ylabel("Env Sum")
    if reward_airl_sum is not None:
        l2 = plot_series(ax_p, epochs, reward_airl_sum, "airl_sum", "#c05621")
        lines += l2
        labels += [l.get_label() for l in l2]
    ax_p_t = ax_p.twinx()
    if reward_sum_ratio is not None:
        l3 = plot_series(ax_p_t, epochs, reward_sum_ratio, "sum_ratio", "#4a5568")
        lines += l3
        labels += [l.get_label() for l in l3]
        ax_p_t.set_ylabel("AIRL/Env")
    ax_p.set_title("Reward Totals")
    if lines:
        ax_p.legend(lines, labels, fontsize=8, loc="upper left")
        style(ax_p)
    else:
        ax_p.axis("off")
        ax_p.text(0.5, 0.5, "No reward totals", ha="center", va="center", fontsize=10)

    # Panel Q: Terminal margin
    ax_q = fig.add_subplot(grid[5, 1])
    terminal_margin = series_if_present(train_rows, "terminal_margin")
    if terminal_margin is not None:
        plot_series(ax_q, epochs, terminal_margin, "terminal_margin", "#4a5568")
        ax_q.set_title("Terminal Margin")
        style(ax_q)
    else:
        ax_q.axis("off")
        ax_q.text(0.5, 0.5, "No terminal margin", ha="center", va="center", fontsize=10)

    # Panel R: RND loss
    ax_r = fig.add_subplot(grid[5, 2])
    rnd_loss = series_if_present(train_rows, "rnd_loss")
    if rnd_loss is not None:
        plot_series(ax_r, epochs, rnd_loss, "rnd_loss", "#2b6cb0")
        ax_r.set_title("RND Loss")
        style(ax_r)
    else:
        ax_r.axis("off")
        ax_r.text(0.5, 0.5, "No RND loss", ha="center", va="center", fontsize=10, color="#718096")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = out_dir / "training_dashboard.png"
    fig_pdf = out_dir / "training_dashboard.pdf"
    fig.savefig(fig_path, dpi=200)
    fig.savefig(fig_pdf, dpi=300)
    plt.close(fig)
    print(f"[analyze_training_metrics] wrote {fig_path} and {fig_pdf}")


if __name__ == "__main__":
    main()
