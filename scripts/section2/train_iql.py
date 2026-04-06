#!/usr/bin/env python3
"""Train IQL baseline on fixed offline dataset.

This script implements Section 2 Part 1:
- Train IQL on sparse environment reward only
- Expect it to struggle (motivates reward shaping in later sections)
- Track representation collapse metrics as supporting evidence
"""
from __future__ import annotations

# Suppress ALSA audio warnings on headless systems (must be before pygame import)
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
import runpy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.iql import IQL, DR3Loss
from src.airl import AIRLReward
from src.envs import make_env
from src.utils import load_config
from src.viz import save_video


def run_eval_rollouts(
    agent: IQL,
    env_cfg: dict,
    layouts: list[str],
    num_episodes: int,
    max_steps: int,
    fps: int,
    writer: SummaryWriter,
    global_step: int,
    video_dir: str | None = None,
) -> dict:
    agent.eval()
    eval_summary = {}
    for layout_name in layouts:
        env = make_env(env_cfg, render_mode="rgb_array", layout_name=layout_name)
        successes = []
        for ep in range(num_episodes):
            obs, info = env.reset(seed=10000 + ep)
            frames = []
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = False
            steps = 0
            episode_return = 0.0
            while not done and steps < max_steps:
                grid = obs["image"]
                agent_pos = np.array([env.agent_pos[0], env.agent_pos[1]])
                agent_dir = env.agent_dir
                carry = np.array([0, 0])
                action = agent.get_action(grid, agent_pos, agent_dir, carry, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_return += reward
                steps += 1
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            successes.append(episode_return > 0)

            # Log only the first episode video per layout
            if ep == 0 and frames:
                # Ensure at least 2 frames so TensorBoard logs as video
                if len(frames) == 1:
                    frames = frames + frames
                video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)
                writer.add_video(
                    f"eval/{layout_name}",
                    video,
                    fps=fps,
                    global_step=global_step,
                )
                # Also save as MP4 file
                if video_dir:
                    os.makedirs(video_dir, exist_ok=True)
                    mp4_path = os.path.join(video_dir, f"epoch{global_step}_{layout_name}.mp4")
                    save_video(frames, mp4_path, fps)
                    print(f"  Saved video: {mp4_path}")

        success_rate = float(np.mean(successes)) if successes else 0.0
        writer.add_scalar(
            f"eval/{layout_name}_success_rate",
            success_rate,
            global_step,
            new_style=False,
        )
        eval_summary[layout_name] = success_rate
    agent.train()
    return eval_summary


def append_csv_row(path: str, row: dict) -> None:
    file_exists = os.path.exists(path)
    if not file_exists:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        existing_rows = [{k: v for k, v in r.items() if k is not None} for r in reader]

    new_fields = [k for k in row.keys() if k not in fieldnames]
    if new_fields:
        fieldnames.extend(new_fields)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({k: existing.get(k, "") for k in fieldnames})
            writer.writerow(row)
        os.replace(tmp_path, path)
        return

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_steps_to_terminal(episode_id: np.ndarray) -> np.ndarray:
    steps_to_terminal = np.empty_like(episode_id, dtype=np.int32)
    if episode_id.size == 0:
        return steps_to_terminal
    start = 0
    current = episode_id[0]
    for idx in range(1, episode_id.size):
        if episode_id[idx] != current:
            length = idx - start
            steps_to_terminal[start:idx] = np.arange(length - 1, -1, -1, dtype=np.int32)
            start = idx
            current = episode_id[idx]
    length = episode_id.size - start
    steps_to_terminal[start:] = np.arange(length - 1, -1, -1, dtype=np.int32)
    return steps_to_terminal


def load_dataset(dataset_path: str, metadata_path: str):
    """Load offline dataset from NPZ and metadata.

    Returns:
        dataset: TensorDataset
        metadata: dict
        data_stats: dict with normalization stats
    """
    print(f"Loading dataset from {dataset_path}")
    data = np.load(dataset_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Convert to tensors
    s_grid = torch.from_numpy(data["s_grid"])
    s_agent_pos = torch.from_numpy(data["s_agent_pos"])
    s_agent_dir = torch.from_numpy(data["s_agent_dir"])
    s_carry = torch.from_numpy(data["s_carry"])
    a = torch.from_numpy(data["a"])
    r = torch.from_numpy(data["r"])
    sp_grid = torch.from_numpy(data["sp_grid"])
    sp_agent_pos = torch.from_numpy(data["sp_agent_pos"])
    sp_agent_dir = torch.from_numpy(data["sp_agent_dir"])
    sp_carry = torch.from_numpy(data["sp_carry"])
    done = torch.from_numpy(data["done"])
    steps_to_terminal = None
    if "episode_id" in data:
        steps_to_terminal = torch.from_numpy(_compute_steps_to_terminal(data["episode_id"]))

    print(f"Dataset loaded:")
    print(f"  - Transitions: {len(a):,}")
    print(f"  - Episodes: {len(metadata['episodes']):,}")
    print(f"  - Grid shape: {s_grid.shape}")
    print(f"  - Reward stats: mean={r.mean():.4f}, std={r.std():.4f}, max={r.max():.4f}")

    # Create dataset
    dataset_tensors = [
        s_grid, s_agent_pos, s_agent_dir, s_carry,
        a, r,
        sp_grid, sp_agent_pos, sp_agent_dir, sp_carry,
        done,
    ]
    if steps_to_terminal is not None:
        dataset_tensors.append(steps_to_terminal)
    dataset = TensorDataset(*dataset_tensors)

    data_stats = {
        "num_transitions": len(a),
        "num_episodes": len(metadata["episodes"]),
        "reward_mean": r.mean().item(),
        "reward_std": r.std().item(),
    }

    return dataset, metadata, data_stats


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def compute_reward_alignment_metrics(
    r_env: torch.Tensor,
    r_airl: torch.Tensor,
    done: torch.Tensor,
    steps_to_terminal: torch.Tensor | None,
    near_terminal_steps: int,
) -> dict:
    eps = 1e-3
    r_env_f = r_env.detach().float()
    r_airl_f = r_airl.detach().float()

    if steps_to_terminal is not None:
        near_mask = steps_to_terminal.to(r_env_f.device) <= near_terminal_steps
    else:
        near_mask = done.bool()

    near_env = r_env_f[near_mask]
    near_airl = r_airl_f[near_mask]
    corr = _safe_corrcoef(near_env.cpu().numpy(), near_airl.cpu().numpy()) if near_env.numel() else 0.0

    done_mask = done.bool()
    if done_mask.any():
        sign_agree = (torch.sign(r_env_f[done_mask]) == torch.sign(r_airl_f[done_mask])).float().mean().item()
    else:
        sign_agree = 0.0

    env_abs_mean = r_env_f.abs().mean().item()
    airl_abs_mean = r_airl_f.abs().mean().item()
    airl_abs_median = r_airl_f.abs().median().item()
    denom = max(env_abs_mean, eps)
    scale_ratio_mean = airl_abs_mean / denom
    scale_ratio_median = airl_abs_median / denom
    dominance_rate = (r_airl_f.abs() > r_env_f.abs()).float().mean().item()
    variance_ratio = r_airl_f.std().item() / (r_env_f.std().item() + eps)

    return {
        "reward_corr_near_terminal": corr,
        "reward_sign_agree_terminal": sign_agree,
        "reward_scale_ratio_mean": scale_ratio_mean,
        "reward_scale_ratio_median": scale_ratio_median,
        "reward_dominance_rate": dominance_rate,
        "reward_variance_ratio": variance_ratio,
        "reward_env_mean": r_env_f.mean().item(),
        "reward_airl_mean": r_airl_f.mean().item(),
        "reward_env_sum": r_env_f.sum().item(),
        "reward_airl_sum": r_airl_f.sum().item(),
        "reward_sum_ratio": (r_airl_f.sum().item() / (r_env_f.sum().item() + eps)),
        "reward_terminal_count": int(done_mask.sum().item()),
        "reward_near_terminal_count": int(near_mask.sum().item()),
    }


@torch.no_grad()
def compute_policy_sensitivity_metrics(
    agent: IQL,
    ref_encoder: torch.nn.Module,
    ref_actor: torch.nn.Module,
    batch: dict,
    r_env: torch.Tensor,
    r_airl: torch.Tensor,
) -> dict:
    features = agent.encode_state(
        batch["s_grid"],
        batch["s_agent_pos"],
        batch["s_agent_dir"],
        batch["s_carry"],
    )
    next_features = agent.encode_state(
        batch["sp_grid"],
        batch["sp_agent_pos"],
        batch["sp_agent_dir"],
        batch["sp_carry"],
    )
    next_v = agent.v_target(next_features)
    v = agent.v(features)
    done_f = batch["done"].float()
    adv_airl = (r_airl + agent.discount * (1 - done_f) * next_v) - v
    adv_env = (r_env + agent.discount * (1 - done_f) * next_v) - v
    adv_shift = adv_airl - adv_env

    cur_logits = agent.actor(features)
    cur_logp = F.log_softmax(cur_logits, dim=1)
    cur_p = cur_logp.exp()

    ref_features = ref_encoder(
        batch["s_grid"],
        batch["s_agent_pos"],
        batch["s_agent_dir"],
        batch["s_carry"],
    )
    ref_logits = ref_actor(ref_features)
    ref_logp = F.log_softmax(ref_logits, dim=1)
    kl = (cur_p * (cur_logp - ref_logp)).sum(dim=1)

    return {
        "adv_shift_mean": adv_shift.mean().item(),
        "adv_shift_abs": adv_shift.abs().mean().item(),
        "action_kl_mean": kl.mean().item(),
    }


@torch.no_grad()
def compute_terminal_margin(agent: IQL, batch: dict) -> float:
    features = agent.encode_state(
        batch["s_grid"],
        batch["s_agent_pos"],
        batch["s_agent_dir"],
        batch["s_carry"],
    )
    q1 = agent.q1(features)
    q2 = agent.q2(features)
    q_min = torch.min(q1, q2)
    done_mask = batch["done"].bool()
    non_done_mask = ~done_mask
    if not done_mask.any() or not non_done_mask.any():
        return 0.0
    q_goal = q_min[done_mask].max(dim=1).values
    q_nongoal = q_min[non_done_mask].max(dim=1).values
    return (q_goal.mean() - q_nongoal.mean()).item()


@torch.no_grad()
def compute_representation_metrics(
    agent: IQL,
    batch: dict,
    max_points: int = 512,
) -> dict:
    batch = {k: v.to(agent.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    features = agent.encode_state(
        batch["s_grid"],
        batch["s_agent_pos"],
        batch["s_agent_dir"],
        batch["s_carry"],
    )
    sp_features = agent.encode_state(
        batch["sp_grid"],
        batch["sp_agent_pos"],
        batch["sp_agent_dir"],
        batch["sp_carry"],
    )

    feats = features.detach().float().cpu().numpy()
    if feats.shape[0] > max_points:
        feats = feats[:max_points]

    # Spectrum metrics
    centered = feats - feats.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True) + 1e-8
    normed = centered / std
    cov = (normed.T @ normed) / max(1, normed.shape[0])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    p = eigvals / eigvals.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    top_eig_ratio = float(eigvals[-1] / eigvals.sum())
    feat_std_mean = float(std.mean())

    # Correlation redundancy (mean abs off-diagonal)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(normed, rowvar=False)
    if corr.shape[0] > 1:
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        if np.any(np.isnan(off_diag)):
            mean_off_diag = 0.0
        else:
            mean_off_diag = float(np.mean(np.abs(off_diag)))
    else:
        mean_off_diag = 0.0

    # Temporal smoothness via s -> s'
    s = features.detach().float()
    sp = sp_features.detach().float()
    cos = F.cosine_similarity(s, sp, dim=1).mean().item()

    return {
        "rep_effective_rank": eff_rank / feats.shape[1],
        "rep_top_eig_ratio": top_eig_ratio,
        "rep_feat_std_mean": feat_std_mean,
        "rep_mean_offdiag_corr": mean_off_diag,
        "rep_temporal_cos": cos,
    }


@torch.no_grad()
def compute_representation_pca(
    agent: IQL,
    batch: dict,
    max_points: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    batch = {k: v.to(agent.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    features = agent.encode_state(
        batch["s_grid"],
        batch["s_agent_pos"],
        batch["s_agent_dir"],
        batch["s_carry"],
    ).detach().float().cpu().numpy()
    labels = batch["done"].detach().float().cpu().numpy()
    if features.shape[0] > max_points:
        features = features[:max_points]
        labels = labels[:max_points]
    centered = features - features.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    return coords, labels


def train_iql(
    agent: IQL,
    dataset: TensorDataset,
    config: dict,
    output_dir: str,
    use_dr3: bool = False,
    env_cfg: dict | None = None,
    eval_layouts: list[str] | None = None,
    eval_episodes: int = 1,
    eval_max_steps: int = 200,
    eval_fps: int = 10,
    video_dir: str | None = None,
    dashboard_every: int = 0,
    dashboard_dir: str | None = None,
    dashboard_title: str | None = None,
    reward_model: AIRLReward | None = None,
):
    """Train IQL agent on offline dataset.

    Args:
        agent: IQL agent
        dataset: Offline dataset
        config: Training config
        output_dir: Directory to save checkpoints and logs
        use_dr3: Whether to use DR3 regularization
        video_dir: Directory to save evaluation videos as MP4 files
    """
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_freq = config["eval_freq"]
    save_freq = config["save_freq"]
    near_terminal_steps = int(config.get("reward_near_terminal_steps", 5))
    rep_dashboard_every = int(config.get("rep_dashboard_every", 0))
    rep_max_points = int(config.get("rep_max_points", 512))

    # DataLoader
    dataloader_workers = int(config.get("dataloader_workers", 4))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=bool(config.get("pin_memory", True)),
        persistent_workers=bool(config.get("persistent_workers", True)) and dataloader_workers > 0,
        prefetch_factor=int(config.get("prefetch_factor", 2)),
    )

    # DR3 loss (optional)
    dr3_loss = DR3Loss(
        eps=config.get("dr3_eps", 1e-4),
        alpha=config.get("dr3_alpha", 1.0),
    ) if use_dr3 else None

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))

    # Reference policy snapshot for KL (epoch 0)
    ref_encoder = None
    ref_actor = None
    if reward_model is not None:
        ref_encoder = copy.deepcopy(agent.encoder).eval()
        ref_actor = copy.deepcopy(agent.actor).eval()
        ref_encoder.to(agent.device)
        ref_actor.to(agent.device)

    # Training loop
    metrics_history = []
    step = 0
    printed_input_stats = False
    train_csv_path = os.path.join(output_dir, "train_metrics.csv")
    train_json_path = os.path.join(output_dir, "train_metrics_latest.json")
    eval_csv_path = os.path.join(output_dir, "eval_metrics.csv")
    eval_json_path = os.path.join(output_dir, "eval_metrics_latest.json")
    rep_csv_path = os.path.join(output_dir, "rep_metrics.csv")
    rep_json_path = os.path.join(output_dir, "rep_metrics_latest.json")
    rep_dir = os.path.join(output_dir, "rep_diagnostics")

    if rep_dashboard_every > 0:
        # Start a clean rep-metrics stream per training run to avoid mixing runs.
        for path in (rep_csv_path, rep_json_path):
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(rep_dir):
            for name in os.listdir(rep_dir):
                if name.startswith("rep_pca_epoch_") or name.startswith("rep_dashboard."):
                    try:
                        os.remove(os.path.join(rep_dir, name))
                    except OSError:
                        pass

    steps_per_epoch = int(config.get("steps_per_epoch", 0))

    for epoch in range(num_epochs):
        epoch_metrics = []
        last_batch = None

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step_in_epoch, batch_data in enumerate(pbar, start=1):
            # Unpack batch
            steps_to_terminal = None
            if len(batch_data) == 12:
                (
                    s_grid, s_agent_pos, s_agent_dir, s_carry,
                    a, r,
                    sp_grid, sp_agent_pos, sp_agent_dir, sp_carry,
                    done,
                    steps_to_terminal,
                ) = batch_data
            else:
                (
                    s_grid, s_agent_pos, s_agent_dir, s_carry,
                    a, r,
                    sp_grid, sp_agent_pos, sp_agent_dir, sp_carry,
                    done,
                ) = batch_data

            if not printed_input_stats:
                grid_min = s_grid.min().item()
                grid_max = s_grid.max().item()
                grid_mean = s_grid.float().mean().item()
                print(f"[input_stats] grid min: {grid_min}, max: {grid_max}, mean: {grid_mean:.4f}")
                printed_input_stats = True

            batch = {
                "s_grid": s_grid,
                "s_agent_pos": s_agent_pos,
                "s_agent_dir": s_agent_dir,
                "s_carry": s_carry,
                "a": a,
                "r": r,
                "sp_grid": sp_grid,
                "sp_agent_pos": sp_agent_pos,
                "sp_agent_dir": sp_agent_dir,
                "sp_carry": sp_carry,
                "done": done,
            }
            if steps_to_terminal is not None:
                batch["steps_to_terminal"] = steps_to_terminal

            r_env = batch["r"].clone()  # Original env reward for logging
            shaping = None
            # Potential-based reward shaping: r_total = r_env + alpha * (gamma * h(s') - h(s))
            # This is provably safe: telescoping sum means shaping can't be exploited
            # The agent must still reach the goal to maximize total reward
            if reward_model is not None:
                batch = {k: v.to(agent.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                shaping_alpha = float(config.get("shaping_alpha", 0.1))
                shaping_gamma = float(config.get("discount", 0.99))
                with torch.no_grad():
                    s_feat = reward_model.encode_state(
                        batch["s_grid"],
                        batch["s_agent_pos"],
                        batch["s_agent_dir"],
                        batch["s_carry"],
                    )
                    sp_feat = reward_model.encode_state(
                        batch["sp_grid"],
                        batch["sp_agent_pos"],
                        batch["sp_agent_dir"],
                        batch["sp_carry"],
                    )
                    h_s = reward_model.h(s_feat)
                    h_sp = reward_model.h(sp_feat)
                    # Potential-based shaping: F(s,s') = gamma * h(s') - h(s)
                    shaping = shaping_gamma * h_sp - h_s
                    # Total reward = environment + scaled shaping
                    batch["r"] = r_env.to(agent.device) + shaping_alpha * shaping

            # IQL update
            metrics = agent.update(batch)

            # DR3 regularization (optional)
            if use_dr3 and dr3_loss is not None:
                # Get features for current batch (allow encoder gradients)
                batch = {k: v.to(agent.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                features = agent.encode_state(
                    batch["s_grid"],
                    batch["s_agent_pos"],
                    batch["s_agent_dir"],
                    batch["s_carry"],
                )

                # Compute DR3 loss
                dr3_reg = dr3_loss(features)

                # Add to encoder optimizer
                agent.encoder_optimizer.zero_grad()
                dr3_reg.backward()
                agent.encoder_optimizer.step()

                metrics["dr3_loss"] = dr3_reg.item()

                # Get DR3 diagnostics
                dr3_metrics = dr3_loss.get_metrics(features)
                metrics.update({f"dr3_{k}": v for k, v in dr3_metrics.items()})

            if reward_model is not None:
                r_env_dev = r_env.to(agent.device)
                steps_term = batch.get("steps_to_terminal")
                metrics.update(
                    compute_reward_alignment_metrics(
                        r_env_dev,
                        batch["r"],
                        batch["done"],
                        steps_term,
                        near_terminal_steps,
                    )
                )
                if ref_encoder is not None and ref_actor is not None:
                    metrics.update(
                        compute_policy_sensitivity_metrics(
                            agent,
                            ref_encoder,
                            ref_actor,
                            batch,
                            r_env_dev,
                            batch["r"],
                        )
                    )
                done_mask = batch["done"].bool()
                metrics["terminal_margin"] = compute_terminal_margin(agent, batch)
                metrics["terminal_margin_valid"] = float(done_mask.any().item()) * float((~done_mask).any().item())

            epoch_metrics.append(metrics)
            step += 1
            last_batch = batch

            # Update progress bar
            avg_metrics = {k: np.nanmean([m[k] for m in epoch_metrics[-100:]]) for k in metrics.keys()}
            pbar.set_postfix({k: f"{v:.4f}" for k, v in avg_metrics.items()})

            # TensorBoard (step-level)
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, step, new_style=False)
            if reward_model is not None:
                writer.add_scalar("train/reward_mean", batch["r"].mean().item(), step, new_style=False)

            if steps_per_epoch and step_in_epoch >= steps_per_epoch:
                break

        # Epoch summary
        epoch_summary = {k: np.nanmean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0].keys()}
        if "reward_env_sum" in epoch_metrics[0] and "reward_airl_sum" in epoch_metrics[0]:
            env_sum_total = float(np.nansum([m["reward_env_sum"] for m in epoch_metrics]))
            airl_sum_total = float(np.nansum([m["reward_airl_sum"] for m in epoch_metrics]))
            epoch_summary["reward_env_sum_total"] = env_sum_total
            epoch_summary["reward_airl_sum_total"] = airl_sum_total
            epoch_summary["reward_sum_ratio_total"] = airl_sum_total / (env_sum_total + 1e-3)
        epoch_summary["epoch"] = epoch + 1
        metrics_history.append(epoch_summary)

        print(f"\nEpoch {epoch+1} summary:")
        for k, v in epoch_summary.items():
            if k != "epoch":
                print(f"  {k}: {v:.4f}")
                writer.add_scalar(f"epoch/{k}", v, epoch + 1, new_style=False)
        append_csv_row(train_csv_path, epoch_summary)
        write_json(train_json_path, epoch_summary)

        # Periodic eval rollouts
        if env_cfg and eval_layouts and (epoch + 1) % eval_freq == 0:
            eval_summary = run_eval_rollouts(
                agent,
                env_cfg,
                eval_layouts,
                eval_episodes,
                eval_max_steps,
                eval_fps,
                writer,
                global_step=epoch + 1,
                video_dir=video_dir,
            )
            eval_row = {"epoch": epoch + 1, **eval_summary}
            append_csv_row(eval_csv_path, eval_row)
            write_json(eval_json_path, eval_row)

        # Auto-generate dashboard every N epochs
        if dashboard_every and (epoch + 1) % dashboard_every == 0:
            out_dir = dashboard_dir or os.path.join(output_dir, "training_diagnostics")
            os.makedirs(out_dir, exist_ok=True)
            script = ROOT / "scripts" / "section2" / "analyze_training_metrics.py"
            argv = [
                str(script),
                "--train-csv",
                train_csv_path,
                "--eval-csv",
                eval_csv_path,
                "--out-dir",
                out_dir,
                "--title",
                dashboard_title or "Training Diagnostics (Epoch-Level)",
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                runpy.run_path(str(script), run_name="__main__")
            finally:
                sys.argv = old_argv

        if rep_dashboard_every and (epoch + 1) % rep_dashboard_every == 0 and last_batch is not None:
            rep_metrics = compute_representation_metrics(agent, last_batch, max_points=rep_max_points)
            rep_metrics["epoch"] = epoch + 1
            append_csv_row(rep_csv_path, rep_metrics)
            write_json(rep_json_path, rep_metrics)
            coords, labels = compute_representation_pca(agent, last_batch, max_points=rep_max_points)
            os.makedirs(rep_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(rep_dir, f"rep_pca_epoch_{epoch+1}.npz"),
                coords=coords,
                labels=labels,
            )
            script = ROOT / "scripts" / "section2" / "analyze_representation.py"
            argv = [
                str(script),
                "--rep-csv",
                rep_csv_path,
                "--rep-dir",
                rep_dir,
                "--title",
                f"Representation Diagnostics - {Path(output_dir).name}",
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                runpy.run_path(str(script), run_name="__main__")
            finally:
                sys.argv = old_argv

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    agent.save(final_path)
    print(f"\nTraining complete. Final model saved: {final_path}")
    writer.flush()
    writer.close()

    return metrics_history


def main():
    parser = argparse.ArgumentParser(description="Train IQL on offline dataset")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--use-dr3", action="store_true", help="Enable DR3 regularization")
    parser.add_argument("--env-config", default=None, help="Env config for eval rollouts")
    parser.add_argument("--eval-layouts", default="", help="Comma-separated layouts for eval rollouts")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Eval episodes per layout")
    parser.add_argument("--eval-max-steps", type=int, default=200, help="Max steps for eval rollouts")
    parser.add_argument("--eval-fps", type=int, default=10, help="FPS for eval videos")
    parser.add_argument("--video-dir", default=None, help="Directory to save eval videos as MP4 files")
    parser.add_argument("--dashboard-every", type=int, default=0, help="Generate dashboard every N epochs")
    parser.add_argument("--dashboard-dir", default=None, help="Directory to save training dashboards")
    parser.add_argument("--reward-model", default=None, help="Path to AIRL reward_model.pt")
    parser.add_argument("--reward-config", default=None, help="Path to AIRL config for reward model")
    parser.add_argument("--fast-test", action="store_true", help="Override config for quick test run")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.fast_test:
        config["num_epochs"] = 3
        config["steps_per_epoch"] = 100
        config["eval_freq"] = 1
        config["rep_dashboard_every"] = 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load dataset
    dataset, metadata, data_stats = load_dataset(args.dataset, args.metadata)

    # Save data stats
    with open(os.path.join(args.output_dir, "data_stats.json"), "w") as f:
        json.dump(data_stats, f, indent=2)

    # Create agent
    agent = IQL(
        grid_channels=config["grid_channels"],
        num_actions=config["num_actions"],
        feature_dim=config.get("feature_dim", 256),
        hidden_dim=config.get("hidden_dim", 256),
        discount=config.get("discount", 0.99),
        tau=config.get("tau", 0.7),
        beta=config.get("beta", 3.0),
        learning_rate=float(config.get("learning_rate", 3e-4)),
        target_update_freq=config.get("target_update_freq", 2),
        polyak_tau=config.get("polyak_tau", None),
        rnd_alpha=float(config.get("rnd_alpha", 0.0)),
        rnd_output_dim=int(config.get("rnd_output_dim", config.get("feature_dim", 256))),
        use_target_encoder=bool(config.get("use_target_encoder", True)),
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )

    use_dr3 = bool(args.use_dr3 or config.get("use_dr3", False))

    print(f"\nAgent created:")
    print(f"  - Device: {agent.device}")
    print(f"  - Feature dim: {config.get('feature_dim', 256)}")
    print(f"  - Discount: {config.get('discount', 0.99)}")
    print(f"  - Tau (expectile): {config.get('tau', 0.7)}")
    print(f"  - DR3 enabled: {use_dr3}")

    env_cfg = None
    eval_layouts = None
    if args.env_config and args.eval_layouts:
        env_cfg = load_config(args.env_config)["env"]
        eval_layouts = [x.strip() for x in args.eval_layouts.split(",") if x.strip()]
    dataset_name = Path(args.metadata).parent.name
    dashboard_title = f"IQL Training Diagnostics - {dataset_name}"

    reward_model = None
    if args.reward_model and args.reward_config:
        reward_cfg = load_config(args.reward_config)
        reward_model = AIRLReward(
            grid_channels=reward_cfg["model"]["grid_channels"],
            num_actions=reward_cfg["model"]["num_actions"],
            feature_dim=reward_cfg["model"]["feature_dim"],
            hidden_dim=reward_cfg["model"]["hidden_dim"],
            gamma=reward_cfg["airl"]["gamma"],
            state_only_reward=bool(reward_cfg["airl"].get("state_only_reward", False)),
        ).to(agent.device)
        # Initialize encoder lazy layers before loading
        with torch.no_grad():
            dummy_grid = torch.zeros(1, 25, 25, reward_cfg["model"]["grid_channels"], device=agent.device)
            dummy_pos = torch.zeros(1, 2, device=agent.device, dtype=torch.int64)
            dummy_dir = torch.zeros(1, device=agent.device, dtype=torch.int64)
            dummy_carry = torch.zeros(1, 2, device=agent.device, dtype=torch.int64)
            _ = reward_model.encode_state(dummy_grid, dummy_pos, dummy_dir, dummy_carry)
        try:
            state = torch.load(args.reward_model, map_location=agent.device, weights_only=True)
        except TypeError:
            state = torch.load(args.reward_model, map_location=agent.device)
        reward_state = state.get("reward", state)
        current_state = reward_model.state_dict()
        filtered = {}
        for k, v in reward_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                filtered[k] = v
        reward_model.load_state_dict(filtered, strict=False)
        reward_model.eval()

    # Train
    print("\nStarting training...")
    video_dir = args.video_dir or os.path.join(args.output_dir, "eval_videos")
    metrics = train_iql(
        agent,
        dataset,
        config,
        args.output_dir,
        use_dr3=use_dr3,
        env_cfg=env_cfg,
        eval_layouts=eval_layouts,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        eval_fps=args.eval_fps,
        video_dir=video_dir,
        dashboard_every=args.dashboard_every,
        dashboard_dir=args.dashboard_dir,
        dashboard_title=dashboard_title,
        reward_model=reward_model,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
