#!/usr/bin/env python3
"""Probe g(s,a) values at goal states vs non-goal states.

This answers the key question: does g(s,a) learn the task reward,
or is all the discriminative signal in h(s)?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.airl.models import AIRLReward
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe g(s,a) at goal vs non-goal states")
    parser.add_argument("--config", required=True, help="Path to AIRL config yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to reward model checkpoint")
    parser.add_argument("--expert-dataset", required=True, help="Path to expert dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Load reward model
    reward = AIRLReward(
        grid_channels=cfg["model"]["grid_channels"],
        num_actions=cfg["model"]["num_actions"],
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        gamma=cfg["airl"]["gamma"],
        state_only_reward=bool(cfg["airl"].get("state_only_reward", False)),
    ).to(device)

    # Load checkpoint - need to handle dynamically created fc_grid
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Initialize fc_grid with correct dimensions from checkpoint
    fc_grid_weight = ckpt["reward"]["encoder.fc_grid.weight"]
    reward.encoder.conv_out_dim = fc_grid_weight.shape[1]
    reward.encoder.fc_grid = torch.nn.Linear(fc_grid_weight.shape[1], 128).to(device)

    reward.load_state_dict(ckpt["reward"])
    reward.eval()

    # Load expert dataset
    data = np.load(args.expert_dataset, allow_pickle=True)

    # Get rewards from dataset
    r_env = data["r"]

    # Separate goal (r > 0) and non-goal (r == 0) indices
    goal_mask = r_env > 0
    non_goal_mask = r_env == 0

    print(f"Total transitions: {len(r_env)}")
    print(f"Goal transitions (r > 0): {goal_mask.sum()}")
    print(f"Non-goal transitions (r == 0): {non_goal_mask.sum()}")
    print(f"Env reward at goal states - mean: {r_env[goal_mask].mean():.4f}, std: {r_env[goal_mask].std():.4f}")
    print()

    # Sample subset for analysis (avoid OOM)
    n_goal = min(goal_mask.sum(), 1000)
    n_non_goal = min(non_goal_mask.sum(), 5000)

    goal_idx = np.where(goal_mask)[0]
    non_goal_idx = np.where(non_goal_mask)[0]

    np.random.seed(42)
    goal_sample = np.random.choice(goal_idx, size=n_goal, replace=False)
    non_goal_sample = np.random.choice(non_goal_idx, size=n_non_goal, replace=False)

    def compute_reward_components(idx: np.ndarray) -> dict:
        """Compute g(s,a), h(s), h(s'), and f(s,a,s') for given indices."""
        batch = {
            "s_grid": torch.from_numpy(data["s_grid"][idx]).to(device),
            "s_agent_pos": torch.from_numpy(data["s_agent_pos"][idx]).to(device),
            "s_agent_dir": torch.from_numpy(data["s_agent_dir"][idx]).to(device),
            "s_carry": torch.from_numpy(data["s_carry"][idx]).to(device),
            "a": torch.from_numpy(data["a"][idx]).to(device),
            "sp_grid": torch.from_numpy(data["sp_grid"][idx]).to(device),
            "sp_agent_pos": torch.from_numpy(data["sp_agent_pos"][idx]).to(device),
            "sp_agent_dir": torch.from_numpy(data["sp_agent_dir"][idx]).to(device),
            "sp_carry": torch.from_numpy(data["sp_carry"][idx]).to(device),
        }

        with torch.no_grad():
            s_feat = reward.encode_state(
                batch["s_grid"],
                batch["s_agent_pos"],
                batch["s_agent_dir"],
                batch["s_carry"],
            )
            sp_feat = reward.encode_state(
                batch["sp_grid"],
                batch["sp_agent_pos"],
                batch["sp_agent_dir"],
                batch["sp_carry"],
            )

            g_vals = reward.g(s_feat, batch["a"]).cpu().numpy()
            h_s = reward.h(s_feat).cpu().numpy()
            h_sp = reward.h(sp_feat).cpu().numpy()
            f_vals = reward.f(s_feat, batch["a"], sp_feat).cpu().numpy()
            shaping = cfg["airl"]["gamma"] * h_sp - h_s

        return {
            "g": g_vals,
            "h_s": h_s,
            "h_sp": h_sp,
            "f": f_vals,
            "shaping": shaping,
            "r_env": r_env[idx],
        }

    goal_results = compute_reward_components(goal_sample)
    non_goal_results = compute_reward_components(non_goal_sample)

    print("=" * 60)
    print("GOAL STATES (r_env > 0)")
    print("=" * 60)
    print(f"  g(s,a):    mean={goal_results['g'].mean():.4f}, std={goal_results['g'].std():.4f}")
    print(f"  h(s):      mean={goal_results['h_s'].mean():.4f}, std={goal_results['h_s'].std():.4f}")
    print(f"  h(s'):     mean={goal_results['h_sp'].mean():.4f}, std={goal_results['h_sp'].std():.4f}")
    print(f"  shaping:   mean={goal_results['shaping'].mean():.4f}, std={goal_results['shaping'].std():.4f}")
    print(f"  f(s,a,s'): mean={goal_results['f'].mean():.4f}, std={goal_results['f'].std():.4f}")
    print(f"  r_env:     mean={goal_results['r_env'].mean():.4f}, std={goal_results['r_env'].std():.4f}")
    print()

    print("=" * 60)
    print("NON-GOAL STATES (r_env == 0)")
    print("=" * 60)
    print(f"  g(s,a):    mean={non_goal_results['g'].mean():.4f}, std={non_goal_results['g'].std():.4f}")
    print(f"  h(s):      mean={non_goal_results['h_s'].mean():.4f}, std={non_goal_results['h_s'].std():.4f}")
    print(f"  h(s'):     mean={non_goal_results['h_sp'].mean():.4f}, std={non_goal_results['h_sp'].std():.4f}")
    print(f"  shaping:   mean={non_goal_results['shaping'].mean():.4f}, std={non_goal_results['shaping'].std():.4f}")
    print(f"  f(s,a,s'): mean={non_goal_results['f'].mean():.4f}, std={non_goal_results['f'].std():.4f}")
    print()

    print("=" * 60)
    print("KEY COMPARISONS")
    print("=" * 60)
    g_gap = goal_results['g'].mean() - non_goal_results['g'].mean()
    f_gap = goal_results['f'].mean() - non_goal_results['f'].mean()
    h_s_gap = goal_results['h_s'].mean() - non_goal_results['h_s'].mean()
    shaping_gap = goal_results['shaping'].mean() - non_goal_results['shaping'].mean()

    print(f"  g(s,a) gap (goal - non-goal):    {g_gap:.4f}")
    print(f"  f(s,a,s') gap (goal - non-goal): {f_gap:.4f}")
    print(f"  h(s) gap (goal - non-goal):      {h_s_gap:.4f}")
    print(f"  shaping gap (goal - non-goal):   {shaping_gap:.4f}")
    print()

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if abs(g_gap) > 0.1:
        print("  g(s,a) shows significant difference at goal vs non-goal states")
        print("  -> g(s,a) HAS learned task-relevant signal")
        print("  -> Consider using f(s,a,s') directly as reward")
    else:
        print("  g(s,a) is nearly flat across goal/non-goal states")
        print("  -> g(s,a) has NOT learned task reward structure")
        print("  -> Only h(s) provides useful shaping signal")
        print("  -> MUST use r_env + alpha * (gamma*h(s')-h(s)) for IQL")

    print()
    if abs(f_gap) > 0.1:
        print(f"  f(s,a,s') shows {f_gap:.2f} gap between goal and non-goal")
        print("  -> This comes primarily from shaping (h terms), not g(s,a)")


if __name__ == "__main__":
    main()
