#!/usr/bin/env python3
"""Estimate alpha scaling for AIRL reward to match target magnitude."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.airl import AIRLReward
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate alpha for AIRL reward scaling")
    parser.add_argument("--reward-model", required=True, help="Path to reward_model.pt")
    parser.add_argument("--reward-config", required=True, help="Path to AIRL config yaml")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--target-mean", type=float, default=0.3, help="Target mean |alpha*r_airl|")
    parser.add_argument("--sample-size", type=int, default=20000, help="Number of transitions to sample")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.reward_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset, allow_pickle=True)
    r_env = data["r"]
    n = len(r_env)
    sample_n = min(args.sample_size, n)
    idx = np.random.choice(n, size=sample_n, replace=False)

    s_grid = torch.from_numpy(data["s_grid"][idx]).to(device)
    s_pos = torch.from_numpy(data["s_agent_pos"][idx]).to(device)
    s_dir = torch.from_numpy(data["s_agent_dir"][idx]).to(device)
    s_carry = torch.from_numpy(data["s_carry"][idx]).to(device)
    a = torch.from_numpy(data["a"][idx]).to(device)
    sp_grid = torch.from_numpy(data["sp_grid"][idx]).to(device)
    sp_pos = torch.from_numpy(data["sp_agent_pos"][idx]).to(device)
    sp_dir = torch.from_numpy(data["sp_agent_dir"][idx]).to(device)
    sp_carry = torch.from_numpy(data["sp_carry"][idx]).to(device)

    reward_model = AIRLReward(
        grid_channels=cfg["model"]["grid_channels"],
        num_actions=cfg["model"]["num_actions"],
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        gamma=cfg["airl"]["gamma"],
        state_only_reward=bool(cfg["airl"].get("state_only_reward", False)),
    ).to(device)
    with torch.no_grad():
        dummy_grid = torch.zeros(1, 25, 25, cfg["model"]["grid_channels"], device=device)
        dummy_pos = torch.zeros(1, 2, device=device, dtype=torch.int64)
        dummy_dir = torch.zeros(1, device=device, dtype=torch.int64)
        dummy_carry = torch.zeros(1, 2, device=device, dtype=torch.int64)
        _ = reward_model.encode_state(dummy_grid, dummy_pos, dummy_dir, dummy_carry)
    state = torch.load(args.reward_model, map_location=device)
    reward_model.load_state_dict(state["reward"], strict=False)
    reward_model.eval()

    with torch.no_grad():
        s_feat = reward_model.encode_state(s_grid, s_pos, s_dir, s_carry)
        sp_feat = reward_model.encode_state(sp_grid, sp_pos, sp_dir, sp_carry)
        r_airl = reward_model.f(s_feat, a, sp_feat).cpu().numpy()

    mean_abs = float(np.mean(np.abs(r_airl)))
    stats = {
        "sample_size": int(sample_n),
        "r_airl_mean": float(np.mean(r_airl)),
        "r_airl_std": float(np.std(r_airl)),
        "r_airl_p10": float(np.percentile(r_airl, 10)),
        "r_airl_p50": float(np.percentile(r_airl, 50)),
        "r_airl_p90": float(np.percentile(r_airl, 90)),
        "r_airl_mean_abs": mean_abs,
        "r_env_mean": float(np.mean(r_env)),
        "r_env_std": float(np.std(r_env)),
        "alpha_target_mean": float(args.target_mean),
        "alpha_suggested": float(args.target_mean / mean_abs) if mean_abs > 0 else 0.0,
        "alpha_for_0.1": float(0.1 / mean_abs) if mean_abs > 0 else 0.0,
        "alpha_for_0.3": float(0.3 / mean_abs) if mean_abs > 0 else 0.0,
        "alpha_for_0.5": float(0.5 / mean_abs) if mean_abs > 0 else 0.0,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[estimate_reward_alpha] wrote {out_path}")
    print(f"[estimate_reward_alpha] mean_abs={mean_abs:.4f} alpha={stats['alpha_suggested']:.4f}")


if __name__ == "__main__":
    main()
