#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import run_episode
from src.envs import make_env
from src.planner import BehaviorConfig, CostConfig, PlannerConfig, plan_actions
from src.utils import ensure_dir, load_config
from src.viz import render_path_overlay, save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Section 1 dataset behaviors")
    parser.add_argument("--config", required=True, help="Path to section1 config yaml")
    parser.add_argument("--episodes", type=int, default=30, help="Number of validation episodes")
    parser.add_argument("--seed", type=int, default=777, help="RNG seed for validation runs")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per validation episode")
    parser.add_argument("--out-dir", default="outputs/section1_validation", help="Output directory")
    parser.add_argument("--require-success", action="store_true", help="Retry until success")
    parser.add_argument("--max-attempts", type=int, default=25, help="Max env retries per episode")
    return parser.parse_args()


def pick_behavior(rng: np.random.Generator, mix: Dict[str, float]) -> str:
    names = list(mix.keys())
    probs = np.array(list(mix.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def make_behavior_cfg(cfg: Dict, name: str) -> BehaviorConfig:
    bcfg = cfg["behaviors"][name]
    return BehaviorConfig(
        random_action_prob=bcfg["random_action_prob"],
        heuristic_noise=bcfg["heuristic_noise"],
        cost_noise_std=bcfg["cost_noise_std"],
    )


def _collect_frames(env, actions: List[int]) -> List[np.ndarray]:
    frames = []
    for action in actions:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return frames


def _build_deterministic_cost_cfg(base_costs: Dict) -> CostConfig:
    return CostConfig(
        step=base_costs["step"],
        turn=base_costs["turn"],
        toggle=base_costs["toggle"],
        lava_penalty=base_costs["lava_penalty"],
        noise_std=0.0,
    )


def _find_solvable_seed(
    env_cfg: Dict,
    base_costs: Dict,
    planner_cfg: PlannerConfig,
    rng: np.random.Generator,
    max_attempts: int,
) -> Optional[int]:
    for _ in range(max_attempts):
        seed = int(rng.integers(0, 1_000_000))
        env = make_env(env_cfg)
        env.reset(seed=seed)
        plan = plan_actions(env, _build_deterministic_cost_cfg(base_costs), planner_cfg, rng)
        if plan:
            return seed
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "paths"))
    ensure_dir(os.path.join(out_dir, "videos"))
    ensure_dir(os.path.join(out_dir, "grids"))

    env_cfg = cfg["env"]
    base_costs = cfg["costs"]
    planner_cfg = PlannerConfig(**cfg["planner"])
    mix = cfg["dataset"]["planner_mix"]

    summary = []

    print(f"[validate] episodes={args.episodes} seed={args.seed} out={out_dir}")
    pbar = tqdm(total=args.episodes, desc="Validation episodes")

    for ep in range(args.episodes):
        seed = _find_solvable_seed(env_cfg, base_costs, planner_cfg, rng, args.max_attempts)
        if seed is None:
            summary.append(
                {
                    "episode": ep,
                    "seed": None,
                    "behavior": None,
                    "status": "no_solvable_env",
                }
            )
            pbar.update(1)
            continue

        env = make_env(env_cfg, render_mode="rgb_array")
        env.reset(seed=seed)

        behavior_name = pick_behavior(rng, mix)
        behavior_cfg = make_behavior_cfg(cfg, behavior_name)

        attempt = 0
        result = None
        while attempt < args.max_attempts:
            result = run_episode(
                env,
                base_costs=base_costs,
                base_planner=planner_cfg,
                behavior=behavior_cfg,
                rng=rng,
                max_steps=args.max_steps,
            )
            if result and (result.success or not args.require_success):
                break
            attempt += 1
            seed = _find_solvable_seed(env_cfg, base_costs, planner_cfg, rng, args.max_attempts)
            if seed is None:
                break
            env = make_env(env_cfg, render_mode="rgb_array")
            env.reset(seed=seed)

        if result is None or not result.transitions:
            summary.append(
                {
                    "episode": ep,
                    "seed": seed,
                    "behavior": behavior_name,
                    "status": "failed_plan_or_rollout",
                }
            )
            pbar.update(1)
            continue

        positions = [t[0].agent_pos for t in result.transitions]
        replay_env = make_env(env_cfg, render_mode="rgb_array")
        replay_env.reset(seed=seed)
        frames = _collect_frames(replay_env, result.actions)
        if frames:
            video_path = os.path.join(out_dir, "videos", f"episode_{ep}_{behavior_name}.mp4")
            save_video(frames, video_path, fps=10)

        base_env = make_env(env_cfg, render_mode="rgb_array")
        base_env.reset(seed=seed)
        frame = base_env.render()
        if frame is not None:
            overlay = render_path_overlay(frame, positions, (base_env.grid.width, base_env.grid.height))
            overlay.save(os.path.join(out_dir, "paths", f"episode_{ep}_{behavior_name}.png"))

        summary.append(
            {
                "episode": ep,
                "seed": seed,
                "behavior": behavior_name,
                "status": "ok",
                "length": len(result.transitions),
                "success": bool(result.success),
            }
        )
        pbar.update(1)

    pbar.close()

    success_rate = np.mean([s.get("success", False) for s in summary if s.get("status") == "ok"])
    print(f"[validate] success_rate={success_rate:.3f}")
    print(f"[validate] outputs in {out_dir}")

    # Lightweight log file for review.
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("episode,seed,behavior,status,length,success\n")
        for row in summary:
            f.write(
                f"{row.get('episode')},{row.get('seed')},{row.get('behavior')},"
                f"{row.get('status')},{row.get('length','')},{row.get('success','')}\n"
            )


if __name__ == "__main__":
    main()
