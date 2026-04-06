#!/usr/bin/env python3
"""Evaluate trained IQL policy in the environment.

Metrics:
- Success rate on training layouts
- Success rate on held-out seeds
- Episode lengths
- Comparison against behavior policies (near_optimal baseline)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minigrid.core.world_object import Lava

from src.envs import make_env
from src.iql import IQL
from src.utils import load_config
from src.viz import save_video


def evaluate_policy(
    agent: IQL,
    env_cfg: dict,
    layout_names: list[str],
    num_episodes_per_layout: int,
    seed_offset: int,
    max_steps: int,
    deterministic: bool = True,
    video_dir: str | None = None,
    max_videos: int = 5,
    fps: int = 10,
    tb_logdir: str | None = None,
) -> dict:
    """Evaluate policy on multiple layouts.

    Args:
        agent: Trained IQL agent
        env_cfg: Environment config
        layout_names: List of layout names to evaluate
        num_episodes_per_layout: Episodes per layout
        seed_offset: Seed offset for evaluation (use high values for held-out seeds)
        max_steps: Max steps per episode
        deterministic: Whether to use deterministic policy

    Returns:
        Dictionary with evaluation metrics
    """
    results = defaultdict(list)

    writer = None
    if tb_logdir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tb_logdir)

    for layout_name in layout_names:
        print(f"\nEvaluating on layout: {layout_name}")

        render_mode = "rgb_array" if video_dir else None
        env = make_env(env_cfg, render_mode=render_mode, layout_name=layout_name)

        layout_successes = []
        layout_lengths = []
        layout_returns = []
        layout_lava_contacts = []
        saved_success = 0
        saved_failure = 0

        for ep in tqdm(range(num_episodes_per_layout), desc=f"{layout_name}"):
            seed = seed_offset + ep
            obs, info = env.reset(seed=seed)

            episode_return = 0
            done = False
            steps = 0
            lava_contacts = 0
            frames = []

            while not done and steps < max_steps:
                # Get observation components
                grid = obs["image"]
                agent_pos = np.array([info["agent_pos"][0], info["agent_pos"][1]])
                agent_dir = info["agent_dir"]
                carry = np.array([0, 0])  # Default carry state

                # Get action from agent
                action = agent.get_action(
                    grid, agent_pos, agent_dir, carry, deterministic=deterministic
                )

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_return += reward
                steps += 1

                if isinstance(env.grid.get(env.agent_pos[0], env.agent_pos[1]), Lava):
                    lava_contacts += 1
                if video_dir:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

            # Record metrics
            success = episode_return > 0  # Sparse reward: +1 at goal
            layout_successes.append(success)
            layout_lengths.append(steps)
            layout_returns.append(episode_return)
            layout_lava_contacts.append(lava_contacts)

            if video_dir and frames:
                if success and saved_success < max_videos:
                    path = Path(video_dir) / f"{layout_name}_success_{saved_success}.mp4"
                    save_video(frames, str(path), fps=fps)
                    saved_success += 1
                elif (not success) and saved_failure < max_videos:
                    path = Path(video_dir) / f"{layout_name}_fail_{saved_failure}.mp4"
                    save_video(frames, str(path), fps=fps)
                    saved_failure += 1

            if writer is not None and frames:
                if success and saved_success <= max_videos:
                    tag = f"{layout_name}/success_{saved_success-1}"
                elif (not success) and saved_failure <= max_videos:
                    tag = f"{layout_name}/fail_{saved_failure-1}"
                else:
                    tag = None
                if tag:
                    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)
                    writer.add_video(tag, video, fps=fps)

        # Aggregate layout metrics
        success_rate = np.mean(layout_successes)
        avg_length = np.mean(layout_lengths)
        avg_return = np.mean(layout_returns)

        print(f"  - Success rate: {success_rate:.2%} ({sum(layout_successes)}/{len(layout_successes)})")
        print(f"  - Avg length: {avg_length:.1f} steps")
        print(f"  - Avg return: {avg_return:.4f}")

        results[f"{layout_name}_success_rate"] = success_rate
        results[f"{layout_name}_avg_length"] = avg_length
        results[f"{layout_name}_avg_return"] = avg_return
        results[f"{layout_name}_lava_contacts"] = layout_lava_contacts
        results[f"{layout_name}_successes"] = layout_successes
        results[f"{layout_name}_lengths"] = layout_lengths

    # Overall metrics
    all_successes = []
    all_lengths = []
    for layout_name in layout_names:
        all_successes.extend(results[f"{layout_name}_successes"])
        all_lengths.extend(results[f"{layout_name}_lengths"])

    results["overall_success_rate"] = np.mean(all_successes)
    results["overall_avg_length"] = np.mean(all_lengths)

    print(f"\n{'='*60}")
    print(f"Overall Results:")
    print(f"  - Success rate: {results['overall_success_rate']:.2%}")
    print(f"  - Avg length: {results['overall_avg_length']:.1f} steps")
    print(f"{'='*60}")

    if writer is not None:
        writer.flush()
        writer.close()
    return dict(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained IQL policy")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--env-config", required=True, help="Path to environment config")
    parser.add_argument("--layouts", nargs="+", required=True, help="Layouts to evaluate")
    parser.add_argument("--num-episodes", type=int, default=100, help="Episodes per layout")
    parser.add_argument("--seed-offset", type=int, default=10000, help="Seed offset (use high for held-out)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--video-dir", default=None, help="Directory for rollout videos")
    parser.add_argument("--max-videos", type=int, default=5, help="Max success/fail videos per layout")
    parser.add_argument("--video-fps", type=int, default=10, help="FPS for rollout videos")
    parser.add_argument("--tb-logdir", default=None, help="TensorBoard logdir for rollout videos")
    parser.add_argument(
        "--goal-indices",
        default="",
        help="Comma-separated goal indices to use (e.g., '0,1,2'). Empty = all.",
    )
    args = parser.parse_args()

    # Load environment config
    env_cfg = load_config(args.env_config)["env"]
    if args.goal_indices:
        env_cfg["goal_region_indices"] = [int(x) for x in args.goal_indices.split(",") if x.strip()]
    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    if args.tb_logdir:
        Path(args.tb_logdir).mkdir(parents=True, exist_ok=True)

    # Create agent
    agent = IQL(
        grid_channels=3,
        num_actions=7,  # MiniGrid actions
        feature_dim=256,
        hidden_dim=256,
    )

    # Load model
    print(f"Loading model from {args.model}")
    agent.load(args.model)
    agent.eval()

    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_policy(
        agent,
        env_cfg,
        args.layouts,
        args.num_episodes,
        args.seed_offset,
        args.max_steps,
        deterministic=not args.stochastic,
        video_dir=args.video_dir,
        max_videos=args.max_videos,
        fps=args.video_fps,
        tb_logdir=args.tb_logdir,
    )

    # Save results
    # Convert numpy types to Python types for JSON
    json_results = {}
    for k, v in results.items():
        if isinstance(v, (list, np.ndarray)):
            json_results[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            json_results[k] = float(v)
        else:
            json_results[k] = v

    with open(args.output, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
