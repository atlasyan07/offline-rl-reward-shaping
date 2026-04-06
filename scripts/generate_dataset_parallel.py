#!/usr/bin/env python3
"""FAST parallel dataset generation using multiprocessing.

Key optimization: Generate all (layout, behavior, episode) combinations in parallel,
then sort deterministically for reproducibility.

Expected speedup: ~8-16x on multi-core CPU (L4 VM has strong CPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import run_episode
from src.envs import make_env
from src.planner import BehaviorConfig, CostConfig, PlannerConfig
from src.utils import ensure_dir, load_config, save_json, seed_numpy
from src.viz import plot_action_distribution, plot_episode_lengths, plot_success_rate, save_video


def assemble_dataset(episodes: List[Dict], fixed_train_layouts: List[str]) -> Tuple[Dict, Dict, List[int], List[bool], List[int]]:
    all_s_grid = []
    all_s_agent_pos = []
    all_s_agent_dir = []
    all_s_carry = []
    all_a = []
    all_r = []
    all_sp_grid = []
    all_sp_agent_pos = []
    all_sp_agent_dir = []
    all_sp_carry = []
    all_done = []
    all_episode_id = []
    all_planner_type = []

    episode_metadata = []
    episode_lengths = []
    episode_successes = []
    all_actions = []

    for episode_id, episode_data in enumerate(episodes):
        num_transitions = len(episode_data["actions"])

        all_s_grid.append(episode_data["s_grid"])
        all_s_agent_pos.append(episode_data["s_agent_pos"])
        all_s_agent_dir.append(episode_data["s_agent_dir"])
        all_s_carry.append(episode_data["s_carry"])
        all_a.append(episode_data["actions"])
        all_r.append(episode_data["rewards"])
        all_sp_grid.append(episode_data["sp_grid"])
        all_sp_agent_pos.append(episode_data["sp_agent_pos"])
        all_sp_agent_dir.append(episode_data["sp_agent_dir"])
        all_sp_carry.append(episode_data["sp_carry"])
        all_done.append(episode_data["dones"])
        all_episode_id.append(np.full(num_transitions, episode_id, dtype=np.int32))
        all_planner_type.extend([episode_data["behavior_name"]] * num_transitions)

        episode_metadata.append({
            "episode_id": episode_id,
            "seed": int(episode_data["seed"]),
            "planner_type": episode_data["behavior_name"],
            "length": num_transitions,
            "success": bool(episode_data["success"]),
            "layout_name": episode_data["layout_name"],
            "split": "train" if episode_data["layout_name"] in fixed_train_layouts else "eval",
            "goal_region_id": int(episode_data.get("goal_region_id", 0)),
        })

        episode_lengths.append(num_transitions)
        episode_successes.append(episode_data["success"])
        all_actions.extend(episode_data["actions"].tolist())

    dataset = {
        "s_grid": np.concatenate(all_s_grid, axis=0),
        "s_agent_pos": np.concatenate(all_s_agent_pos, axis=0),
        "s_agent_dir": np.concatenate(all_s_agent_dir, axis=0),
        "s_carry": np.concatenate(all_s_carry, axis=0),
        "a": np.concatenate(all_a, axis=0),
        "r": np.concatenate(all_r, axis=0),
        "sp_grid": np.concatenate(all_sp_grid, axis=0),
        "sp_agent_pos": np.concatenate(all_sp_agent_pos, axis=0),
        "sp_agent_dir": np.concatenate(all_sp_agent_dir, axis=0),
        "sp_carry": np.concatenate(all_sp_carry, axis=0),
        "done": np.concatenate(all_done, axis=0),
        "episode_id": np.concatenate(all_episode_id, axis=0),
        "planner_type": np.array(all_planner_type, dtype=object),
    }

    metadata = {
        "transitions": int(len(dataset["a"])),
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "episodes": episode_metadata,
    }
    return dataset, metadata, episode_lengths, episode_successes, all_actions


def run_diagnostics(metadata_path: str, out_dir: str, title: str) -> None:
    import runpy

    script = ROOT / "scripts" / "analyze_behavior_dataset.py"
    argv = [
        str(script),
        "--metadata",
        metadata_path,
        "--out-dir",
        out_dir,
        "--title",
        title,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def make_behavior_cfg(cfg: Dict, name: str) -> BehaviorConfig:
    bcfg = cfg["behaviors"][name]
    return BehaviorConfig(
        random_action_prob=bcfg["random_action_prob"],
        heuristic_noise=bcfg["heuristic_noise"],
        cost_noise_std=bcfg["cost_noise_std"],
    )


def generate_single_episode(args: Tuple) -> Dict:
    """Generate a single episode (worker function for multiprocessing).

    Args:
        args: (layout_idx, behavior_idx, episode_idx, layout_name, behavior_name,
               env_cfg, base_costs, planner_cfg, behavior_cfg, seed, max_attempts, max_steps)

    Returns:
        Episode data dict
    """
    (
        layout_idx,
        behavior_idx,
        episode_idx,
        layout_name,
        behavior_name,
        env_cfg,
        base_costs,
        planner_cfg,
        behavior_cfg,
        seed,
        max_attempts,
        max_steps,
    ) = args

    # Create RNG
    rng = np.random.default_rng(seed)

    # Run episode with retries
    for attempt in range(max_attempts):
        try:
            # Create fresh environment for each attempt
            env = make_env(env_cfg, render_mode=None, layout_name=layout_name)
            env_seed = int(rng.integers(0, 2**31))
            env.reset(seed=env_seed)

            result = run_episode(
                env=env,
                base_costs=base_costs,
                base_planner=planner_cfg,
                behavior=behavior_cfg,
                rng=rng,
                max_steps=max_steps,
            )

            if result is None or not result.transitions:
                continue

            transitions = result.transitions
            s_grid = np.stack([t[0].grid for t in transitions]).astype(np.uint8)
            s_agent_pos = np.stack([t[0].agent_pos for t in transitions]).astype(np.int16)
            s_agent_dir = np.array([t[0].agent_dir for t in transitions], dtype=np.int8)
            s_carry = np.stack([t[0].carrying for t in transitions]).astype(np.int16)

            sp_grid = np.stack([t[3].grid for t in transitions]).astype(np.uint8)
            sp_agent_pos = np.stack([t[3].agent_pos for t in transitions]).astype(np.int16)
            sp_agent_dir = np.array([t[3].agent_dir for t in transitions], dtype=np.int8)
            sp_carry = np.stack([t[3].carrying for t in transitions]).astype(np.int16)

            actions = np.array([t[1] for t in transitions], dtype=np.int8)
            rewards = np.array([t[2] for t in transitions], dtype=np.float32)
            dones = np.array([t[4] for t in transitions], dtype=np.bool_)

            episode_data = {
                "s_grid": s_grid,
                "s_agent_pos": s_agent_pos,
                "s_agent_dir": s_agent_dir,
                "s_carry": s_carry,
                "actions": actions,
                "rewards": rewards,
                "sp_grid": sp_grid,
                "sp_agent_pos": sp_agent_pos,
                "sp_agent_dir": sp_agent_dir,
                "sp_carry": sp_carry,
                "dones": dones,
                "success": result.success,
                "seed": seed,
                "env_seed": env_seed,
                "layout_idx": layout_idx,
                "behavior_idx": behavior_idx,
                "episode_idx": episode_idx,
                "layout_name": layout_name,
                "behavior_name": behavior_name,
                "goal_region_id": int(getattr(env, "goal_region_id", 0)),
            }

            return episode_data

        except Exception as e:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Failed after {max_attempts} attempts: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Fast parallel dataset generation")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (default: CPU count)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Setup
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "plots"))
    ensure_dir(os.path.join(output_dir, "videos"))
    ensure_dir(os.path.join(output_dir, "diagnostics"))

    dataset_cfg = cfg["dataset"]
    episode_videos_cfg = cfg.get("episode_videos", {"enabled": False})
    env_cfg = cfg["env"]

    # Planner and costs
    planner_cfg = PlannerConfig(
        algorithm=cfg["planner"]["algorithm"],
        max_nodes=cfg["planner"]["max_nodes"],
        heuristic_scale_range=tuple(cfg["planner"]["heuristic_scale_range"]),
        allow_lava=cfg["planner"]["allow_lava"],
        max_time_s=cfg["planner"]["max_time_s"],
    )

    base_costs = {
        "step": cfg["costs"]["step"],
        "turn": cfg["costs"]["turn"],
        "toggle": cfg["costs"]["toggle"],
        "lava_penalty": cfg["costs"]["lava_penalty"],
        "noise_std": cfg["costs"]["noise_std"],
    }

    # Get fixed layouts and behaviors
    fixed_train_layouts = dataset_cfg.get("fixed_layouts_train", [])
    fixed_eval_layouts = dataset_cfg.get("fixed_layouts_eval", [])
    all_layouts = fixed_train_layouts + fixed_eval_layouts

    behavior_order = dataset_cfg.get("fixed_behavior_order", [])
    episodes_per_behavior = dataset_cfg.get("episodes_per_behavior", 50)
    min_transitions = int(dataset_cfg.get("min_transitions", 0))
    avg_steps_target = dataset_cfg.get("avg_steps_target", [50, 150])
    max_attempts = dataset_cfg.get("max_env_attempts", 25)
    max_steps = env_cfg.get("max_steps", 500)

    # Generate episodes in rounds until we meet min_transitions
    print("Building episode task list...")
    base_seed = cfg.get("seed", 42)
    episodes = []
    total_transitions = 0
    round_idx = 0

    def build_tasks(episodes_per_behavior_local: int, offset: int):
        task_list = []
        for layout_idx, layout_name in enumerate(all_layouts):
            for behavior_idx, behavior_name in enumerate(behavior_order):
                behavior_cfg = make_behavior_cfg(cfg, behavior_name)
                for episode_idx in range(episodes_per_behavior_local):
                    seed = base_seed + layout_idx * 10000 + behavior_idx * 1000 + (offset + episode_idx)
                    task_list.append((
                        layout_idx,
                        behavior_idx,
                        offset + episode_idx,
                        layout_name,
                        behavior_name,
                        env_cfg,
                        base_costs,
                        planner_cfg,
                        behavior_cfg,
                        seed,
                        max_attempts,
                        max_steps,
                    ))
        return task_list

    if min_transitions > 0:
        if isinstance(avg_steps_target, (list, tuple)) and len(avg_steps_target) >= 2:
            expected_steps = float(avg_steps_target[0] + avg_steps_target[1]) / 2.0
        else:
            expected_steps = float(avg_steps_target) if avg_steps_target else 100.0
        expected_steps = max(expected_steps, 1.0)
        total_episodes_needed = int(np.ceil(min_transitions / expected_steps))
        per_behavior = int(np.ceil(total_episodes_needed / max(1, len(all_layouts) * len(behavior_order))))
        episodes_per_behavior = max(1, per_behavior)
        print(
            f"Derived episodes_per_behavior={episodes_per_behavior} "
            f"from min_transitions={min_transitions} and expected_steps={expected_steps:.1f}"
        )

    num_workers = args.workers or cpu_count()
    print(f"\nUsing {num_workers} workers for parallel generation...")

    checkpoint_cfg = cfg.get("checkpoint", {})
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", False))
    checkpoint_factor = float(checkpoint_cfg.get("factor", 2.0))
    checkpoint_title = checkpoint_cfg.get("title", "Behavior Dataset Diagnostics (Checkpoint)")
    if checkpoint_enabled:
        if min_transitions > 0:
            start_default = max(1000, int(min_transitions // 10))
        else:
            start_default = 1000
        next_checkpoint = int(checkpoint_cfg.get("start_transitions", start_default))
    else:
        next_checkpoint = 0

    while min_transitions <= 0 or total_transitions < min_transitions:
        offset = round_idx * episodes_per_behavior
        tasks = build_tasks(episodes_per_behavior, offset)
        print(f"Round {round_idx + 1}: {len(tasks)} episodes")

        with Pool(processes=num_workers) as pool:
            for episode_data in tqdm(
                pool.imap_unordered(generate_single_episode, tasks),
                total=len(tasks),
                desc=f"Generating episodes (round {round_idx + 1})",
            ):
                episodes.append(episode_data)
                total_transitions += int(len(episode_data["actions"]))

        print(f"Collected {total_transitions:,} transitions so far")
        if checkpoint_enabled and total_transitions >= next_checkpoint:
            print(f"[checkpoint] Writing snapshot at {total_transitions:,} transitions")
            dataset, metadata, episode_lengths, episode_successes, all_actions = assemble_dataset(
                episodes, fixed_train_layouts
            )
            metadata["config"] = cfg
            np.savez_compressed(os.path.join(output_dir, "dataset.npz"), **dataset)
            save_json(os.path.join(output_dir, "metadata.json"), metadata)
            if checkpoint_cfg.get("diagnostics", True):
                run_diagnostics(
                    os.path.join(output_dir, "metadata.json"),
                    os.path.join(output_dir, "diagnostics"),
                    checkpoint_title,
                )
            next_checkpoint = int(max(next_checkpoint + 1, next_checkpoint * checkpoint_factor))
        round_idx += 1
        if min_transitions <= 0:
            break

    print(f"\nGenerated {len(episodes)} episodes successfully!")

    # Sort episodes for reproducibility (layout, behavior, episode order)
    print("Sorting episodes...")
    episodes.sort(key=lambda ep: (ep["layout_idx"], ep["behavior_idx"], ep["episode_idx"]))

    # Combine into dataset
    print("Combining into dataset...")
    dataset, metadata, episode_lengths, episode_successes, all_actions = assemble_dataset(
        episodes, fixed_train_layouts
    )

    total_transitions = len(dataset["a"])
    success_rate = np.mean(episode_successes)

    print(f"\nDataset complete:")
    print(f"  - Episodes: {len(episodes):,}")
    print(f"  - Transitions: {total_transitions:,}")
    print(f"  - Success rate: {success_rate:.2%}")
    print(f"  - Avg episode length: {np.mean(episode_lengths):.1f}")

    # Save dataset
    print("\nSaving dataset...")
    np.savez_compressed(
        os.path.join(output_dir, "dataset.npz"),
        **dataset,
    )

    # Save metadata
    metadata["config"] = cfg
    save_json(os.path.join(output_dir, "metadata.json"), metadata)

    # Optional: save episode videos (post-processing)
    if episode_videos_cfg.get("enabled", False):
        max_videos = int(episode_videos_cfg.get("max_episodes", 0))
        video_fps = int(episode_videos_cfg.get("fps", 10))
        if max_videos > 0:
            print("\nSaving episode videos...")
            for idx, episode_data in enumerate(episodes[:max_videos]):
                layout_name = episode_data["layout_name"]
                behavior_name = episode_data["behavior_name"]
                env_seed = int(episode_data.get("env_seed", episode_data["seed"]))
                env = make_env(env_cfg, render_mode="rgb_array", layout_name=layout_name)
                env.reset(seed=env_seed)
                frames = []
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                for action in episode_data["actions"].tolist():
                    _, _, terminated, truncated, _ = env.step(int(action))
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    if terminated or truncated:
                        break
                video_path = os.path.join(
                    output_dir,
                    "videos",
                    f"episode_{idx:04d}_{layout_name}_{behavior_name}.mp4",
                )
                save_video(frames, video_path, fps=video_fps)
                print(f"[video] saved {video_path}")

    # Plots
    if cfg.get("plots", {}).get("enabled", False):
        print("Generating plots...")
        plot_episode_lengths(episode_lengths, os.path.join(output_dir, "plots", "episode_lengths.png"))
        plot_action_distribution(all_actions, os.path.join(output_dir, "plots", "action_distribution.png"))
        plot_success_rate(episode_successes, os.path.join(output_dir, "plots", "success_rate.png"))

    # Episode videos
    episode_videos_cfg = cfg.get("episode_videos", {})
    if episode_videos_cfg.get("enabled", False):
        max_video_episodes = episode_videos_cfg.get("max_episodes", 100)
        video_fps = episode_videos_cfg.get("fps", 10)

        ensure_dir(os.path.join(output_dir, "episode_videos"))

        print(f"\nGenerating episode videos (first {max_video_episodes} episodes)...")
        for idx in tqdm(range(min(max_video_episodes, len(episodes))), desc="Rendering videos"):
            episode_data = episodes[idx]

            # Replay episode to get frames
            env = make_env(env_cfg, render_mode="rgb_array", layout_name=episode_data["layout_name"])
            env.reset(seed=episode_data["seed"])

            frames = []
            for action in episode_data["actions"]:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                env.step(int(action))

            # Final frame
            final_frame = env.render()
            if final_frame is not None:
                frames.append(final_frame)

            # Save video
            video_path = os.path.join(
                output_dir,
                "episode_videos",
                f"episode_{idx:04d}_{episode_data['layout_name']}_{episode_data['behavior_name']}.mp4"
            )
            if frames:
                save_video(frames, video_path, fps=video_fps)

    print(f"\nDataset saved to: {output_dir}")
    print("✓ Done!")


if __name__ == "__main__":
    main()
