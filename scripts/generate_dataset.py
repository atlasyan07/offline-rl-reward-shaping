#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import run_episode
from src.envs_harder import make_env
from src.planner import ACTIONS, BehaviorConfig, CostConfig, PlannerConfig, plan_actions
from src.utils import ensure_dir, load_config, save_json, seed_numpy
from src.viz import (
    plot_action_distribution,
    plot_episode_lengths,
    plot_success_rate,
    render_path_overlay,
    save_video,
)


def _write_dataset(output_dir, transitions, total_transitions, episode_metadata, episode_successes, cfg, suffix=""):
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
    episode_ids = np.array([t[5] for t in transitions], dtype=np.int32)
    planner_types = np.array([t[6] for t in transitions])

    dataset_name = f"dataset{suffix}.npz"
    dataset_path = os.path.join(output_dir, dataset_name)
    np.savez_compressed(
        dataset_path,
        s_grid=s_grid,
        s_agent_pos=s_agent_pos,
        s_agent_dir=s_agent_dir,
        s_carry=s_carry,
        a=actions,
        r=rewards,
        sp_grid=sp_grid,
        sp_agent_pos=sp_agent_pos,
        sp_agent_dir=sp_agent_dir,
        sp_carry=sp_carry,
        done=dones,
        episode_id=episode_ids,
        planner_type=planner_types,
    )

    save_json(
        os.path.join(output_dir, f"metadata{suffix}.json"),
        {
            "config": cfg,
            "episodes": episode_metadata,
            "transitions": int(total_transitions),
            "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline dataset for MiniGrid MultiRoom")
    parser.add_argument("--config", required=True, help="Path to section1 config yaml")
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


def _build_cost_cfg(base_costs: Dict, behavior_cfg: BehaviorConfig) -> CostConfig:
    return CostConfig(
        step=base_costs["step"],
        turn=base_costs["turn"],
        toggle=base_costs["toggle"],
        lava_penalty=base_costs["lava_penalty"],
        noise_std=behavior_cfg.cost_noise_std,
    )


def _build_deterministic_cost_cfg(base_costs: Dict) -> CostConfig:
    return CostConfig(
        step=base_costs["step"],
        turn=base_costs["turn"],
        toggle=base_costs["toggle"],
        lava_penalty=base_costs["lava_penalty"],
        noise_std=0.0,
    )

def _build_planner_cfg(base_planner: PlannerConfig, behavior_cfg: BehaviorConfig) -> PlannerConfig:
    base_min, base_max = base_planner.heuristic_scale_range
    scale_min = max(0.0, base_min * (1.0 - behavior_cfg.heuristic_noise))
    scale_max = base_max * (1.0 + behavior_cfg.heuristic_noise)
    return PlannerConfig(
        algorithm=base_planner.algorithm,
        max_nodes=base_planner.max_nodes,
        heuristic_scale_range=(scale_min, scale_max),
        allow_lava=base_planner.allow_lava,
    )


def create_video_episode(
    env_cfg: Dict,
    base_costs: Dict,
    planner_cfg: PlannerConfig,
    behavior_cfg: BehaviorConfig,
    rng: np.random.Generator,
    max_steps: int,
    fps: int,
    path: str,
    layout_name: str | None = None,
) -> None:
    env = make_env(env_cfg, render_mode="rgb_array", layout_name=layout_name)
    env.reset(seed=int(rng.integers(0, 1_000_000)))

    frames: List[np.ndarray] = []
    done = False
    steps = 0
    plan = []
    behavior_planner_cfg = _build_planner_cfg(planner_cfg, behavior_cfg)

    while not done and steps < max_steps:
        if not plan:
            from src.planner import plan_actions

            plan = plan_actions(env, _build_cost_cfg(base_costs, behavior_cfg), behavior_planner_cfg, rng) or []
            if not plan:
                break

        if rng.random() < behavior_cfg.random_action_prob:
            action = int(rng.choice([int(a) for a in ACTIONS]))
        else:
            action = int(plan.pop(0))

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        _, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated)
        steps += 1
        if truncated:
            break

    if frames:
        save_video(frames, path, fps=fps)


def _pick_behavior(rng: np.random.Generator, mix: Dict[str, float]) -> str:
    names = list(mix.keys())
    probs = np.array(list(mix.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def _make_behavior_cfg(behaviors_cfg: Dict, name: str) -> BehaviorConfig:
    bcfg = behaviors_cfg[name]
    return BehaviorConfig(
        random_action_prob=bcfg["random_action_prob"],
        heuristic_noise=bcfg["heuristic_noise"],
        cost_noise_std=bcfg["cost_noise_std"],
    )


def _generate_batch(args: Tuple) -> List[Tuple]:
    (
        env_cfg,
        base_costs,
        planner_cfg_dict,
        behaviors_cfg,
        mix,
        avg_steps_target,
        max_env_attempts,
        seed,
        episodes_per_worker,
    ) = args

    rng = np.random.default_rng(seed)
    planner_cfg = PlannerConfig(**planner_cfg_dict)
    results: List[Tuple] = []

    for _ in range(episodes_per_worker):
        env, env_seed = build_solvable_env(env_cfg, base_costs, planner_cfg, rng, max_attempts=max_env_attempts)
        if env is None or env_seed is None:
            continue

        behavior_name = _pick_behavior(rng, mix)
        behavior_cfg = _make_behavior_cfg(behaviors_cfg, behavior_name)

        target_min, target_max = avg_steps_target
        sampled_max = int(rng.integers(target_min, target_max + 1))

        result = run_episode(
            env,
            base_costs=base_costs,
            base_planner=planner_cfg,
            behavior=behavior_cfg,
            rng=rng,
            max_steps=min(env.max_steps, sampled_max),
        )

        if result is None or not result.transitions:
            continue

        results.append(
            (
                result.transitions,
                env_seed,
                behavior_name,
                bool(result.success),
            )
        )

    return results


def _generate_fixed_layout_batch(args: Tuple) -> Dict[str, object]:
    """Generate episodes for a specific (layout, behavior) combination."""
    (
        env_cfg,
        base_costs,
        planner_cfg_dict,
        behaviors_cfg,
        layout_name,
        behavior_name,
        avg_steps_target,
        seed,
        num_episodes,
    ) = args

    rng = np.random.default_rng(seed)
    planner_cfg = PlannerConfig(**planner_cfg_dict)
    behavior_cfg = _make_behavior_cfg(behaviors_cfg, behavior_name)
    results: List[Tuple] = []
    fail_counts: Dict[str, int] = {"no_plan": 0, "empty_transitions": 0}

    for _ in range(num_episodes):
        env_seed = int(rng.integers(0, 1_000_000))
        env = make_env(env_cfg, layout_name=layout_name)
        env.reset(seed=env_seed)

        target_min, target_max = avg_steps_target
        sampled_max = int(rng.integers(target_min, target_max + 1))

        result = run_episode(
            env,
            base_costs=base_costs,
            base_planner=planner_cfg,
            behavior=behavior_cfg,
            rng=rng,
            max_steps=min(env.max_steps, sampled_max),
        )

        if result is None:
            fail_counts["no_plan"] += 1
            continue
        if not result.transitions:
            fail_counts["empty_transitions"] += 1
            continue

        goal_region_id = getattr(env, "goal_region_id", None)
        results.append(
            (
                result.transitions,
                env_seed,
                behavior_name,
                bool(result.success),
                layout_name,
                goal_region_id,
                result.actions,  # For video replay
            )
        )

    return {"results": results, "fail_counts": fail_counts}


def _save_episode_video(env_cfg: Dict, seed: int, actions: List[int], path: str, fps: int, layout_name: str | None = None) -> None:
    env = make_env(env_cfg, render_mode="rgb_array", layout_name=layout_name)
    env.reset(seed=seed)
    frames: List[np.ndarray] = []
    for action in actions:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    if frames:
        save_video(frames, path, fps=fps)

def build_solvable_env(
    env_cfg: Dict,
    base_costs: Dict,
    planner_cfg: PlannerConfig,
    rng: np.random.Generator,
    max_attempts: int = 25,
) -> Tuple[Optional[object], Optional[int]]:
    for _ in range(max_attempts):
        env = make_env(env_cfg)
        seed = int(rng.integers(0, 1_000_000))
        env.reset(seed=seed)
        plan = plan_actions(env, _build_deterministic_cost_cfg(base_costs), planner_cfg, rng)
        if plan:
            return env, seed
    return None, None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "plots"))
    ensure_dir(os.path.join(output_dir, "videos"))
    ensure_dir(os.path.join(output_dir, "paths"))

    rng = seed_numpy(cfg["seed"])

    env_cfg = cfg["env"]
    base_costs = cfg["costs"]
    planner_cfg = PlannerConfig(**cfg["planner"])
    dataset_cfg = cfg["dataset"]
    episode_videos_cfg = cfg.get("episode_videos", {"enabled": False})

    transitions = []
    episode_metadata = []
    all_actions = []
    episode_lengths = []
    episode_successes = []

    total_transitions = 0
    episode_id = 0

    workers = int(dataset_cfg.get("workers", cpu_count() or 1))
    episodes_per_worker = int(dataset_cfg.get("episodes_per_worker", 2))
    max_attempts = int(dataset_cfg.get("max_env_attempts", 25))
    fixed_train_seeds = dataset_cfg.get("fixed_seeds_train", [])
    fixed_eval_seeds = dataset_cfg.get("fixed_seeds_eval", [])
    fixed_train_layouts = dataset_cfg.get("fixed_layouts_train", [])
    fixed_eval_layouts = dataset_cfg.get("fixed_layouts_eval", [])
    episodes_per_seed = int(dataset_cfg.get("episodes_per_seed", 1))
    episodes_per_layout = int(dataset_cfg.get("episodes_per_layout", 1))
    behavior_order = dataset_cfg.get("fixed_behavior_order", [])
    episodes_per_behavior = int(dataset_cfg.get("episodes_per_behavior", 1))
    use_fixed_seeds = bool(fixed_train_seeds or fixed_eval_seeds)
    use_fixed_layouts = bool(fixed_train_layouts or fixed_eval_layouts)

    checkpoint_every_transitions = int(dataset_cfg.get("checkpoint_every_transitions", 0))
    checkpoint_every_seconds = float(dataset_cfg.get("checkpoint_every_seconds", 0.0))
    checkpoint_dir = None
    next_checkpoint = None
    last_checkpoint_time = time.time()
    if checkpoint_every_transitions > 0 or checkpoint_every_seconds > 0:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        ensure_dir(checkpoint_dir)
        if checkpoint_every_transitions > 0:
            next_checkpoint = checkpoint_every_transitions

    def maybe_checkpoint() -> None:
        nonlocal next_checkpoint, last_checkpoint_time
        if checkpoint_dir is None:
            return
        now = time.time()
        hit_transitions = next_checkpoint is not None and total_transitions >= next_checkpoint
        hit_time = checkpoint_every_seconds > 0 and (now - last_checkpoint_time) >= checkpoint_every_seconds
        if not hit_transitions and not hit_time:
            return
        suffix = f"_{total_transitions}"
        _write_dataset(checkpoint_dir, transitions, total_transitions, episode_metadata, episode_successes, cfg, suffix)
        last_checkpoint_time = now
        if next_checkpoint is not None:
            while total_transitions >= next_checkpoint:
                next_checkpoint += checkpoint_every_transitions
        print(f"[checkpoint] saved checkpoints{suffix}.npz")

    pbar = tqdm(total=dataset_cfg["min_transitions"], desc="Generating transitions")
    heartbeat_seconds = float(dataset_cfg.get("heartbeat_seconds", 30))
    last_heartbeat = time.time()
    completed_work_items = 0
    last_batch_transitions = 0
    fail_counts_total = {"no_plan": 0, "empty_transitions": 0}
    fail_counts_since = {"no_plan": 0, "empty_transitions": 0}
    consecutive_failures = 0
    ran_parallel_fixed_layouts = False

    # Parallel fixed-layout generation
    if use_fixed_layouts and workers > 1:
        print(f"[info] fixed layouts with {workers} workers (parallel generation)")
        planner_cfg_dict = cfg["planner"]
        behaviors_cfg = cfg["behaviors"]
        planner_mix = dataset_cfg.get("planner_mix") or {name: 1.0 for name in behaviors_cfg.keys()}
        avg_steps_target = dataset_cfg["avg_steps_target"]

        ordered_layouts = [str(n) for n in fixed_train_layouts] + [str(n) for n in fixed_eval_layouts]
        layout_split = {str(n): "train" for n in fixed_train_layouts}
        layout_split.update({str(n): "eval" for n in fixed_eval_layouts})

        if behavior_order:
            behavior_names = behavior_order
            repeats = episodes_per_behavior
        else:
            behavior_names = list(cfg["behaviors"].keys())
            repeats = episodes_per_layout

        # Build work items: chunk into smaller batches for progress visibility
        # Each work item processes `chunk_size` episodes (default 10)
        chunk_size = int(dataset_cfg.get("chunk_size", 10))
        work_items = []
        for layout_name in ordered_layouts:
            for behavior_name in behavior_names:
                # Split repeats into chunks
                remaining = repeats
                while remaining > 0:
                    batch_size = min(chunk_size, remaining)
                    seed = int(rng.integers(0, 1_000_000_000))
                    work_items.append((
                        env_cfg,
                        base_costs,
                        planner_cfg_dict,
                        behaviors_cfg,
                        layout_name,
                        behavior_name,
                        avg_steps_target,
                        seed,
                        batch_size,
                    ))
                    remaining -= batch_size

        print(f"[info] dispatching {len(work_items)} work items ({chunk_size} eps/chunk) to {workers} workers")

        video_enabled = bool(episode_videos_cfg.get("enabled", False))
        video_fps = int(episode_videos_cfg.get("fps", 10))
        recorded_videos: set[tuple[str, str]] = set()

        with Pool(processes=workers) as pool:
            for batch in pool.imap_unordered(_generate_fixed_layout_batch, work_items):
                if not batch or not batch.get("results"):
                    consecutive_failures += 1
                    if consecutive_failures % 10 == 0:
                        print(f"[warn] {consecutive_failures} empty batches")
                    continue
                consecutive_failures = 0
                completed_work_items += 1

                batch_fail_counts = batch.get("fail_counts") or {}
                for key, value in batch_fail_counts.items():
                    if key not in fail_counts_total:
                        fail_counts_total[key] = 0
                        fail_counts_since[key] = 0
                    fail_counts_total[key] += int(value)
                    fail_counts_since[key] += int(value)

                for item in batch["results"]:
                    transitions_batch, seed, behavior_name, success, layout_name, goal_region_id, actions_list = item

                    for (s, a, r, sp, done) in transitions_batch:
                        transitions.append((s, a, r, sp, done, episode_id, behavior_name))
                        all_actions.append(a)

                    episode_lengths.append(len(transitions_batch))
                    episode_successes.append(success)
                    episode_metadata.append({
                        "episode_id": episode_id,
                        "seed": seed,
                        "planner_type": behavior_name,
                        "length": len(transitions_batch),
                        "success": bool(success),
                        "layout_name": layout_name,
                        "split": layout_split.get(layout_name, "train"),
                        "goal_region_id": goal_region_id,
                    })

                    # Record one video per (layout, behavior) combo
                    if video_enabled:
                        video_key = (layout_name, behavior_name)
                        if video_key not in recorded_videos:
                            video_path = os.path.join(output_dir, "videos", f"{layout_name}_{behavior_name}.mp4")
                            _save_episode_video(env_cfg, seed, actions_list, video_path, fps=video_fps, layout_name=layout_name)
                            recorded_videos.add(video_key)
                            print(f"[video] saved {video_path}")

                    total_transitions += len(transitions_batch)
                    last_batch_transitions += len(transitions_batch)
                    episode_id += 1
                    pbar.update(len(transitions_batch))
                    maybe_checkpoint()

                if total_transitions >= dataset_cfg["min_transitions"] and episode_id >= dataset_cfg["min_episodes"]:
                    break
                now = time.time()
                if now - last_heartbeat >= heartbeat_seconds:
                    elapsed_min = (now - last_heartbeat) / 60.0
                    fail_summary = " ".join(
                        f"{k}={fail_counts_since[k]}" for k in sorted(fail_counts_since)
                    )
                    print(
                        "[heartbeat] elapsed={:.1f}m items={}/{} transitions={} last_batch={} fails={}".format(
                            elapsed_min,
                            completed_work_items,
                            len(work_items),
                            total_transitions,
                            last_batch_transitions,
                            fail_summary,
                        )
                    )
                    last_heartbeat = now
                    last_batch_transitions = 0
                    for key in fail_counts_since:
                        fail_counts_since[key] = 0

        ran_parallel_fixed_layouts = True

    elif use_fixed_seeds and workers > 1:
        print("[info] fixed seeds enabled; forcing single-worker generation")
        workers = 1
        # Fall through to single-worker generation below

    # Single-worker generation (fixed seeds, fixed layouts with workers=1, or random envs)
    if not ran_parallel_fixed_layouts:
        video_enabled = bool(episode_videos_cfg.get("enabled", False))
        video_max = int(episode_videos_cfg.get("max_episodes", 0))
        video_fps = int(episode_videos_cfg.get("fps", 10))
        # Track which (layout, behavior) combos have been recorded
        recorded_videos: set[tuple[str, str]] = set()
        if use_fixed_layouts:
            ordered_layouts = [str(n) for n in fixed_train_layouts] + [str(n) for n in fixed_eval_layouts]
            layout_split = {str(n): "train" for n in fixed_train_layouts}
            layout_split.update({str(n): "eval" for n in fixed_eval_layouts})

            for layout_name in ordered_layouts:
                if total_transitions >= dataset_cfg["min_transitions"] and episode_id >= dataset_cfg["min_episodes"]:
                    break

                if behavior_order:
                    behavior_names = behavior_order
                    repeats = episodes_per_behavior
                else:
                    behavior_names = [None]
                    repeats = episodes_per_layout

                for behavior_name in behavior_names:
                    for _ in range(repeats):
                        env = make_env(env_cfg, render_mode=None, layout_name=layout_name)
                        seed = int(rng.integers(0, 1_000_000))
                        env.reset(seed=seed)

                        if behavior_name is None:
                            behavior_name = pick_behavior(rng, planner_mix)
                        behavior_cfg = make_behavior_cfg(cfg, behavior_name)

                        target_min, target_max = dataset_cfg["avg_steps_target"]
                        sampled_max = int(rng.integers(target_min, target_max + 1))
                        result = run_episode(
                            env,
                            base_costs=base_costs,
                            base_planner=planner_cfg,
                            behavior=behavior_cfg,
                            rng=rng,
                            max_steps=min(env.max_steps, sampled_max),
                        )

                        if result is None or not result.transitions:
                            consecutive_failures += 1
                            if consecutive_failures % 20 == 0:
                                print(f"[warn] {consecutive_failures} consecutive failed rollouts")
                            continue
                        consecutive_failures = 0

                        for (s, a, r, sp, done) in result.transitions:
                            transitions.append((s, a, r, sp, done, episode_id, behavior_name))
                            all_actions.append(a)

                        episode_lengths.append(len(result.transitions))
                        episode_successes.append(result.success)
                        episode_metadata.append(
                            {
                                "episode_id": episode_id,
                                "seed": seed,
                                "planner_type": behavior_name,
                                "length": len(result.transitions),
                                "success": bool(result.success),
                                "layout_name": layout_name,
                                "split": layout_split.get(layout_name, "train"),
                                "goal_region_id": getattr(env, "goal_region_id", None),
                            }
                        )

                        # Record one video per (layout, behavior) combo
                        video_key = (layout_name, behavior_name)
                        if video_enabled and video_key not in recorded_videos:
                            video_path = os.path.join(output_dir, "videos", f"{layout_name}_{behavior_name}.mp4")
                            _save_episode_video(env_cfg, seed, result.actions, video_path, fps=video_fps, layout_name=layout_name)
                            recorded_videos.add(video_key)
                            print(f"[video] saved {video_path}")

                        total_transitions += len(result.transitions)
                        episode_id += 1
                        pbar.update(len(result.transitions))
                        maybe_checkpoint()

                        if total_transitions >= dataset_cfg["min_transitions"] and episode_id >= dataset_cfg["min_episodes"]:
                            break
                    if total_transitions >= dataset_cfg["min_transitions"] and episode_id >= dataset_cfg["min_episodes"]:
                        break
        elif use_fixed_seeds:
            seed_plan_checked = set()
            seed_split = {int(s): "train" for s in fixed_train_seeds}
            seed_split.update({int(s): "eval" for s in fixed_eval_seeds})
            ordered_seeds = [int(s) for s in fixed_train_seeds] + [int(s) for s in fixed_eval_seeds]

            for seed in ordered_seeds:
                if total_transitions >= dataset_cfg["min_transitions"] and episode_id >= dataset_cfg["min_episodes"]:
                    break

                if seed not in seed_plan_checked:
                    env = make_env(env_cfg)
                    env.reset(seed=seed)
                    plan = plan_actions(env, _build_deterministic_cost_cfg(base_costs), planner_cfg, rng)
                    if not plan:
                        print(f"[warn] seed {seed} unsolvable; skipping")
                        seed_plan_checked.add(seed)
                        continue
                    seed_plan_checked.add(seed)

                for _ in range(episodes_per_seed):
                    env = make_env(env_cfg)
                    env.reset(seed=seed)

                    behavior_name = pick_behavior(rng, planner_mix)
                    behavior_cfg = make_behavior_cfg(cfg, behavior_name)

                    target_min, target_max = dataset_cfg["avg_steps_target"]
                    sampled_max = int(rng.integers(target_min, target_max + 1))
                    result = run_episode(
                        env,
                        base_costs=base_costs,
                        base_planner=planner_cfg,
                        behavior=behavior_cfg,
                        rng=rng,
                        max_steps=min(env.max_steps, sampled_max),
                    )

                    if result is None or not result.transitions:
                        consecutive_failures += 1
                        if consecutive_failures % 20 == 0:
                            print(f"[warn] {consecutive_failures} consecutive failed rollouts")
                        continue
                    consecutive_failures = 0

                    for (s, a, r, sp, done) in result.transitions:
                        transitions.append((s, a, r, sp, done, episode_id, behavior_name))
                        all_actions.append(a)

                    episode_lengths.append(len(result.transitions))
                    episode_successes.append(result.success)
                    episode_metadata.append(
                        {
                            "episode_id": episode_id,
                            "seed": seed,
                            "planner_type": behavior_name,
                            "length": len(result.transitions),
                            "success": bool(result.success),
                            "split": seed_split.get(int(seed), "train"),
                        }
                    )

                    if video_enabled and episode_id < video_max:
                        video_path = os.path.join(output_dir, "videos", f"episode_{episode_id}_{behavior_name}.mp4")
                        _save_episode_video(env_cfg, seed, result.actions, video_path, fps=video_fps)
                        print(f"[video] saved {video_path}")

                    total_transitions += len(result.transitions)
                    episode_id += 1
                    pbar.update(len(result.transitions))
                    maybe_checkpoint()

                    if total_transitions >= dataset_cfg["min_transitions"] and episode_id >= dataset_cfg["min_episodes"]:
                        break
        else:
            while total_transitions < dataset_cfg["min_transitions"] or episode_id < dataset_cfg["min_episodes"]:
                env, seed = build_solvable_env(env_cfg, base_costs, planner_cfg, rng, max_attempts=max_attempts)
                if env is None or seed is None:
                    continue

                behavior_name = pick_behavior(rng, planner_mix)
                behavior_cfg = make_behavior_cfg(cfg, behavior_name)

                target_min, target_max = dataset_cfg["avg_steps_target"]
                sampled_max = int(rng.integers(target_min, target_max + 1))
                result = run_episode(
                    env,
                    base_costs=base_costs,
                    base_planner=planner_cfg,
                    behavior=behavior_cfg,
                    rng=rng,
                    max_steps=min(env.max_steps, sampled_max),
                )

                if result is None or not result.transitions:
                    consecutive_failures += 1
                    if consecutive_failures % 20 == 0:
                        print(f"[warn] {consecutive_failures} consecutive failed rollouts")
                    continue
                consecutive_failures = 0

                for (s, a, r, sp, done) in result.transitions:
                    transitions.append((s, a, r, sp, done, episode_id, behavior_name))
                    all_actions.append(a)

                episode_lengths.append(len(result.transitions))
                episode_successes.append(result.success)
                episode_metadata.append(
                    {
                        "episode_id": episode_id,
                        "seed": seed,
                        "planner_type": behavior_name,
                        "length": len(result.transitions),
                        "success": bool(result.success),
                    }
                )

                if video_enabled and episode_id < video_max:
                    video_path = os.path.join(output_dir, "videos", f"episode_{episode_id}_{behavior_name}.mp4")
                    _save_episode_video(env_cfg, seed, result.actions, video_path, fps=video_fps)
                    print(f"[video] saved {video_path}")

                total_transitions += len(result.transitions)
                episode_id += 1
                pbar.update(len(result.transitions))
                maybe_checkpoint()
    else:
        planner_cfg_dict = cfg["planner"]
        mix = planner_mix
        avg_steps_target = dataset_cfg["avg_steps_target"]
        behaviors_cfg = cfg["behaviors"]

        with Pool(processes=workers) as pool:
            while total_transitions < dataset_cfg["min_transitions"] or episode_id < dataset_cfg["min_episodes"]:
                args_list = []
                for _ in range(workers):
                    seed = int(rng.integers(0, 1_000_000_000))
                    args_list.append(
                        (
                            env_cfg,
                            base_costs,
                            planner_cfg_dict,
                            behaviors_cfg,
                            mix,
                            avg_steps_target,
                            max_attempts,
                            seed,
                            episodes_per_worker,
                        )
                    )

                for batch in pool.imap_unordered(_generate_batch, args_list):
                    if not batch:
                        consecutive_failures += 1
                        if consecutive_failures % 10 == 0:
                            print(f"[warn] {consecutive_failures} empty batches")
                        continue
                    consecutive_failures = 0
                    for transitions_batch, seed, behavior_name, success in batch:
                        for (s, a, r, sp, done) in transitions_batch:
                            transitions.append((s, a, r, sp, done, episode_id, behavior_name))
                            all_actions.append(a)

                        episode_lengths.append(len(transitions_batch))
                        episode_successes.append(success)
                        episode_metadata.append(
                            {
                                "episode_id": episode_id,
                                "seed": seed,
                                "planner_type": behavior_name,
                                "length": len(transitions_batch),
                                "success": bool(success),
                            }
                        )

                        total_transitions += len(transitions_batch)
                        episode_id += 1
                        pbar.update(len(transitions_batch))
                        maybe_checkpoint()

                        if total_transitions >= dataset_cfg["min_transitions"] and episode_id >= dataset_cfg["min_episodes"]:
                            break
                    if total_transitions >= dataset_cfg["min_transitions"] and episode_id >= dataset_cfg["min_episodes"]:
                        break

    pbar.close()

    # Serialize final dataset
    _write_dataset(output_dir, transitions, total_transitions, episode_metadata, episode_successes, cfg)

    if cfg["plots"]["enabled"]:
        plot_episode_lengths(episode_lengths, os.path.join(output_dir, "plots", "episode_lengths.png"))
        plot_action_distribution(all_actions, os.path.join(output_dir, "plots", "action_distribution.png"))
        plot_success_rate(episode_successes, os.path.join(output_dir, "plots", "success_rate.png"))

    if cfg["videos"]["enabled"]:
        video_types = ["near_optimal", "noisy", "suboptimal"]
        # Use first training layout for example videos
        example_layout = fixed_train_layouts[0] if fixed_train_layouts else None
        for vt in video_types:
            behavior_cfg = make_behavior_cfg(cfg, vt)
            create_video_episode(
                env_cfg,
                base_costs,
                planner_cfg,
                behavior_cfg,
                rng,
                max_steps=cfg["videos"]["max_steps"],
                fps=cfg["videos"]["fps"],
                path=os.path.join(output_dir, "videos", f"{vt}.mp4"),
                layout_name=example_layout,
            )

    if cfg["paths"]["enabled"]:
        max_paths = cfg["paths"]["max_episodes"]
        for idx in range(min(max_paths, len(episode_metadata))):
            layout_name = episode_metadata[idx].get("layout_name")
            env = make_env(env_cfg, render_mode="rgb_array", layout_name=layout_name)
            env.reset(seed=episode_metadata[idx]["seed"])
            positions = []
            for transition in transitions:
                if transition[5] != idx:
                    continue
                positions.append(transition[0].agent_pos)
            frame = env.render()
            if frame is None:
                continue
            overlay = render_path_overlay(frame, positions, (env.grid.width, env.grid.height))
            overlay.save(os.path.join(output_dir, "paths", f"episode_{idx}.png"))


if __name__ == "__main__":
    main()
