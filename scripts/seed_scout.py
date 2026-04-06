#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minigrid.core.world_object import Door, Key, Lava

from src.envs import make_env
from src.planner import PlannerConfig, CostConfig, plan_actions
from src.utils import ensure_dir, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scout solvable MiniGrid seeds and rank by difficulty")
    parser.add_argument("--config", required=True, help="Path to section1 config yaml")
    parser.add_argument("--seed-start", type=int, default=0, help="Seed start (inclusive)")
    parser.add_argument("--seed-end", type=int, default=500, help="Seed end (exclusive)")
    parser.add_argument("--num-train", type=int, default=3, help="Number of train seeds")
    parser.add_argument("--num-eval", type=int, default=2, help="Number of eval seeds")
    parser.add_argument("--out-dir", default="outputs/seed_scout", help="Output directory")
    parser.add_argument("--max-time-s", type=float, default=1.0, help="Planner timeout per seed")
    parser.add_argument("--no-render", action="store_true", help="Skip saving grid images")
    return parser.parse_args()


def count_objects(env) -> Dict[str, int]:
    counts = {
        "doors": 0,
        "locked_doors": 0,
        "keys": 0,
        "lava": 0,
    }
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if obj is None:
                continue
            if isinstance(obj, Door):
                counts["doors"] += 1
                if obj.is_locked:
                    counts["locked_doors"] += 1
            elif isinstance(obj, Key):
                counts["keys"] += 1
            elif isinstance(obj, Lava):
                counts["lava"] += 1
    return counts


def score_difficulty(plan_len: int, counts: Dict[str, int], rooms: int) -> float:
    return (
        1.0 * plan_len
        + 8.0 * counts["locked_doors"]
        + 3.0 * counts["doors"]
        + 0.5 * counts["lava"]
        + 2.0 * rooms
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "images"))

    env_cfg = cfg["env"]
    planner_cfg = PlannerConfig(**cfg["planner"])
    planner_cfg.max_time_s = args.max_time_s
    base_costs = cfg["costs"]
    cost_cfg = CostConfig(
        step=base_costs["step"],
        turn=base_costs["turn"],
        toggle=base_costs["toggle"],
        lava_penalty=base_costs["lava_penalty"],
        noise_std=0.0,
    )

    results = []
    seed_range = range(args.seed_start, args.seed_end)
    for seed in tqdm(seed_range, desc="Scanning seeds"):
        env = make_env(env_cfg, render_mode=None if args.no_render else "rgb_array")
        env.reset(seed=seed)

        plan = plan_actions(env, cost_cfg, planner_cfg, np.random.default_rng(seed))
        if not plan:
            continue

        counts = count_objects(env)
        rooms = len(getattr(env, "rooms", []))
        plan_len = len(plan)
        score = score_difficulty(plan_len, counts, rooms)

        if not args.no_render:
            frame = env.render()
            if frame is not None:
                img = Image.fromarray(frame)
                img.save(os.path.join(out_dir, "images", f"seed_{seed}.png"))

        results.append(
            {
                "seed": seed,
                "plan_len": plan_len,
                "rooms": rooms,
                **counts,
                "score": float(score),
                "image": f"images/seed_{seed}.png",
            }
        )

    results.sort(key=lambda r: r["score"])
    train = results[: args.num_train]
    evals = results[-args.num_eval :] if args.num_eval > 0 else []

    summary = {
        "scanned": args.seed_end - args.seed_start,
        "solvable": len(results),
        "train_seeds": [r["seed"] for r in train],
        "eval_seeds": [r["seed"] for r in evals],
        "train_details": train,
        "eval_details": evals,
    }

    save_json(os.path.join(out_dir, "seed_scout.json"), summary)

    print("[seed_scout] train seeds:", summary["train_seeds"])
    print("[seed_scout] eval seeds:", summary["eval_seeds"])
    print("[seed_scout] details saved to", os.path.join(out_dir, "seed_scout.json"))


if __name__ == "__main__":
    main()
