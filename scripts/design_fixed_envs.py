#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minigrid.core.world_object import Door, Wall

from src.envs import make_env
from src.planner import CostConfig, PlannerConfig, plan_actions
from src.utils import ensure_dir, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design fixed solvable environments with controlled difficulty")
    parser.add_argument("--config", required=True, help="Path to section1 config yaml")
    parser.add_argument("--seed-start", type=int, default=0, help="Seed start (inclusive)")
    parser.add_argument("--max-tries", type=int, default=300, help="Max seeds to try")
    parser.add_argument("--num-envs", type=int, default=5, help="Number of environments to select")
    parser.add_argument("--max-rooms", type=int, default=4, help="Maximum rooms allowed")
    parser.add_argument("--max-lava", type=int, default=2, help="Maximum lava tiles per env")
    parser.add_argument("--out-dir", default="outputs/fixed_envs", help="Output directory")
    return parser.parse_args()


def room_bounds(room) -> Tuple[int, int, int, int]:
    x, y = room.top
    w, h = room.size
    return x, y, x + w - 1, y + h - 1


def find_adjacent_wall_opening(env) -> Optional[Tuple[int, int]]:
    rooms = getattr(env, "rooms", [])
    for i, r1 in enumerate(rooms):
        x1a, y1a, x1b, y1b = room_bounds(r1)
        for j, r2 in enumerate(rooms):
            if i >= j:
                continue
            x2a, y2a, x2b, y2b = room_bounds(r2)

            # r2 to the right of r1
            if x2a == x1b and not (y2b < y1a or y2a > y1b):
                ys = list(range(max(y1a + 1, y2a + 1), min(y1b, y2b)))
                if ys:
                    y = ys[len(ys) // 2]
                    return (x1b, y)
            # r2 to the left of r1
            if x1a == x2b and not (y2b < y1a or y2a > y1b):
                ys = list(range(max(y1a + 1, y2a + 1), min(y1b, y2b)))
                if ys:
                    y = ys[len(ys) // 2]
                    return (x1a, y)
            # r2 below r1
            if y2a == y1b and not (x2b < x1a or x2a > x1b):
                xs = list(range(max(x1a + 1, x2a + 1), min(x1b, x2b)))
                if xs:
                    x = xs[len(xs) // 2]
                    return (x, y1b)
            # r2 above r1
            if y1a == y2b and not (x2b < x1a or x2a > x1b):
                xs = list(range(max(x1a + 1, x2a + 1), min(x1b, x2b)))
                if xs:
                    x = xs[len(xs) // 2]
                    return (x, y1a)
    return None


def open_all_doors(env) -> None:
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if isinstance(obj, Door):
                obj.is_open = True
                obj.is_locked = False


def is_passable(env, pos: Tuple[int, int], ignore_lava: bool) -> bool:
    x, y = pos
    obj = env.grid.get(x, y)
    if obj is None:
        return True
    if isinstance(obj, Wall):
        return False
    if isinstance(obj, Door):
        return True
    return True


def shortest_path(env, start: Tuple[int, int], goal: Tuple[int, int], ignore_lava: bool) -> Optional[List[Tuple[int, int]]]:
    from collections import deque

    q = deque([start])
    came = {start: None}
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            break
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in came:
                continue
            if 0 <= nx < env.grid.width and 0 <= ny < env.grid.height:
                if is_passable(env, (nx, ny), ignore_lava=ignore_lava):
                    came[(nx, ny)] = (x, y)
                    q.append((nx, ny))

    if goal not in came:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path



def ensure_multiple_routes(env) -> bool:
    start = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)
    base_path = shortest_path(env, start, goal, ignore_lava=True)
    if not base_path or len(base_path) < 4:
        return False
    # Block a single tile on the shortest path to see if an alternative exists.
    for pos in base_path[1:-1]:
        x, y = pos
        obj = env.grid.get(x, y)
        if isinstance(obj, Wall):
            continue
        env.grid.set(x, y, Wall())
        alt = shortest_path(env, start, goal, ignore_lava=True)
        env.grid.set(x, y, obj)
        if alt:
            return True
    return False


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "images"))

    env_cfg = dict(cfg["env"])
    env_cfg["max_rooms"] = min(env_cfg.get("max_rooms", args.max_rooms), args.max_rooms)

    planner_cfg = PlannerConfig(**cfg["planner"])
    base_costs = cfg["costs"]
    cost_cfg = CostConfig(
        step=base_costs["step"],
        turn=base_costs["turn"],
        toggle=base_costs["toggle"],
        lava_penalty=base_costs["lava_penalty"],
        noise_std=0.0,
    )

    selected = []
    tries = 0
    for seed in tqdm(range(args.seed_start, args.seed_start + args.max_tries), desc="Designing envs"):
        if len(selected) >= args.num_envs:
            break
        tries += 1
        env = make_env(env_cfg, render_mode="rgb_array")
        env.reset(seed=seed)

        if len(getattr(env, "rooms", [])) > args.max_rooms:
            continue

        open_all_doors(env)

        opening = find_adjacent_wall_opening(env)
        if opening is None:
            continue
        if isinstance(env.grid.get(opening[0], opening[1]), Wall):
            env.grid.set(opening[0], opening[1], None)

        if not ensure_multiple_routes(env):
            continue

        # Ensure solvable by planner
        plan = plan_actions(env, cost_cfg, planner_cfg, np.random.default_rng(seed))
        if not plan:
            continue

        frame = env.render()
        if frame is not None:
            Image.fromarray(frame).save(os.path.join(out_dir, "images", f"seed_{seed}.png"))

        selected.append(
            {
                "seed": seed,
                "rooms": len(getattr(env, "rooms", [])),
                "plan_len": len(plan),
                "image": f"images/seed_{seed}.png",
            }
        )

    save_json(os.path.join(out_dir, "fixed_envs.json"), {"selected": selected, "tries": tries})
    print("[design_fixed_envs] selected seeds:", [s["seed"] for s in selected])
    print("[design_fixed_envs] details saved to", os.path.join(out_dir, "fixed_envs.json"))


if __name__ == "__main__":
    main()
