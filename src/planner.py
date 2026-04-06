from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
from minigrid.core.actions import Actions

from .utils import StateSnapshot, snapshot_state, state_key


@dataclass
class CostConfig:
    step: float
    turn: float
    toggle: float
    lava_penalty: float
    noise_std: float


@dataclass
class PlannerConfig:
    algorithm: str
    max_nodes: int
    heuristic_scale_range: Tuple[float, float]
    allow_lava: bool
    max_time_s: float | None = None


@dataclass
class BehaviorConfig:
    random_action_prob: float
    heuristic_noise: float
    cost_noise_std: float


ACTIONS = [
    Actions.left,
    Actions.right,
    Actions.forward,
    Actions.pickup,
    Actions.toggle,
]

TURN_ACTIONS = {Actions.left, Actions.right}


def clone_env(env):
    return copy.deepcopy(env)


def find_goal_pos(env) -> Tuple[int, int]:
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if obj is not None and obj.type == "goal":
                return (x, y)
    return (-1, -1)


def action_cost(action: Actions, cost_cfg: CostConfig) -> float:
    base = cost_cfg.step
    if action in TURN_ACTIONS:
        base += cost_cfg.turn
    if action == Actions.toggle:
        base += cost_cfg.toggle
    return base


def is_lava(env) -> bool:
    cell = env.grid.get(env.agent_pos[0], env.agent_pos[1])
    return cell is not None and cell.type == "lava"


def astar_plan(env, cost_cfg: CostConfig, planner_cfg: PlannerConfig, rng: np.random.Generator) -> Optional[List[int]]:
    start_snapshot = snapshot_state(env)
    start_key = state_key(start_snapshot)
    goal_pos = find_goal_pos(env)
    start_time = time.time()

    frontier = []
    heapq.heappush(frontier, (0.0, 0, start_key, clone_env(env)))
    came_from: Dict = {start_key: None}
    action_from: Dict = {start_key: None}
    cost_so_far: Dict = {start_key: 0.0}

    nodes_expanded = 0

    while frontier:
        if planner_cfg.max_time_s is not None and (time.time() - start_time) > planner_cfg.max_time_s:
            return None
        _, _, current_key, current_env = heapq.heappop(frontier)
        nodes_expanded += 1
        if nodes_expanded > planner_cfg.max_nodes:
            return None
        if tuple(current_env.agent_pos) == goal_pos:
            return reconstruct_actions(came_from, action_from, current_key)

        for action in ACTIONS:
            next_env = clone_env(current_env)
            _, _, terminated, truncated, _ = next_env.step(action)
            next_snapshot = snapshot_state(next_env)
            next_key = state_key(next_snapshot)

            if next_key == current_key:
                continue
            if not planner_cfg.allow_lava and is_lava(next_env):
                continue

            step_cost = action_cost(action, cost_cfg)
            if is_lava(next_env):
                step_cost += cost_cfg.lava_penalty
            if cost_cfg.noise_std > 0:
                step_cost = max(0.0, step_cost + rng.normal(0.0, cost_cfg.noise_std))

            new_cost = cost_so_far[current_key] + step_cost
            if next_key not in cost_so_far or new_cost < cost_so_far[next_key]:
                cost_so_far[next_key] = new_cost
                priority = new_cost + heuristic(next_env, goal_pos, planner_cfg, rng)
                heapq.heappush(frontier, (priority, nodes_expanded, next_key, next_env))
                came_from[next_key] = current_key
                action_from[next_key] = action

            if terminated and not truncated:
                return reconstruct_actions(came_from, action_from, next_key)

    return None


def heuristic(env, goal_pos: Tuple[int, int], planner_cfg: PlannerConfig, rng: np.random.Generator) -> float:
    if goal_pos == (-1, -1):
        return 0.0
    scale_min, scale_max = planner_cfg.heuristic_scale_range
    scale = rng.uniform(scale_min, scale_max)
    manhattan = abs(env.agent_pos[0] - goal_pos[0]) + abs(env.agent_pos[1] - goal_pos[1])
    return float(manhattan) * scale


def reconstruct_actions(came_from: Dict, action_from: Dict, goal_key) -> List[int]:
    actions = []
    current = goal_key
    while came_from[current] is not None:
        actions.append(int(action_from[current]))
        current = came_from[current]
    actions.reverse()
    return actions


def plan_actions(env, cost_cfg: CostConfig, planner_cfg: PlannerConfig, rng: np.random.Generator) -> Optional[List[int]]:
    if planner_cfg.algorithm.lower() not in {"astar", "dijkstra"}:
        raise ValueError(f"Unknown planner algorithm: {planner_cfg.algorithm}")

    cfg = PlannerConfig(
        algorithm=planner_cfg.algorithm,
        max_nodes=planner_cfg.max_nodes,
        heuristic_scale_range=(0.0, 0.0) if planner_cfg.algorithm.lower() == "dijkstra" else planner_cfg.heuristic_scale_range,
        allow_lava=planner_cfg.allow_lava,
    )
    return astar_plan(env, cost_cfg, cfg, rng)
