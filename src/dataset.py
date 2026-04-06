from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .planner import BehaviorConfig, CostConfig, PlannerConfig, plan_actions, ACTIONS
from .utils import StateSnapshot, snapshot_state


@dataclass
class EpisodeResult:
    transitions: List[Tuple[StateSnapshot, int, float, StateSnapshot, bool]]
    actions: List[int]
    success: bool


def _behavior_cost_cfg(base_costs: Dict, behavior: BehaviorConfig) -> CostConfig:
    return CostConfig(
        step=base_costs["step"],
        turn=base_costs["turn"],
        toggle=base_costs["toggle"],
        lava_penalty=base_costs["lava_penalty"],
        noise_std=behavior.cost_noise_std,
    )


def _behavior_planner_cfg(base_planner: PlannerConfig, behavior: BehaviorConfig) -> PlannerConfig:
    base_min, base_max = base_planner.heuristic_scale_range
    scale_min = max(0.0, base_min * (1.0 - behavior.heuristic_noise))
    scale_max = base_max * (1.0 + behavior.heuristic_noise)
    return PlannerConfig(
        algorithm=base_planner.algorithm,
        max_nodes=base_planner.max_nodes,
        heuristic_scale_range=(scale_min, scale_max),
        allow_lava=base_planner.allow_lava,
    )


def _random_action(rng: np.random.Generator) -> int:
    return int(rng.choice([int(a) for a in ACTIONS]))


def run_episode(
    env,
    base_costs: Dict,
    base_planner: PlannerConfig,
    behavior: BehaviorConfig,
    rng: np.random.Generator,
    max_steps: int,
) -> Optional[EpisodeResult]:
    transitions: List[Tuple[StateSnapshot, int, float, StateSnapshot, bool]] = []
    actions_taken: List[int] = []

    cost_cfg = _behavior_cost_cfg(base_costs, behavior)
    planner_cfg = _behavior_planner_cfg(base_planner, behavior)

    plan: List[int] = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        if not plan:
            plan = plan_actions(env, cost_cfg, planner_cfg, rng) or []
            if not plan:
                return None

        used_random = False
        if rng.random() < behavior.random_action_prob:
            action = _random_action(rng)
            used_random = True
        else:
            action = int(plan.pop(0))

        state = snapshot_state(env)
        _, reward, terminated, truncated, _ = env.step(action)
        next_state = snapshot_state(env)
        done = bool(terminated)
        transitions.append((state, action, float(reward), next_state, done))
        actions_taken.append(action)
        steps += 1

        if used_random:
            plan = []

        if truncated:
            break

    return EpisodeResult(transitions=transitions, actions=actions_taken, success=done)
