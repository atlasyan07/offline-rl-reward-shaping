import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import yaml
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX


@dataclass
class StateSnapshot:
    grid: np.ndarray  # (W, H, 3)
    agent_pos: Tuple[int, int]
    agent_dir: int
    carrying: Tuple[int, int]  # (obj_idx, color_idx) or (-1, -1)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def seed_numpy(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def encode_carrying(carrying_obj) -> Tuple[int, int]:
    if carrying_obj is None:
        return (-1, -1)
    obj_idx = OBJECT_TO_IDX.get(carrying_obj.type, -1)
    color_idx = COLOR_TO_IDX.get(carrying_obj.color, -1)
    return (int(obj_idx), int(color_idx))


def snapshot_state(env) -> StateSnapshot:
    grid = env.grid.encode().astype(np.uint8)
    agent_pos = tuple(int(x) for x in env.agent_pos)
    agent_dir = int(env.agent_dir)
    carrying = encode_carrying(env.carrying)
    return StateSnapshot(grid=grid, agent_pos=agent_pos, agent_dir=agent_dir, carrying=carrying)


def state_key(snapshot: StateSnapshot) -> Tuple[Any, ...]:
    return (
        snapshot.grid.tobytes(),
        snapshot.grid.shape,
        snapshot.agent_pos,
        snapshot.agent_dir,
        snapshot.carrying,
    )
