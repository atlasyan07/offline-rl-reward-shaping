from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Dict, Tuple

import numpy as np
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Floor, Goal, Key, Lava, Wall
from minigrid.core.grid import Grid
from minigrid.core.constants import COLOR_NAMES
from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs.multiroom import MultiRoomEnv, MultiRoom

try:
    from src.layouts import FIXED_LAYOUTS, GOAL_CELLS, LAVA_STRIPS
except ModuleNotFoundError:
    from layouts import FIXED_LAYOUTS, GOAL_CELLS, LAVA_STRIPS
from src.planner import CostConfig, PlannerConfig, plan_actions


@dataclass
class DoorConfig:
    open_prob: float
    locked_prob: float
    closed_prob: float


@dataclass
class LavaConfig:
    enabled: bool
    tiles_per_room: Tuple[int, int]


@dataclass
class MultiRoomShiftEnv(MultiRoomEnv):
    def __init__(
        self,
        min_rooms: int,
        max_rooms: int,
        max_room_size: int = 10,
        max_steps: int | None = None,
        door_cfg: DoorConfig | None = None,
        lava_cfg: LavaConfig | None = None,
        **kwargs,
    ):
        self.door_cfg = door_cfg or DoorConfig(0.3, 0.3, 0.4)
        self.lava_cfg = lava_cfg or LavaConfig(True, (0, 3))
        super().__init__(min_rooms, max_rooms, maxRoomSize=max_room_size, max_steps=max_steps, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self._configure_doors_and_keys()
        self._open_all_doors()
        if self.lava_cfg.enabled:
            self._add_lava_tiles()

    def _configure_doors_and_keys(self) -> None:
        probs = np.array(
            [self.door_cfg.open_prob, self.door_cfg.locked_prob, self.door_cfg.closed_prob],
            dtype=np.float32,
        )
        probs = probs / probs.sum()

        entry_positions = {room.entryDoorPos: idx for idx, room in enumerate(self.rooms) if idx > 0}
        for pos, room_idx in entry_positions.items():
            door_obj = self.grid.get(pos[0], pos[1])
            if not isinstance(door_obj, Door):
                continue
            roll = self._rand_float(0.0, 1.0)
            if roll < probs[0]:
                door_obj.is_open = True
                door_obj.is_locked = False
            elif roll < probs[0] + probs[1]:
                door_obj.is_open = False
                door_obj.is_locked = True
                self._place_key_for_door(door_obj.color, room_idx)
            else:
                door_obj.is_open = False
                door_obj.is_locked = False

    def _place_key_for_door(self, color: str, room_idx: int) -> None:
        key = Key(color)
        candidate_rooms = self.rooms[:1]
        if not candidate_rooms:
            return
        room = self._rand_elem(candidate_rooms)
        self.place_obj(key, room.top, room.size)

    def _open_all_doors(self) -> None:
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, Door):
                    obj.is_open = True
                    obj.is_locked = False

    def _add_lava_tiles(self) -> None:
        min_tiles, max_tiles = self.lava_cfg.tiles_per_room
        avoid = self._lava_avoid_positions()
        for room in self.rooms:
            count = self._rand_int(min_tiles, max_tiles + 1)
            for _ in range(count):
                x = self._rand_int(room.top[0] + 1, room.top[0] + room.size[0] - 1)
                y = self._rand_int(room.top[1] + 1, room.top[1] + room.size[1] - 1)
                if (x, y) == self.agent_pos or (x, y) == self.goal_pos:
                    continue
                if (x, y) in avoid:
                    continue
                obj = self.grid.get(x, y)
                if obj is None:
                    self.grid.set(x, y, Lava())

    def _lava_avoid_positions(self) -> set[tuple[int, int]]:
        avoid: set[tuple[int, int]] = set()

        def add_with_neighbors(pos: tuple[int, int]) -> None:
            px, py = pos
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    avoid.add((nx, ny))

        add_with_neighbors(tuple(self.agent_pos))
        add_with_neighbors(tuple(self.goal_pos))

        for room in self.rooms:
            if hasattr(room, "entryDoorPos") and room.entryDoorPos is not None:
                add_with_neighbors(tuple(room.entryDoorPos))
            if hasattr(room, "exitDoorPos") and room.exitDoorPos is not None:
                add_with_neighbors(tuple(room.exitDoorPos))

        return avoid


class FixedLayoutEnv(MiniGridEnv):
    def __init__(
        self,
        layout_name: str,
        goal_region_indices: list[int] | None = None,
        max_steps: int | None = None,
        **kwargs,
    ):
        if layout_name not in FIXED_LAYOUTS:
            raise ValueError(f"Unknown fixed layout: {layout_name}")
        self.layout_name = layout_name
        self.layout_grid = [list(row) for row in FIXED_LAYOUTS[layout_name]]
        self.height = len(self.layout_grid)
        self.width = len(self.layout_grid[0])
        if max_steps is None:
            max_steps = 4 * self.width * self.height
        if goal_region_indices is None:
            self.goal_region_indices = None
        else:
            self.goal_region_indices = [int(x) for x in goal_region_indices]

        mission_space = MissionSpace(lambda: "reach the goal")
        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            **kwargs,
        )

    def _entrance_avoid_positions(self) -> set[tuple[int, int]]:
        internal_rows = {1, 8, 10, 17}
        internal_cols = {1, 8, 10, 17}
        entrances: set[tuple[int, int]] = set()
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                cell = self.layout_grid[y][x]
                if cell in {".", "S", "G"} and (x in internal_cols or y in internal_rows):
                    entrances.add((x, y))

        avoid: set[tuple[int, int]] = set()
        for x, y in entrances:
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    avoid.add((nx, ny))
        return avoid

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        start_pos = None
        for y, row in enumerate(self.layout_grid):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall())
                else:
                    self.grid.set(x, y, Floor())
                    if cell in {"S", ">"}:
                        start_pos = (x, y)

        if start_pos is None:
            raise ValueError(f"Layout {self.layout_name} missing start")
        self.agent_pos = start_pos
        self.agent_dir = 0

        avoid = self._entrance_avoid_positions()
        for (x, y) in LAVA_STRIPS.get(self.layout_name, []):
            if (x, y) in avoid or (x, y) == self.agent_pos:
                continue
            if isinstance(self.grid.get(x, y), Floor):
                self.grid.set(x, y, Lava())

        candidates = GOAL_CELLS.get(self.layout_name, [])
        if not candidates:
            raise ValueError(f"Layout {self.layout_name} missing goal cells")
        if self.goal_region_indices:
            candidates = [candidates[i] for i in self.goal_region_indices if i < len(candidates)]
            if not candidates:
                candidates = GOAL_CELLS[self.layout_name]

        pick = self._rand_int(0, len(candidates))
        gx, gy = candidates[pick]
        self.goal_region_id = GOAL_CELLS[self.layout_name].index(candidates[pick])

        if not isinstance(self.grid.get(gx, gy), Floor) or (gx, gy) == self.agent_pos:
            raise ValueError(f"Layout {self.layout_name} has no valid goal cell")
        self.goal_pos = (gx, gy)
        self.grid.set(gx, gy, Goal())

LAYOUT_SPECS = {
    "train_easy": {
        "start_room": 5,
        "start_offset": (1, 1),
        "goal_regions": [
            # Full interior of room 2 (top-right) for random goal placement
            (2, 1, 1, 4, 4),
        ],
        "lava_strategy": "optimal_path",
        "openings": {
            # FORK: Two paths from start (room 5)
            "4-5": [2],     # Path 1: Start -> Room 4 (bottom-middle)
            "5-0": [4],     # Path 2: Start -> Room 0 (top-left)
            # Path 1 continuation: Room 4 -> Room 3 -> Room 2 (goal)
            "3-4": [3],
            "2-3": [3],
            # Path 2 continuation: Room 0 -> Room 1 -> Room 2 (goal)
            "0-1": [3],
            "1-2": [3],
        },
        "blocks": [
            (0, 2, 2, 2, 2),  # Block in room 0 to extend path
            (1, 2, 2, 2, 2),  # Block in room 1
            (2, 2, 2, 2, 2),  # Block in room 2
            (3, 2, 2, 2, 2),  # Block in room 3
            (4, 2, 2, 2, 2),  # Block in room 4
        ],
    },
    "train_medium": {
        "start_room": 5,
        "start_offset": (1, 1),
        "goal_regions": [
            # Full interior of room 2 (top-right) for random goal placement
            (2, 1, 1, 4, 4),
        ],
        "lava_strategy": "optimal_path",
        "openings": {
            # FORK at start
            "4-5": [2, 3, 4],    # SHORT path (with lava): Start -> Room 4 (wide corridor)
            "5-0": [4],          # LONG path (safe): Start -> Room 0
            # SHORT path: Room 4 -> Room 3 -> Room 2 (goal)
            "3-4": [2, 3, 4],    # Lava at offset 3
            "2-3": [3],
            # LONG path: Room 0 -> Room 1 -> Room 2 (goal)
            "0-1": [3],
            "1-2": [3],
        },
        "blocks": [
            (0, 2, 2, 2, 2),
            (1, 2, 2, 2, 2),
            (2, 2, 2, 2, 2),
            (3, 2, 2, 2, 2),
            (4, 2, 2, 2, 2),
        ],
    },
    "train_hard": {
        "start_room": 5,
        "start_offset": (1, 1),
        "goal_regions": [
            # Full interior of room 2 (top-right) for random goal placement
            (2, 1, 1, 4, 4),
        ],
        "lava_strategy": "optimal_path",
        "openings": {
            # FORK at start
            "4-5": [2, 3, 4],    # SHORT path with lava at offset 2
            "5-0": [4],          # LONG path (safe)
            # SHORT path: Room 4 -> Room 3 -> Room 2 (goal)
            "3-4": [3],
            "2-3": [3],
            # LONG path: Room 0 -> Room 1 -> Room 2 (goal)
            "0-1": [3],
            "1-2": [3],
        },
        "blocks": [
            (0, 2, 2, 2, 2),
            (1, 2, 2, 2, 2),
            (2, 2, 2, 2, 2),
            (3, 2, 2, 2, 2),
            (4, 2, 2, 2, 2),
        ],
    },
    "eval_harder": {
        "start_room": 5,
        "start_offset": (1, 1),
        "goal_regions": [
            # Full interior of room 2 (top-right) for random goal placement
            (2, 1, 1, 4, 4),
        ],
        "lava_strategy": "optimal_path",
        "openings": {
            # FORK at start
            "4-5": [2],          # Path to bottom corridor
            "5-0": [4],          # Path to top corridor
            # SHORT risky path: Room 4 -> Room 3 (with lava)
            "3-4": [2, 3, 4],    # Lava at offset 2
            # LONG safe path: Room 0 -> Room 1 -> Room 2 -> Room 3
            "0-1": [3],
            "1-2": [3],
            "2-3": [3],
        },
        "blocks": [
            (0, 2, 2, 2, 2),
            (1, 2, 2, 2, 2),
            (2, 2, 2, 2, 2),
            (4, 2, 2, 2, 2),
        ],
    },
    "eval_hardest": {
        "start_room": 5,
        "start_offset": (1, 1),
        "goal_regions": [
            # Full interior of room 2 (top-right) for random goal placement
            (2, 1, 1, 4, 4),
        ],
        "lava_strategy": "optimal_path",
        "openings": {
            # FORK at start
            "5-0": [2, 3, 4],    # SHORT path with lava at offset 2
            "4-5": [2],          # LONG path entrance
            # SHORT path: Room 0 -> Room 1 -> Room 2 (goal)
            "0-1": [2, 3, 4],    # Lava at offset 2
            "1-2": [3],
            # LONG path: Room 4 -> Room 3 -> Room 2 (goal)
            "3-4": [3],
            "2-3": [3],
        },
        "blocks": [
            (1, 2, 2, 2, 2),
            (2, 2, 2, 2, 2),
            (3, 2, 2, 2, 2),
            (4, 2, 2, 2, 2),
        ],
    },
}


class StructuredMultiRoomEnv(MultiRoomEnv):
    def __init__(
        self,
        min_rooms: int,
        max_rooms: int,
        max_room_size: int = 6,
        max_steps: int | None = None,
        layout_name: str | None = None,
        goal_region_indices: list[int] | None = None,
        **kwargs,
    ):
        if max_steps is None:
            max_steps = 200
        self.max_room_size = max_room_size
        self.fixed_room_count = 6
        self.layout_name = layout_name or "base"
        if self.layout_name not in LAYOUT_SPECS:
            raise ValueError(f"Unknown layout_name: {self.layout_name}")
        if goal_region_indices is None:
            self.goal_region_indices = None
        else:
            self.goal_region_indices = [int(x) for x in goal_region_indices]
        if max_rooms < self.fixed_room_count:
            raise ValueError("max_rooms must be >= 6 for structured layout")
        super().__init__(min_rooms, max_rooms, maxRoomSize=max_room_size, max_steps=max_steps, **kwargs)

    def _room_positions(self):
        # 2x3 loop of 6 rooms with shared wall cells (size 6)
        return [
            (1, 1),   # 0 top-left (start)
            (6, 1),   # 1 top-mid (risk)
            (11, 1),  # 2 top-right (door)
            (11, 6),  # 3 bottom-right (goal)
            (6, 6),   # 4 bottom-mid (neutral)
            (1, 6),   # 5 bottom-left (safe)
        ]

    def _shared_door_pos(self, top_a, top_b, size, offset):
        ax, ay = top_a
        bx, by = top_b
        if ay == by and bx > ax:
            return (ax + size - 1, ay + offset)
        if ay == by and bx < ax:
            return (ax, ay + offset)
        if ax == bx and by > ay:
            return (ax + offset, ay + size - 1)
        if ax == bx and by < ay:
            return (ax + offset, ay)
        return (ax + size - 1, ay + offset)

    def _build_rooms(self):
        size = self.max_room_size
        positions = self._room_positions()
        rooms = []
        n = len(positions)
        for i in range(n):
            rooms.append(MultiRoom(positions[i], (size, size), None, None))
        return rooms

    def _draw_rooms(self, rooms):
        grid = Grid(self.width, self.height)
        wall = Wall()
        for room in rooms:
            x0, y0 = room.top
            w, h = room.size
            for i in range(w):
                grid.set(x0 + i, y0, wall)
                grid.set(x0 + i, y0 + h - 1, wall)
            for j in range(h):
                grid.set(x0, y0 + j, wall)
                grid.set(x0 + w - 1, y0 + j, wall)
        self.grid = grid

    def _carve_openings(self, rooms):
        offsets = LAYOUT_SPECS[self.layout_name]["openings"]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        for a_idx, b_idx in edges:
            edge_key = f"{a_idx}-{b_idx}"
            for offset in offsets.get(edge_key, []):
                pos = self._shared_door_pos(rooms[a_idx].top, rooms[b_idx].top, rooms[a_idx].size[0], offset)
                self.grid.set(pos[0], pos[1], None)

    def _assign_roles(self, rooms):
        spec = LAYOUT_SPECS[self.layout_name]
        start_room = spec["start_room"]
        self.room_roles = {
            start_room: "start_room",
        }
        for idx in range(len(rooms)):
            if idx not in self.room_roles:
                self.room_roles[idx] = "room"

    def _add_detour_block(self, room, rel_x: int, rel_y: int, width: int, height: int) -> None:
        x0, y0 = room.top
        for y in range(y0 + rel_y, y0 + rel_y + height):
            for x in range(x0 + rel_x, x0 + rel_x + width):
                self.grid.set(x, y, Wall())

    def _apply_layout_blocks(self) -> None:
        blocks = LAYOUT_SPECS[self.layout_name]["blocks"]
        for room_idx, rel_x, rel_y, width, height in blocks:
            self._add_detour_block(self.rooms[room_idx], rel_x, rel_y, width, height)

    def _add_fixed_lava(self, path_positions: list[tuple[int, int]]) -> None:
        spec = LAYOUT_SPECS[self.layout_name]
        if spec.get("lava_strategy") != "optimal_path":
            return
        if not path_positions or len(path_positions) < 8:
            return

        # Avoid doorway tiles on edges
        opening_positions = self._opening_positions(spec)

        # Exclude positions too close to start/goal
        min_start = 1
        min_goal = 2
        candidates = []
        for idx, pos in enumerate(path_positions):
            if idx <= min_start:
                continue
            if idx >= len(path_positions) - min_goal:
                continue
            if pos in opening_positions:
                continue
            candidates.append(pos)

        if not candidates:
            return

        # Bias lava closer to the goal to increase risk
        lava_pos = candidates[int(len(candidates) * 0.75)]
        if self.grid.get(lava_pos[0], lava_pos[1]) is None:
            self.grid.set(lava_pos[0], lava_pos[1], Lava())

    def _opening_positions(self, spec: Dict) -> set[tuple[int, int]]:
        positions: set[tuple[int, int]] = set()
        offsets = spec.get("openings", {})
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        for a_idx, b_idx in edges:
            edge_key = f"{a_idx}-{b_idx}"
            for offset in offsets.get(edge_key, []):
                pos = self._shared_door_pos(
                    self.rooms[a_idx].top,
                    self.rooms[b_idx].top,
                    self.rooms[a_idx].size[0],
                    offset,
                )
                positions.add(pos)
        return positions

    def _optimal_path_positions(self) -> list[tuple[int, int]]:
        env_copy = copy.deepcopy(self)
        # Ensure planner steps don't fail before reset finishes
        if not hasattr(env_copy, "step_count"):
            env_copy.step_count = 0
        cost_cfg = CostConfig(step=1.0, turn=0.2, toggle=2.0, lava_penalty=10.0, noise_std=0.0)
        planner_cfg = PlannerConfig(
            algorithm="astar",
            max_nodes=60000,
            heuristic_scale_range=(1.0, 1.0),
            allow_lava=True,
            max_time_s=2.0,
        )
        rng = np.random.default_rng(0)
        actions = plan_actions(env_copy, cost_cfg, planner_cfg, rng)
        if not actions:
            return []

        positions = [tuple(env_copy.agent_pos)]
        for action in actions:
            env_copy.step(action)
            positions.append(tuple(env_copy.agent_pos))
        return positions

    def _gen_grid(self, width, height):
        self.rooms = self._build_rooms()
        self._draw_rooms(self.rooms)
        self._carve_openings(self.rooms)
        self._assign_roles(self.rooms)

        # Start and goal placement
        spec = LAYOUT_SPECS[self.layout_name]
        start_room = self.rooms[spec["start_room"]]
        start_dx, start_dy = spec.get("start_offset", (1, 1))
        self.agent_pos = (start_room.top[0] + start_dx, start_room.top[1] + start_dy)
        self.agent_dir = 0
        self._apply_layout_blocks()

        regions = spec.get("goal_regions", [])
        if not regions:
            raise ValueError("No goal_regions defined in layout spec")
        if self.goal_region_indices:
            candidates = [regions[i] for i in self.goal_region_indices]
        else:
            candidates = regions

        region_pick = self._rand_int(0, len(candidates))
        room_idx, rx, ry, rw, rh = candidates[region_pick]
        self.goal_region_id = regions.index(candidates[region_pick])
        goal_room = self.rooms[room_idx]

        placed = False
        for _ in range(rw * rh):
            gx = goal_room.top[0] + rx + self._rand_int(0, rw)
            gy = goal_room.top[1] + ry + self._rand_int(0, rh)
            if self.grid.get(gx, gy) is None:
                self.goal_pos = (gx, gy)
                self.grid.set(gx, gy, Goal())
                placed = True
                break
        if not placed:
            for y in range(goal_room.top[1] + ry, goal_room.top[1] + ry + rh):
                for x in range(goal_room.top[0] + rx, goal_room.top[0] + rx + rw):
                    if self.grid.get(x, y) is None:
                        self.goal_pos = (x, y)
                        self.grid.set(x, y, Goal())
                        placed = True
                        break
                if placed:
                    break
        # Compute optimal path (no lava placed yet)
        path_positions = self._optimal_path_positions()
        self._add_fixed_lava(path_positions)
        # Note: max_steps is set by config, not overridden here


def make_env(cfg: Dict, render_mode: str | None = None, layout_name: str | None = None, difficulty: int = 0):
    if layout_name in LAYOUT_SPECS:
        return StructuredMultiRoomEnv(
            min_rooms=cfg["min_rooms"],
            max_rooms=cfg["max_rooms"],
            max_room_size=cfg.get("max_room_size", 6),
            max_steps=cfg.get("max_steps"),
            layout_name=layout_name,
            goal_region_indices=cfg.get("goal_region_indices"),
            render_mode=render_mode,
        )
    if layout_name in FIXED_LAYOUTS:
        return FixedLayoutEnv(
            layout_name=layout_name,
            goal_region_indices=cfg.get("goal_region_indices"),
            max_steps=cfg.get("max_steps"),
            render_mode=render_mode,
        )
    return StructuredMultiRoomEnv(
        min_rooms=cfg["min_rooms"],
        max_rooms=cfg["max_rooms"],
        max_room_size=cfg.get("max_room_size", 6),
        max_steps=cfg.get("max_steps"),
        layout_name=layout_name,
        goal_region_indices=cfg.get("goal_region_indices"),
        render_mode=render_mode,
    )
