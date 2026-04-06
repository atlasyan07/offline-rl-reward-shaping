#!/usr/bin/env python3
"""Extract expert trajectories from an offline dataset with data-driven heuristics."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


TURN_ACTIONS = {0, 1}  # MiniGrid: left, right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract expert trajectories from dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for expert dataset")
    parser.add_argument("--top-pct", type=float, default=0.60, help="Fraction of shortest per group")
    parser.add_argument("--feature-pct", type=float, default=0.40, help="Fraction of feature-guided per group")
    parser.add_argument("--turn-rate-quantile", type=float, default=0.5, help="Quantile cutoff for turn rate")
    parser.add_argument("--max-per-group", type=int, default=50, help="Max episodes per (layout, goal) group")
    parser.add_argument("--cap-near-optimal", type=float, default=0.60, help="Max fraction of near_optimal")
    return parser.parse_args()


def compute_turn_rate(actions: np.ndarray) -> float:
    if actions.size == 0:
        return 1.0
    return float(np.isin(actions, list(TURN_ACTIONS)).mean())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    episodes = metadata.get("episodes", [])
    if not episodes:
        raise SystemExit("No episodes found in metadata")

    episode_ids = data["episode_id"]
    actions = data["a"]

    # Build per-episode stats
    per_ep = []
    for ep in episodes:
        ep_id = ep["episode_id"]
        ep_mask = episode_ids == ep_id
        ep_actions = actions[ep_mask]
        turn_rate = compute_turn_rate(ep_actions)
        per_ep.append(
            {
                "episode_id": ep_id,
                "length": int(ep["length"]),
                "success": bool(ep["success"]),
                "layout_name": ep["layout_name"],
                "goal_region_id": int(ep.get("goal_region_id", 0)),
                "planner_type": ep.get("planner_type", "unknown"),
                "turn_rate": turn_rate,
            }
        )

    # Global turn-rate cutoff from successful episodes
    success_turn_rates = [e["turn_rate"] for e in per_ep if e["success"]]
    if not success_turn_rates:
        raise SystemExit("No successful episodes found")
    turn_cutoff = float(np.quantile(success_turn_rates, args.turn_rate_quantile))

    # Group by (layout, goal_region)
    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for ep in per_ep:
        if not ep["success"]:
            continue
        key = (ep["layout_name"], ep["goal_region_id"])
        grouped[key].append(ep)

    selected_ids = []
    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda x: x["length"])
        n_total = len(group_sorted)
        n_short = max(1, int(np.ceil(n_total * args.top_pct)))
        n_feat = max(1, int(np.ceil(n_total * args.feature_pct)))

        shortest = group_sorted[:n_short]
        longest = group_sorted[-n_feat:]
        # Feature-guided: prefer low turn-rate among longer successes
        longest = [g for g in longest if g["turn_rate"] <= turn_cutoff]

        combined = shortest + longest

        # Cap near_optimal dominance within group
        max_near = int(np.floor(args.max_per_group * args.cap_near_optimal))
        near = [g for g in combined if g["planner_type"] == "near_optimal"][:max_near]
        rest = [g for g in combined if g["planner_type"] != "near_optimal"]
        combined = near + rest
        combined = combined[: args.max_per_group]
        selected_ids.extend([g["episode_id"] for g in combined])

    selected_ids = sorted(set(selected_ids))
    if not selected_ids:
        raise SystemExit("No expert episodes selected; try relaxing thresholds")

    # Build expert dataset
    mask = np.isin(episode_ids, selected_ids)
    expert = {k: v[mask] if k != "planner_type" else v[mask] for k, v in data.items()}

    # Remap episode IDs to contiguous
    old_to_new = {old: i for i, old in enumerate(selected_ids)}
    expert["episode_id"] = np.array([old_to_new[int(e)] for e in expert["episode_id"]], dtype=np.int32)

    # Build expert metadata
    expert_episodes = []
    for ep in per_ep:
        if ep["episode_id"] not in old_to_new:
            continue
        expert_episodes.append(
            {
                "episode_id": old_to_new[ep["episode_id"]],
                "length": ep["length"],
                "success": ep["success"],
                "layout_name": ep["layout_name"],
                "goal_region_id": ep["goal_region_id"],
                "planner_type": ep["planner_type"],
                "turn_rate": ep["turn_rate"],
            }
        )

    expert_metadata = {
        "source_dataset": os.path.abspath(args.dataset),
        "selection": {
            "top_pct": args.top_pct,
            "feature_pct": args.feature_pct,
            "turn_rate_quantile": args.turn_rate_quantile,
            "turn_rate_cutoff": turn_cutoff,
            "max_per_group": args.max_per_group,
            "cap_near_optimal": args.cap_near_optimal,
        },
        "episodes": expert_episodes,
        "transitions": int(len(expert["a"])),
        "success_rate": float(np.mean([e["success"] for e in expert_episodes])),
    }

    np.savez_compressed(out_dir / "expert_dataset.npz", **expert)
    with open(out_dir / "expert_metadata.json", "w", encoding="utf-8") as f:
        json.dump(expert_metadata, f, indent=2)

    print(f"[extract_expert_dataset] experts={len(expert_episodes)} transitions={len(expert['a'])}")
    print(f"[extract_expert_dataset] turn_rate_cutoff={turn_cutoff:.3f} out_dir={out_dir}")


if __name__ == "__main__":
    main()
