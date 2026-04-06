#!/usr/bin/env python3
"""Merge behavior dataset with expert successes to reach a target success rate.

Behavior data: keep all non-exploratory episodes, keep exploratory successes,
drop a fraction of exploratory failures, then add expert successes to reach
the target success rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge behavior + expert datasets")
    parser.add_argument("--behavior-dataset", required=True, help="Path to behavior dataset.npz")
    parser.add_argument("--behavior-metadata", required=True, help="Path to behavior metadata.json")
    parser.add_argument("--expert-dataset", required=True, help="Path to expert dataset.npz")
    parser.add_argument("--expert-metadata", required=True, help="Path to expert metadata.json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--target-success-rate", type=float, default=0.70,
                        help="Target episode success rate after merge (default: 0.70)")
    parser.add_argument(
        "--drop-failure-behaviors",
        default="exploratory,noisy,suboptimal_major",
        help="Comma-separated failure behaviors to drop in order if success is still low",
    )
    parser.add_argument(
        "--extra-failure-drop",
        type=int,
        default=0,
        help="Additional failure episodes to drop after exploratory filtering",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    return parser.parse_args()


def load_metadata(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def build_episode_maps(episodes: List[Dict]) -> Tuple[Dict[int, Dict], List[int]]:
    episode_by_id: Dict[int, Dict] = {}
    ids: List[int] = []
    for ep in episodes:
        eid = int(ep["episode_id"])
        episode_by_id[eid] = ep
        ids.append(eid)
    return episode_by_id, ids


def filter_dataset(data: Dict[str, np.ndarray], keep_episode_ids: np.ndarray) -> Dict[str, np.ndarray]:
    episode_ids = data["episode_id"].astype(np.int64)
    keep_mask = np.isin(episode_ids, keep_episode_ids)
    filtered = {}
    for key, arr in data.items():
        if arr.shape[0] != episode_ids.shape[0]:
            filtered[key] = arr
            continue
        filtered[key] = arr[keep_mask]
    return filtered


def remap_episode_ids(episodes: List[Dict], new_offset: int = 0) -> Tuple[List[Dict], Dict[int, int]]:
    mapping: Dict[int, int] = {}
    new_episodes: List[Dict] = []
    for idx, ep in enumerate(episodes):
        old_id = int(ep["episode_id"])
        new_id = new_offset + idx
        mapping[old_id] = new_id
        new_ep = dict(ep)
        new_ep["episode_id"] = new_id
        new_episodes.append(new_ep)
    return new_episodes, mapping


def apply_episode_id_mapping(data: Dict[str, np.ndarray], mapping: Dict[int, int]) -> None:
    episode_ids = data["episode_id"].astype(np.int64)
    new_ids = np.array([mapping[int(eid)] for eid in episode_ids], dtype=np.int32)
    data["episode_id"] = new_ids


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    behavior_data = load_npz(Path(args.behavior_dataset))
    behavior_meta = load_metadata(Path(args.behavior_metadata))
    expert_data = load_npz(Path(args.expert_dataset))
    expert_meta = load_metadata(Path(args.expert_metadata))

    behavior_episodes = behavior_meta.get("episodes", [])
    expert_episodes = expert_meta.get("episodes", [])

    beh_by_id, beh_ids = build_episode_maps(behavior_episodes)
    exp_by_id, exp_ids = build_episode_maps(expert_episodes)

    exploratory_success = []
    exploratory_fail = []
    non_exploratory = []
    total_success = 0

    for eid in beh_ids:
        ep = beh_by_id[eid]
        success = bool(ep.get("success", False))
        if success:
            total_success += 1
        if ep.get("planner_type") == "exploratory":
            if success:
                exploratory_success.append(eid)
            else:
                exploratory_fail.append(eid)
        else:
            non_exploratory.append(eid)

    nB = len(beh_ids)
    sB = total_success
    target = args.target_success_rate

    # Expert successes only
    expert_success_ids = [eid for eid in exp_ids if bool(exp_by_id[eid].get("success", False))]
    rng.shuffle(expert_success_ids)

    # Determine how many exploratory failures to drop and how many expert successes to add
    max_drop = len(exploratory_fail)
    chosen_drop = max_drop
    chosen_add = len(expert_success_ids)
    achieved_rate = None

    for drop in range(0, max_drop + 1):
        nB_prime = nB - drop
        sB_prime = sB
        required_add = int(np.ceil((target * nB_prime - sB_prime) / (1.0 - target)))
        if required_add < 0:
            required_add = 0
        if required_add <= len(expert_success_ids):
            chosen_drop = drop
            chosen_add = required_add
            achieved_rate = (sB_prime + chosen_add) / (nB_prime + chosen_add)
            break

    if achieved_rate is None:
        # Even after dropping all exploratory failures, we can't hit target
        nB_prime = nB - max_drop
        sB_prime = sB
        chosen_drop = max_drop
        chosen_add = len(expert_success_ids)
        achieved_rate = (sB_prime + chosen_add) / (nB_prime + chosen_add)

    keep_exploratory_fail = exploratory_fail.copy()
    rng.shuffle(keep_exploratory_fail)
    keep_exploratory_fail = keep_exploratory_fail[chosen_drop:]

    keep_behavior_ids = np.array(
        non_exploratory + exploratory_success + keep_exploratory_fail,
        dtype=np.int64,
    )

    drop_order = [b.strip() for b in args.drop_failure_behaviors.split(",") if b.strip()]
    extra_drop_applied = 0
    if args.extra_failure_drop > 0 and drop_order:
        # Build ordered list of failure episode ids by behavior
        ordered_fail_ids: List[int] = []
        for behavior in drop_order:
            for eid in beh_ids:
                ep = beh_by_id[eid]
                if ep.get("planner_type") != behavior:
                    continue
                if bool(ep.get("success", False)):
                    continue
                ordered_fail_ids.append(eid)

        ordered_fail_ids = list(dict.fromkeys(ordered_fail_ids))
        keep_set = set(int(eid) for eid in keep_behavior_ids.tolist())
        droppable = [eid for eid in ordered_fail_ids if eid in keep_set]
        extra_drop_applied = min(args.extra_failure_drop, len(droppable))
        if extra_drop_applied > 0:
            drop_set = set(droppable[:extra_drop_applied])
            keep_behavior_ids = np.array([eid for eid in keep_set if eid not in drop_set], dtype=np.int64)
            keep_behavior_ids.sort()

    add_expert_ids = np.array(expert_success_ids[:chosen_add], dtype=np.int64)

    # Filter datasets
    behavior_filtered = filter_dataset(behavior_data, keep_behavior_ids)
    expert_filtered = filter_dataset(expert_data, add_expert_ids)

    # Ensure planner_type exists for expert data
    if "planner_type" not in expert_filtered:
        expert_filtered["planner_type"] = np.array(["expert_clean"] * len(expert_filtered["episode_id"]), dtype=object)
    else:
        expert_filtered["planner_type"] = np.array(
            ["expert_clean"] * len(expert_filtered["episode_id"]), dtype=object
        )

    # Filter metadata and remap episode ids
    behavior_kept_eps = [beh_by_id[int(eid)] for eid in keep_behavior_ids.tolist()]
    expert_kept_eps = [exp_by_id[int(eid)] for eid in add_expert_ids.tolist()]

    behavior_kept_eps, beh_map = remap_episode_ids(behavior_kept_eps, new_offset=0)
    apply_episode_id_mapping(behavior_filtered, beh_map)

    expert_kept_eps, exp_map = remap_episode_ids(expert_kept_eps, new_offset=len(behavior_kept_eps))
    apply_episode_id_mapping(expert_filtered, exp_map)

    # Merge datasets
    merged = {}
    for key in behavior_filtered.keys():
        if key not in expert_filtered:
            merged[key] = behavior_filtered[key]
            continue
        merged[key] = np.concatenate([behavior_filtered[key], expert_filtered[key]], axis=0)

    # Save
    np.savez_compressed(out_dir / "dataset.npz", **merged)
    total_eps = len(behavior_kept_eps) + len(expert_kept_eps)
    total_success = sum(1 for ep in behavior_kept_eps if ep.get("success")) + sum(
        1 for ep in expert_kept_eps if ep.get("success")
    )
    final_success_rate = (total_success / total_eps) if total_eps else 0.0
    merged_meta = {
        "config": {
            "behavior_dataset": args.behavior_dataset,
            "expert_dataset": args.expert_dataset,
            "target_success_rate": target,
            "dropped_exploratory_failures": int(chosen_drop),
            "added_expert_successes": int(chosen_add),
            "extra_failure_drop": int(extra_drop_applied),
            "drop_failure_behaviors": drop_order,
        },
        "episodes": behavior_kept_eps + expert_kept_eps,
        "transitions": int(len(merged["episode_id"])),
        "success_rate": float(final_success_rate),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2)

    print(
        "[merge] behavior_eps={} expert_eps_added={} dropped_exploratory_failures={} "
        "success_rate={:.3f}".format(
            len(behavior_kept_eps),
            len(expert_kept_eps),
            chosen_drop,
            final_success_rate,
        )
    )


if __name__ == "__main__":
    main()
