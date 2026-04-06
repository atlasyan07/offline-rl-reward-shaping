#!/usr/bin/env python3
"""Build a labeled dataset for AIRL discriminator training (expert vs negative).

Positives: expert-like successes from the behavior dataset.
Negatives: failures + worst successes (longest / highest turn-rate / lowest reward).
"""
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
    parser = argparse.ArgumentParser(description="Build AIRL discriminator dataset")
    parser.add_argument("--dataset", required=True, help="Path to behavior dataset.npz")
    parser.add_argument("--metadata", required=True, help="Path to behavior metadata.json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--expert-dataset", help="Path to expert dataset.npz for augmentation")
    parser.add_argument("--expert-metadata", help="Path to expert metadata.json for augmentation")
    parser.add_argument("--target-goal-ratio", type=float, default=0.05,
                        help="Target goal-transition ratio after expert augmentation")

    # Positive (expert) selection
    parser.add_argument("--pos-top-pct", type=float, default=0.60,
                        help="Fraction of shortest per group")
    parser.add_argument("--pos-feature-pct", type=float, default=0.40,
                        help="Fraction of feature-guided per group")
    parser.add_argument("--pos-turn-rate-quantile", type=float, default=0.5,
                        help="Quantile cutoff for turn rate")
    parser.add_argument("--pos-max-per-group", type=int, default=50,
                        help="Max episodes per (layout, goal) group")
    parser.add_argument("--pos-cap-near-optimal", type=float, default=0.60,
                        help="Max fraction of near_optimal")

    # Negative selection
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="Negatives per positive")
    parser.add_argument("--neg-worst-success-pct", type=float, default=0.20,
                        help="Worst success fraction to allow as negatives")
    parser.add_argument("--neg-max-steps", type=int,
                        help="Cap negative episode length to first N transitions")
    return parser.parse_args()


def compute_turn_rate(actions: np.ndarray) -> float:
    if actions.size == 0:
        return 1.0
    return float(np.isin(actions, list(TURN_ACTIONS)).mean())


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
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
    rewards = data["r"]

    # Per-episode stats
    per_ep: List[Dict] = []
    for ep in episodes:
        ep_id = ep["episode_id"]
        ep_mask = episode_ids == ep_id
        ep_actions = actions[ep_mask]
        ep_rewards = rewards[ep_mask]
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
                "reward_sum": float(ep_rewards.sum()),
            }
        )

    success_eps = [e for e in per_ep if e["success"]]
    if not success_eps:
        raise SystemExit("No successful episodes found")

    # Positive selection (expert-like)
    success_turn_rates = [e["turn_rate"] for e in success_eps]
    turn_cutoff = float(np.quantile(success_turn_rates, args.pos_turn_rate_quantile))

    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for ep in success_eps:
        key = (ep["layout_name"], ep["goal_region_id"])
        grouped[key].append(ep)

    positive_ids: List[int] = []
    for _, group in grouped.items():
        group_sorted = sorted(group, key=lambda x: x["length"])
        n_total = len(group_sorted)
        n_short = max(1, int(np.ceil(n_total * args.pos_top_pct)))
        n_feat = max(1, int(np.ceil(n_total * args.pos_feature_pct)))

        shortest = group_sorted[:n_short]
        longest = group_sorted[-n_feat:]
        longest = [g for g in longest if g["turn_rate"] <= turn_cutoff]

        combined = shortest + longest
        max_near = int(np.floor(args.pos_max_per_group * args.pos_cap_near_optimal))
        near = [g for g in combined if g["planner_type"] == "near_optimal"][:max_near]
        rest = [g for g in combined if g["planner_type"] != "near_optimal"]
        combined = (near + rest)[: args.pos_max_per_group]
        positive_ids.extend([g["episode_id"] for g in combined])

    positive_ids = sorted(set(positive_ids))
    if not positive_ids:
        raise SystemExit("No positive episodes selected; relax thresholds")

    # Negative selection: failures + worst successes
    failure_eps = [e for e in per_ep if not e["success"]]
    failure_ids = [e["episode_id"] for e in failure_eps]

    remaining_success = [e for e in success_eps if e["episode_id"] not in positive_ids]
    if remaining_success:
        remaining_success = sorted(
            remaining_success,
            key=lambda e: (e["length"], e["turn_rate"], -e["reward_sum"]),
            reverse=True,
        )
    worst_k = int(np.ceil(len(remaining_success) * args.neg_worst_success_pct))
    worst_success_ids = [e["episode_id"] for e in remaining_success[:worst_k]]

    target_neg = int(np.ceil(len(positive_ids) * args.neg_ratio))
    rng.shuffle(failure_ids)
    negatives = []
    for ep_id in failure_ids:
        if len(negatives) >= target_neg:
            break
        negatives.append(ep_id)
    for ep_id in worst_success_ids:
        if len(negatives) >= target_neg:
            break
        negatives.append(ep_id)
    negatives = sorted(set(negatives))

    if not negatives:
        raise SystemExit("No negatives selected; adjust settings")

    # Build base labeled dataset (optionally truncate negative episodes)
    keep_ids = sorted(set(positive_ids + negatives))
    keep_mask = np.zeros(len(episode_ids), dtype=bool)
    kept_len: Dict[int, int] = {}
    for ep_id in keep_ids:
        ep_indices = np.flatnonzero(episode_ids == ep_id)
        if ep_id in positive_ids or args.neg_max_steps is None or ep_id not in negatives:
            selected = ep_indices
        else:
            selected = ep_indices[: args.neg_max_steps]
        keep_mask[selected] = True
        kept_len[int(ep_id)] = int(selected.size)

    labeled = {k: v[keep_mask] if k != "planner_type" else v[keep_mask] for k, v in data.items()}

    label_lookup = {int(ep_id): 1 for ep_id in positive_ids}
    labels = np.array([label_lookup.get(int(e), 0) for e in labeled["episode_id"]], dtype=np.int64)
    labeled["label"] = labels

    labeled_episodes = []
    for ep in per_ep:
        if ep["episode_id"] not in keep_ids:
            continue
        labeled_episodes.append(
            {
                "episode_id": ep["episode_id"],
                "length": kept_len.get(ep["episode_id"], ep["length"]),
                "success": ep["success"],
                "layout_name": ep["layout_name"],
                "goal_region_id": ep["goal_region_id"],
                "planner_type": ep["planner_type"],
                "turn_rate": ep["turn_rate"],
                "reward_sum": ep["reward_sum"],
                "label": int(label_lookup.get(ep["episode_id"], 0)),
                "source": "behavior",
            }
        )

    # Optional: augment positives with expert episodes to reach target goal ratio
    if args.expert_dataset and args.expert_metadata:
        expert_data = np.load(args.expert_dataset, allow_pickle=True)
        with open(args.expert_metadata, "r", encoding="utf-8") as f:
            expert_meta = json.load(f)
        expert_eps = expert_meta.get("episodes", [])
        if not expert_eps:
            raise SystemExit("No episodes found in expert metadata")

        goal_mask = labeled["r"] > 0
        goal_count = int(goal_mask.sum())
        total_count = int(len(labeled["r"]))
        current_ratio = goal_count / max(1, total_count)

        if current_ratio < args.target_goal_ratio:
            rng.shuffle(expert_eps)
            selected_expert_ids = []
            added = 0
            for ep in expert_eps:
                selected_expert_ids.append(ep["episode_id"])
                ep_mask = expert_data["episode_id"] == ep["episode_id"]
                added += int((expert_data["r"][ep_mask] > 0).sum())
                total_count += int(ep_mask.sum())
                goal_count += int((expert_data["r"][ep_mask] > 0).sum())
                if goal_count / max(1, total_count) >= args.target_goal_ratio:
                    break

            if selected_expert_ids:
                exp_mask = np.isin(expert_data["episode_id"], selected_expert_ids)
                for k in labeled.keys():
                    if k not in expert_data:
                        continue
                    labeled[k] = np.concatenate([labeled[k], expert_data[k][exp_mask]], axis=0)
                labeled["label"] = np.concatenate(
                    [labeled["label"], np.ones(exp_mask.sum(), dtype=np.int64)], axis=0
                )
                for ep in expert_eps:
                    if ep["episode_id"] in selected_expert_ids:
                        labeled_episodes.append(
                            {
                                "episode_id": ep["episode_id"],
                                "length": ep["length"],
                                "success": ep["success"],
                                "layout_name": ep["layout_name"],
                                "goal_region_id": ep.get("goal_region_id", 0),
                                "planner_type": ep.get("planner_type", "expert"),
                                "turn_rate": ep.get("turn_rate", 0.0),
                                "reward_sum": ep.get("reward_sum", 0.0),
                                "label": 1,
                                "source": "expert",
                            }
                        )

    # Remap episode IDs to contiguous
    all_ids = [int(ep["episode_id"]) for ep in labeled_episodes]
    old_to_new = {old: i for i, old in enumerate(sorted(set(all_ids)))}
    labeled["episode_id"] = np.array([old_to_new[int(e)] for e in labeled["episode_id"]], dtype=np.int32)
    for ep in labeled_episodes:
        ep["episode_id"] = old_to_new[int(ep["episode_id"])]

    meta = {
        "source_dataset": os.path.abspath(args.dataset),
        "selection": {
            "pos_top_pct": args.pos_top_pct,
            "pos_feature_pct": args.pos_feature_pct,
            "pos_turn_rate_quantile": args.pos_turn_rate_quantile,
            "pos_turn_rate_cutoff": turn_cutoff,
            "pos_max_per_group": args.pos_max_per_group,
            "pos_cap_near_optimal": args.pos_cap_near_optimal,
            "neg_ratio": args.neg_ratio,
            "neg_worst_success_pct": args.neg_worst_success_pct,
            "neg_max_steps": args.neg_max_steps,
        },
        "counts": {
            "positives": len(positive_ids),
            "negatives": len(negatives),
            "total_eps": len(labeled_episodes),
            "total_transitions": int(len(labeled["a"])),
        },
        "episodes": labeled_episodes,
    }

    np.savez_compressed(out_dir / "discriminator_dataset.npz", **labeled)
    with open(out_dir / "discriminator_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        "[build_discriminator_dataset] "
        f"positives={len(positive_ids)} negatives={len(negatives)} "
        f"transitions={len(labeled['a'])} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
