#!/usr/bin/env python3
"""Extract positive expert transitions from a discriminator dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract expert transitions from discriminator dataset")
    parser.add_argument("--dataset", required=True, help="Path to discriminator_dataset.npz")
    parser.add_argument("--metadata", required=True, help="Path to discriminator_metadata.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for expert dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "label" not in data:
        raise SystemExit("Discriminator dataset missing 'label' array")

    labels = data["label"]
    pos_mask = labels == 1
    if pos_mask.sum() == 0:
        raise SystemExit("No positive transitions found (label==1)")

    expert: Dict[str, np.ndarray] = {k: v[pos_mask] for k, v in data.items()}

    # Remap episode ids to contiguous
    episode_ids = expert["episode_id"].astype(np.int64)
    unique_eps = np.unique(episode_ids)
    remap = {int(old): int(new) for new, old in enumerate(unique_eps)}
    expert["episode_id"] = np.array([remap[int(e)] for e in episode_ids], dtype=np.int32)

    # Filter metadata episodes to positives only and remap IDs
    expert_episodes = []
    for ep in metadata.get("episodes", []):
        if ep.get("label") != 1:
            continue
        old_id = int(ep["episode_id"])
        if old_id not in remap:
            continue
        ep_copy = dict(ep)
        ep_copy["episode_id"] = remap[old_id]
        expert_episodes.append(ep_copy)

    expert_meta = {
        "source_dataset": str(Path(args.dataset).resolve()),
        "selection": {
            "label": 1,
        },
        "counts": {
            "episodes": len(expert_episodes),
            "transitions": int(len(expert["a"])),
        },
        "episodes": expert_episodes,
        "success_rate": float(np.mean([e.get("success", False) for e in expert_episodes])) if expert_episodes else 0.0,
    }

    np.savez_compressed(out_dir / "expert_dataset.npz", **expert)
    with open(out_dir / "expert_metadata.json", "w", encoding="utf-8") as f:
        json.dump(expert_meta, f, indent=2)

    print(f"[extract_expert_from_discriminator] episodes={len(expert_episodes)} transitions={len(expert['a'])}")
    print(f"[extract_expert_from_discriminator] out_dir={out_dir}")


if __name__ == "__main__":
    main()
