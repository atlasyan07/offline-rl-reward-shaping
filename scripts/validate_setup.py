#!/usr/bin/env python3
"""Pre-flight validation before running full data collection."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config


def validate_config(config_path: str) -> bool:
    """Validate configuration meets Section 1 requirements."""
    print(f"Validating: {config_path}")
    print("=" * 60)

    cfg = load_config(config_path)

    # Check dataset targets
    dataset_cfg = cfg["dataset"]
    min_episodes = dataset_cfg.get("min_episodes", 0)
    min_transitions = dataset_cfg.get("min_transitions", 0)
    episodes_per_behavior = dataset_cfg.get("episodes_per_behavior", 0)

    print(f"\n✓ Dataset Targets:")
    print(f"  - Min episodes: {min_episodes:,}")
    print(f"  - Min transitions: {min_transitions:,}")
    print(f"  - Episodes per behavior: {episodes_per_behavior}")

    # Check layouts
    train_layouts = dataset_cfg.get("fixed_layouts_train", [])
    eval_layouts = dataset_cfg.get("fixed_layouts_eval", [])

    print(f"\n✓ Layouts:")
    print(f"  - Training layouts: {len(train_layouts)} {train_layouts}")
    print(f"  - Eval layouts: {len(eval_layouts)} {eval_layouts}")

    # Check behaviors
    behaviors = cfg.get("behaviors", {})
    behavior_order = dataset_cfg.get("fixed_behavior_order", [])

    print(f"\n✓ Behaviors:")
    print(f"  - Defined: {list(behaviors.keys())}")
    print(f"  - Execution order: {behavior_order}")

    # Calculate expected episodes and transitions
    num_layouts = len(train_layouts)
    num_behaviors = len(behavior_order)
    total_episodes = num_layouts * num_behaviors * episodes_per_behavior

    avg_steps_target = dataset_cfg.get("avg_steps_target", [100, 200])
    est_avg_steps = sum(avg_steps_target) / 2
    est_transitions = total_episodes * est_avg_steps

    print(f"\n✓ Expected Output:")
    print(f"  - Total episodes: {total_episodes:,}")
    print(f"  - Avg steps per episode: ~{est_avg_steps:.0f}")
    print(f"  - Est. transitions: ~{est_transitions:,.0f}")

    # Validate against requirements
    print(f"\n✓ Section 1 Requirements:")
    meets_episodes = total_episodes >= 1000
    meets_transitions = est_transitions >= 100000
    has_diversity = num_behaviors >= 3

    print(f"  - ≥1,000 episodes: {meets_episodes} ({total_episodes:,})")
    print(f"  - ≥100k transitions: {meets_transitions} ({est_transitions:,.0f})")
    print(f"  - ≥3 behavior types: {has_diversity} ({num_behaviors})")

    # Check diagnostics
    videos_enabled = cfg.get("videos", {}).get("enabled", False)
    plots_enabled = cfg.get("plots", {}).get("enabled", False)
    paths_enabled = cfg.get("paths", {}).get("enabled", False)
    episode_videos_enabled = cfg.get("episode_videos", {}).get("enabled", False)

    print(f"\n✓ Diagnostics Enabled:")
    print(f"  - Example videos: {videos_enabled}")
    print(f"  - Episode videos: {episode_videos_enabled}")
    print(f"  - Plots: {plots_enabled}")
    print(f"  - Path overlays: {paths_enabled}")

    # Overall status
    all_checks = meets_episodes and meets_transitions and has_diversity

    print("\n" + "=" * 60)
    if all_checks:
        print("✅ VALIDATION PASSED - Ready for data collection!")
        return True
    else:
        print("❌ VALIDATION FAILED - Adjust configuration before running")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate dataset configuration")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    success = validate_config(args.config)
    sys.exit(0 if success else 1)
