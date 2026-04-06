#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs import make_env
from src.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render structured MultiRoom layouts for sanity checks")
    parser.add_argument("--config", required=True, help="Path to section1 config yaml")
    parser.add_argument("--count", type=int, default=10, help="Number of layouts to render")
    parser.add_argument("--layouts", default="", help="Comma-separated layout names (overrides count)")
    parser.add_argument("--out-dir", default="outputs/structured_layouts", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    env_cfg = cfg["env"]

    ensure_dir(args.out_dir)
    if args.layouts:
        layout_names = [x.strip() for x in args.layouts.split(",") if x.strip()]
    else:
        layout_names = []
        layout_names.extend([str(n) for n in cfg["dataset"].get("fixed_layouts_train", [])])
        layout_names.extend([str(n) for n in cfg["dataset"].get("fixed_layouts_eval", [])])
        if not layout_names:
            layout_names = ["train_easy"] * args.count

    for i, name in enumerate(layout_names[: args.count]):
        env = make_env(env_cfg, render_mode="rgb_array", layout_name=name)
        env.reset(seed=1000 + i)
        frame = env.render()
        if frame is None:
            raise SystemExit("render returned None")
        region_id = getattr(env, "goal_region_id", "x")
        Image.fromarray(frame).save(os.path.join(args.out_dir, f"layout_{i:02d}_{name}_g{region_id}.png"))

    print("[render_structured_layouts] wrote", args.out_dir)


if __name__ == "__main__":
    main()
