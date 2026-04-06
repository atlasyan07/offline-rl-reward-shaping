#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.envs_harder as envs
from src.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render harder structured layouts from envs_harder")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument(
        "--layouts",
        default="train_easy,train_medium,train_hard,eval_harder,eval_hardest",
        help="Comma-separated layout names",
    )
    parser.add_argument("--out-dir", default="outputs/structured_layouts_harder", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    ensure_dir(args.out_dir)

    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    for i, name in enumerate(layouts):
        env = envs.make_env(env_cfg, render_mode="rgb_array", layout_name=name)
        env.reset(seed=1000 + i)
        frame = env.render()
        if frame is None:
            raise SystemExit("render returned None")
        Image.fromarray(frame).save(os.path.join(args.out_dir, f"{name}.png"))

    print("[render_harder_layouts] wrote", args.out_dir)


if __name__ == "__main__":
    main()
