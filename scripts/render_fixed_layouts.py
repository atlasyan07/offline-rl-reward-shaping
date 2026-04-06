#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs import make_env
from src.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fixed layouts to images")
    parser.add_argument("--config", required=True, help="Path to section1 config yaml")
    parser.add_argument("--out-dir", default="outputs/fixed_layouts", help="Output directory")
    return parser.parse_args()


def render_layouts(env_cfg, layout_names: List[str], out_dir: str) -> None:
    for name in layout_names:
        env = make_env(env_cfg, render_mode="rgb_array", layout_name=name)
        env.reset(seed=0)
        frame = env.render()
        if frame is None:
            continue
        Image.fromarray(frame).save(os.path.join(out_dir, f"{name}.png"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    dataset_cfg = cfg["dataset"]
    out_dir = args.out_dir
    ensure_dir(out_dir)

    layouts = []
    layouts.extend([str(n) for n in dataset_cfg.get("fixed_layouts_train", [])])
    layouts.extend([str(n) for n in dataset_cfg.get("fixed_layouts_eval", [])])
    if not layouts:
        raise SystemExit("No fixed_layouts_* entries found in config")

    render_layouts(env_cfg, layouts, out_dir)
    print("[render_fixed_layouts] wrote", out_dir)


if __name__ == "__main__":
    main()
