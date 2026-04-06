from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .planner import ACTIONS


def save_video(frames: List[np.ndarray], path: str, fps: int) -> None:
    imageio.mimsave(path, frames, fps=fps)


def plot_episode_lengths(lengths: Sequence[int], path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=30, color="#2b6cb0", alpha=0.8)
    plt.title("Episode Lengths")
    plt.xlabel("Steps")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_action_distribution(actions: Sequence[int], path: str) -> None:
    action_ids = [int(a) for a in actions]
    if not action_ids:
        return
    counts = {a: 0 for a in range(max(action_ids) + 1)}
    for a in action_ids:
        counts[a] += 1
    labels = [str(a) for a in counts.keys()]
    values = [counts[a] for a in counts.keys()]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#2f855a", alpha=0.8)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_success_rate(successes: Sequence[bool], path: str) -> None:
    rate = np.mean(successes) if successes else 0.0
    plt.figure(figsize=(4, 4))
    plt.bar(["success", "failure"], [rate, 1.0 - rate], color=["#3182ce", "#c53030"])
    plt.ylim(0, 1)
    plt.title("Success Rate")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def render_path_overlay(frame: np.ndarray, path_positions: List[Tuple[int, int]], grid_size: Tuple[int, int]) -> Image.Image:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")

    w, h = img.size
    grid_w, grid_h = grid_size
    cell_w = w / grid_w
    cell_h = h / grid_h

    for idx, (x, y) in enumerate(path_positions):
        x0 = int(x * cell_w)
        y0 = int(y * cell_h)
        x1 = int((x + 1) * cell_w)
        y1 = int((y + 1) * cell_h)
        alpha = 80 + int(120 * (idx / max(1, len(path_positions) - 1)))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 215, 0, alpha), width=2)

    return img
