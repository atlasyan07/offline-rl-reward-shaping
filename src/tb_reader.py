"""TensorBoard log reader utilities for analysis."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader


def read_tb_scalars(
    path: str | Path,
    tags: Optional[List[str]] = None,
    tag_prefix: Optional[str] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    """Read scalar values from a TensorBoard event file.

    Args:
        path: Path to event file or directory containing event files
        tags: List of specific tags to read (None = all)
        tag_prefix: Filter tags by prefix (e.g., "eval/" or "train/")

    Returns:
        Dict mapping tag names to list of (step, value) tuples
    """
    path = Path(path)

    # Find event files
    if path.is_dir():
        event_files = sorted(
            path.glob("events.out.tfevents.*"),
            key=lambda p: p.stat().st_mtime,
        )
    else:
        event_files = [path]

    if not event_files:
        raise FileNotFoundError(f"No event files found at {path}")

    scalars: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    for event_file in event_files:
        loader = EventFileLoader(str(event_file))
        for event in loader.Load():
            if not event.summary.value:
                continue
            step = int(event.step)
            for v in event.summary.value:
                tag = v.tag

                # Filter by specific tags
                if tags and tag not in tags:
                    continue
                # Filter by prefix
                if tag_prefix and not tag.startswith(tag_prefix):
                    continue

                # Extract value - handle both simple_value and tensor formats
                value = None
                if v.HasField("simple_value"):
                    value = float(v.simple_value)
                elif v.HasField("tensor") and v.tensor.float_val:
                    value = float(v.tensor.float_val[0])

                if value is not None:
                    scalars[tag].append((step, value))

    return dict(scalars)


def scalars_to_dataframe(scalars: Dict[str, List[Tuple[int, float]]]) -> pd.DataFrame:
    """Convert scalar dict to a pandas DataFrame with step as index.

    Args:
        scalars: Dict from read_tb_scalars()

    Returns:
        DataFrame with step as index and tags as columns
    """
    if not scalars:
        return pd.DataFrame()

    # Collect all data
    records = []
    for tag, values in scalars.items():
        for step, value in values:
            records.append({"step": step, "tag": tag, "value": value})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Pivot to wide format
    df_wide = df.pivot_table(index="step", columns="tag", values="value", aggfunc="last")
    df_wide = df_wide.sort_index()
    return df_wide


def get_tb_summary(scalars: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics for each scalar tag.

    Args:
        scalars: Dict from read_tb_scalars()

    Returns:
        Dict mapping tag to summary stats (min, max, mean, first, last, etc.)
    """
    summary = {}
    for tag, values in scalars.items():
        if not values:
            continue
        steps = np.array([v[0] for v in values])
        vals = np.array([v[1] for v in values])
        order = np.argsort(steps)
        steps = steps[order]
        vals = vals[order]

        summary[tag] = {
            "count": len(vals),
            "first_step": int(steps[0]),
            "last_step": int(steps[-1]),
            "first": float(vals[0]),
            "last": float(vals[-1]),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "delta": float(vals[-1] - vals[0]),
        }
    return summary


def print_tb_summary(path: str | Path, tag_prefix: Optional[str] = None) -> None:
    """Print a formatted summary of TensorBoard logs.

    Args:
        path: Path to event file or tb directory
        tag_prefix: Optional filter by tag prefix
    """
    scalars = read_tb_scalars(path, tag_prefix=tag_prefix)
    summary = get_tb_summary(scalars)

    if not summary:
        print("No scalar data found.")
        return

    print(f"\nTensorBoard Summary ({len(summary)} tags)")
    print("=" * 80)

    for tag in sorted(summary.keys()):
        s = summary[tag]
        print(f"\n{tag}:")
        print(f"  Steps: {s['first_step']} -> {s['last_step']} ({s['count']} points)")
        print(f"  Value: {s['first']:.4f} -> {s['last']:.4f} (delta: {s['delta']:+.4f})")
        print(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]  Mean: {s['mean']:.4f} ± {s['std']:.4f}")


def load_training_curves(
    tb_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training curves separated by category.

    Args:
        tb_path: Path to TensorBoard event file or directory

    Returns:
        Tuple of (train_df, eval_df, epoch_df) DataFrames
    """
    scalars = read_tb_scalars(tb_path)

    train_scalars = {k: v for k, v in scalars.items() if k.startswith("train/")}
    eval_scalars = {k: v for k, v in scalars.items() if k.startswith("eval/")}
    epoch_scalars = {k: v for k, v in scalars.items() if k.startswith("epoch/")}

    return (
        scalars_to_dataframe(train_scalars),
        scalars_to_dataframe(eval_scalars),
        scalars_to_dataframe(epoch_scalars),
    )


def analyze_tb(path: str | Path) -> Dict:
    """Quick analysis of TensorBoard logs.

    Args:
        path: Path to event file or tb directory

    Returns:
        Dict with scalars, summary, and dataframes
    """
    scalars = read_tb_scalars(path)
    return {
        "scalars": scalars,
        "summary": get_tb_summary(scalars),
        "df": scalars_to_dataframe(scalars),
        "train_df": scalars_to_dataframe({k: v for k, v in scalars.items() if k.startswith("train/")}),
        "eval_df": scalars_to_dataframe({k: v for k, v in scalars.items() if k.startswith("eval/")}),
    }
