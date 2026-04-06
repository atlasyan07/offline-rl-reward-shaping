#!/usr/bin/env python3
"""Parse TensorBoard scalar logs into compact CSV/JSON summaries."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TensorBoard scalar logs")
    parser.add_argument("--logdir", required=True, help="TensorBoard logdir")
    parser.add_argument("--out-dir", required=True, help="Output directory for summaries")
    parser.add_argument("--prefix", action="append", default=[], help="Filter tags by prefix (repeatable)")
    parser.add_argument("--latest-only", action="store_true", help="Use only the newest event file")
    parser.add_argument("--max-events", type=int, default=0, help="Max events to read per file (0=all)")
    parser.add_argument("--progress", action="store_true", help="Show a tqdm progress bar")
    return parser.parse_args()


def should_keep(tag: str, prefixes: List[str]) -> bool:
    if not prefixes:
        return True
    return any(tag.startswith(p) for p in prefixes)


def read_events(
    path: str,
    prefixes: List[str],
    max_events: int,
    show_progress: bool,
) -> Dict[str, List[Tuple[int, float]]]:
    scalars: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    loader = EventFileLoader(path)
    count = 0
    iterator = loader.Load()
    if show_progress:
        total = max_events if max_events else None
        iterator = tqdm(iterator, total=total, unit="event", desc=f"Reading {os.path.basename(path)}")
    for event in iterator:
        if max_events and count >= max_events:
            break
        if not event.summary.value:
            continue
        step = int(event.step)
        for v in event.summary.value:
            if not v.HasField("simple_value"):
                continue
            tag = v.tag
            if should_keep(tag, prefixes):
                scalars[tag].append((step, float(v.simple_value)))
        count += 1
    return scalars


def summarize(values: List[Tuple[int, float]]) -> Dict[str, float]:
    steps = np.array([v[0] for v in values], dtype=np.int64)
    vals = np.array([v[1] for v in values], dtype=np.float64)
    if len(vals) == 0:
        return {}
    order = np.argsort(steps)
    steps = steps[order]
    vals = vals[order]
    return {
        "count": int(len(vals)),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first": float(vals[0]),
        "last": float(vals[-1]),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "delta": float(vals[-1] - vals[0]),
    }


def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_files = sorted(
        [p for p in logdir.iterdir() if p.name.startswith("events.out.tfevents")],
        key=lambda p: p.stat().st_mtime,
    )
    if not event_files:
        raise SystemExit(f"No event files found in {logdir}")
    if args.latest_only:
        event_files = [event_files[-1]]

    merged: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for path in event_files:
        if args.progress:
            print(f"[parse_tb_scalars] reading {path.name}")
        scalars = read_events(str(path), args.prefix, args.max_events, args.progress)
        for tag, vals in scalars.items():
            merged[tag].extend(vals)

    # Summaries
    summary_rows = []
    summary_json = {}
    for tag in sorted(merged.keys()):
        stats = summarize(merged[tag])
        if not stats:
            continue
        row = {"tag": tag, **stats}
        summary_rows.append(row)
        summary_json[tag] = stats

    # Write CSV
    csv_path = out_dir / "tb_scalar_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    # Write JSON
    json_path = out_dir / "tb_scalar_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"[parse_tb_scalars] tags={len(summary_rows)} csv={csv_path} json={json_path}")


if __name__ == "__main__":
    main()
