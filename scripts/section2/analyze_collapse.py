#!/usr/bin/env python3
"""Analyze representation collapse in trained IQL models.

This script provides diagnostics to observe (not engineer) collapse:
1. Compute effective rank of feature representations
2. Visualize eigenvalue spectrum of feature covariance
3. Compare IQL vs IQL+DR3
4. Show that collapse correlates with poor performance
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.iql import IQL, DR3Loss


def load_dataset(dataset_path: str):
    """Load offline dataset."""
    print(f"Loading dataset from {dataset_path}")
    data = np.load(dataset_path)

    s_grid = torch.from_numpy(data["s_grid"])
    s_agent_pos = torch.from_numpy(data["s_agent_pos"])
    s_agent_dir = torch.from_numpy(data["s_agent_dir"])
    s_carry = torch.from_numpy(data["s_carry"])

    dataset = TensorDataset(s_grid, s_agent_pos, s_agent_dir, s_carry)
    return dataset


def compute_feature_statistics(agent: IQL, dataset: TensorDataset, num_samples: int = 10000):
    """Compute feature statistics over dataset.

    Returns:
        features: (N, D) feature matrix
        metrics: dict with collapse diagnostics
    """
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    all_features = []
    count = 0

    agent.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing features"):
            s_grid, s_agent_pos, s_agent_dir, s_carry = batch

            # Move to device
            s_grid = s_grid.to(agent.device)
            s_agent_pos = s_agent_pos.to(agent.device)
            s_agent_dir = s_agent_dir.to(agent.device)
            s_carry = s_carry.to(agent.device)

            # Encode
            features = agent.encode_state(s_grid, s_agent_pos, s_agent_dir, s_carry)
            all_features.append(features.cpu())

            count += features.shape[0]
            if count >= num_samples:
                break

    # Concatenate
    features = torch.cat(all_features, dim=0)[:num_samples]

    # Compute metrics using DR3Loss
    dr3 = DR3Loss()
    metrics = dr3.get_metrics(features)

    # Compute eigenvalue spectrum
    features_centered = features - features.mean(dim=0, keepdim=True)
    cov = (features_centered.T @ features_centered) / features.shape[0]
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0].cpu().numpy()
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    metrics["eigenvalues"] = eigenvalues.tolist()

    return features.cpu().numpy(), metrics


def plot_eigenvalue_spectrum(results: dict, output_path: str):
    """Plot eigenvalue spectrum comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Eigenvalue spectrum
    ax = axes[0]
    for model_name, metrics in results.items():
        eigenvalues = np.array(metrics["eigenvalues"])
        ax.plot(eigenvalues, label=model_name, marker="o", markersize=3)

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale("log")
    ax.set_title("Eigenvalue Spectrum of Feature Covariance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative variance explained
    ax = axes[1]
    for model_name, metrics in results.items():
        eigenvalues = np.array(metrics["eigenvalues"])
        cumsum = np.cumsum(eigenvalues) / eigenvalues.sum()
        ax.plot(cumsum, label=model_name, marker="o", markersize=3)

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Cumulative Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved eigenvalue spectrum: {output_path}")
    plt.close()


def plot_metrics_comparison(results: dict, output_path: str):
    """Plot bar chart comparing collapse metrics."""
    metrics_to_plot = ["rank", "eigenvalue_ratio", "feature_std"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric_name in enumerate(metrics_to_plot):
        ax = axes[i]
        values = [results[name][metric_name] for name in model_names]
        ax.bar(model_names, values)
        ax.set_title(metric_name.replace("_", " ").title())
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x labels if needed
        if len(model_names) > 2:
            ax.set_xticklabels(model_names, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved metrics comparison: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze representation collapse")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to model checkpoints")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for models")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to use")
    args = parser.parse_args()

    if len(args.models) != len(args.labels):
        raise ValueError("Number of models must match number of labels")

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Analyze each model
    results = {}

    for model_path, label in zip(args.models, args.labels):
        print(f"\n{'='*60}")
        print(f"Analyzing: {label}")
        print(f"{'='*60}")

        # Load model
        # Create dummy agent (will load weights)
        agent = IQL(
            grid_channels=3,
            num_actions=7,
            feature_dim=256,
            hidden_dim=256,
        )
        agent.load(model_path)
        agent.eval()

        # Compute statistics
        features, metrics = compute_feature_statistics(agent, dataset, args.num_samples)

        results[label] = metrics

        print(f"\nMetrics for {label}:")
        print(f"  - Effective rank: {metrics['rank']:.4f}")
        print(f"  - Top eigenvalue: {metrics['top_eigenvalue']:.4f}")
        print(f"  - Eigenvalue ratio: {metrics['eigenvalue_ratio']:.4f}")
        print(f"  - Feature std: {metrics['feature_std']:.4f}")

    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "collapse_analysis.json")
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in v.items()}
                       for k, v in results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # Plot comparisons
    spectrum_path = os.path.join(args.output_dir, "eigenvalue_spectrum.png")
    plot_eigenvalue_spectrum(results, spectrum_path)

    metrics_path = os.path.join(args.output_dir, "collapse_metrics.png")
    plot_metrics_comparison(results, metrics_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
