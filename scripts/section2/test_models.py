#!/usr/bin/env python3
"""Quick test that IQL models can be instantiated and run forward passes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.iql import IQL, DR3Loss, StateEncoder, QNetwork, VNetwork


def test_encoder():
    """Test StateEncoder forward pass."""
    print("Testing StateEncoder...")
    encoder = StateEncoder(grid_channels=3, feature_dim=256, hidden_dim=128)

    # Dummy input
    batch_size = 4
    grid = torch.randint(0, 256, (batch_size, 10, 10, 3), dtype=torch.uint8)
    agent_pos = torch.randint(0, 10, (batch_size, 2), dtype=torch.int16)
    agent_dir = torch.randint(0, 4, (batch_size,), dtype=torch.int8)
    carry = torch.zeros((batch_size, 2), dtype=torch.int16)

    # Forward pass
    features = encoder(grid, agent_pos, agent_dir, carry)

    assert features.shape == (batch_size, 256), f"Expected (4, 256), got {features.shape}"
    print(f"  ✓ Output shape: {features.shape}")
    print(f"  ✓ Feature range: [{features.min():.4f}, {features.max():.4f}]")


def test_q_network():
    """Test QNetwork forward pass."""
    print("\nTesting QNetwork...")
    q_net = QNetwork(feature_dim=256, num_actions=7, hidden_dim=256)

    # Dummy input
    batch_size = 4
    features = torch.randn(batch_size, 256)

    # Forward pass
    q_values = q_net(features)

    assert q_values.shape == (batch_size, 7), f"Expected (4, 7), got {q_values.shape}"
    print(f"  ✓ Output shape: {q_values.shape}")
    print(f"  ✓ Q-value range: [{q_values.min():.4f}, {q_values.max():.4f}]")


def test_v_network():
    """Test VNetwork forward pass."""
    print("\nTesting VNetwork...")
    v_net = VNetwork(feature_dim=256, hidden_dim=256)

    # Dummy input
    batch_size = 4
    features = torch.randn(batch_size, 256)

    # Forward pass
    values = v_net(features)

    assert values.shape == (batch_size,), f"Expected (4,), got {values.shape}"
    print(f"  ✓ Output shape: {values.shape}")
    print(f"  ✓ Value range: [{values.min():.4f}, {values.max():.4f}]")


def test_iql_agent():
    """Test IQL agent instantiation and update."""
    print("\nTesting IQL agent...")
    agent = IQL(
        grid_channels=3,
        num_actions=7,
        feature_dim=256,
        hidden_dim=256,
        device="cpu",
    )

    # Dummy batch
    batch_size = 4
    batch = {
        "s_grid": torch.randint(0, 256, (batch_size, 10, 10, 3), dtype=torch.uint8),
        "s_agent_pos": torch.randint(0, 10, (batch_size, 2), dtype=torch.int16),
        "s_agent_dir": torch.randint(0, 4, (batch_size,), dtype=torch.int8),
        "s_carry": torch.zeros((batch_size, 2), dtype=torch.int16),
        "a": torch.randint(0, 7, (batch_size,), dtype=torch.int8),
        "r": torch.randn(batch_size),
        "sp_grid": torch.randint(0, 256, (batch_size, 10, 10, 3), dtype=torch.uint8),
        "sp_agent_pos": torch.randint(0, 10, (batch_size, 2), dtype=torch.int16),
        "sp_agent_dir": torch.randint(0, 4, (batch_size,), dtype=torch.int8),
        "sp_carry": torch.zeros((batch_size, 2), dtype=torch.int16),
        "done": torch.zeros(batch_size, dtype=torch.bool),
    }

    # Update
    metrics = agent.update(batch)

    print(f"  ✓ Update successful")
    print(f"  ✓ Metrics: {list(metrics.keys())}")
    for k, v in metrics.items():
        print(f"      {k}: {v:.4f}")

    # Test action selection
    grid = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    agent_pos = np.array([5, 5], dtype=np.int16)
    agent_dir = 0
    carry = np.array([0, 0], dtype=np.int16)

    action = agent.get_action(grid, agent_pos, agent_dir, carry)
    assert 0 <= action < 7, f"Invalid action: {action}"
    print(f"  ✓ Action selection: {action}")


def test_dr3_loss():
    """Test DR3 loss computation."""
    print("\nTesting DR3Loss...")
    dr3 = DR3Loss(eps=1e-4, alpha=1.0)

    # Dummy features
    batch_size = 100
    feature_dim = 256
    features = torch.randn(batch_size, feature_dim)

    # Compute loss
    loss = dr3(features)
    print(f"  ✓ DR3 loss: {loss.item():.4f}")

    # Get metrics
    metrics = dr3.get_metrics(features)
    print(f"  ✓ Metrics:")
    for k, v in metrics.items():
        print(f"      {k}: {v:.4f}")


def main():
    print("="*60)
    print("Testing Section 2 Models")
    print("="*60)

    test_encoder()
    test_q_network()
    test_v_network()
    test_iql_agent()
    test_dr3_loss()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    main()
