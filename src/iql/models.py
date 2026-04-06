"""Neural network models for IQL."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Simple IMPALA-style residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x + residual)
        return x


class StateEncoder(nn.Module):
    """Encode MiniGrid observation to feature vector.

    Takes:
        - grid: (B, H, W, C) uint8 grid
        - agent_pos: (B, 2) int16 position
        - agent_dir: (B,) int8 direction
        - carry: (B, 2) int16 inventory

    Returns:
        - features: (B, feature_dim) float32
    """

    def __init__(
        self,
        grid_channels: int = 3,
        feature_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # CNN for grid encoding (IMPALA-style)
        self.conv1 = nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res3 = ResidualBlock(64)

        # Will be set dynamically based on input size
        self.conv_out_dim = None
        self.fc_grid = None

        # Learned spatial embeddings for position + direction
        self.pos_embed_x = nn.Embedding(32, 32)
        self.pos_embed_y = nn.Embedding(32, 32)
        self.dir_embed = nn.Embedding(4, 16)
        self.fc_carry = nn.Linear(2, 16)

        # Combine all features
        # 128 (grid) + 32 (pos) + 16 (dir) + 16 (carry) = 192
        combined_dim = 128 + 32 + 16 + 16  # = 192
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def _init_conv_out(self, grid: torch.Tensor) -> None:
        """Initialize conv output layer based on input size."""
        if self.fc_grid is not None:
            return

        with torch.no_grad():
            # Run a dummy forward pass to get conv output size
            x = grid.float() / 10.0
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = F.relu(self.conv1(x))
            x = self.res1(x)
            x = F.relu(self.conv2(x))
            x = self.res2(x)
            x = F.relu(self.conv3(x))
            x = self.res3(x)
            x = F.adaptive_avg_pool2d(x, (5, 5))
            self.conv_out_dim = x.shape[1] * x.shape[2] * x.shape[3]

        # Create the linear layer
        self.fc_grid = nn.Linear(self.conv_out_dim, 128).to(grid.device)

    def forward(
        self,
        grid: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_dir: torch.Tensor,
        carry: torch.Tensor,
    ) -> torch.Tensor:
        """Encode state to feature vector.

        Args:
            grid: (B, H, W, C) uint8
            agent_pos: (B, 2) int/float
            agent_dir: (B,) int
            carry: (B, 2) int/float

        Returns:
            features: (B, feature_dim)
        """
        batch_size = grid.shape[0]

        # Initialize conv output layer if needed
        self._init_conv_out(grid)

        # Process grid with CNN
        x = grid.float() / 10.0
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = x.reshape(batch_size, -1)
        grid_feat = F.relu(self.fc_grid(x))

        # Process position (learned embeddings)
        pos_x = torch.clamp(agent_pos[:, 0].long(), 0, 31)
        pos_y = torch.clamp(agent_pos[:, 1].long(), 0, 31)
        pos_feat = self.pos_embed_x(pos_x) + self.pos_embed_y(pos_y)

        # Process direction (learned embedding)
        dir_feat = self.dir_embed(agent_dir.long() % 4)

        # Process carry
        carry_feat = F.relu(self.fc_carry(carry.float()))

        # Combine all features
        combined = torch.cat([grid_feat, pos_feat, dir_feat, carry_feat], dim=1)
        features = self.fc_combined(combined)

        return features


class QNetwork(nn.Module):
    """Q-function: Q(s, a)."""

    def __init__(self, feature_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            features: (B, feature_dim)

        Returns:
            q_values: (B, num_actions)
        """
        return self.net(features)


class VNetwork(nn.Module):
    """Value function: V(s)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            features: (B, feature_dim)

        Returns:
            values: (B,)
        """
        return self.net(features).squeeze(-1)


class PolicyNetwork(nn.Module):
    """Policy network: logits for π(a|s)."""

    def __init__(self, feature_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
