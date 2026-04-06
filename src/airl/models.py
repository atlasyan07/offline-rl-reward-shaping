from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.iql.models import StateEncoder


class AIRLReward(nn.Module):
    """AIRL reward with g(s,a) + gamma*h(s') - h(s).

    - g(s,a) is an MLP that outputs reward given state features and action
    - h(s) is the shaping potential (2-layer MLP)
    """

    def __init__(
        self,
        grid_channels: int,
        num_actions: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        state_only_reward: bool = False,
        g_clip: float | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.num_actions = num_actions
        self.state_only_reward = state_only_reward
        self.g_clip = g_clip
        self.encoder = StateEncoder(grid_channels, feature_dim, hidden_dim)

        # g(s,a) or g(s) - reward depends on state (and action if enabled)
        if state_only_reward:
            self.g_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.g_head = nn.Sequential(
                nn.Linear(feature_dim + num_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        # h(s) - shaping potential (state only)
        self.h_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_state(
        self,
        grid: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_dir: torch.Tensor,
        carry: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(grid, agent_pos, agent_dir, carry)

    def g(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Reward function g(s,a)."""
        if self.state_only_reward:
            g_vals = self.g_head(features).squeeze(1)
            if self.g_clip is not None:
                g_vals = torch.tanh(g_vals) * self.g_clip
            return g_vals
        action_onehot = torch.zeros(features.shape[0], self.num_actions, device=features.device)
        action_onehot.scatter_(1, actions.long().unsqueeze(1), 1.0)
        x = torch.cat([features, action_onehot], dim=1)
        g_vals = self.g_head(x).squeeze(1)
        if self.g_clip is not None:
            g_vals = torch.tanh(g_vals) * self.g_clip
        return g_vals

    def h(self, features: torch.Tensor) -> torch.Tensor:
        return self.h_head(features).squeeze(1)

    def f(
        self,
        s_feat: torch.Tensor,
        a: torch.Tensor,
        sp_feat: torch.Tensor,
        done: torch.Tensor | None = None,
    ) -> torch.Tensor:
        g_sa = self.g(s_feat, a)
        h_s = self.h(s_feat)
        h_sp = self.h(sp_feat)
        if done is not None:
            h_sp = h_sp * (1.0 - done.float())
        return g_sa + self.gamma * h_sp - h_s

    def discriminate(
        self,
        f_sa: torch.Tensor,
        log_pi: torch.Tensor,
        use_logsumexp: bool = False,
    ) -> torch.Tensor:
        # D = sigmoid(f - log_pi) or exp(f - logsumexp(f, log_pi))
        if use_logsumexp:
            log_pq = torch.logsumexp(torch.stack([f_sa, log_pi], dim=1), dim=1)
            return torch.exp(f_sa - log_pq)
        return torch.sigmoid(f_sa - log_pi)
