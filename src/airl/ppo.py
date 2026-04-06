from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iql.models import StateEncoder


@dataclass
class PPOBatch:
    s_grid: torch.Tensor
    s_agent_pos: torch.Tensor
    s_agent_dir: torch.Tensor
    s_carry: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class PPOPolicy(nn.Module):
    def __init__(
        self,
        grid_channels: int,
        num_actions: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = StateEncoder(grid_channels, feature_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, grid, pos, direction, carry) -> torch.Tensor:
        return self.encoder(grid, pos, direction, carry)

    def forward(self, grid, pos, direction, carry) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode(grid, pos, direction, carry)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(1)
        return logits, value

    @torch.no_grad()
    def act(self, grid, pos, direction, carry) -> Tuple[int, float, float]:
        logits, value = self.forward(grid, pos, direction, carry)
        probs = F.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs.gather(1, torch.tensor([[action]], device=probs.device)).squeeze(1))
        return action, float(log_prob.item()), float(value.item())

    def evaluate_actions(self, batch: PPOBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(
            batch.s_grid,
            batch.s_agent_pos,
            batch.s_agent_dir,
            batch.s_carry,
        )
        log_probs = F.log_softmax(logits, dim=1)
        action_logp = log_probs.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        return action_logp, values, entropy
