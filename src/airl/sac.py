from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iql.models import StateEncoder


class SACReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.pos = 0
        self.size = 0
        self.storage: Dict[str, np.ndarray] | None = None

    def _init_storage(self, sample: Dict[str, np.ndarray]) -> None:
        self.storage = {}
        for key, value in sample.items():
            shape = (self.capacity,) + value.shape[1:]
            self.storage[key] = np.zeros(shape, dtype=value.dtype)

    def add_rollout(self, rollout: Dict[str, np.ndarray]) -> None:
        if self.storage is None:
            self._init_storage(rollout)
        assert self.storage is not None
        n = len(rollout["a"])
        for i in range(n):
            for key, value in rollout.items():
                self.storage[key][self.pos] = value[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        assert self.storage is not None
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: v[idx] for k, v in self.storage.items()}


class SACAgent(nn.Module):
    def __init__(
        self,
        grid_channels: int,
        num_actions: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        alpha: float = 0.2,
        lr: float = 3e-4,
        target_tau: float = 0.005,
        actor_updates_encoder: bool = False,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.alpha = float(alpha)
        self.target_tau = float(target_tau)
        self.actor_updates_encoder = bool(actor_updates_encoder)

        self.encoder = StateEncoder(grid_channels, feature_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.q1_target = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.q2_target = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        q_params = list(self.encoder.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters())
        self.q_optimizer = torch.optim.Adam(q_params, lr=lr)

        actor_params = list(self.actor.parameters())
        if self.actor_updates_encoder:
            actor_params += list(self.encoder.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr)

    def encode(self, grid, pos, direction, carry) -> torch.Tensor:
        return self.encoder(grid, pos, direction, carry)

    def forward(self, grid, pos, direction, carry) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode(grid, pos, direction, carry)
        logits = self.actor(feat)
        value = torch.zeros(logits.shape[0], device=logits.device)
        return logits, value

    @torch.no_grad()
    def act(self, grid, pos, direction, carry) -> Tuple[int, float, float]:
        logits, _ = self.forward(grid, pos, direction, carry)
        probs = F.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs.gather(1, torch.tensor([[action]], device=probs.device)).squeeze(1))
        return action, float(log_prob.item()), 0.0

    def log_prob(self, grid, pos, direction, carry, actions) -> torch.Tensor:
        logits, _ = self.forward(grid, pos, direction, carry)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    def entropy(self, grid, pos, direction, carry) -> torch.Tensor:
        logits, _ = self.forward(grid, pos, direction, carry)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=1).mean()

    def update(self, batch: Dict[str, torch.Tensor], rewards: torch.Tensor, gamma: float) -> Dict[str, float]:
        s_feat = self.encode(
            batch["s_grid"],
            batch["s_agent_pos"],
            batch["s_agent_dir"],
            batch["s_carry"],
        )
        sp_feat = self.encode(
            batch["sp_grid"],
            batch["sp_agent_pos"],
            batch["sp_agent_dir"],
            batch["sp_carry"],
        )
        actions = batch["a"].long()
        done = batch["done"].float()

        with torch.no_grad():
            next_logits = self.actor(sp_feat)
            next_log_probs = F.log_softmax(next_logits, dim=1)
            next_probs = next_log_probs.exp()
            q1_next = self.q1_target(sp_feat)
            q2_next = self.q2_target(sp_feat)
            q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + (1.0 - done) * gamma * v_next

        q1_pred = self.q1(s_feat).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(s_feat).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        actor_feat = s_feat if self.actor_updates_encoder else s_feat.detach()
        logits = self.actor(actor_feat)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        q1_vals = self.q1(actor_feat).detach()
        q2_vals = self.q2(actor_feat).detach()
        q_min = torch.min(q1_vals, q2_vals)
        actor_loss = (probs * (self.alpha * log_probs - q_min)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update()

        return {
            "actor_loss": float(actor_loss.item()),
            "q_loss": float(q_loss.item()),
        }

    def _soft_update(self) -> None:
        tau = self.target_tau
        with torch.no_grad():
            for tgt, src in zip(self.q1_target.parameters(), self.q1.parameters()):
                tgt.data.mul_(1 - tau).add_(src.data, alpha=tau)
            for tgt, src in zip(self.q2_target.parameters(), self.q2.parameters()):
                tgt.data.mul_(1 - tau).add_(src.data, alpha=tau)
