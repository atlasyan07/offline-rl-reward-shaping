"""Implicit Q-Learning (IQL) algorithm.

Reference: https://arxiv.org/abs/2110.06169
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

from .models import StateEncoder, QNetwork, VNetwork, PolicyNetwork


class IQL(nn.Module):
    """Implicit Q-Learning agent.

    IQL avoids querying Q(s', a') for unseen actions by learning V(s')
    via expectile regression on Q(s, a) for actions in the dataset.

    Loss components:
    1. V-learning: Expectile regression E_τ[Q(s,a) - V(s)]
    2. Q-learning: MSE on r + γV(s') - Q(s,a)
    3. Policy extraction: Advantage-weighted behavioral cloning
    """

    def __init__(
        self,
        grid_channels: int,
        num_actions: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        discount: float = 0.99,
        tau: float = 0.7,  # Expectile for V-learning (0.7 = upper expectile)
        beta: float = 3.0,  # Temperature for advantage-weighted BC
        learning_rate: float = 3e-4,
        target_update_freq: int = 2,
        polyak_tau: float | None = None,
        rnd_alpha: float = 0.0,
        rnd_output_dim: int | None = None,
        use_target_encoder: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.num_actions = num_actions
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.target_update_freq = target_update_freq
        self.polyak_tau = polyak_tau
        self.rnd_alpha = rnd_alpha
        self.rnd_output_dim = rnd_output_dim or feature_dim
        self.use_target_encoder = use_target_encoder
        self.device = device

        # Networks
        self.encoder = StateEncoder(grid_channels, feature_dim, hidden_dim).to(device)
        self.q1 = QNetwork(feature_dim, num_actions, hidden_dim).to(device)
        self.q2 = QNetwork(feature_dim, num_actions, hidden_dim).to(device)
        self.v = VNetwork(feature_dim, hidden_dim).to(device)
        self.actor = PolicyNetwork(feature_dim, num_actions, hidden_dim).to(device)

        # Target networks (for V only, IQL doesn't need Q targets)
        self.v_target = VNetwork(feature_dim, hidden_dim).to(device)
        self.v_target.load_state_dict(self.v.state_dict())
        self.encoder_target = StateEncoder(grid_channels, feature_dim, hidden_dim).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        for p in self.encoder_target.parameters():
            p.requires_grad = False
        # Prime lazy layers so encoder_target matches encoder's grid shape.
        with torch.no_grad():
            dummy_grid = torch.zeros(1, 25, 25, grid_channels, device=device)
            dummy_pos = torch.zeros(1, 2, device=device)
            dummy_dir = torch.zeros(1, device=device, dtype=torch.long)
            dummy_carry = torch.zeros(1, 2, device=device)
            _ = self.encoder_target(dummy_grid, dummy_pos, dummy_dir, dummy_carry)

        # RND regularization (optional)
        self.rnd_target = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.rnd_output_dim),
        ).to(device)
        self.rnd_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.rnd_output_dim),
        ).to(device)
        for p in self.rnd_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate
        )
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.rnd_optimizer = torch.optim.Adam(self.rnd_predictor.parameters(), lr=learning_rate)

        self.update_count = 0

    def _soft_update_v(self) -> None:
        tau = self.polyak_tau
        if tau is None:
            return
        with torch.no_grad():
            for tgt, src in zip(self.v_target.parameters(), self.v.parameters()):
                tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)
            if self.use_target_encoder:
                for tgt, src in zip(self.encoder_target.parameters(), self.encoder.parameters()):
                    tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)

    def encode_state(
        self,
        grid: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_dir: torch.Tensor,
        carry: torch.Tensor,
    ) -> torch.Tensor:
        """Encode state to features."""
        return self.encoder(grid, agent_pos, agent_dir, carry)

    def expectile_loss(self, diff: torch.Tensor, tau: float) -> torch.Tensor:
        """Asymmetric squared loss for expectile regression.

        For τ=0.7:
        - Penalize (Q - V) > 0 more (fitting upper expectile)
        - Penalize (Q - V) < 0 less

        This makes V(s) ≈ τ-quantile of Q(s,a) over dataset actions.
        """
        weight = torch.where(diff > 0, tau, 1 - tau)
        return weight * (diff**2)

    def update_v(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update V via expectile regression on Q(s,a).

        V_τ(s) = argmin_V E_a~D[(Q(s,a) - V)²_τ]

        where (·)²_τ is asymmetric squared loss.
        """
        # Encode states
        features = self.encode_state(
            batch["s_grid"],
            batch["s_agent_pos"],
            batch["s_agent_dir"],
            batch["s_carry"],
        )

        # Get Q-values for dataset actions
        q1_all = self.q1(features)
        q2_all = self.q2(features)
        actions = batch["a"].long()
        q1_a = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_min = torch.min(q1_all, q2_all)
        q_a = q_min.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get V predictions
        v_pred = self.v(features)

        # Expectile loss
        v_loss = self.expectile_loss(q_a - v_pred, self.tau).mean()

        rnd_loss = None
        if self.rnd_alpha > 0.0:
            with torch.no_grad():
                target = self.rnd_target(features)
            pred = self.rnd_predictor(features)
            rnd_loss = F.mse_loss(pred, target)
            v_loss = v_loss + self.rnd_alpha * rnd_loss

        # Update
        self.v_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        if self.rnd_alpha > 0.0:
            self.rnd_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        self.encoder_optimizer.step()
        if self.rnd_alpha > 0.0:
            self.rnd_optimizer.step()

        q_var = q_min.var(dim=1).mean()
        adv_abs_mean = (q_a - v_pred).abs().mean()
        q_spread = (q_min.max(dim=1).values - q_min.min(dim=1).values).mean()
        adv_all = q_min - v_pred.unsqueeze(1)
        adv_spread = (adv_all.max(dim=1).values - adv_all.min(dim=1).values).mean()
        pos_adv_mass = torch.relu(adv_all).mean()
        pos_adv_frac = (adv_all > 0).float().mean()

        return {
            "v_loss": v_loss.item(),
            "v_mean": v_pred.mean().item(),
            "q_mean": q_a.mean().item(),
            "q_var": q_var.item(),
            "adv_abs_mean": adv_abs_mean.item(),
            "q_spread": q_spread.item(),
            "adv_spread": adv_spread.item(),
            "pos_adv_mass": pos_adv_mass.item(),
            "pos_adv_frac": pos_adv_frac.item(),
            **({"rnd_loss": rnd_loss.item()} if rnd_loss is not None else {}),
        }

    def update_q(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update Q via Bellman backup with V(s').

        Q(s,a) ← r + γV(s')

        No max over actions - IQL uses V(s') directly.
        """
        # Encode current states
        features = self.encode_state(
            batch["s_grid"],
            batch["s_agent_pos"],
            batch["s_agent_dir"],
            batch["s_carry"],
        )

        # Encode next states
        with torch.no_grad():
            if self.use_target_encoder:
                next_features = self.encoder_target(
                    batch["sp_grid"],
                    batch["sp_agent_pos"],
                    batch["sp_agent_dir"],
                    batch["sp_carry"],
                )
            else:
                next_features = self.encode_state(
                    batch["sp_grid"],
                    batch["sp_agent_pos"],
                    batch["sp_agent_dir"],
                    batch["sp_carry"],
                )
            # Use target V for stability
            next_v = self.v_target(next_features)
            target_q = batch["r"] + self.discount * (1 - batch["done"].float()) * next_v

        # Get Q predictions
        q1_all = self.q1(features)
        q2_all = self.q2(features)
        actions = batch["a"].long()
        q1_a = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD loss
        q1_loss = F.mse_loss(q1_a, target_q)
        q2_loss = F.mse_loss(q2_a, target_q)
        q_loss = q1_loss + q2_loss

        # Update
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        td_abs = (target_q - q1_a).abs()
        td_abs_mean = td_abs.mean()
        td_abs_p90 = torch.quantile(td_abs, 0.9)
        return {
            "q_loss": q_loss.item(),
            "q1_mean": q1_a.mean().item(),
            "q2_mean": q2_a.mean().item(),
            "target_q_mean": target_q.mean().item(),
            "td_abs_mean": td_abs_mean.item(),
            "td_abs_p90": td_abs_p90.item(),
        }

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single update step (V then Q).

        Args:
            batch: Dictionary with keys:
                - s_grid: (B, H, W, C)
                - s_agent_pos: (B, 2)
                - s_agent_dir: (B,)
                - s_carry: (B, 2)
                - a: (B,)
                - r: (B,)
                - sp_grid: (B, H, W, C)
                - sp_agent_pos: (B, 2)
                - sp_agent_dir: (B,)
                - sp_carry: (B, 2)
                - done: (B,)

        Returns:
            Dictionary of losses and metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Update V
        v_metrics = self.update_v(batch)

        # Update Q
        q_metrics = self.update_q(batch)

        # Update actor (advantage-weighted regression)
        actor_metrics = self.update_actor(batch)

        # Update target V
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            if self.polyak_tau is None:
                self.v_target.load_state_dict(self.v.state_dict())
                if self.use_target_encoder:
                    self.encoder_target.load_state_dict(self.encoder.state_dict())
            else:
                self._soft_update_v()

        return {**v_metrics, **q_metrics, **actor_metrics}

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update actor via advantage-weighted regression."""
        with torch.no_grad():
            features = self.encode_state(
                batch["s_grid"],
                batch["s_agent_pos"],
                batch["s_agent_dir"],
                batch["s_carry"],
            )
            q1 = self.q1(features)
            q2 = self.q2(features)
            q_min = torch.min(q1, q2)
            v = self.v(features)
            actions = batch["a"].long()
            q_a = q_min.gather(1, actions.unsqueeze(1)).squeeze(1)
            adv = q_a - v
            weights = torch.exp(adv / self.beta).clamp(max=100.0)

        features_detached = features.detach()
        logits = self.actor(features_detached)
        log_probs = F.log_softmax(logits, dim=1)
        action_logp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        actor_loss = -(weights * action_logp).mean()
        probs = torch.exp(log_probs)
        policy_entropy = -(probs * log_probs).sum(dim=1).mean()
        unique_actions = torch.unique(actions).numel()
        action_diversity = unique_actions / float(self.num_actions)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "adv_mean": adv.mean().item(),
            "policy_entropy": policy_entropy.item(),
            "action_diversity": float(action_diversity),
        }

    @torch.no_grad()
    def get_action(
        self,
        grid: np.ndarray,
        agent_pos: np.ndarray,
        agent_dir: int,
        carry: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """Select action using learned Q-function.

        Args:
            grid: (H, W, C) uint8
            agent_pos: (2,)
            agent_dir: scalar
            carry: (2,)
            deterministic: If True, take argmax. If False, sample from softmax.

        Returns:
            action: int
        """
        # Add batch dimension and convert to tensors
        grid_t = torch.from_numpy(grid).unsqueeze(0).to(self.device)
        pos_t = torch.from_numpy(agent_pos).unsqueeze(0).to(self.device)
        dir_t = torch.tensor([agent_dir]).to(self.device)
        carry_t = torch.from_numpy(carry).unsqueeze(0).to(self.device)

        # Encode and get Q-values
        features = self.encode_state(grid_t, pos_t, dir_t, carry_t)
        logits = self.actor(features)

        if deterministic:
            action = logits.argmax(dim=1).item()
        else:
            probs = F.softmax(logits, dim=1)
            action = torch.multinomial(probs, 1).item()

        return action

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "encoder_target": self.encoder_target.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "v": self.v.state_dict(),
                "v_target": self.v_target.state_dict(),
                "actor": self.actor.state_dict(),
                "encoder_optimizer": self.encoder_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "update_count": self.update_count,
                "polyak_tau": self.polyak_tau,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        if "encoder_target" in checkpoint:
            self.encoder_target.load_state_dict(checkpoint["encoder_target"])
        else:
            self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.v.load_state_dict(checkpoint["v"])
        self.v_target.load_state_dict(checkpoint["v_target"])
        if "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"])
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.v_optimizer.load_state_dict(checkpoint["v_optimizer"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.update_count = checkpoint["update_count"]
        if "polyak_tau" in checkpoint:
            self.polyak_tau = checkpoint["polyak_tau"]
