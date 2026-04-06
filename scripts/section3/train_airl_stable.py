#!/usr/bin/env python3
"""Simplified, stability-first AIRL training (PPO policy).

Design goals:
- Limit moving parts: PPO only, modest discriminator strength.
- Use policy replay for more stable negatives.
- Keep defaults conservative and easy to reason about.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.airl import AIRLReward, PPOPolicy, PPOBatch
from src.envs import make_env
from src.utils import load_config, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AIRL reward (stable PPO)")
    parser.add_argument("--config", required=True, help="Path to AIRL config yaml")
    return parser.parse_args()


def _steps_to_done(episode_ids: np.ndarray, done: np.ndarray) -> np.ndarray:
    steps = np.full(len(done), np.inf, dtype=np.float32)
    for ep_id in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == ep_id)
        if idx.size == 0:
            continue
        done_idx = idx[done[idx].astype(bool)]
        if done_idx.size == 0:
            continue
        last_done = int(done_idx[-1])
        steps[idx] = (last_done - idx).astype(np.float32)
    return steps


def load_expert_dataset(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    expert = {k: data[k] for k in data.files}
    r = expert["r"]
    n_goal = (r > 0).sum()
    print(f"[airl] Expert dataset: {len(r)} transitions, {n_goal} goal ({100*n_goal/len(r):.1f}%)")

    episode_ids = expert.get("episode_id")
    done = expert.get("done")
    if episode_ids is not None and done is not None:
        expert["steps_to_done"] = _steps_to_done(episode_ids, done)
    expert["goal_indices"] = np.flatnonzero(r > 0)
    expert["non_goal_indices"] = np.flatnonzero(r <= 0)
    return expert


def sample_expert(
    batch_size: int,
    expert: Dict[str, np.ndarray],
    goal_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    goal_idx = expert.get("goal_indices")
    non_goal_idx = expert.get("non_goal_indices")
    if goal_idx is None or non_goal_idx is None or goal_ratio <= 0.0:
        n = len(expert["a"])
        idx = rng.integers(0, n, size=batch_size)
    else:
        n_goal = int(np.round(batch_size * goal_ratio))
        n_non = batch_size - n_goal
        goal_pick = rng.choice(goal_idx, size=n_goal, replace=goal_idx.size < n_goal)
        non_pick = rng.choice(non_goal_idx, size=n_non, replace=non_goal_idx.size < n_non)
        idx = np.concatenate([goal_pick, non_pick])
        rng.shuffle(idx)
    skip = {"planner_type", "episode_id", "goal_indices", "non_goal_indices"}
    return {k: v[idx] for k, v in expert.items() if k not in skip}


def compute_near_terminal_weights(
    batch: Dict[str, torch.Tensor],
    near_steps: int,
    weight: float,
) -> torch.Tensor | None:
    if near_steps <= 0 or weight <= 0.0:
        return None
    if "steps_to_done" in batch:
        mask = batch["steps_to_done"] <= float(near_steps)
    else:
        mask = batch["done"].bool()
    weights = torch.ones_like(batch["a"], dtype=torch.float32)
    weights = weights + weight * mask.float()
    return weights


def compute_log_pi(policy: PPOPolicy, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits, _ = policy.forward(
        batch["s_grid"],
        batch["s_agent_pos"],
        batch["s_agent_dir"],
        batch["s_carry"],
    )
    log_probs = F.log_softmax(logits, dim=1)
    return log_probs.gather(1, batch["a"].long().unsqueeze(1)).squeeze(1)


def collect_rollouts(
    env_cfg: dict,
    layouts: List[str],
    policy: PPOPolicy,
    rollout_steps: int,
    max_steps: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    env = make_env(env_cfg, render_mode=None, layout_name=rng.choice(layouts))
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    data = {k: [] for k in [
        "s_grid", "s_agent_pos", "s_agent_dir", "s_carry",
        "a", "logp", "value",
        "sp_grid", "sp_agent_pos", "sp_agent_dir", "sp_carry",
        "done",
    ]}

    steps = 0
    ep_steps = 0
    while steps < rollout_steps:
        grid = obs["image"]
        pos = env.agent_pos
        direction = env.agent_dir
        carry = np.array([0, 0], dtype=np.int16)

        grid_t = torch.from_numpy(grid).unsqueeze(0).to(device)
        pos_t = torch.from_numpy(np.array(pos, dtype=np.int16)).unsqueeze(0).to(device)
        dir_t = torch.tensor([direction], device=device)
        carry_t = torch.from_numpy(carry).unsqueeze(0).to(device)

        action, logp, value = policy.act(grid_t, pos_t, dir_t, carry_t)

        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        data["s_grid"].append(grid)
        data["s_agent_pos"].append(pos)
        data["s_agent_dir"].append(direction)
        data["s_carry"].append(carry)
        data["a"].append(action)
        data["logp"].append(logp)
        data["value"].append(value)

        data["sp_grid"].append(next_obs["image"])
        data["sp_agent_pos"].append(env.agent_pos)
        data["sp_agent_dir"].append(env.agent_dir)
        data["sp_carry"].append(carry)
        data["done"].append(done)

        obs = next_obs
        steps += 1
        ep_steps += 1

        if done or ep_steps >= max_steps:
            env = make_env(env_cfg, render_mode=None, layout_name=rng.choice(layouts))
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            ep_steps = 0

    return {k: np.array(v) for k, v in data.items()}


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * (1.0 - dones[t]) * next_value - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def build_heatmap(
    reward: AIRLReward,
    env_cfg: dict,
    layout_name: str,
    device: torch.device,
) -> np.ndarray:
    env = make_env(env_cfg, render_mode=None, layout_name=layout_name)
    env.reset(seed=0)
    grid = env.grid.encode().astype(np.uint8)
    h_map = np.full((grid.shape[0], grid.shape[1]), np.nan, dtype=np.float32)

    positions = []
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            obj = env.grid.get(x, y)
            if obj is None:
                positions.append((x, y))

    if not positions:
        return h_map

    pos_arr = np.array(positions, dtype=np.int16)
    grid_batch = np.repeat(grid[None, ...], len(positions), axis=0)
    dir_arr = np.zeros(len(positions), dtype=np.int64)
    carry_arr = np.zeros((len(positions), 2), dtype=np.int16)

    with torch.no_grad():
        feat = reward.encode_state(
            torch.from_numpy(grid_batch).to(device),
            torch.from_numpy(pos_arr).to(device),
            torch.from_numpy(dir_arr).to(device),
            torch.from_numpy(carry_arr).to(device),
        )
        h_vals = reward.h(feat).cpu().numpy()

    for (x, y), val in zip(positions, h_vals):
        h_map[y, x] = float(val)
    return h_map


def save_heatmap(path: str, heatmap: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def compute_disc_grad_penalty(
    reward: AIRLReward,
    s_feat: torch.Tensor,
    sp_feat: torch.Tensor,
    actions: torch.Tensor,
    log_pi: torch.Tensor,
    use_logsumexp: bool,
    eps: float,
) -> torch.Tensor:
    s_feat = s_feat.detach().requires_grad_(True)
    sp_feat = sp_feat.detach().requires_grad_(True)
    f_sa = reward.f(s_feat, actions, sp_feat)
    d_val = reward.discriminate(f_sa, log_pi, use_logsumexp=use_logsumexp)
    grad_s, grad_sp = autograd.grad(
        outputs=d_val.sum(),
        inputs=[s_feat, sp_feat],
        create_graph=True,
    )
    grad_norm = torch.sqrt((grad_s ** 2).sum(dim=1) + (grad_sp ** 2).sum(dim=1) + eps)
    return ((grad_norm - 1.0) ** 2).mean()


class PolicyReplay:
    def __init__(self, max_iters: int):
        self.max_iters = max_iters
        self.rollouts: deque[Dict[str, np.ndarray]] = deque(maxlen=max_iters)
        self.cum_sizes: np.ndarray = np.array([], dtype=np.int64)
        self.total = 0

    def add(self, rollout: Dict[str, np.ndarray]) -> None:
        slim = {k: rollout[k] for k in [
            "s_grid", "s_agent_pos", "s_agent_dir", "s_carry",
            "sp_grid", "sp_agent_pos", "sp_agent_dir", "sp_carry",
            "a", "done",
        ]}
        self.rollouts.append(slim)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self.rollouts:
            self.cum_sizes = np.array([], dtype=np.int64)
            self.total = 0
            return
        sizes = [len(r["a"]) for r in self.rollouts]
        self.cum_sizes = np.cumsum(sizes)
        self.total = int(self.cum_sizes[-1])

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if self.total == 0:
            raise ValueError("Replay buffer is empty")
        idx = rng.integers(0, self.total, size=batch_size)
        rollout_idx = np.searchsorted(self.cum_sizes, idx, side="right")
        starts = np.concatenate(([0], self.cum_sizes[:-1]))
        step_idx = idx - starts[rollout_idx]

        batch = {}
        for k in self.rollouts[0].keys():
            samples = [self.rollouts[r_i][k][s_i] for r_i, s_i in zip(rollout_idx, step_idx)]
            batch[k] = np.stack(samples, axis=0)
        return batch


def ppo_update(
    policy: PPOPolicy,
    policy_opt: torch.optim.Optimizer,
    batch: PPOBatch,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    updates_per_iter: int,
    minibatches: int,
    grad_clip: float | None,
) -> float:
    n = batch.actions.shape[0]
    indices = np.arange(n)
    entropy_accum = 0.0
    steps = 0

    for _ in range(updates_per_iter):
        np.random.shuffle(indices)
        split = np.array_split(indices, minibatches)
        for mb_idx in split:
            mb = PPOBatch(
                s_grid=batch.s_grid[mb_idx],
                s_agent_pos=batch.s_agent_pos[mb_idx],
                s_agent_dir=batch.s_agent_dir[mb_idx],
                s_carry=batch.s_carry[mb_idx],
                actions=batch.actions[mb_idx],
                log_probs=batch.log_probs[mb_idx],
                returns=batch.returns[mb_idx],
                advantages=batch.advantages[mb_idx],
            )
            new_logp, values_pred, entropy = policy.evaluate_actions(mb)
            ratio = (new_logp - mb.log_probs).exp()
            surr1 = ratio * mb.advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb.advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, mb.returns)
            ppo_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            policy_opt.zero_grad()
            ppo_loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            policy_opt.step()
            entropy_accum += float(entropy.item())
            steps += 1

    return entropy_accum / max(steps, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "checkpoints"))
    ensure_dir(os.path.join(output_dir, "heatmaps"))
    ensure_dir(os.path.join(output_dir, "diagnostics"))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    expert = load_expert_dataset(cfg["expert_dataset"])
    env_cfg = load_config(cfg["env_config"])["env"]
    layouts = cfg["layouts"]

    reward = AIRLReward(
        grid_channels=cfg["model"]["grid_channels"],
        num_actions=cfg["model"]["num_actions"],
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        gamma=cfg["airl"]["gamma"],
        state_only_reward=bool(cfg["airl"].get("state_only_reward", False)),
        g_clip=float(cfg["airl"]["g_clip"]) if cfg["airl"].get("g_clip") is not None else None,
    ).to(device)

    policy = PPOPolicy(
        grid_channels=cfg["model"]["grid_channels"],
        num_actions=cfg["model"]["num_actions"],
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=float(cfg["ppo"]["lr"]))
    reward_opt = torch.optim.Adam(reward.parameters(), lr=float(cfg["airl"]["lr"]), weight_decay=1e-4)

    log_path = os.path.join(output_dir, "airl_metrics.csv")
    log_exists = os.path.exists(log_path)

    replay_iters = int(cfg["airl"].get("replay_buffer_iters", 0))
    replay = PolicyReplay(replay_iters) if replay_iters > 0 else None

    disc_updates_per_iter = int(cfg["airl"].get("disc_updates_per_iter", 1))
    ppo_warmup_iters = int(cfg["airl"].get("ppo_warmup_iters", 0))
    use_logsumexp = bool(cfg["airl"].get("use_logsumexp", False))
    grad_penalty_weight = float(cfg["airl"].get("disc_grad_penalty", 0.0))
    grad_penalty_eps = float(cfg["airl"].get("disc_grad_penalty_eps", 1e-12))
    policy_reward_mode = cfg["airl"].get("policy_reward", "proxy").lower()
    reward_reg_lambda = float(cfg["airl"].get("reward_reg_lambda", 0.0))
    policy_freeze_iters = int(cfg["airl"].get("policy_freeze_iters", 0))
    goal_batch_ratio = float(cfg["airl"].get("goal_batch_ratio", 0.0))
    near_terminal_steps = int(cfg["airl"].get("near_terminal_steps", 0))
    near_terminal_weight = float(cfg["airl"].get("near_terminal_weight", 0.0))
    disc_grad_clip = cfg["airl"].get("grad_clip")

    ppo_updates = int(cfg["ppo"].get("updates_per_iter", 1))
    ppo_minibatches = int(cfg["ppo"].get("minibatches", 1))
    ppo_grad_clip = cfg["ppo"].get("grad_clip")

    prev_heat = None
    pbar = tqdm(range(1, cfg["airl"]["iterations"] + 1), desc="AIRL Iterations")
    for it in pbar:
        rollout = collect_rollouts(
            env_cfg,
            layouts,
            policy,
            rollout_steps=cfg["ppo"]["rollout_steps"],
            max_steps=cfg["ppo"]["max_steps"],
            device=device,
            rng=rng,
        )

        if replay is not None:
            replay.add(rollout)

        policy_batch = {k: torch.from_numpy(v).to(device) for k, v in rollout.items()}

        disc_losses = []
        reg_loss = torch.tensor(0.0, device=device)
        for _ in range(disc_updates_per_iter):
            expert_batch = sample_expert(cfg["airl"]["batch_size"], expert, goal_batch_ratio, rng)
            expert_batch = {k: torch.from_numpy(v).to(device) for k, v in expert_batch.items()}

            if replay is not None and replay.total >= cfg["airl"]["batch_size"]:
                policy_sample_np = replay.sample(cfg["airl"]["batch_size"], rng)
                policy_sample = {k: torch.from_numpy(v).to(device) for k, v in policy_sample_np.items()}
            else:
                n_rollout = len(rollout["a"])
                idx = rng.integers(0, n_rollout, size=cfg["airl"]["batch_size"])
                policy_sample = {k: torch.from_numpy(v[idx]).to(device) for k, v in rollout.items()}

            s_feat_p = reward.encode_state(
                policy_sample["s_grid"],
                policy_sample["s_agent_pos"],
                policy_sample["s_agent_dir"],
                policy_sample["s_carry"],
            )
            sp_feat_p = reward.encode_state(
                policy_sample["sp_grid"],
                policy_sample["sp_agent_pos"],
                policy_sample["sp_agent_dir"],
                policy_sample["sp_carry"],
            )
            f_policy = reward.f(s_feat_p, policy_sample["a"], sp_feat_p, done=policy_sample["done"])

            s_feat_e = reward.encode_state(
                expert_batch["s_grid"],
                expert_batch["s_agent_pos"],
                expert_batch["s_agent_dir"],
                expert_batch["s_carry"],
            )
            sp_feat_e = reward.encode_state(
                expert_batch["sp_grid"],
                expert_batch["sp_agent_pos"],
                expert_batch["sp_agent_dir"],
                expert_batch["sp_carry"],
            )
            f_expert = reward.f(s_feat_e, expert_batch["a"], sp_feat_e, done=expert_batch["done"])
            g_expert = reward.g(s_feat_e, expert_batch["a"])

            log_pi_expert = compute_log_pi(policy, expert_batch).detach()
            log_pi_policy = compute_log_pi(policy, policy_sample).detach()

            d_expert = reward.discriminate(f_expert, log_pi_expert, use_logsumexp=use_logsumexp)
            d_policy = reward.discriminate(f_policy, log_pi_policy, use_logsumexp=use_logsumexp)

            eps = 1e-6
            w_expert = compute_near_terminal_weights(expert_batch, near_terminal_steps, near_terminal_weight)
            w_policy = compute_near_terminal_weights(policy_sample, near_terminal_steps, near_terminal_weight)
            expert_loss = -torch.log(d_expert + eps)
            policy_loss = -torch.log(1 - d_policy + eps)
            if w_expert is not None:
                expert_loss = (expert_loss * w_expert).sum() / w_expert.sum().clamp_min(1.0)
            else:
                expert_loss = expert_loss.mean()
            if w_policy is not None:
                policy_loss = (policy_loss * w_policy).sum() / w_policy.sum().clamp_min(1.0)
            else:
                policy_loss = policy_loss.mean()
            disc_loss = expert_loss + policy_loss

            reg_loss = torch.tensor(0.0, device=device)
            if reward_reg_lambda > 0.0:
                done_mask = expert_batch["done"].bool()
                if done_mask.any():
                    reg_loss = F.mse_loss(g_expert[done_mask], expert_batch["r"][done_mask])
                disc_loss = disc_loss + reward_reg_lambda * reg_loss

            grad_pen = torch.tensor(0.0, device=device)
            if grad_penalty_weight > 0.0:
                grad_pen = compute_disc_grad_penalty(
                    reward,
                    s_feat_p,
                    sp_feat_p,
                    policy_sample["a"],
                    log_pi_policy,
                    use_logsumexp,
                    grad_penalty_eps,
                )
                grad_pen += compute_disc_grad_penalty(
                    reward,
                    s_feat_e,
                    sp_feat_e,
                    expert_batch["a"],
                    log_pi_expert,
                    use_logsumexp,
                    grad_penalty_eps,
                )
                grad_pen = grad_pen * 0.5

            reward_opt.zero_grad()
            (disc_loss + grad_penalty_weight * grad_pen).backward()
            if disc_grad_clip is not None and disc_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(reward.parameters(), disc_grad_clip)
            reward_opt.step()

            disc_losses.append(disc_loss.item())

        disc_loss_avg = np.mean(disc_losses)
        disc_acc = float(((d_expert > 0.5).float().mean() + (d_policy < 0.5).float().mean()) / 2.0)

        skip_ppo = (it <= ppo_warmup_iters) or (it <= policy_freeze_iters)
        entropy_val = 0.0
        if not skip_ppo:
            with torch.no_grad():
                s_feat_ppo = reward.encode_state(
                    policy_batch["s_grid"],
                    policy_batch["s_agent_pos"],
                    policy_batch["s_agent_dir"],
                    policy_batch["s_carry"],
                )
                sp_feat_ppo = reward.encode_state(
                    policy_batch["sp_grid"],
                    policy_batch["sp_agent_pos"],
                    policy_batch["sp_agent_dir"],
                    policy_batch["sp_carry"],
                )
                f_ppo = reward.f(s_feat_ppo, policy_batch["a"], sp_feat_ppo, done=policy_batch["done"])
                log_pi_ppo = compute_log_pi(policy, policy_batch)
                if policy_reward_mode == "g":
                    ppo_rewards = reward.g(s_feat_ppo, policy_batch["a"]).cpu().numpy().astype(np.float32)
                elif policy_reward_mode == "f":
                    ppo_rewards = f_ppo.cpu().numpy().astype(np.float32)
                else:
                    d_ppo = reward.discriminate(f_ppo, log_pi_ppo, use_logsumexp=use_logsumexp)
                    eps = 1e-6
                    ppo_rewards = (torch.log(d_ppo + eps) - torch.log(1 - d_ppo + eps)).cpu().numpy().astype(np.float32)

            values = rollout["value"].astype(np.float32)
            dones = rollout["done"].astype(np.float32)
            adv, ret = compute_gae(ppo_rewards, values, dones, cfg["airl"]["gamma"], cfg["ppo"]["gae_lambda"])
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            batch = PPOBatch(
                s_grid=policy_batch["s_grid"],
                s_agent_pos=policy_batch["s_agent_pos"],
                s_agent_dir=policy_batch["s_agent_dir"],
                s_carry=policy_batch["s_carry"],
                actions=policy_batch["a"].long(),
                log_probs=policy_batch["logp"].detach(),
                returns=torch.from_numpy(ret).to(device),
                advantages=torch.from_numpy(adv).to(device),
            )
            entropy_val = ppo_update(
                policy,
                policy_opt,
                batch,
                clip_ratio=float(cfg["ppo"]["clip_ratio"]),
                value_coef=float(cfg["ppo"]["value_coef"]),
                entropy_coef=float(cfg["ppo"]["entropy_coef"]),
                updates_per_iter=ppo_updates,
                minibatches=ppo_minibatches,
                grad_clip=ppo_grad_clip,
            )

        with torch.no_grad():
            s_feat_curr = reward.encode_state(
                policy_batch["s_grid"],
                policy_batch["s_agent_pos"],
                policy_batch["s_agent_dir"],
                policy_batch["s_carry"],
            )
            g_expert_vals = reward.g(s_feat_e, expert_batch["a"]).cpu().numpy()
            g_policy_vals = reward.g(s_feat_curr, policy_batch["a"]).cpu().numpy()
            h_expert_vals = reward.h(s_feat_e).cpu().numpy()
            h_policy_vals = reward.h(s_feat_curr).cpu().numpy()

        row = {
            "iteration": it,
            "disc_loss": float(disc_loss_avg),
            "disc_acc": disc_acc,
            "disc_grad_pen": float(grad_pen.item()) if grad_penalty_weight > 0.0 else 0.0,
            "g_reg_loss": float(reg_loss.item()),
            "g_expert_mean": float(np.mean(g_expert_vals)),
            "g_policy_mean": float(np.mean(g_policy_vals)),
            "g_gap": float(np.mean(g_expert_vals) - np.mean(g_policy_vals)),
            "g_expert_std": float(np.std(g_expert_vals)),
            "g_policy_std": float(np.std(g_policy_vals)),
            "h_expert_mean": float(np.mean(h_expert_vals)),
            "h_policy_mean": float(np.mean(h_policy_vals)),
            "h_gap": float(np.mean(h_expert_vals) - np.mean(h_policy_vals)),
            "h_expert_std": float(np.std(h_expert_vals)),
            "h_policy_std": float(np.std(h_policy_vals)),
            "policy_entropy": entropy_val,
            "ppo_active": int(not skip_ppo),
        }

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not log_exists:
                writer.writeheader()
                log_exists = True
            writer.writerow(row)

        pbar.set_postfix(
            {
                "disc": f"{row['disc_loss']:.3f}",
                "acc": f"{row['disc_acc']:.3f}",
                "g_gap": f"{row['g_gap']:.3f}",
                "h_gap": f"{row['h_gap']:.3f}",
                "h_std": f"{row['h_expert_std']:.3f}",
                "ppo": "ON" if not skip_ppo else "frozen",
            }
        )

        if it % cfg["airl"]["heatmap_every"] == 0:
            for layout_name in cfg["heatmap_layouts"]:
                h_map = build_heatmap(reward, env_cfg, layout_name, device)
                save_heatmap(os.path.join(output_dir, "heatmaps", f"iter{it}_{layout_name}.png"), h_map)
                if prev_heat is not None:
                    _ = np.nanmean((h_map - prev_heat) ** 2)
                prev_heat = h_map
            script = ROOT / "scripts" / "section3" / "analyze_airl_metrics.py"
            argv = [
                str(script),
                "--metrics-csv",
                log_path,
                "--heatmap-dir",
                os.path.join(output_dir, "heatmaps"),
                "--iteration",
                str(it),
                "--out-dir",
                os.path.join(output_dir, "diagnostics"),
                "--title",
                f"AIRL Diagnostics - {Path(cfg['expert_dataset']).parent.name}",
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                import runpy
                runpy.run_path(str(script), run_name="__main__")
            finally:
                sys.argv = old_argv

        if it % cfg["airl"]["save_every"] == 0:
            ckpt = os.path.join(output_dir, "checkpoints", f"airl_iter_{it}.pt")
            torch.save(
                {
                    "reward": reward.state_dict(),
                    "policy": policy.state_dict(),
                },
                ckpt,
            )

    final_path = os.path.join(output_dir, "reward_model.pt")
    torch.save({"reward": reward.state_dict()}, final_path)
    print(f"[airl] saved reward model to {final_path}")


if __name__ == "__main__":
    main()
