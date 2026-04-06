#!/usr/bin/env python3
"""Train DPO on Anthropic HH-RLHF with Qwen2.5-0.5B-Instruct.

Usage:
    python scripts/section4/train_dpo.py --beta 0.1 --output-dir outputs/section4_dpo_beta01
    python scripts/section4/train_dpo.py --beta 0.5 --output-dir outputs/section4_dpo_beta05
"""
from __future__ import annotations

import argparse
import csv
import json
import copy
from pathlib import Path

import subprocess

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dpo.dpo import dpo_loss, compute_sequence_logps
from src.dpo.data import HHRLHFDPODataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training on HH-RLHF")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--beta", type=float, required=True,
                        help="KL penalty coefficient (e.g. 0.1, 0.5)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Stop after this many optimiser steps (0 = full epoch)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dashboard-every", type=int, default=10,
                        help="Regenerate dashboard every N optimiser steps (0 = off)")
    parser.add_argument("--train-samples", type=int, default=0,
                        help="Limit training samples (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── Save config ───────────────────────────────────────────────────────────
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load model and tokenizer ──────────────────────────────────────────────
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Reference model: frozen copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading Anthropic HH-RLHF dataset...")
    raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    if args.train_samples > 0:
        raw_dataset = raw_dataset.select(range(min(args.train_samples, len(raw_dataset))))

    dataset = HHRLHFDPODataset(raw_dataset, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} pairs, batch_size={args.batch_size}, "
          f"grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}")
    print(f"Beta: {args.beta}, LR: {args.lr}, Max steps: {args.max_steps}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ── Training loop ─────────────────────────────────────────────────────────
    metrics_file = out_dir / "train_metrics.csv"
    fieldnames = [
        "step", "loss", "reward_margin", "chosen_reward_mean",
        "rejected_reward_mean", "accuracy",
    ]
    csv_file = open(metrics_file, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_margin = 0.0
    accum_chosen_r = 0.0
    accum_rejected_r = 0.0
    accum_acc = 0.0
    accum_count = 0

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            # Policy log-probs
            policy_chosen_logps = compute_sequence_logps(model, chosen_ids, chosen_mask, chosen_labels)
            policy_rejected_logps = compute_sequence_logps(model, rejected_ids, rejected_mask, rejected_labels)

            # Reference log-probs (no grad)
            with torch.no_grad():
                ref_chosen_logps = compute_sequence_logps(ref_model, chosen_ids, chosen_mask, chosen_labels)
                ref_rejected_logps = compute_sequence_logps(ref_model, rejected_ids, rejected_mask, rejected_labels)

            # DPO loss
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.beta,
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            # Track metrics
            margin = (chosen_rewards - rejected_rewards).mean().item()
            acc = (chosen_rewards > rejected_rewards).float().mean().item()
            accum_loss += loss.item()
            accum_margin += margin
            accum_chosen_r += chosen_rewards.mean().item()
            accum_rejected_r += rejected_rewards.mean().item()
            accum_acc += acc
            accum_count += 1

            # Optimiser step
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % args.log_every == 0:
                    row = {
                        "step": global_step,
                        "loss": accum_loss / accum_count,
                        "reward_margin": accum_margin / accum_count,
                        "chosen_reward_mean": accum_chosen_r / accum_count,
                        "rejected_reward_mean": accum_rejected_r / accum_count,
                        "accuracy": accum_acc / accum_count,
                    }
                    writer.writerow(row)
                    csv_file.flush()
                    print(f"[step {global_step}] loss={row['loss']:.4f} "
                          f"margin={row['reward_margin']:.4f} "
                          f"acc={row['accuracy']:.3f}")
                    accum_loss = accum_margin = accum_chosen_r = 0.0
                    accum_rejected_r = accum_acc = 0.0
                    accum_count = 0

                # Save checkpoint
                if global_step % args.save_every == 0:
                    ckpt_dir = out_dir / f"checkpoint_{global_step}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"Saved checkpoint to {ckpt_dir}")

                # Dashboard
                if args.dashboard_every > 0 and global_step % args.dashboard_every == 0:
                    dash_dir = out_dir / "training_diagnostics"
                    dash_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        subprocess.run(
                            [
                                "python3",
                                "scripts/section4/analyze_dpo_metrics.py",
                                "--metrics", str(metrics_file),
                                "--output-dir", str(dash_dir),
                                "--title", f"DPO Training — β = {args.beta} (step {global_step})",
                            ],
                            timeout=30,
                            capture_output=True,
                        )
                        print(f"Dashboard updated at step {global_step}")
                    except Exception as e:
                        print(f"Dashboard generation failed: {e}")

                # Early stop
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ── Save final ────────────────────────────────────────────────────────────
    csv_file.close()
    final_dir = out_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save final metrics summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "model": args.model_name,
            "beta": args.beta,
            "total_steps": global_step,
            "lr": args.lr,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "max_length": args.max_length,
        }, f, indent=2)

    print(f"\nDone. {global_step} steps. Output in {out_dir}")


if __name__ == "__main__":
    main()
