#!/usr/bin/env python3
"""
IQL Training Telemetry Interpreter

This script helps you understand what's happening during IQL training.
Run it on your TensorBoard logs to get a human-readable diagnosis.

Usage:
    python scripts/section2/interpret_training.py --tb-path outputs/section2_iql_baseline/tb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tb_reader import read_tb_scalars, get_tb_summary


# =============================================================================
# METRIC REFERENCE GUIDE
# =============================================================================
METRIC_GUIDE = """
================================================================================
                    IQL TRAINING METRICS - REFERENCE GUIDE
================================================================================

WHAT IS IQL?
------------
IQL (Implicit Q-Learning) learns from offline data without querying the
environment. It learns three things:
  1. V(s) - Value function: "How good is this state?"
  2. Q(s,a) - Q function: "How good is this action in this state?"
  3. Policy - "What action should I take?"

The key insight: Advantage = Q(s,a) - V(s) tells us if an action is
better or worse than average.

================================================================================
                              METRIC MEANINGS
================================================================================

┌─────────────────┬────────────────────────────────────────────────────────────┐
│ METRIC          │ WHAT IT MEANS                                              │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ v_loss          │ How well V-network predicts expected returns               │
│                 │ GOOD: Low and stable (0.001 - 0.1)                         │
│                 │ BAD:  Very high (>1) or erratic                            │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ q_loss          │ How well Q-network predicts action values                  │
│                 │ GOOD: Decreasing, then stable (0.01 - 0.5)                 │
│                 │ BAD:  Increasing or very high (>2)                         │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ actor_loss      │ Policy learning signal (cross-entropy style)               │
│                 │ GOOD: Stable around 1.0-1.5 (for 7 actions)                │
│                 │ BAD:  Very low (<0.5) = overconfident                      │
│                 │ BAD:  Very high (>2.5) = confused policy                   │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ v_mean          │ Average predicted value across states                      │
│                 │ GOOD: Matches your reward scale (0-1 for sparse)           │
│                 │ BAD:  Exploding (>10) or collapsing to 0                   │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ q_mean          │ Average predicted Q-value                                  │
│                 │ GOOD: Similar to v_mean but slightly higher                │
│                 │ BAD:  Diverging wildly from v_mean                         │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ q_var           │ Variance in Q predictions                                  │
│                 │ GOOD: Some variance (0.05 - 0.5) = differentiated values   │
│                 │ BAD:  Near zero = REPRESENTATION COLLAPSE                  │
│                 │ BAD:  Very high (>2) = unstable learning                   │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ adv_abs_mean    │ Average |Q - V| = action differentiation                   │
│                 │ GOOD: Meaningful (0.05 - 0.5) = can rank actions           │
│                 │ BAD:  Near zero (<0.01) = ALL ACTIONS LOOK THE SAME        │
│                 │       This is the #1 sign of failure!                      │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ adv_mean        │ Average (Q - V), usually slightly negative                 │
│                 │ GOOD: Small negative (-0.1 to 0)                           │
│                 │ BAD:  Large magnitude = something wrong                    │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ target_q_mean   │ Target network Q (slowly updated copy)                     │
│                 │ GOOD: Tracks q_mean with some lag                          │
│                 │ BAD:  Diverging from q_mean                                │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ success_rate    │ % of eval episodes reaching the goal                       │
│                 │ GOOD: Increasing over time, >50% eventually                │
│                 │ BAD:  Stuck at 0% = not learning useful policy             │
└─────────────────┴────────────────────────────────────────────────────────────┘

================================================================================
                           DIAGNOSIS PATTERNS
================================================================================

HEALTHY TRAINING:
  ✓ Losses decrease then stabilize
  ✓ adv_abs_mean stays meaningful (>0.05)
  ✓ q_var stays meaningful (>0.05)
  ✓ success_rate gradually improves
  ✓ v_mean and q_mean stay bounded

REPRESENTATION COLLAPSE (what you're seeing):
  ✗ adv_abs_mean → 0 (actions become indistinguishable)
  ✗ q_var → small (all states get same value)
  ✗ v_mean ≈ q_mean (no advantage signal)
  ✗ success_rate stuck at 0%

  WHY: Sparse rewards don't give enough signal. The network learns
       "everything is equally mediocre" because it rarely sees rewards.

  FIX: Add reward shaping (Section 3) to provide intermediate signals.

DIVERGENCE / INSTABILITY:
  ✗ Losses increasing
  ✗ v_mean or q_mean exploding (>10 or <-10)
  ✗ Erratic oscillations

  FIX: Lower learning rate, check data normalization.

OVERCONFIDENT POLICY:
  ✗ actor_loss very low (<0.5)
  ✗ Policy always picks same action

  FIX: Increase temperature/beta, more exploration in data.

================================================================================
"""


def diagnose_training(summary: dict) -> None:
    """Print diagnosis based on training metrics."""

    print("\n" + "="*80)
    print("                         TRAINING DIAGNOSIS")
    print("="*80)

    issues = []
    good_signs = []

    # Check success rate
    success_tags = [k for k in summary if 'success_rate' in k]
    if success_tags:
        avg_success = sum(summary[k]['mean'] for k in success_tags) / len(success_tags)
        final_success = sum(summary[k]['last'] for k in success_tags) / len(success_tags)
        if final_success == 0:
            issues.append(("CRITICAL", "Success rate is 0% - agent learned nothing useful"))
        elif final_success < 0.3:
            issues.append(("WARNING", f"Success rate is low ({final_success*100:.1f}%)"))
        else:
            good_signs.append(f"Success rate: {final_success*100:.1f}%")

    # Check advantage collapse
    if 'epoch/adv_abs_mean' in summary:
        adv = summary['epoch/adv_abs_mean']
        if adv['last'] < 0.01:
            issues.append(("CRITICAL", f"Advantage collapsed to {adv['last']:.4f} - actions are indistinguishable"))
        elif adv['last'] < 0.05:
            issues.append(("WARNING", f"Advantage is low ({adv['last']:.4f}) - weak action differentiation"))
        else:
            good_signs.append(f"Advantage magnitude: {adv['last']:.4f} (healthy)")

    # Check Q variance
    if 'epoch/q_var' in summary:
        qvar = summary['epoch/q_var']
        if qvar['last'] < 0.05:
            issues.append(("WARNING", f"Q variance is low ({qvar['last']:.4f}) - possible representation collapse"))
        else:
            good_signs.append(f"Q variance: {qvar['last']:.4f} (has diversity)")

    # Check Q-V alignment
    if 'epoch/q_mean' in summary and 'epoch/v_mean' in summary:
        q = summary['epoch/q_mean']['last']
        v = summary['epoch/v_mean']['last']
        diff = abs(q - v)
        if diff < 0.01:
            issues.append(("WARNING", f"Q ≈ V ({q:.4f} ≈ {v:.4f}) - no advantage signal"))

    # Check for divergence
    if 'epoch/q_mean' in summary:
        q = summary['epoch/q_mean']
        if abs(q['last']) > 10:
            issues.append(("CRITICAL", f"Q values diverging: {q['last']:.2f}"))
        elif q['last'] < -1:
            issues.append(("WARNING", f"Q values negative: {q['last']:.4f}"))

    # Check losses
    if 'epoch/q_loss' in summary:
        ql = summary['epoch/q_loss']
        if ql['last'] > ql['first']:
            issues.append(("WARNING", f"Q loss increased: {ql['first']:.4f} -> {ql['last']:.4f}"))
        elif ql['delta'] < 0:
            good_signs.append(f"Q loss decreased: {ql['first']:.4f} -> {ql['last']:.4f}")

    if 'epoch/v_loss' in summary:
        vl = summary['epoch/v_loss']
        if vl['last'] < 0.01:
            good_signs.append(f"V loss converged: {vl['last']:.6f}")

    # Print results
    if issues:
        print("\n🚨 ISSUES DETECTED:")
        for severity, msg in issues:
            icon = "❌" if severity == "CRITICAL" else "⚠️"
            print(f"  {icon} [{severity}] {msg}")

    if good_signs:
        print("\n✅ GOOD SIGNS:")
        for msg in good_signs:
            print(f"  • {msg}")

    # Overall verdict
    print("\n" + "-"*80)
    critical_count = sum(1 for s, _ in issues if s == "CRITICAL")
    if critical_count > 0:
        print("VERDICT: ❌ TRAINING FAILED")
        print("         The model did not learn a useful policy.")
        print("         This is EXPECTED for sparse rewards (Section 2 baseline).")
        print("         Proceed to Section 3: Reward Shaping to fix this.")
    elif len(issues) > 0:
        print("VERDICT: ⚠️ TRAINING PARTIALLY SUCCESSFUL")
        print("         Some issues detected - review warnings above.")
    else:
        print("VERDICT: ✅ TRAINING LOOKS HEALTHY")
    print("-"*80)


def print_metric_table(summary: dict) -> None:
    """Print a formatted table of final metric values."""

    print("\n" + "="*80)
    print("                         FINAL METRIC VALUES")
    print("="*80)

    # Group metrics
    groups = {
        "Losses": ["v_loss", "q_loss", "actor_loss"],
        "Values": ["v_mean", "q_mean", "q1_mean", "q2_mean", "target_q_mean"],
        "Diagnostics": ["q_var", "adv_mean", "adv_abs_mean"],
        "Eval": ["success_rate"],
    }

    for group_name, patterns in groups.items():
        matching = []
        for tag in sorted(summary.keys()):
            for pattern in patterns:
                if pattern in tag:
                    matching.append(tag)
                    break

        if matching:
            print(f"\n{group_name}:")
            print("-" * 60)
            for tag in matching:
                s = summary[tag]
                short_tag = tag.replace("epoch/", "").replace("train/", "").replace("eval/", "")
                trend = "↑" if s['delta'] > 0.001 else "↓" if s['delta'] < -0.001 else "→"
                print(f"  {short_tag:30s} {s['first']:>10.4f} {trend} {s['last']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Interpret IQL training metrics")
    parser.add_argument("--tb-path", required=True, help="Path to TensorBoard logs (file or dir)")
    parser.add_argument("--guide", action="store_true", help="Print the metric reference guide")
    args = parser.parse_args()

    if args.guide:
        print(METRIC_GUIDE)
        return

    # Load data
    print(f"Loading TensorBoard data from: {args.tb_path}")
    scalars = read_tb_scalars(args.tb_path)
    summary = get_tb_summary(scalars)

    if not summary:
        print("No scalar data found!")
        return

    # Print guide header
    print(METRIC_GUIDE)

    # Print metric table
    print_metric_table(summary)

    # Run diagnosis
    diagnose_training(summary)

    print("\n" + "="*80)
    print("TIP: Run with --guide to see just the metric reference guide")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
