"""DPO: Direct Preference Optimisation.

Implements the Bradley-Terry preference loss from Rafailov et al. (2023).
No dependency on TRL — the loss, data handling, and logging are explicit.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the DPO loss under the Bradley-Terry preference model.

    The implicit reward for a completion y given prompt x is:
        r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))

    The preference probability under Bradley-Terry is:
        p(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))

    The loss is the negative log-likelihood of the observed preferences:
        L = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
                                - log pi(y_l|x)/pi_ref(y_l|x)))]

    Args:
        policy_chosen_logps: Log-probs of chosen completions under pi.
        policy_rejected_logps: Log-probs of rejected completions under pi.
        ref_chosen_logps: Log-probs of chosen completions under pi_ref.
        ref_rejected_logps: Log-probs of rejected completions under pi_ref.
        beta: KL penalty coefficient.

    Returns:
        loss: Scalar DPO loss.
        chosen_rewards: Implicit rewards for chosen completions.
        rejected_rewards: Implicit rewards for rejected completions.
    """
    # Log-ratios: how much the policy has moved from the reference
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # Implicit rewards
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    # Bradley-Terry logits: reward margin
    logits = chosen_rewards - rejected_rewards

    # Loss: negative log-sigmoid of the margin
    loss = -F.logsigmoid(logits).mean()

    return loss, chosen_rewards.detach(), rejected_rewards.detach()


def compute_sequence_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sequence log-probabilities for a batch.

    Args:
        model: Causal LM.
        input_ids: (B, L) token ids.
        attention_mask: (B, L) attention mask.
        labels: (B, L) labels with -100 for prompt tokens.

    Returns:
        logps: (B,) total log-prob of the non-masked tokens per sequence.
    """
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Shift: predict next token from current position
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = (shift_labels != -100)

    # Per-token log-probs
    per_token_logps = F.log_softmax(shift_logits, dim=-1)
    token_logps = per_token_logps.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)

    # Mask out prompt tokens and sum over completion
    token_logps = token_logps * shift_mask.float()
    sequence_logps = token_logps.sum(dim=1)

    return sequence_logps
