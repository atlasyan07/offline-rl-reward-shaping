"""Data loading and preprocessing for DPO on Anthropic HH-RLHF."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


def extract_prompt_and_response(text: str) -> tuple[str, str]:
    """Split an HH-RLHF conversation into prompt and final response.

    The dataset format uses "\\n\\nHuman: ..." and "\\n\\nAssistant: ..."
    We treat everything up to (and including) the last "Assistant:" as the
    prompt, and the text after it as the response.
    """
    # Find the last "Assistant:" marker
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        # Fallback: treat entire text as response
        return "", text
    prompt = text[: idx + len(marker)]
    response = text[idx + len(marker) :]
    return prompt, response


class HHRLHFDPODataset(Dataset):
    """Tokenised preference pairs from Anthropic HH-RLHF.

    Each item returns tokenised chosen and rejected sequences with labels
    masked over the prompt portion (set to -100).
    """

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def _tokenise_with_labels(self, prompt: str, response: str) -> dict[str, torch.Tensor]:
        """Tokenise prompt + response; mask prompt tokens in labels."""
        # Tokenise prompt alone to find its length
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Tokenise full sequence
        full_text = prompt + response
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: copy of input_ids with prompt tokens masked
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), self.max_length)
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        chosen_prompt, chosen_response = extract_prompt_and_response(item["chosen"])
        _, rejected_response = extract_prompt_and_response(item["rejected"])

        chosen = self._tokenise_with_labels(chosen_prompt, chosen_response)
        rejected = self._tokenise_with_labels(chosen_prompt, rejected_response)

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
        }
