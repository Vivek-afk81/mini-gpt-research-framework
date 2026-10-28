"""
LexiFlow data pipeline.

Handles JSONL loading, train/val/test splitting, prompt formatting,
and the PyTorch Dataset with label masking (-100 on prompt tokens).
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Load examples from a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples, path):
    """Save examples to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(raw_path, train_path, val_path, test_path,
                  train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Shuffle the raw JSONL and write train / val / test splits.

    Returns the three lists for inspection.
    """
    examples = load_jsonl(raw_path)
    random.seed(seed)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]

    save_jsonl(train, train_path)
    save_jsonl(val, val_path)
    save_jsonl(test, test_path)

    print(f"Split {n} examples -> train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(example, config):
    """
    Convert a dataset example into the tagged prompt string.

    Format:
        [TASK] category [TONE] tone [INPUT] hinglish [OUTPUT] formal english
    """
    parts = [
        f"{config.task_tag} {example['category']}",
        f"{config.tone_tag} {example.get('tone', 'formal')}",
        f"{config.input_tag} {example['input']}",
        f"{config.output_tag} {example['output']}",
    ]
    return " ".join(parts)


def format_prompt_prefix(example, config):
    """
    Build the prompt-only portion (everything up to and including [OUTPUT]).

    Used to determine where label masking ends.
    """
    return (
        f"{config.task_tag} {example['category']} "
        f"{config.tone_tag} {example.get('tone', 'formal')} "
        f"{config.input_tag} {example['input']} "
        f"{config.output_tag}"
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class LexiFlowDataset(Dataset):
    """
    Tokenises JSONL examples into (input_ids, labels) pairs.

    Label masking: the prompt portion of labels is set to -100 so that
    cross-entropy loss is computed only on the formal-English output tokens.

    Padding: sequences shorter than context_length are right-padded with
    eos_token_id (input) and -100 (labels).  Because the causal mask
    prevents earlier tokens from attending to later positions, padding
    tokens do not affect predictions on real tokens.
    """

    def __init__(self, jsonl_path, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.examples = load_jsonl(jsonl_path)
        self.max_length = config.model.context_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Full prompt (input + output + EOS)
        full_text = format_prompt(example, self.config)
        full_ids = self.tokenizer.encode(full_text, add_eos=True)

        # Prompt-only portion (for computing where to start loss)
        prompt_text = format_prompt_prefix(example, self.config)
        prompt_len = len(self.tokenizer.encode(prompt_text))

        # Truncate to context_length + 1 (need one extra for the target shift)
        full_ids = full_ids[: self.max_length + 1]

        # Shift: input is all-but-last, target is all-but-first
        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]

        # Mask prompt tokens in target with -100
        labels = list(target_ids)
        for i in range(min(prompt_len - 1, len(labels))):
            labels[i] = -100

        # Right-pad to max_length
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.eos_token_id] * pad_len
        labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
