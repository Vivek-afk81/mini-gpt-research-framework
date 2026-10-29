"""Leakage-aware, template-grouped dataset splitting."""

import random
import re
from collections import Counter

from .clean import load_records, clean, save_records


HONORIFICS = re.compile(r"\b(sir|ma'?am|madam|please|plz|pls|kripya)\b", re.IGNORECASE)
SUBJECTS = re.compile(
    r"\b(maths?|dsa|os|dbms|java|python|physics|chemistry|computer networks|"
    r"machine learning|machinelearning|english|economics|electronics)\b",
    re.IGNORECASE,
)


def template_key(text):
    value = HONORIFICS.sub(" ", text.lower())
    value = SUBJECTS.sub("<subject>", value)
    value = re.sub(r"\d+", "<number>", value)
    value = re.sub(r"[^\w<>\s]", " ", value)
    return " ".join(value.split())


def grouped_split(records, seed=42, train_ratio=0.70, val_ratio=0.15):
    """Split complete template groups so near-duplicates cannot cross splits."""
    groups = {}
    for record in records:
        groups.setdefault(template_key(record["input"]), []).append(record)
    group_values = list(groups.values())
    random.Random(seed).shuffle(group_values)
    train_target = int(len(records) * train_ratio)
    val_target = int(len(records) * val_ratio)
    train, validation, test = [], [], []
    for group in group_values:
        if len(train) + len(group) <= train_target or not train:
            train.extend(group)
        elif len(validation) + len(group) <= val_target or not validation:
            validation.extend(group)
        else:
            test.extend(group)
    return train, validation, test


def process_file(source, output_dir="data/processed", seed=42):
    records = clean(load_records(source))
    train, validation, test = grouped_split(records, seed=seed)
    save_records(train, f"{output_dir}/train.jsonl")
    save_records(validation, f"{output_dir}/validation.jsonl")
    save_records(test, f"{output_dir}/test.jsonl")
    return {"total": len(records), "train": len(train), "validation": len(validation), "test": len(test),
            "categories": dict(Counter(record["category"] for record in records))}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean and split LexiFlow JSONL data")
    parser.add_argument("source", default="data/campus_messages_v2.jsonl", nargs="?")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    print(process_file(args.source, args.output_dir))