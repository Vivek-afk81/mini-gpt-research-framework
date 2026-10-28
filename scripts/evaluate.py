"""
LexiFlow evaluation script.

Computes:
1. Quantitative loss and perplexity on validation and test sets
2. Generation quality metrics across all six categories:
   - Empty output rate
   - Repetition rate (Distinct-1, Distinct-2)
   - Formality score (presence of formal markers, honorifics, polite modals)
   - Meaning preservation (intent & keyword overlap)
3. Qualitative evaluation of sample outputs

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --split test
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.tokenizer import GPT2Tokenizer
from src.data_loader import LexiFlowDataset, load_jsonl
from src.model.gpt import GPT
from src.generation import TextGenerator


FORMAL_MARKERS = [
    r"\bsir\b", r"\bma'?am\b", r"\bplease\b", r"\bkindly\b", r"\brequest\b",
    r"\bcould you\b", r"\bwould you\b", r"\bwould it be possible\b",
    r"\binquire\b", r"\bclarify\b", r"\bassistance\b", r"\bregarding\b",
    r"\bapologize\b", r"\bgrateful\b", r"\brespectfully\b"
]

CATEGORY_KEYWORDS = {
    "assignment_query": ["assignment", "submit", "submission", "due", "topic", "page", "format", "group", "pdf"],
    "attendance_request": ["attendance", "short", "shortage", "percent", "medical", "class", "improve"],
    "leave_request": ["leave", "absence", "sick", "ill", "fever", "family", "attend", "approve"],
    "exam_query": ["exam", "examination", "mid", "end", "sem", "room", "schedule", "date", "hall"],
    "deadline_extension": ["extension", "deadline", "extend", "delay", "time", "due", "complete"],
    "complaint": ["issue", "complaint", "working", "repair", "resolve", "ac", "wifi", "lab", "portal"],
}


def compute_loss_and_perplexity(model, dataloader, device):
    """Compute average loss and perplexity on dataset."""
    model.eval()
    total_loss = 0.0
    num_tokens = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            # Count non-masked tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item()
            num_tokens += valid_tokens

    avg_loss = total_loss / max(1, num_tokens)
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, perplexity


def compute_distinct_ngrams(tokens: List[str], n: int) -> float:
    """Calculate distinct-n ratio."""
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(1, len(ngrams))


def evaluate_sample(inp: str, out: str, ref: str, category: str) -> Dict[str, Any]:
    """Evaluate a single generated response against quality criteria."""
    out_stripped = out.strip()
    is_empty = (len(out_stripped) == 0)

    words = re.findall(r'\b\w+\b', out_stripped.lower())

    # Repetition metrics
    distinct_1 = compute_distinct_ngrams(words, 1)
    distinct_2 = compute_distinct_ngrams(words, 2)
    has_word_repetition = False
    word_counts = Counter(words)
    for w, count in word_counts.items():
        if len(w) > 3 and count >= 3:
            has_word_repetition = True

    # Formality score (0 to 1 based on formal markers present)
    matches = sum(1 for pat in FORMAL_MARKERS if re.search(pat, out_stripped, re.IGNORECASE))
    formality_score = min(1.0, matches / 2.0)

    # Meaning preservation (keyword overlap with category / input)
    cat_keys = CATEGORY_KEYWORDS.get(category, [])
    ref_words = set(re.findall(r'\b\w+\b', ref.lower()))
    input_words = set(re.findall(r'\b\w+\b', inp.lower()))

    # Check how many domain or reference concepts are preserved
    preserved_keywords = [kw for kw in cat_keys if kw in words]
    meaning_overlap = len(set(words) & (ref_words | set(cat_keys))) / max(1, min(len(words), 8))

    return {
        "is_empty": is_empty,
        "length_chars": len(out_stripped),
        "word_count": len(words),
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "has_word_repetition": has_word_repetition,
        "formality_score": formality_score,
        "meaning_overlap": min(1.0, meaning_overlap),
        "preserved_keywords": preserved_keywords,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LexiFlow model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--val-path", type=str, default="data/processed/validation.jsonl")
    parser.add_argument("--test-path", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-json", type=str, default="results/evaluation_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n============================================================")
    print(f"  LEXIFLOW MODEL EVALUATION ({device.upper()})")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"============================================================\n")

    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found at {args.checkpoint}. Run training first.")
        sys.exit(1)

    config = Config()
    tokenizer = GPT2Tokenizer()

    # Load model for loss computation
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        context_length=config.model.context_length,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Measure Loss & Perplexity on Validation & Test sets
    val_loss, val_ppl = None, None
    test_loss, test_ppl = None, None

    if os.path.exists(args.val_path):
        val_dataset = LexiFlowDataset(args.val_path, tokenizer, config)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        val_loss, val_ppl = compute_loss_and_perplexity(model, val_loader, device)
        print(f"[Validation Set] Loss: {val_loss:.4f}  |  Perplexity: {val_ppl:.2f}")

    if os.path.exists(args.test_path):
        test_dataset = LexiFlowDataset(args.test_path, tokenizer, config)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        test_loss, test_ppl = compute_loss_and_perplexity(model, test_loader, device)
        print(f"[Test Set]       Loss: {test_loss:.4f}  |  Perplexity: {test_ppl:.2f}")

    # 2. Text Generation Evaluation across categories
    generator = TextGenerator(checkpoint_path=args.checkpoint, device=device)

    eval_examples = []
    if os.path.exists(args.test_path):
        eval_examples.extend(load_jsonl(args.test_path))
    if os.path.exists(args.val_path):
        eval_examples.extend(load_jsonl(args.val_path))

    # De-duplicate by id
    seen_ids = set()
    unique_examples = []
    for ex in eval_examples:
        if ex["id"] not in seen_ids:
            seen_ids.add(ex["id"])
            unique_examples.append(ex)

    print(f"\nEvaluating generation on {len(unique_examples)} held-out examples...")

    per_category = {}
    sample_records = []

    for ex in unique_examples:
        cat = ex["category"]
        if cat not in per_category:
            per_category[cat] = {
                "count": 0, "empty": 0, "repetition": 0,
                "formality_sum": 0.0, "meaning_sum": 0.0,
                "distinct_1_sum": 0.0, "distinct_2_sum": 0.0
            }

        gen_res = generator.generate(
            input_text=ex["input"],
            category=cat,
            tone="formal",
            temperature=args.temperature,
            repetition_penalty=1.25,
        )

        metrics = evaluate_sample(ex["input"], gen_res["output"], ex["output"], cat)

        per_category[cat]["count"] += 1
        if metrics["is_empty"]:
            per_category[cat]["empty"] += 1
        if metrics["has_word_repetition"]:
            per_category[cat]["repetition"] += 1
        per_category[cat]["formality_sum"] += metrics["formality_score"]
        per_category[cat]["meaning_sum"] += metrics["meaning_overlap"]
        per_category[cat]["distinct_1_sum"] += metrics["distinct_1"]
        per_category[cat]["distinct_2_sum"] += metrics["distinct_2"]

        sample_records.append({
            "id": ex["id"],
            "category": cat,
            "input": ex["input"],
            "reference": ex["output"],
            "generated": gen_res["output"],
            "metrics": metrics,
        })

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Category':<22} {'Count':<6} {'Empty%':<8} {'Repeat%':<9} {'Formality':<11} {'Meaning'}")
    print("-" * 80)

    total_count = sum(d["count"] for d in per_category.values())
    total_empty = sum(d["empty"] for d in per_category.values())
    total_rep = sum(d["repetition"] for d in per_category.values())
    total_formality = sum(d["formality_sum"] for d in per_category.values())
    total_meaning = sum(d["meaning_sum"] for d in per_category.values())

    cat_summaries = {}
    for cat, d in sorted(per_category.items()):
        cnt = d["count"]
        emp_rate = (d["empty"] / cnt) * 100
        rep_rate = (d["repetition"] / cnt) * 100
        avg_form = d["formality_sum"] / cnt
        avg_mean = d["meaning_sum"] / cnt
        cat_summaries[cat] = {
            "count": cnt,
            "empty_rate_pct": round(emp_rate, 2),
            "repetition_rate_pct": round(rep_rate, 2),
            "avg_formality": round(avg_form, 3),
            "avg_meaning_overlap": round(avg_mean, 3),
        }
        print(f"{cat:<22} {cnt:<6} {emp_rate:<7.1f}% {rep_rate:<8.1f}% {avg_form:<10.2f} {avg_mean:.2f}")

    print("-" * 80)
    overall_empty_pct = (total_empty / max(1, total_count)) * 100
    overall_rep_pct = (total_rep / max(1, total_count)) * 100
    overall_formality = total_formality / max(1, total_count)
    overall_meaning = total_meaning / max(1, total_count)
    print(f"{'OVERALL':<22} {total_count:<6} {overall_empty_pct:<7.1f}% {overall_rep_pct:<8.1f}% {overall_formality:<10.2f} {overall_meaning:.2f}")
    print("=" * 80)

    # 3. Print 10 representative qualitative examples
    print("\n--- SAMPLE GENERATIONS (Input vs Reference vs Model Output) ---")
    for item in sample_records[:10]:
        print(f"\n[{item['category']}] (ID: {item['id']})")
        print(f"  Input    : {item['input']}")
        print(f"  Reference: {item['reference']}")
        print(f"  Model    : {item['generated']}")

    # Save to JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "overall": {
            "evaluated_samples": total_count,
            "empty_output_rate_pct": round(overall_empty_pct, 2),
            "repetition_rate_pct": round(overall_rep_pct, 2),
            "average_formality": round(overall_formality, 3),
            "average_meaning_overlap": round(overall_meaning, 3),
        },
        "per_category": cat_summaries,
        "samples": sample_records,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed evaluation report saved to: {args.output_json}\n")


if __name__ == "__main__":
    main()
