"""
LexiFlow — QLoRA fine-tuning script for Qwen2.5-1.5B-Instruct.

Designed to run on Google Colab (free tier, T4 GPU).

Usage:
    python src/finetune/train_qlora.py
    python src/finetune/train_qlora.py --epochs 5 --lr 2e-4
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def train_qlora(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="data/finetuning/train_chat.jsonl",
    output_dir="checkpoints/qlora",
    epochs=3,
    batch_size=4,
    lr=2e-4,
    max_length=512,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
):
    """Run QLoRA fine-tuning."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    # ── Check GPU ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. QLoRA training requires a GPU.")

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\nLoading data from {data_path} ...")
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Training examples: {len(examples)}")

    # ── Load tokenizer ────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Format with chat template ─────────────────────────────────────
    def format_example(example):
        """Apply the chat template to format messages."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(format_example)
    print(f"Sample:\n{dataset[0]['text'][:300]}...\n")

    # ── Load model with 4-bit quantization ────────────────────────────
    print(f"Loading model: {base_model} (4-bit quantized)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA configuration ────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Training arguments ────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=1.0,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_length,
    )

    print(f"\n{'=' * 60}")
    print(f"  QLoRA Fine-tuning: {base_model}")
    print(f"  epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"  LoRA r={lora_r}  alpha={lora_alpha}")
    print(f"{'=' * 60}\n")

    # ── Train ─────────────────────────────────────────────────────────
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nAdapter saved to {output_dir}")

    # ── Quick test ────────────────────────────────────────────────────
    print("\n--- Quick Test ---")
    test_messages = [
        {"role": "system", "content": "You are LexiFlow, a campus assistant. Help the student."},
        {"role": "user", "content": "sir meri attendance short ho gayi hai kya karu"},
    ]
    test_input = tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Input: sir meri attendance short ho gayi hai kya karu")
    print(f"Output: {response}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for LexiFlow")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", type=str, default="data/finetuning/train_chat.jsonl")
    parser.add_argument("--output", type=str, default="checkpoints/qlora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    train_qlora(
        base_model=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
    )


if __name__ == "__main__":
    main()
