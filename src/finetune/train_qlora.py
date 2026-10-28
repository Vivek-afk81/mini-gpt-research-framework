"""
LexiFlow — QLoRA fine-tuning script for Qwen2.5-1.5B-Instruct.

Designed to run on Google Colab (free tier, T4 GPU).

Usage:
    python src/finetune/train_qlora.py
    python src/finetune/train_qlora.py --epochs 5 --lr 2e-4
"""

import argparse
import inspect
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
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {vram:.1f} GB")
    else:
        raise RuntimeError(
            "No GPU detected. QLoRA training with 4-bit bitsandbytes requires CUDA "
            "(recommended: Google Colab T4 runtime)."
        )

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\nLoading data from {data_path} ...")
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if not examples:
        raise ValueError(f"No training examples found in: {data_path}")
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
    # Remove original 'messages' column so SFTTrainer uses pre-formatted 'text' directly
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
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

    # Crucial for T4 GPUs (Compute Capability 7.5):
    # Force model config to float16 so PEFT and TRL never fall back to Qwen's native bfloat16
    model.config.torch_dtype = torch.float16
    if hasattr(model.config, "dtype"):
        model.config.dtype = torch.float16

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False

    # ── LoRA configuration ────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # 1. Trainable parameters MUST be float32 for PyTorch AMP GradScaler (fp16=True)
    # 2. Frozen non-quantized parameters must be float16 (never bfloat16) for Tesla T4
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
        elif param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for buf in model.buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"\nTrainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    # ── Training arguments ────────────────────────────────────────────
    try:
        from trl import SFTConfig

        sft_config_params = inspect.signature(SFTConfig.__init__).parameters
        ConfigClass = SFTConfig
    except ImportError:
        sft_config_params = {}
        ConfigClass = TrainingArguments

    config_kwargs = dict(
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
        bf16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if ConfigClass is not TrainingArguments:
        if "dataset_text_field" in sft_config_params:
            config_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sft_config_params:
            config_kwargs["max_seq_length"] = max_length
        elif "max_length" in sft_config_params:
            config_kwargs["max_length"] = max_length
        if "packing" in sft_config_params:
            config_kwargs["packing"] = False
        if "use_liger_kernel" in sft_config_params:
            config_kwargs["use_liger_kernel"] = False

    training_args = ConfigClass(**config_kwargs)

    # ── Trainer ───────────────────────────────────────────────────────
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters

    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    if (
        "dataset_text_field" in trainer_signature
        and "dataset_text_field" not in config_kwargs
    ):
        trainer_kwargs["dataset_text_field"] = "text"
    if (
        "max_seq_length" in trainer_signature
        and "max_seq_length" not in config_kwargs
        and "max_length" not in config_kwargs
    ):
        trainer_kwargs["max_seq_length"] = max_length
    if "packing" in trainer_signature and "packing" not in config_kwargs:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)

    # Disable TRL's custom chunked_logprob kernel if present
    if hasattr(trainer, "_use_chunked_ce"):
        trainer._use_chunked_ce = False

    print(f"\n{'=' * 60}")
    print(f"  QLoRA Fine-tuning: {base_model}")
    print(f"  epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"  LoRA r={lora_r}  alpha={lora_alpha}")
    print(f"{'=' * 60}\n")

    # ── Train ─────────────────────────────────────────────────────────
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nAdapter saved to {output_dir}")

    # ── Quick test ────────────────────────────────────────────────────
    print("\n--- Quick Test ---")
    model.config.use_cache = True
    test_messages = [
        {
            "role": "system",
            "content": "You are LexiFlow, a campus assistant. Help the student.",
        },
        {
            "role": "user",
            "content": "sir meri attendance short ho gayi hai kya karu",
        },
    ]
    test_input = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_device = next(model.parameters()).device
    inputs = tokenizer(test_input, return_tensors="pt").to(model_device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    print("Input: sir meri attendance short ho gayi hai kya karu")
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