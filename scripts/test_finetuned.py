"""
Smoke-test script for LexiFlow Qwen + QLoRA adapter inference.
"""

import argparse
from pathlib import Path

import torch


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = "checkpoints/qlora"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


PROMPTS = [
    "sir meri attendance short ho gayi hai kya karu",
    "maam assignment ki last date kya hai",
    "sir exam me calculator allowed hai kya",
]


def _validate_adapter(path: Path):
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"QLoRA adapter not found or incomplete at '{path}'. "
            f"Missing: {', '.join(missing)}. "
            "Train in Colab first, then place extracted adapter files in checkpoints/qlora."
        )


def _load_model(base_model: str, adapter_path: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_path)
    _validate_adapter(adapter_dir)

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model_kwargs = {"torch_dtype": dtype}
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    if not use_cuda:
        model = model.to("cpu")

    model.eval()
    return model, tokenizer


def _input_device(model):
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        first = next(iter(device_map.values()))
        if isinstance(first, int):
            return f"cuda:{first}"
        if isinstance(first, str) and first not in {"disk", "meta"}:
            return first
    return next(model.parameters()).device


def _generate(model, tokenizer, prompt: str, max_new_tokens: int):
    messages = [
        {
            "role": "system",
            "content": (
                "You are LexiFlow, a helpful campus assistant. "
                "Convert the student's Hinglish message into polite formal English."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(_input_device(model))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Smoke-test Qwen + LexiFlow QLoRA adapter")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    model, tokenizer = _load_model(args.base_model, args.adapter_path)
    print(f"Loaded base model: {args.base_model}")
    print(f"Loaded adapter: {args.adapter_path}")
    print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")

    for idx, prompt in enumerate(PROMPTS, 1):
        response = _generate(model, tokenizer, prompt, args.max_new_tokens)
        print(f"[{idx}] Prompt: {prompt}")
        print(f"    Response: {response}\n")


if __name__ == "__main__":
    main()
