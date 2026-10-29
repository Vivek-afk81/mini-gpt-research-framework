"""Convert processed records into chat-format JSONL for LoRA training."""

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "Convert the Hinglish campus message to formal English and classify it. "
    "Return JSON with english, category, and tone only."
)


def convert_file(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as source_handle, destination.open("w", encoding="utf-8") as output_handle:
        for line in source_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": record["input"]},
                    {"role": "assistant", "content": json.dumps({
                        "english": record["output"],
                        "category": record["category"],
                        "tone": record["tone"],
                    }, ensure_ascii=False)},
                ]
            }
            output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare(processed_dir="data/processed", output_dir="data/finetuning"):
    counts = {}
    for source_name, destination_name in (
        ("train.jsonl", "train_chat.jsonl"),
        ("validation.jsonl", "validation_chat.jsonl"),
        ("test.jsonl", "test_chat.jsonl"),
    ):
        counts[destination_name] = convert_file(
            Path(processed_dir) / source_name,
            Path(output_dir) / destination_name,
        )
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LexiFlow LoRA chat data")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/finetuning")
    args = parser.parse_args()
    print(prepare(args.processed_dir, args.output_dir))