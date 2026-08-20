"""
Configurable training script for GPT language model.

Usage:
    python scripts/train.py --config configs/m.yaml
    python scripts/train.py --d_model 128 --num_layers 4 --num_heads 4 --epochs 10
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Add project root to path so imports work when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import load_raw_text, create_datasets
from src.tokenizer import GPT2Tokenizer
from src.model.gpt import GPT


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, dataloader, vocab_size, device):
    """Compute average cross-entropy loss on a dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader)


def train(config):
    """Run a full training loop with the given config dict."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    raw_text = load_raw_text()
    tokenizer = GPT2Tokenizer()
    token_ids = tokenizer.encode(raw_text)
    vocab_size = tokenizer.tokenizer.n_vocab

    context_length = config["model"]["context_length"]
    train_dataset, test_dataset = create_datasets(
        token_ids, max_length=context_length, stride=context_length
    )

    print(f"Total tokens: {len(token_ids):,}")
    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Test sequences: {len(test_dataset):,}")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────
    model = GPT(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        context_length=context_length,
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))

    # ── Training ──────────────────────────────────────────────────────
    epochs = config["training"]["epochs"]
    best_test_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        test_loss = evaluate(model, test_loader, vocab_size, device)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")

        # Save best checkpoint
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = os.path.join("checkpoints", f"{config['name']}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "test_loss": test_loss,
            }, checkpoint_path)

    elapsed = time.time() - start_time

    # ── Results ───────────────────────────────────────────────────────
    result = {
        "name": config["name"],
        "d_model": config["model"]["d_model"],
        "num_layers": config["model"]["num_layers"],
        "num_heads": config["model"]["num_heads"],
        "params": num_params,
        "train_loss": f"{train_loss:.4f}",
        "test_loss": f"{best_test_loss:.4f}",
        "epochs": epochs,
        "time_seconds": f"{elapsed:.1f}",
    }

    # Append to results CSV
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "scaling_runs.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"\nDone in {elapsed:.1f}s — best test loss: {best_test_loss:.4f}")
    print(f"Results appended to {csv_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train GPT language model")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--name", type=str, default="run")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Use the config filename (without extension) as the run name
        config["name"] = os.path.splitext(os.path.basename(args.config))[0]
    else:
        config = {
            "name": args.name,
            "model": {
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "context_length": args.context_length,
            },
            "training": {
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
            },
        }

    print(f"\n{'='*50}")
    print(f"  Training: {config['name']}")
    print(f"  d_model={config['model']['d_model']}  layers={config['model']['num_layers']}  heads={config['model']['num_heads']}")
    print(f"{'='*50}\n")

    train(config)


if __name__ == "__main__":
    main()
