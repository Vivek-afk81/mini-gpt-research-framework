"""
LexiFlow training script.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 30 --batch-size 8 --lr 3e-4
"""

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path so imports work when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.tokenizer import GPT2Tokenizer
from src.data_loader import LexiFlowDataset, split_dataset
from src.model.gpt import GPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device):
    """Compute average loss and perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(min(avg_loss, 100))  # clamp to avoid overflow
    model.train()
    return avg_loss, perplexity


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config):
    """Run a full training loop with the given configuration."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    tokenizer = GPT2Tokenizer()

    # Auto-split raw dataset if processed splits don't exist
    if not os.path.exists(config.data.train_path):
        print("Processed splits not found — splitting raw dataset ...")
        split_dataset(
            config.data.raw_path,
            config.data.train_path,
            config.data.val_path,
            config.data.test_path,
            config.data.train_ratio,
            config.data.val_ratio,
        )

    train_dataset = LexiFlowDataset(config.data.train_path, tokenizer, config)
    val_dataset = LexiFlowDataset(config.data.val_path, tokenizer, config)

    print(f"Train examples : {len(train_dataset)}")
    print(f"Val examples   : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────
    config.model.vocab_size = tokenizer.vocab_size

    model = GPT(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        context_length=config.model.context_length,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)

    num_params = count_parameters(model)
    print(f"Parameters     : {num_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    total_steps = len(train_loader) * config.training.epochs
    scheduler = get_lr_scheduler(
        optimizer, config.training.warmup_steps, total_steps,
    )

    # ── Training ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = []
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"  Training LexiFlow")
    print(f"  d_model={config.model.d_model}  layers={config.model.num_layers}  "
          f"heads={config.model.num_heads}  params={num_params:,}")
    print(f"  epochs={config.training.epochs}  batch_size={config.training.batch_size}  "
          f"lr={config.training.learning_rate}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip,
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(min(train_loss, 100))

        # Evaluate on validation set
        val_loss, val_ppl = evaluate(model, val_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{config.training.epochs}  "
            f"train_loss={train_loss:.4f}  train_ppl={train_ppl:8.1f}  "
            f"val_loss={val_loss:.4f}  val_ppl={val_ppl:8.1f}  "
            f"lr={lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "train_ppl": f"{train_ppl:.1f}",
            "val_loss": f"{val_loss:.4f}",
            "val_ppl": f"{val_ppl:.1f}",
            "lr": f"{lr:.2e}",
        })

        # ── Checkpoints ──────────────────────────────────────────────
        os.makedirs("checkpoints", exist_ok=True)

        # Always save latest
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
        }, "checkpoints/latest_model.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config,
            }, "checkpoints/best_model.pt")
            print(f"         -> New best model (val_loss={val_loss:.4f})")

    elapsed = time.time() - start_time

    # ── Save training history ─────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/training_loss.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"\n{'=' * 60}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Best validation loss : {best_val_loss:.4f}")
    print(f"  Best val perplexity  : {math.exp(min(best_val_loss, 100)):.1f}")
    print(f"  History saved to     : {csv_path}")
    print(f"{'=' * 60}")

    return best_val_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LexiFlow model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()

    config = Config()

    # Override defaults with any CLI arguments
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.d_model is not None:
        config.model.d_model = args.d_model
    if args.num_layers is not None:
        config.model.num_layers = args.num_layers
    if args.num_heads is not None:
        config.model.num_heads = args.num_heads
    if args.context_length is not None:
        config.model.context_length = args.context_length
    if args.dropout is not None:
        config.model.dropout = args.dropout

    train(config)


if __name__ == "__main__":
    main()
