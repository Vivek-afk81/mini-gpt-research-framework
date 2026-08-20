"""
Attention head visualization script.

Loads a trained checkpoint, feeds a sample text through the model,
and generates per-head, per-layer attention heatmaps.

Usage:
    python scripts/visualize_attention.py --checkpoint checkpoints/m.pt
    python scripts/visualize_attention.py --checkpoint checkpoints/m.pt --text "Buck did not read the newspapers"
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tokenizer import GPT2Tokenizer
from src.model.gpt import GPT


def load_model(checkpoint_path, device):
    """Load a trained model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    tokenizer = GPT2Tokenizer()
    vocab_size = tokenizer.tokenizer.n_vocab

    model = GPT(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        context_length=config["model"]["context_length"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, tokenizer


def plot_attention_heatmap(weights, tokens, layer_idx, head_idx, save_dir):
    """Plot a single attention head's weights as a heatmap."""
    # weights shape: (T, T)
    T = len(tokens)
    attn = weights[:T, :T].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        vmin=0,
        vmax=attn.max(),
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Layer {layer_idx + 1}, Head {head_idx + 1}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Key (attends to)", fontsize=11)
    ax.set_ylabel("Query (token)", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    path = os.path.join(save_dir, f"layer{layer_idx + 1}_head{head_idx + 1}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_attention_grid(all_weights, tokens, num_layers, num_heads, save_dir):
    """Plot all heads in a single grid figure."""
    T = len(tokens)
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4 * num_heads, 4 * num_layers))

    if num_layers == 1:
        axes = axes.reshape(1, -1)
    if num_heads == 1:
        axes = axes.reshape(-1, 1)

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attn = all_weights[layer_idx][0, head_idx, :T, :T].cpu().numpy()
            ax = axes[layer_idx][head_idx]
            sns.heatmap(
                attn,
                xticklabels=tokens if layer_idx == num_layers - 1 else False,
                yticklabels=tokens if head_idx == 0 else False,
                cmap="viridis",
                vmin=0,
                vmax=attn.max(),
                square=True,
                ax=ax,
                cbar=False,
            )
            ax.set_title(f"L{layer_idx + 1} H{head_idx + 1}", fontsize=9)
            ax.tick_params(axis="x", rotation=45, labelsize=5)
            ax.tick_params(axis="y", rotation=0, labelsize=5)

    fig.suptitle("Attention Head Patterns Across Layers", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "attention_grid.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grid saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize attention heads")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default=None,
                        help="Sample text to visualize (default: first 20 tokens from data)")
    parser.add_argument("--max_tokens", type=int, default=20,
                        help="Max tokens to visualize (keep small for readable heatmaps)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading checkpoint...")
    model, config, tokenizer = load_model(args.checkpoint, device)
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]
    print(f"Model: d_model={config['model']['d_model']}, layers={num_layers}, heads={num_heads}")

    # Prepare input text
    if args.text:
        text = args.text
    else:
        from src.dataset import load_raw_text
        text = load_raw_text()[:500]  # First 500 chars

    token_ids = tokenizer.encode(text)[:args.max_tokens]
    token_labels = [tokenizer.decode([tid]).replace("\n", "\\n") for tid in token_ids]

    print(f"Tokens ({len(token_ids)}): {token_labels}")

    # Forward pass with attention weights
    input_tensor = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        logits, all_weights = model(input_tensor, return_weights=True)

    # Create output directory
    save_dir = os.path.join("figures", "attention_heatmaps")
    os.makedirs(save_dir, exist_ok=True)

    # Plot individual heatmaps
    print(f"\nGenerating {num_layers * num_heads} individual heatmaps...")
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            head_weights = all_weights[layer_idx][0, head_idx]  # (T, T)
            plot_attention_heatmap(head_weights, token_labels, layer_idx, head_idx, save_dir)

    # Plot combined grid
    print(f"\nGenerating combined grid...")
    plot_attention_grid(all_weights, token_labels, num_layers, num_heads, save_dir)

    print(f"\nDone! All figures saved to {save_dir}/")


if __name__ == "__main__":
    main()
