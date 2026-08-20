"""
Scaling law curve plotter.

Reads results/scaling_runs.csv and generates a log-log plot of
test loss vs. parameter count, with a power-law fit.

Usage:
    python scripts/plot_scaling.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_results(csv_path="results/scaling_runs.csv"):
    """Load scaling run results from CSV."""
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "name": row["name"],
                "params": int(row["params"]),
                "test_loss": float(row["test_loss"]),
                "train_loss": float(row["train_loss"]),
            })
    return results


def fit_power_law(params, losses):
    """Fit log(loss) = a * log(params) + b, return slope and intercept."""
    log_params = np.log(params)
    log_losses = np.log(losses)
    coeffs = np.polyfit(log_params, log_losses, 1)
    slope, intercept = coeffs

    # R² calculation
    predicted = np.polyval(coeffs, log_params)
    ss_res = np.sum((log_losses - predicted) ** 2)
    ss_tot = np.sum((log_losses - np.mean(log_losses)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return slope, intercept, r_squared


def plot_scaling_curve(results, save_path="figures/scaling_curve.png"):
    """Generate a publication-quality scaling law plot."""
    params = np.array([r["params"] for r in results])
    test_losses = np.array([r["test_loss"] for r in results])
    names = [r["name"] for r in results]

    slope, intercept, r_squared = fit_power_law(params, test_losses)

    # Generate fitted line
    log_params_range = np.linspace(np.log(params.min()) - 0.3, np.log(params.max()) + 0.3, 100)
    fitted_log_losses = slope * log_params_range + intercept

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter points
    ax.scatter(params, test_losses, s=120, c="#2563eb", edgecolors="white",
               linewidths=1.5, zorder=5)

    # Fitted line
    ax.plot(np.exp(log_params_range), np.exp(fitted_log_losses),
            color="#dc2626", linewidth=2, linestyle="--", alpha=0.8,
            label=f"Power law fit: α = {abs(slope):.4f}")

    # Label each point
    for i, name in enumerate(names):
        ax.annotate(
            name.upper(),
            (params[i], test_losses[i]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight="bold",
            color="#374151",
        )

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (non-embedding)", fontsize=13)
    ax.set_ylabel("Test Loss (cross-entropy)", fontsize=13)
    ax.set_title("Scaling Law: Test Loss vs. Model Size", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=12, loc="upper right")

    # Annotation box with fit stats
    textstr = (
        f"Fit: L(N) ∝ N^(−{abs(slope):.4f})\n"
        f"R² = {r_squared:.4f}\n"
        f"Kaplan et al. ref: α ≈ 0.076"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#f3f4f6", edgecolor="#d1d5db", alpha=0.9)
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", bbox=props)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scaling curve to {save_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Scaling Law Results")
    print(f"{'='*50}")
    print(f"  Slope (α):  {abs(slope):.4f}")
    print(f"  R²:         {r_squared:.4f}")
    print(f"  Kaplan ref: α ≈ 0.076")
    print(f"{'='*50}")
    print(f"\n  {'Name':<6} {'Params':>12} {'Test Loss':>12}")
    print(f"  {'-'*6} {'-'*12} {'-'*12}")
    for r in sorted(results, key=lambda x: x["params"]):
        print(f"  {r['name']:<6} {r['params']:>12,} {r['test_loss']:>12.4f}")


def main():
    csv_path = "results/scaling_runs.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run training first.")
        sys.exit(1)

    results = load_results(csv_path)
    if len(results) < 2:
        print(f"Need at least 2 runs for a scaling plot. Found {len(results)}.")
        print("Run more configs with: python scripts/train.py --config configs/<size>.yaml")
        sys.exit(1)

    plot_scaling_curve(results)


if __name__ == "__main__":
    main()
