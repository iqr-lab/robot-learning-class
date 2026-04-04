import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_metrics(csv_path, output_path=None):
    df = pd.read_csv(csv_path)

    # Use GAIL round as x-axis
    x = df["mean/disc/global_step"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- Panel 1: Generator imitation loss (PPO policy gradient loss) ---
    ax = axes[0]
    col = "mean/gen/train/policy_gradient_loss"
    ax.plot(x, df[col], color="#8b5cf6", lw=1.5, label="policy gradient loss")
    ax.set_ylabel("Loss")
    ax.set_title("Generator Imitation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Discriminator accuracy on expert vs generated ---
    # Per-source loss is not logged; accuracy is the closest available split.
    ax = axes[1]
    ax.plot(x, df["mean/disc/disc_acc_expert"], color="#06b6d4", lw=1.5, label="acc on demonstrations")
    ax.plot(x, df["mean/disc/disc_acc_gen"],    color="#f43f5e", lw=1.5, label="acc on generated")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="random chance (0.5)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Discriminator Accuracy: Demonstrations vs Generated")
    ax.set_xlabel("GAIL Round")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_file", required=True, help="path to progress.csv")
    parser.add_argument("-o", "--output", default=None, help="optional output image path")
    args = parser.parse_args()

    plot_metrics(args.log_file, args.output)
