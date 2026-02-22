"""Plotting utilities for convergence and latent sensitivity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def plot_training_curves(history: Dict[str, Iterable[float]], save_path: Path) -> None:
    """Plot train and validation total loss across epochs."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(list(history["train_total"]), label="Train Total")
    plt.plot(list(history["val_total"]), label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_latent_trends(df: pd.DataFrame, save_path: Path) -> None:
    """Plot latent variables against key inputs for sensitivity analysis."""
    required = {
        "BMEP",
        "H2_percentage",
        "Spark_Ignition_Timing",
        "Lambda",
        "L1_FMEP",
        "L2_Indicated_Efficiency",
        "L3_Temp_Potential",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for latent plotting: {sorted(missing)}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey="row")
    latents = [
        "L1_FMEP",
        "L2_Indicated_Efficiency",
        "L3_Temp_Potential",
    ]
    inputs = ["BMEP", "H2_percentage", "Spark_Ignition_Timing", "Lambda"]

    for r, latent in enumerate(latents):
        for c, input_col in enumerate(inputs):
            ax = axes[r, c]
            sns.regplot(data=df, x=input_col, y=latent, ax=ax, scatter_kws={"s": 12, "alpha": 0.4}, line_kws={"color": "red"})
            if c == 0:
                ax.set_ylabel(latent)
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
