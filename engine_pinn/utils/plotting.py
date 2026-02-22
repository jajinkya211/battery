"""Plotting utilities for convergence and latent sensitivity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

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


def plot_parity_plots(df: pd.DataFrame, save_path: Path) -> None:
    """Parity plots for BSFC and NOx with R² in title."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target in zip(axes, ["BSFC", "NOx"]):
        true = df[f"{target}_true"]
        pred = df[f"{target}_pred"]
        ax.scatter(true, pred, alpha=0.35, s=14)
        mn, mx = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], "r--")
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{target} Parity (R²={r2_score(true, pred):.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_inputs(df: pd.DataFrame, save_path: Path) -> None:
    """Residual plots against all inputs for BSFC and NOx."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = ["BMEP", "H2_percentage", "Spark_Ignition_Timing", "Lambda"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey="row")

    df = df.copy()
    df["BSFC_resid"] = df["BSFC_pred"] - df["BSFC_true"]
    df["NOx_resid"] = df["NOx_pred"] - df["NOx_true"]

    for i, inp in enumerate(inputs):
        sns.scatterplot(data=df, x=inp, y="BSFC_resid", ax=axes[0, i], s=12, alpha=0.4)
        axes[0, i].axhline(0.0, color="red", linestyle="--", linewidth=1)
        axes[0, i].set_ylabel("BSFC residual" if i == 0 else "")

        sns.scatterplot(data=df, x=inp, y="NOx_resid", ax=axes[1, i], s=12, alpha=0.4)
        axes[1, i].axhline(0.0, color="red", linestyle="--", linewidth=1)
        axes[1, i].set_ylabel("NOx residual" if i == 0 else "")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
