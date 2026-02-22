"""Configuration objects for training and physical constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhysicsConfig:
    """Physical constants and initialization values."""

    lhv: float = 42_000.0
    nox_b_init: float = 5.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    seed: int = 42
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_phys: float = 1.0
    lambda_mono: float = 0.2
    epochs_phase1: int = 25
    epochs_phase2: int = 50
    patience: int = 15
    test_size: float = 0.15
    val_size: float = 0.15
    data_csv: Path = Path("engine_pinn/data/engine_data.csv")
    checkpoint_dir: Path = Path("engine_pinn/checkpoints")
    plots_dir: Path = Path("engine_pinn/artifacts")
