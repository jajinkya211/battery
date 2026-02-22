"""Dataset utilities for PINN-based engine virtual sensing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


FEATURE_COLUMNS = ["BMEP", "H2_percentage", "Spark_Ignition_Timing", "Lambda"]
TARGET_COLUMNS = ["BSFC", "NOx"]


@dataclass
class DatasetBundle:
    """Container for dataloaders and scalers."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    x_scaler: StandardScaler
    y_scaler: StandardScaler


class EngineDataset(Dataset):
    """Torch dataset that stores normalized features/targets and raw features."""

    def __init__(
        self,
        features_scaled: np.ndarray,
        targets_scaled: np.ndarray,
        features_raw: np.ndarray,
        targets_raw: np.ndarray,
    ) -> None:
        if len(features_scaled) != len(targets_scaled):
            raise ValueError("Feature and target lengths must match.")

        self.features_scaled = torch.tensor(features_scaled, dtype=torch.float32)
        self.targets_scaled = torch.tensor(targets_scaled, dtype=torch.float32)
        self.features_raw = torch.tensor(features_raw, dtype=torch.float32)
        self.targets_raw = torch.tensor(targets_raw, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features_scaled)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.features_scaled[idx],
            "y": self.targets_scaled[idx],
            "x_raw": self.features_raw[idx],
            "y_raw": self.targets_raw[idx],
        }


def load_or_generate_dataframe(
    csv_path: Path,
    n_samples: int = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load dataset from CSV or generate synthetic data if missing."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        required_cols = set(FEATURE_COLUMNS + TARGET_COLUMNS)
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        return df

    rng = np.random.default_rng(random_state)
    bmep = rng.uniform(2.0, 18.0, n_samples)
    h2 = rng.uniform(0.0, 40.0, n_samples)
    spark = rng.uniform(-10.0, 30.0, n_samples)
    lamb = rng.uniform(0.85, 1.25, n_samples)

    fmep = 0.4 + 0.08 * bmep + 0.02 * rng.normal(size=n_samples)
    eff = np.clip(0.28 + 0.015 * bmep - 0.0015 * (spark - 10) ** 2 + 0.003 * h2, 0.2, 0.8)
    temp = 800 + 9.0 * spark + 6.5 * bmep + 2.0 * h2 + 20 * rng.normal(size=n_samples)
    temp = np.clip(temp, 300, None)

    lhv = 42_000.0
    bsfc = (3600.0 / lhv) * ((bmep + fmep) / (np.maximum(bmep, 1e-3) * eff))
    bsfc *= 1.0 + 0.02 * rng.normal(size=n_samples)

    a_true, b_true = 1200.0, 4.8
    nox = a_true * np.exp(-b_true / np.maximum(temp / 1000.0, 1e-4))
    nox *= 1 + 0.06 * np.maximum(spark, 0) / 30.0
    nox *= 1.0 + 0.03 * rng.normal(size=n_samples)
    nox = np.clip(nox, 1e-3, None)

    df = pd.DataFrame(
        {
            "BMEP": bmep,
            "H2_percentage": h2,
            "Spark_Ignition_Timing": spark,
            "Lambda": lamb,
            "BSFC": bsfc,
            "NOx": nox,
        }
    )
    return df


def _split_dataframe(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=random_state)
    return train_df, val_df, test_df


def build_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 128,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> DatasetBundle:
    """Build train/val/test dataloaders with standard scaling."""
    train_df, val_df, test_df = _split_dataframe(df, test_size, val_size, random_state)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train_df[FEATURE_COLUMNS])
    y_train = y_scaler.fit_transform(train_df[TARGET_COLUMNS])

    x_val = x_scaler.transform(val_df[FEATURE_COLUMNS])
    y_val = y_scaler.transform(val_df[TARGET_COLUMNS])

    x_test = x_scaler.transform(test_df[FEATURE_COLUMNS])
    y_test = y_scaler.transform(test_df[TARGET_COLUMNS])

    train_ds = EngineDataset(x_train, y_train, train_df[FEATURE_COLUMNS].to_numpy(), train_df[TARGET_COLUMNS].to_numpy())
    val_ds = EngineDataset(x_val, y_val, val_df[FEATURE_COLUMNS].to_numpy(), val_df[TARGET_COLUMNS].to_numpy())
    test_ds = EngineDataset(x_test, y_test, test_df[FEATURE_COLUMNS].to_numpy(), test_df[TARGET_COLUMNS].to_numpy())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return DatasetBundle(train_loader, val_loader, test_loader, x_scaler, y_scaler)
