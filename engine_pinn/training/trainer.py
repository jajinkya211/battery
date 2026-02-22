"""Training loop for two-phase PINN optimization with early stopping."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from engine_pinn.training.loss import PINNLoss


@dataclass
class TrainHistory:
    train_total: List[float]
    val_total: List[float]


class Trainer:
    """Trainer implementing latent pretraining and full optimization phases."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: PINNLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: Path,
        patience: int = 20,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=6)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _step(self, batch: Dict[str, torch.Tensor], training: bool) -> Dict[str, float]:
        x_scaled = batch["x"].to(self.device)
        y_raw = batch["y_raw"].to(self.device)
        x_raw = batch["x_raw"].to(self.device).detach().requires_grad_(True)
        lhv_raw = batch["lhv_raw"].to(self.device)

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        out = self.model(x_scaled, x_raw, lhv_raw)
        losses = self.loss_fn(out, y_raw, x_raw)

        if training:
            losses.total.backward()
            self.optimizer.step()

        return {
            "total": float(losses.total.detach().cpu()),
            "data": float(losses.data.detach().cpu()),
            "physics": float(losses.physics.detach().cpu()),
            "mono": float(losses.monotonicity.detach().cpu()),
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        entries = [self._step(batch, training) for batch in loader]
        return {k: float(np.mean([e[k] for e in entries])) for k in entries[0]}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs_phase1: int = 30,
        epochs_phase2: int = 70,
    ) -> TrainHistory:
        best_val = float("inf")
        no_improve = 0
        history = TrainHistory(train_total=[], val_total=[])

        total_epochs = epochs_phase1 + epochs_phase2
        for epoch in range(total_epochs):
            phase = 1 if epoch < epochs_phase1 else 2
            self.model.set_physics_scalar_training(enabled=(phase == 2))

            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            self.scheduler.step(val_metrics["total"])
            history.train_total.append(train_metrics["total"])
            history.val_total.append(val_metrics["total"])

            self.logger.info(
                "Epoch %d/%d | phase=%d | train=%.6f | val=%.6f | data=%.6f | phys=%.6f | mono=%.6f",
                epoch + 1,
                total_epochs,
                phase,
                train_metrics["total"],
                val_metrics["total"],
                val_metrics["data"],
                val_metrics["physics"],
                val_metrics["mono"],
            )

            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                no_improve = 0
                ckpt = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), ckpt)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        best_ckpt = self.checkpoint_dir / "best_model.pt"
        if best_ckpt.exists():
            self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        return history
