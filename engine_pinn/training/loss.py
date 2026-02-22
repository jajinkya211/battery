"""Composite loss for physics-informed engine modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class LossBreakdown:
    total: torch.Tensor
    data: torch.Tensor
    physics: torch.Tensor
    monotonicity: torch.Tensor


class PINNLoss(nn.Module):
    """Data + physics + monotonicity loss with autograd-based constraints."""

    def __init__(self, lambda_phys: float = 1.0, lambda_mono: float = 0.2) -> None:
        super().__init__()
        self.lambda_phys = lambda_phys
        self.lambda_mono = lambda_mono
        self.mse = nn.MSELoss()

    def _physics_penalty(self, latents: torch.Tensor) -> torch.Tensor:
        l1, l2, l3 = latents[:, 0], latents[:, 1], latents[:, 2]
        penalty = (
            torch.relu(-l1).pow(2).mean()
            + torch.relu(0.2 - l2).pow(2).mean()
            + torch.relu(l2 - 1.0).pow(2).mean()
            + torch.relu(-l3).pow(2).mean()
        )
        return penalty

    def _monotonicity_penalty(
        self,
        x_raw: torch.Tensor,
        bsfc_pred: torch.Tensor,
        nox_pred: torch.Tensor,
    ) -> torch.Tensor:
        ones_bsfc = torch.ones_like(bsfc_pred)
        ones_nox = torch.ones_like(nox_pred)

        grad_bsfc = torch.autograd.grad(
            bsfc_pred,
            x_raw,
            grad_outputs=ones_bsfc,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        grad_nox = torch.autograd.grad(
            nox_pred,
            x_raw,
            grad_outputs=ones_nox,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grad_bsfc is None:
            grad_bsfc = torch.zeros_like(x_raw)
        if grad_nox is None:
            grad_nox = torch.zeros_like(x_raw)

        dbsfc_dbmep = grad_bsfc[:, 0]
        dnox_dbmep = grad_nox[:, 0]
        dnox_dspark = grad_nox[:, 2]

        violation = (
            torch.relu(dbsfc_dbmep).pow(2).mean()
            + torch.relu(-dnox_dbmep).pow(2).mean()
            + torch.relu(-dnox_dspark).pow(2).mean()
        )
        return violation

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        y_true_raw: torch.Tensor,
        x_raw_requires_grad: torch.Tensor,
    ) -> LossBreakdown:
        eps = 1e-6
        bsfc_pred = model_out["bsfc"]
        nox_pred = model_out["nox"]
        latents = model_out["latents"]

        bsfc_true = y_true_raw[:, 0:1]
        nox_true = torch.clamp(y_true_raw[:, 1:2], min=eps)

        data_loss = self.mse(bsfc_pred, bsfc_true) + self.mse(
            torch.log(torch.clamp(nox_pred, min=eps)), torch.log(nox_true)
        )
        physics_loss = self._physics_penalty(latents)
        mono_loss = self._monotonicity_penalty(x_raw_requires_grad, bsfc_pred, nox_pred)

        total = data_loss + self.lambda_phys * physics_loss + self.lambda_mono * mono_loss
        return LossBreakdown(total=total, data=data_loss, physics=physics_loss, monotonicity=mono_loss)
