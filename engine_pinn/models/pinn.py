"""Physics-informed neural network model for engine virtual sensing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from engine_pinn.models.layers import LatentActivationBlock


@dataclass
class ForwardOutput:
    """Structured model output for predictions and latent variables."""

    bsfc: torch.Tensor
    nox: torch.Tensor
    latents: torch.Tensor


class EnginePINN(nn.Module):
    """PINN with latent physical states and deterministic physics equations."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: tuple[int, int] = (64, 32),
        lhv: float = 42_000.0,
        nox_a_init: float = 500.0,
        nox_b_init: float = 5.0,
    ) -> None:
        super().__init__()
        self.lhv = lhv
        h1, h2 = hidden_dims

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
        )

        self.latent_head = nn.Linear(h2, 3)
        self.latent_activation = LatentActivationBlock()

        self.nox_a_raw = nn.Parameter(torch.tensor(float(max(nox_a_init, 1e-3))))
        self.nox_b_raw = nn.Parameter(torch.tensor(float(max(nox_b_init, 1e-3))))

        self._init_latent_biases()

    @property
    def nox_a(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.nox_a_raw)

    @property
    def nox_b(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.nox_b_raw)

    def _init_latent_biases(self) -> None:
        nn.init.xavier_uniform_(self.latent_head.weight)
        with torch.no_grad():
            self.latent_head.bias[0] = 0.5
            l2_target = (0.35 - 0.2) / 0.6
            self.latent_head.bias[1] = torch.log(torch.tensor(l2_target / (1 - l2_target)))
            self.latent_head.bias[2] = 1.0

    def set_physics_scalar_training(self, enabled: bool) -> None:
        """Freeze/unfreeze A and B scalar parameters for multi-phase training."""
        self.nox_a_raw.requires_grad = enabled
        self.nox_b_raw.requires_grad = enabled

    def forward(self, x_scaled: torch.Tensor, x_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass using scaled features and raw features for physics equations."""
        eps = 1e-6
        feats = self.feature_extractor(x_scaled)
        z_unconstrained = self.latent_head(feats)
        latents = self.latent_activation(z_unconstrained)

        bmep = x_raw[:, 0:1]
        l1 = latents[:, 0:1]
        l2 = latents[:, 1:2]
        l3 = latents[:, 2:3]

        bsfc = (3600.0 / self.lhv) * ((bmep + l1) / (torch.clamp(bmep, min=eps) * torch.clamp(l2, min=eps)))
        nox = self.nox_a * torch.exp(-self.nox_b / torch.clamp(l3, min=eps))
        nox = torch.clamp(nox, min=eps)

        return {"bsfc": bsfc, "nox": nox, "latents": latents}
