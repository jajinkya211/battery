"""Physics-informed neural network model for engine virtual sensing."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from engine_pinn.models.layers import LatentActivationBlock


class EnginePINN(nn.Module):
    """PINN with latent physical states and deterministic physics equations."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: tuple[int, int] = (64, 32),
        nox_a_init: float = 500.0,
        nox_b_init: float = 5.0,
    ) -> None:
        super().__init__()
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
        self.nox_c_raw = nn.Parameter(torch.tensor(1.0))

        self._init_latent_biases()

    @property
    def nox_a(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.nox_a_raw)

    @property
    def nox_b(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.nox_b_raw)

    @property
    def nox_c(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.nox_c_raw) + 0.5

    def _init_latent_biases(self) -> None:
        nn.init.xavier_uniform_(self.latent_head.weight)
        with torch.no_grad():
            self.latent_head.bias[0] = 0.5
            l2_target = (0.35 - 0.2) / 0.6
            self.latent_head.bias[1] = torch.log(torch.tensor(l2_target / (1 - l2_target)))
            self.latent_head.bias[2] = 1.0

    def set_physics_scalar_training(self, enabled: bool) -> None:
        """Freeze/unfreeze A, B, C scalar parameters for multi-phase training."""
        self.nox_a_raw.requires_grad = enabled
        self.nox_b_raw.requires_grad = enabled
        self.nox_c_raw.requires_grad = enabled

    def forward(self, x_scaled: torch.Tensor, x_raw: torch.Tensor, lhv_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass using scaled features/raw features and per-sample LHV."""
        eps = 1e-6
        feats = self.feature_extractor(x_scaled)
        z_unconstrained = self.latent_head(feats)
        latents = self.latent_activation(z_unconstrained)

        bmep = x_raw[:, 0:1]
        l1 = latents[:, 0:1]
        l2 = latents[:, 1:2]
        l3 = latents[:, 2:3]

        lhv_kj = torch.clamp(lhv_raw * 1000.0, min=eps)
        bsfc = (3600.0 / lhv_kj) * ((bmep + l1) / (torch.clamp(bmep, min=eps) * torch.clamp(l2, min=eps)))
        l3_safe = torch.clamp(l3, min=eps)
        nox = self.nox_a * torch.pow(l3_safe, self.nox_c) * torch.exp(-self.nox_b / l3_safe)
        nox = torch.clamp(nox, min=eps)

        return {"bsfc": bsfc, "nox": nox, "latents": latents}
