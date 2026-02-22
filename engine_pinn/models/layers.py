"""Custom layers and activations for physical latent constraints."""

from __future__ import annotations

import torch
import torch.nn as nn


class EfficiencyActivation(nn.Module):
    """Map unconstrained values to physically plausible efficiency [low, high]."""

    def __init__(self, low: float = 0.2, high: float = 0.8) -> None:
        super().__init__()
        if high <= low:
            raise ValueError("`high` must be greater than `low`.")
        self.low = low
        self.scale = high - low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.low + self.scale * torch.sigmoid(x)


class LatentActivationBlock(nn.Module):
    """Apply dedicated activations to latent outputs L1, L2, and L3."""

    def __init__(self) -> None:
        super().__init__()
        self.softplus = nn.Softplus()
        self.eff_act = EfficiencyActivation(0.2, 0.8)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != 3:
            raise ValueError("Latent tensor must have exactly 3 channels.")

        l1 = self.softplus(z[:, 0:1])
        l2 = self.eff_act(z[:, 1:2])
        l3 = self.softplus(z[:, 2:3])
        return torch.cat([l1, l2, l3], dim=-1)
