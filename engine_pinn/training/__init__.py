"""Training package with objective and trainer utilities."""

from .loss import PINNLoss
from .trainer import Trainer

__all__ = ["PINNLoss", "Trainer"]
