"""A fixed-size fast-weight network trained on next-token embeddings at test time.

The memory contains a linear layer and an inverse feature-covariance matrix.
It retains no text, question index, examples, token labels, or retrieval slots.
The write error is the gradient of 0.5 * ||k @ W - target_embedding||^2.
Preconditioning and normalizing that gradient protects previously written
directions while allowing a new observation to overwrite the current key.
"""

import math

import torch
from torch import nn
from torch.nn import functional


class NeuralTokenMemory(nn.Module):
    """CPU FP64 optimizer state; FP32 weights/readout, with no example replay."""

    def __init__(self, width: int, *, ridge: float = 1e-4, decay: float = 0.0):
        super().__init__()
        if width < 1 or not math.isfinite(ridge) or ridge <= 0 or not 0 <= decay < 1:
            msg = "Require positive width/ridge and decay in [0, 1)"
            raise ValueError(msg)
        self.ridge = ridge
        self.decay = decay
        self.weight = nn.Parameter(torch.zeros(width, width), requires_grad=False)
        self.register_buffer("inverse_covariance", torch.eye(width, dtype=torch.float64))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = functional.normalize(hidden.to(self.weight), dim=-1)
        return query @ self.weight

    @torch.no_grad()
    def write(self, hidden: torch.Tensor, targets: torch.Tensor, *, repeats: int = 1) -> dict:
        """Update parameters from transient supervision; discard examples on return."""
        if (
            hidden.ndim != 2
            or hidden.shape != targets.shape
            or hidden.shape[1] != self.weight.shape[0]
            or not torch.isfinite(hidden).all()
            or not torch.isfinite(targets).all()
            or (hidden.norm(dim=-1) == 0).any()
            or (targets.norm(dim=-1) == 0).any()
            or repeats < 1
        ):
            msg = "Require matching finite nonzero feature/target rows and positive repeats"
            raise ValueError(msg)
        keys = functional.normalize(hidden.detach().cpu().double(), dim=-1)
        values = functional.normalize(targets.detach().cpu().double(), dim=-1)
        errors = []
        for _ in range(repeats):
            for key, value in zip(keys, values, strict=True):
                self.weight.mul_(1 - self.decay)
                preconditioned = self.inverse_covariance @ key
                curvature = key @ preconditioned
                error = value - key @ self.weight.double()
                errors.append(float(error.square().sum()))
                gain = preconditioned / curvature.clamp_min(1e-12)
                self.weight.add_(torch.outer(gain, error).float())
                self.inverse_covariance.sub_(
                    torch.outer(preconditioned, preconditioned) / (self.ridge + curvature)
                )
        return {
            "writes": len(errors),
            "mean_surprise": sum(errors) / len(errors) if errors else 0.0,
        }
