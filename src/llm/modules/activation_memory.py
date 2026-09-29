"""Activation-driven fast weights with a token-grounded online learning target.

Slow projections learn how statement and question activations address the same
fast network. An online write minimizes next-token embedding reconstruction by
recursive least squares. Only fixed-size weights/optimizer tensors persist.
This is an experimental linear fast memory, not a reproduction of Titans/HOPE.
"""

import torch
from torch import nn
from torch.nn import functional


class ActivationFastMemory(nn.Module):
    """CPU FP64 online solver, with differentiable FP32 episode solves in training."""

    def __init__(
        self,
        input_width: int = 2048,
        value_width: int = 1024,
        key_width: int = 256,
        *,
        ridge: float = 0.01,
    ):
        super().__init__()
        if min(input_width, value_width, key_width) < 1 or ridge <= 0:
            msg = "Require positive widths and ridge"
            raise ValueError(msg)
        self.ridge = ridge
        self.key_width = key_width
        hidden_width = max(256, key_width)
        self.to_key = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, key_width),
        )
        self.to_query = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, key_width),
        )
        self.to_query.load_state_dict(self.to_key.state_dict())
        self.fast_weight = nn.Parameter(torch.zeros(key_width, value_width), requires_grad=False)
        self.register_buffer(
            "inverse_covariance", torch.eye(key_width, dtype=torch.float64) / ridge
        )

    def keys(self, hidden: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.to_key(hidden.float()), dim=-1)

    def queries(self, hidden: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.to_query(hidden.float()), dim=-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.queries(hidden) @ self.fast_weight

    def episode_read(self, source_hidden, query_hidden, values, source_mask):
        """Differentiate through an empty-state inner solve; never a persisted KV store."""
        keys = self.keys(source_hidden) * source_mask[..., None]
        queries = self.queries(query_hidden)
        gram = keys @ keys.transpose(-1, -2)
        gram = gram + self.ridge * torch.eye(keys.shape[1], device=keys.device)
        coefficients = torch.linalg.solve(gram, keys @ queries.transpose(-1, -2)).transpose(-1, -2)
        return coefficients @ values, queries @ keys.transpose(-1, -2)

    @torch.no_grad()
    def reset(self):
        self.fast_weight.zero_()
        self.inverse_covariance.copy_(torch.eye(self.key_width, dtype=torch.float64) / self.ridge)

    @torch.no_grad()
    def write(self, hidden: torch.Tensor, values: torch.Tensor):
        """Update fast weights explicitly; autograd is not needed for this RLS step.

        For one key, the reconstruction gradient is k.T @ (k @ W - v).
        RLS preconditions that gradient by P / (1 + k @ P @ k.T).
        Only W and P change here. Slow projections learn through episode_read
        during outer training, where gradient tracking must remain enabled.
        """
        if (
            hidden.ndim != 2
            or values.ndim != 2
            or len(hidden) != len(values)
            or not len(hidden)
            or not torch.isfinite(hidden).all()
            or not torch.isfinite(values).all()
        ):
            msg = "Require matching nonempty finite hidden and target rows"
            raise ValueError(msg)
        keys = self.keys(hidden).double()
        values = functional.normalize(values.double(), dim=-1)
        preconditioned = self.inverse_covariance @ keys.T
        gram = torch.eye(len(keys), dtype=torch.float64) + keys @ preconditioned
        gain = torch.linalg.solve(gram, preconditioned.T).T
        error = values - keys @ self.fast_weight.double()
        self.fast_weight.add_((gain @ error).float())
        self.inverse_covariance.sub_(gain @ preconditioned.T)
        return {"tokens": len(keys), "surprise": float(error.square().mean())}
