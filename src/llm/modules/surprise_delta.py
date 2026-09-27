"""Online delta-rule memory with two exponentially decaying fast weights.

This is a small HOPE-inspired adapter, not an implementation of the full HOPE
self-modifying architecture. The inner update is a gradient step on the
associative squared error, recomputed from the current state at every token.
"""

import math

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


class SurpriseDeltaMemory(nn.Module):
    """Two-timescale, online key/value memory with explicit persistent state."""

    def __init__(
        self,
        dim: int,
        memory_size: int = 256,
        chunk_size: int = 16,
        checkpoint_chunks: bool = True,
    ):
        super().__init__()
        if min(dim, memory_size, chunk_size) < 1:
            msg = "Memory dimensions and chunk size must be positive"
            raise ValueError(msg)
        self.dim = dim
        self.memory_size = memory_size
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.norm = nn.LayerNorm(dim)
        self.to_k = nn.Linear(dim, memory_size, bias=False)
        self.to_q = nn.Linear(dim, memory_size, bias=False)
        self.to_v = nn.Linear(dim, memory_size, bias=False)
        self.to_out = nn.Linear(memory_size, dim, bias=False)
        # Shared initial geometry lets a freshly written key be queried before
        # outer training discovers a useful embedding.
        with torch.no_grad():
            self.to_q.weight.copy_(self.to_k.weight)
        self.log_decay_fast = nn.Parameter(torch.tensor(math.log(math.log(2) / 512)))
        self.log_decay_slow = nn.Parameter(torch.tensor(math.log(math.log(2) / 8192)))
        self.logit_rate_fast = nn.Parameter(torch.tensor(math.log(4.0)))
        self.logit_rate_slow = nn.Parameter(torch.tensor(-math.log(9.0)))
        self.read_logits = nn.Parameter(torch.tensor([1.0, 0.0]))

    def _initial_state(self, batch, *, device, dtype):
        shape = (batch, self.memory_size, self.memory_size)
        return {
            "params": {
                "fast": torch.zeros(shape, device=device, dtype=dtype),
                "slow": torch.zeros(shape, device=device, dtype=dtype),
            },
            "surprise": {},
        }

    def _check_state(self, state, batch):
        if set(state) != {"params", "surprise"} or state["surprise"]:
            msg = "Invalid delta memory state"
            raise ValueError(msg)
        if set(state["params"]) != {"fast", "slow"} or any(
            value.shape != (batch, self.memory_size, self.memory_size)
            for value in state["params"].values()
        ):
            msg = "Invalid delta memory state shape"
            raise ValueError(msg)

    def _projections(self, x):
        x = self.norm(x.to(self.norm.weight.dtype))
        return (
            functional.normalize(self.to_q(x), dim=-1),
            functional.normalize(self.to_k(x), dim=-1),
            self.to_v(x),
        )

    def _rates(self):
        # A product of exp(-lambda) gives exact exponential retention over any
        # number of valid tokens. Bound rates so a unit key cannot overshoot.
        decay = torch.exp(-torch.stack((self.log_decay_fast, self.log_decay_slow)).exp())
        rate = torch.stack((self.logit_rate_fast, self.logit_rate_slow)).sigmoid()
        mix = self.read_logits.softmax(dim=0)
        return decay, rate, mix

    def _chunk(self, fast, slow, q, k, v, valid):
        decay, rate, mix = self._rates()
        outputs = []
        for position in range(q.shape[1]):
            query = q[:, position].unsqueeze(-1)
            key = k[:, position].unsqueeze(-1)
            value = v[:, position].unsqueeze(-1)
            # Read before write: no current or future token can be retrieved.
            retrieved = mix[0] * (fast @ query) + mix[1] * (slow @ query)
            outputs.append(self.to_out(retrieved.squeeze(-1)))
            # e = v - Wk is the negative output gradient of 1/2 ||Wk-v||^2.
            # Its outer product with k is the exact online weight gradient.
            fast_error = value - fast @ key
            slow_error = value - slow @ key
            next_fast = decay[0] * fast + rate[0] * fast_error @ key.transpose(-1, -2)
            next_slow = decay[1] * slow + rate[1] * slow_error @ key.transpose(-1, -2)
            mask = valid[:, position, None, None]
            fast = torch.where(mask, next_fast, fast)
            slow = torch.where(mask, next_slow, slow)
        return torch.stack(outputs, dim=1) * valid.unsqueeze(-1), fast, slow

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = False,
        token_mask=None,
        *,
        state=None,
        batched_state: bool = False,
    ):
        batch, sequence, dim = x.shape
        if batch < 1 or sequence < 1 or dim != self.dim:
            msg = f"Expected nonempty [batch, sequence, {self.dim}] input"
            raise ValueError(msg)
        if token_mask is None:
            token_mask = torch.ones((batch, sequence), device=x.device, dtype=torch.bool)
        elif token_mask.shape != x.shape[:2]:
            msg = "token_mask must have shape [batch, sequence]"
            raise ValueError(msg)
        token_mask = token_mask.to(device=x.device, dtype=torch.bool)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = self._projections(x)
            state = (
                state
                if state is not None
                else self._initial_state(batch, device=x.device, dtype=q.dtype)
            )
            self._check_state(state, batch)
            fast, slow = state["params"]["fast"], state["params"]["slow"]
            outputs = []
            for start in range(0, sequence, self.chunk_size):
                stop = start + self.chunk_size
                args = (
                    fast,
                    slow,
                    q[:, start:stop],
                    k[:, start:stop],
                    v[:, start:stop],
                    token_mask[:, start:stop],
                )
                if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                    output, fast, slow = checkpoint(
                        self._chunk, *args, use_reentrant=False, preserve_rng_state=False
                    )
                else:
                    output, fast, slow = self._chunk(*args)
                outputs.append(output)
            result = torch.cat(outputs, dim=1)
            final = {"params": {"fast": fast, "slow": slow}, "surprise": {}}
        if not return_state:
            return result
        if batched_state:
            return result, final
        return result, [
            {"params": {"fast": fast[i], "slow": slow[i]}, "surprise": {}} for i in range(batch)
        ]

    def read_state(self, x: torch.Tensor, state: dict | None) -> torch.Tensor:
        """Read one immutable pre-window state from any selected Qwen layer."""
        if state is None:
            return torch.zeros_like(x, dtype=torch.float32)
        self._check_state(state, x.shape[0])
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = self.norm(x.to(self.norm.weight.dtype))
            query = functional.normalize(self.to_q(normalized), dim=-1)
            mix = self.read_logits.softmax(dim=0)
            retrieved = mix[0] * torch.bmm(query, state["params"]["fast"].transpose(1, 2)) + mix[
                1
            ] * torch.bmm(query, state["params"]["slow"].transpose(1, 2))
            return self.to_out(retrieved)
