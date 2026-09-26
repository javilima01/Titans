import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint


class MemoryMLP(nn.Module):
    """MLP whose fast weights hold the sequence's associative memory."""

    def __init__(self, dim: int, hidden_dim: int, depth: int = 2):
        super().__init__()
        if min(dim, hidden_dim, depth) < 1:
            msg = "Memory dimensions and depth must be positive"
            raise ValueError(msg)
        layers = []
        in_dim = dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.SiLU()))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.linear_indices = tuple(range(0, len(layers), 2))

    def forward(self, x):
        return self.net(x)

    def associative_factors(self, params, keys, values):
        """Factor each token's weight gradient as delta @ activation.T.

        The explicit chain rule is differentiable itself: outer backprop retains
        the higher order terms needed to learn keys, values and update gates.
        Keeping these factors avoids allocating [batch, chunk, out, in]
        gradient tensors. Bias gradients are simply delta.
        """
        inputs, preactivations = [], []
        x = keys
        for index in self.linear_indices:
            inputs.append(x)
            x = x @ params[f"net.{index}.weight"].transpose(-1, -2)
            x = x + params[f"net.{index}.bias"].unsqueeze(1)
            preactivations.append(x)
            if index != self.linear_indices[-1]:
                x = functional.silu(x)

        delta = 2 * (x - values)
        factors = []
        for layer_number in range(len(self.linear_indices) - 1, -1, -1):
            index = self.linear_indices[layer_number]
            factors.append((inputs[layer_number], delta))
            if layer_number:
                delta = delta @ params[f"net.{index}.weight"]
                z = preactivations[layer_number - 1]
                sigmoid = z.sigmoid()
                delta = delta * sigmoid * (1 + z * (1 - sigmoid))
        return list(reversed(factors))


class TitansMemory(nn.Module):
    """Causal, batched Titans recurrence with gradients refreshed per chunk.

    Chunk size 1 is the original sequential update. Larger chunks use the
    paper's approximation: all associative gradients in a chunk use the same
    chunk-start weights. Momentum, forgetting and retrieval remain causal at
    every token. Factorized gradients and a dual-form read avoid materializing
    per-token weights entirely. Checkpointing reduces saved activations further.
    """

    def __init__(
        self,
        dim: int,
        hidden_size: int | None = None,
        memory_depth: int = 2,
        max_lr: float = 0.1,
        chunk_size: int = 16,
        checkpoint_chunks: bool = True,
        normalize_qk: bool = True,
        max_inner_grad_norm: float | None = None,
    ):
        super().__init__()
        if chunk_size < 1 or max_lr <= 0:
            msg = "chunk_size and max_lr must be positive"
            raise ValueError(msg)
        self.dim = dim
        self.max_lr = max_lr
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.normalize_qk = normalize_qk
        if max_inner_grad_norm is not None and max_inner_grad_norm <= 0:
            msg = "max_inner_grad_norm must be positive or None"
            raise ValueError(msg)
        self.max_inner_grad_norm = max_inner_grad_norm
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.memory = MemoryMLP(
            dim, hidden_size if hidden_size is not None else 4 * dim, memory_depth
        )
        self.to_alpha = nn.Linear(dim, 1)
        self.to_eta = nn.Linear(dim, 1)
        self.to_theta = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        # Start with slow forgetting and conservative inner steps.
        nn.init.constant_(self.to_alpha.bias, -4.0)
        nn.init.constant_(self.to_eta.bias, 2.0)
        nn.init.constant_(self.to_theta.bias, -4.0)

    @staticmethod
    def _transition(decay):
        """A[t, i] = product(decay[i+1:t+1]), i <= t; avoids dividing prefixes."""
        length = decay.shape[1]
        positions = torch.arange(length, device=decay.device)
        after = positions[:, None] > positions[None, :]
        factors = torch.where(after, decay.unsqueeze(-1), 1.0)
        return factors.cumprod(dim=1).tril()

    def _chunk(self, params, momentum, q, k, v, alpha, eta, theta, valid):
        # Masked tokens preserve BOTH fast weights and momentum.
        decay = 1 - alpha.squeeze(-1) * valid
        eta = torch.where(valid, eta.squeeze(-1), 1.0)
        theta = theta.squeeze(-1) * valid
        momentum_transition = self._transition(eta)
        weight_transition = self._transition(decay)
        momentum_prefix = eta.cumprod(dim=1).unsqueeze(-1)
        weight_prefix = decay.cumprod(dim=1).unsqueeze(-1)
        surprise_coeff = momentum_transition * theta.unsqueeze(1)
        update_coeff = weight_transition @ (surprise_coeff * valid.unsqueeze(-1))
        carry_coeff = weight_transition @ (momentum_prefix * valid.unsqueeze(-1))
        factors = self.memory.associative_factors(params, k, v)
        if self.max_inner_grad_norm is not None:
            # ||delta outer activation||_F = ||delta|| * ||activation||;
            # include bias gradients and clip the entire per-token MLP gradient.
            # The scale remains differentiable for outer meta-learning.
            norm_squared = sum(
                delta.square().sum(-1) * (activations.square().sum(-1) + 1)
                for activations, delta in factors
            )
            scale = (self.max_inner_grad_norm / norm_squared.clamp_min(1e-12).sqrt()).clamp(max=1)
            factors = [(activations, delta * scale.unsqueeze(-1)) for activations, delta in factors]

        new_params, new_momentum = {}, {}
        x = q
        for index, (activations, delta) in zip(self.memory.linear_indices, factors, strict=True):
            weight_name, bias_name = f"net.{index}.weight", f"net.{index}.bias"
            weight, bias = params[weight_name], params[bias_name]
            sw, sb = momentum[weight_name], momentum[bias_name]
            # W_t q_t can be evaluated using low-rank gradient factors instead
            # of constructing every W_t. The +1 accounts for the bias update.
            similarities = x @ activations.transpose(-1, -2) + 1
            x = (
                weight_prefix * (x @ weight.transpose(-1, -2) + bias.unsqueeze(1))
                + carry_coeff * (x @ sw.transpose(-1, -2) + sb.unsqueeze(1))
                - (update_coeff * similarities) @ delta
            )
            weighted_delta = update_coeff[:, -1].unsqueeze(-1) * delta
            momentum_delta = surprise_coeff[:, -1].unsqueeze(-1) * delta
            new_params[weight_name] = (
                weight_prefix[:, -1].unsqueeze(-1) * weight
                + carry_coeff[:, -1].unsqueeze(-1) * sw
                - weighted_delta.transpose(-1, -2) @ activations
            )
            new_params[bias_name] = (
                weight_prefix[:, -1] * bias + carry_coeff[:, -1] * sb - weighted_delta.sum(dim=1)
            )
            new_momentum[weight_name] = (
                momentum_prefix[:, -1].unsqueeze(-1) * sw
                - momentum_delta.transpose(-1, -2) @ activations
            )
            new_momentum[bias_name] = momentum_prefix[:, -1] * sb - momentum_delta.sum(dim=1)
            if index != self.memory.linear_indices[-1]:
                x = functional.silu(x)
        return x * valid.unsqueeze(-1), new_params, new_momentum

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = False,
        token_mask=None,
        *,
        state=None,
        batched_state: bool = False,
    ):
        """Optionally continue batched fast weights/momentum at a chunk boundary.

        State is explicit and never modified in place. Callers must finish whole
        chunks before carrying state to another call to preserve the update rule.
        ``batched_state`` selects a tensor dictionary instead of the legacy list.
        """
        batch, sequence, dim = x.shape
        if dim != self.dim:
            error_msg = f"expected feature dimension {self.dim}, got {dim}"
            raise ValueError(error_msg)
        if batch < 1 or sequence < 1:
            msg = "Memory requires a nonempty batch and sequence"
            raise ValueError(msg)
        if token_mask is None:
            token_mask = torch.ones((batch, sequence), dtype=torch.bool, device=x.device)
        elif token_mask.shape != x.shape[:2]:
            msg = "token_mask must have shape [batch, sequence]"
            raise ValueError(msg)
        token_mask = token_mask.to(device=x.device, dtype=torch.bool)

        # Keep the adapter FP32 when Qwen uses BF16; disable ambient AMP here.
        # Explicit gradients also work under no_grad/inference_mode at test time.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.norm(x.to(self.norm.weight.dtype))
            q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
            if self.normalize_qk:
                q, k = functional.normalize(q, dim=-1), functional.normalize(k, dim=-1)
            alpha = self.to_alpha(x).sigmoid()
            eta = self.to_eta(x).sigmoid()
            theta = self.max_lr * self.to_theta(x).sigmoid()
            if state is None:
                params = {
                    name: p.unsqueeze(0).expand(batch, *p.shape)
                    for name, p in self.memory.named_parameters()
                }
                momentum = {name: torch.zeros_like(p) for name, p in params.items()}
            else:
                params, momentum = state["params"], state["surprise"]
                for name, parameter in self.memory.named_parameters():
                    for values in (params, momentum):
                        if values[name].shape != (batch, *parameter.shape):
                            msg = f"Invalid batched memory state shape for {name}"
                            raise ValueError(msg)
            outputs = []
            for start in range(0, sequence, self.chunk_size):
                stop = start + self.chunk_size
                inputs = (
                    params,
                    momentum,
                    q[:, start:stop],
                    k[:, start:stop],
                    v[:, start:stop],
                    alpha[:, start:stop],
                    eta[:, start:stop],
                    theta[:, start:stop],
                    token_mask[:, start:stop],
                )
                if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                    y, params, momentum = checkpoint(
                        self._chunk, *inputs, use_reentrant=False, preserve_rng_state=False
                    )
                else:
                    y, params, momentum = self._chunk(*inputs)
                outputs.append(y)
            y = torch.cat(outputs, dim=1)
        if return_state:
            if batched_state:
                return y, {"params": params, "surprise": momentum}
            # Keep the existing per-example state format.
            states = [
                {
                    "params": {name: p[b] for name, p in params.items()},
                    "surprise": {name: p[b] for name, p in momentum.items()},
                }
                for b in range(batch)
            ]
            return y, states
        return y


class AttentionWithTitans(nn.Module):
    """Add a gated neural-memory branch to Qwen's attention residual."""

    def __init__(
        self,
        attention: nn.Module,
        memory: TitansMemory,
        gate_init: float = -5.0,
        memory_id: int = 0,
    ):
        super().__init__()
        self.attention = attention
        self.memory = memory
        self.memory_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.memory_id = memory_id

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        memory_mask=None,
        memory_context=None,
        **kwargs,
    ):
        attn_out, attn_weights = self.attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        if memory_context is None:
            memory_out = self.memory(hidden_states, token_mask=memory_mask)
        else:
            mode = memory_context["mode"]
            if mode == "disabled":
                return attn_out, attn_weights
            initial = memory_context["initial"].get(self.memory_id) if mode == "normal" else None
            memory_out, final = self.memory(
                hidden_states,
                token_mask=memory_mask,
                state=initial,
                return_state=True,
                batched_state=True,
            )
            memory_context["final"][self.memory_id] = final
        # Gate and multiply in FP32 before casting back to the backbone dtype.
        out = attn_out + (self.memory_gate.sigmoid() * memory_out).to(attn_out.dtype)
        return out, attn_weights
