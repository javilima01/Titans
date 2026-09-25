import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Tuple


class MemoryMLP(nn.Module):
    """
    Neural memory M(k).

    This network's *weights* are the long-term memory state.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        depth: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = dim

        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TitansMemory(nn.Module):
    """
    Simplified sequential Titans neural-memory block.

    Input:
        x: [B, T, D]

    Output:
        y: [B, T, D]

    Important:
        This is the straightforward sequential/reference implementation.
        It does NOT implement the paper's chunk-parallel training algorithm.
    """

    def __init__(
        self,
        dim: int,
        hidden_size: int = None,
        memory_depth: int = 2,
        max_lr: float = 0.1,
    ):
        super().__init__()

        hidden_size = hidden_size or 4 * dim

        self.dim = dim
        self.max_lr = max_lr

        # Associative memory projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Neural memory M
        self.memory = MemoryMLP(
            dim=dim,
            hidden_dim=hidden_size,
            depth=memory_depth,
        )

        # Data-dependent Titans gates.
        #
        # alpha: forgetting / weight decay
        # eta:   surprise momentum
        # theta: inner-loop learning rate
        self.to_alpha = nn.Linear(dim, 1)
        self.to_eta = nn.Linear(dim, 1)
        self.to_theta = nn.Linear(dim, 1)

        # Optional normalization before memory projections
        self.norm = nn.LayerNorm(dim)

    def _initial_fast_weights(self):
        """
        Clone the learned initialization of the memory network.

        These tensors become the mutable test-time memory state.
        """
        return OrderedDict((name, p.clone()) for name, p in self.memory.named_parameters())

    @staticmethod
    def _functional_memory(
        memory: nn.Module,
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
    ):
        """
        Run MemoryMLP with explicitly supplied fast weights.
        """
        return torch.func.functional_call(memory, params, (x,))

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = False,
    ):
        """
        x: [batch, sequence, dim]

        Memory is independent for every item in the batch.
        For clarity this reference implementation scans batch elements
        independently.
        """

        B, T, D = x.shape

        if D != self.dim:
            raise ValueError(f"expected feature dimension {self.dim}, got {D}")

        x_norm = self.norm(x)

        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        # Titans' input-dependent update coefficients.
        alpha = torch.sigmoid(self.to_alpha(x_norm))

        # momentum coefficient in [0, 1]
        eta = torch.sigmoid(self.to_eta(x_norm))

        # positive bounded inner-loop step size
        theta = self.max_lr * torch.sigmoid(self.to_theta(x_norm))

        batch_outputs = []
        final_states = []

        for b in range(B):
            # M_0
            params = self._initial_fast_weights()

            # S_0 = 0
            surprise_momentum = OrderedDict(
                (name, torch.zeros_like(p)) for name, p in params.items()
            )

            outputs = []

            for t in range(T):
                kt = k[b, t : t + 1]  # [1, D]
                vt = v[b, t : t + 1]
                qt = q[b, t : t + 1]

                alpha_t = alpha[b, t]
                eta_t = eta[b, t]
                theta_t = theta[b, t]

                # -----------------------------------------------------
                # 1. Associative memory loss
                #
                # l(M_{t-1}; x_t)
                #   = ||M_{t-1}(k_t) - v_t||^2
                # -----------------------------------------------------

                predicted_value = self._functional_memory(
                    self.memory,
                    params,
                    kt,
                )

                memory_loss = F.mse_loss(
                    predicted_value,
                    vt,
                    reduction="sum",
                )

                # "Momentary surprise":
                #
                # grad_M l(M_{t-1}; x_t)
                grads = torch.autograd.grad(
                    memory_loss,
                    tuple(params.values()),
                    create_graph=self.training,
                    retain_graph=self.training,
                )

                grads = OrderedDict(zip(params.keys(), grads))

                # -----------------------------------------------------
                # 2. Surprise momentum
                #
                # S_t = eta_t S_{t-1}
                #       - theta_t grad(l_t)
                # -----------------------------------------------------

                new_surprise = OrderedDict()

                for name in params:
                    new_surprise[name] = eta_t * surprise_momentum[name] - theta_t * grads[name]

                # -----------------------------------------------------
                # 3. Forgetting + memory update
                #
                # M_t = (1 - alpha_t) M_{t-1} + S_t
                # -----------------------------------------------------

                new_params = OrderedDict()

                for name in params:
                    new_params[name] = (1.0 - alpha_t) * params[name] + new_surprise[name]

                params = new_params
                surprise_momentum = new_surprise

                # -----------------------------------------------------
                # 4. Retrieve from UPDATED memory
                #
                # y_t = M_t(q_t)
                # -----------------------------------------------------

                recalled = self._functional_memory(
                    self.memory,
                    params,
                    qt,
                )

                outputs.append(recalled)

            outputs = torch.cat(outputs, dim=0)
            batch_outputs.append(outputs)

            final_states.append(
                {
                    "params": params,
                    "surprise": surprise_momentum,
                }
            )

        y = torch.stack(batch_outputs, dim=0)

        if return_state:
            return y, final_states

        return y


class AttentionWithTitans(nn.Module):
    """
    Wraps an existing Qwen attention module:

        Attention(x)

    becomes:

        Attention(x) + sigmoid(gate) * Titans(x)

    Qwen's original decoder residual connection remains untouched.
    """

    def __init__(
        self,
        attention: nn.Module,
        memory: TitansMemory,
        gate_init: float = -5.0,
    ):
        super().__init__()

        self.attention = attention
        self.memory = memory

        # sigmoid(-5) ~= 0.0067, so initially the pretrained model
        # is almost unchanged.
        self.memory_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        # Original pretrained Qwen attention
        attn_out, attn_weights = self.attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

        # Titans branch sees exactly the same normalized hidden state
        # as attention.
        memory_out = self.memory(hidden_states)

        gate = torch.sigmoid(self.memory_gate).to(
            dtype=attn_out.dtype,
            device=attn_out.device,
        )

        out = attn_out + gate * memory_out

        return out, attn_weights
