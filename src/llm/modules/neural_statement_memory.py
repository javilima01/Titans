"""Fixed-size nonlinear fast-weight network for complete UTF-8 statements.

Inputs are semantic embeddings. A fixed random Fourier feature layer feeds a
trainable dense output layer, decoded with a fixed byte embedding layer. Online
writes update dense weights and an inverse-covariance optimizer state. No source
strings, support vectors, example replay, or per-fact slots survive a write.
"""

import math

import torch
from torch import nn
from torch.nn import functional


class NeuralStatementMemory(nn.Module):
    """Research prototype; fixed byte limit, no automatic relevance threshold."""

    def __init__(
        self,
        input_width: int = 384,
        feature_width: int = 1024,
        *,
        max_bytes: int = 192,
        code_width: int = 32,
        bandwidth: float = 4.0,
        ridge: float = 1e-4,
        seed: int = 71,
        latent_width: int = 0,
        latent_sparsity: int = 0,
    ):
        super().__init__()
        if (
            min(input_width, feature_width, max_bytes, code_width) < 1
            or not math.isfinite(bandwidth)
            or bandwidth <= 0
            or not math.isfinite(ridge)
            or ridge <= 0
            or latent_width < 0
            or not 0 <= latent_sparsity <= latent_width
        ):
            msg = "Require positive dimensions, bandwidth and ridge"
            raise ValueError(msg)
        self.ridge = ridge
        self.seed = seed
        self.latent_sparsity = latent_sparsity
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "projection",
            torch.randn(input_width, feature_width, generator=generator) * math.sqrt(2 * bandwidth),
        )
        self.register_buffer("phase", torch.rand(feature_width, generator=generator) * 2 * math.pi)
        self.register_buffer(
            "byte_embedding",
            functional.normalize(torch.randn(257, code_width, generator=generator), dim=-1),
        )
        self.max_bytes = max_bytes
        self.weight = nn.Parameter(
            torch.zeros(feature_width, latent_width or max_bytes * code_width), requires_grad=False
        )
        self.register_buffer("inverse_covariance", torch.eye(feature_width, dtype=torch.float64))
        if latent_width:
            self.decoder = nn.Parameter(
                torch.zeros(latent_width, max_bytes * code_width), requires_grad=False
            )
            self.register_buffer("latent_covariance", torch.eye(latent_width, dtype=torch.float64))
            self.register_buffer("write_count", torch.zeros((), dtype=torch.int64))

    def features(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = functional.normalize(hidden.to(self.projection), dim=-1)
        return functional.normalize(torch.cos(hidden @ self.projection + self.phase), dim=-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        result = self.features(hidden) @ self.weight
        if hasattr(self, "decoder"):
            result = self.cleanup(result) @ self.decoder
        return result.reshape(*hidden.shape[:-1], -1, self.byte_embedding.shape[1])

    def cleanup(self, latent: torch.Tensor, *, steps: int = 12) -> torch.Tensor:
        """Pseudoinverse Hopfield dynamics; no stored pattern vectors at read time."""
        width = self.latent_covariance.shape[0]
        recurrent = torch.eye(width) - self.latent_covariance.float()
        recurrent.fill_diagonal_(0)
        current = self.activate(latent)
        for _ in range(steps):
            current = self.activate(current @ recurrent)
        return functional.normalize(current, dim=-1)

    def activate(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_sparsity:
            indices = latent.topk(self.latent_sparsity, dim=-1).indices
            return torch.zeros_like(latent).scatter_(-1, indices, 1)
        return latent.sign()

    @torch.no_grad()
    def update(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        covariance: torch.Tensor,
    ) -> float:
        key = key.double()
        preconditioned = covariance @ key
        curvature = key @ preconditioned
        residual = target - key.float() @ weight
        gain = (preconditioned / curvature.clamp_min(1e-12)).float()
        weight.add_(torch.outer(gain, residual))
        covariance.sub_(torch.outer(preconditioned, preconditioned) / (self.ridge + curvature))
        return float(residual.square().mean())

    @torch.no_grad()
    def write(self, hidden: torch.Tensor, statement: str) -> dict:
        """Consume one temporary statement; output weights learn its complete bytes."""
        encoded = list(statement.encode("utf-8"))
        max_bytes = self.max_bytes
        if len(encoded) >= max_bytes:
            msg = f"Statement needs {len(encoded) + 1} bytes including EOS; limit is {max_bytes}"
            raise ValueError(msg)
        if hidden.ndim == 1:
            hidden = hidden[None]
        if (
            hidden.ndim != 2
            or hidden.shape[1] != self.projection.shape[0]
            or not torch.isfinite(hidden).all()
            or not len(hidden)
            or (hidden.norm(dim=-1) == 0).any()
        ):
            msg = "Require finite nonzero semantic embedding rows"
            raise ValueError(msg)
        ids = torch.tensor([*encoded, 256])
        target = torch.zeros(max_bytes, self.byte_embedding.shape[1])
        target[: len(ids)] = self.byte_embedding[ids]
        target = target.flatten()
        if hasattr(self, "decoder"):
            generator = torch.Generator().manual_seed(self.seed + 1000 + int(self.write_count))
            code = functional.normalize(
                self.activate(torch.randn(self.decoder.shape[0], generator=generator)), dim=0
            )
            self.update(code, target, self.decoder, self.latent_covariance)
            target = code
            self.write_count.add_(1)
        surprises = [
            self.update(key, target, self.weight, self.inverse_covariance)
            for key in self.features(hidden)
        ]
        return {
            "surprise": sum(surprises) / len(surprises),
            "bytes": len(encoded),
            "keys": len(hidden),
        }

    @torch.no_grad()
    def recall(self, hidden: torch.Tensor) -> tuple[str, list[float]]:
        vectors = self(hidden)
        if vectors.norm() < 1e-8:
            return "", []
        scores = functional.normalize(vectors, dim=-1) @ self.byte_embedding.T
        ids = scores.argmax(-1).tolist()
        end = ids.index(256) if 256 in ids else len(ids)
        text = bytes(ids[:end]).decode("utf-8", errors="replace")
        return text, scores.max(-1).values[: end + 1].tolist()
