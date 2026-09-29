from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional

from src.llm.helpers.activation_memory_probe import token_context
from src.llm.modules.activation_memory import ActivationFastMemory

torch.set_num_threads(1)


class ActivationMemoryTests(unittest.TestCase):
    def test_no_grad_write_matches_preconditioned_autograd_gradient(self):
        torch.manual_seed(47)
        network = ActivationFastMemory(12, 8, 16, ridge=0.02)
        network.write(torch.randn(4, 12), torch.randn(4, 8))
        before = {name: value.clone() for name, value in network.state_dict().items()}
        hidden = torch.randn(1, 12)
        value = functional.normalize(torch.randn(1, 8, dtype=torch.float64), dim=-1)
        key = network.keys(hidden).detach().double()
        reference_weight = before["fast_weight"].double().requires_grad_()
        loss = 0.5 * (key @ reference_weight - value).square().sum()
        (gradient,) = torch.autograd.grad(loss, reference_weight)
        covariance = before["inverse_covariance"]
        expected = reference_weight.detach() - (covariance @ gradient) / (
            1 + key @ covariance @ key.T
        )

        with torch.no_grad():
            network.write(hidden, value)
            updated_loss = 0.5 * (key @ network.fast_weight.double() - value).square().sum()

        torch.testing.assert_close(network.fast_weight, expected.float(), atol=2e-7, rtol=2e-6)
        assert updated_loss < loss.detach()
        assert not torch.equal(network.fast_weight, before["fast_weight"])
        assert not torch.equal(network.inverse_covariance, covariance)
        for name, parameter in network.named_parameters():
            assert parameter.grad is None
            if name != "fast_weight":
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)

    def test_context_pool_cannot_see_future_tokens(self):
        torch.manual_seed(43)
        embedding = torch.nn.Embedding(10, 8)
        model = SimpleNamespace(
            device="cpu", model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding))
        )
        first = token_context(model, [1, 2, 3, 4])
        second = token_context(model, [1, 2, 8, 9])
        torch.testing.assert_close(first[:2], second[:2], rtol=0, atol=0)
        assert not torch.equal(first[2:], second[2:])

    def test_online_writes_equal_joint_ridge_solution_after_reload(self):
        torch.manual_seed(31)
        network = ActivationFastMemory(12, 8, 16, ridge=0.02)
        hidden = torch.randn(9, 12)
        values = functional.normalize(torch.randn(9, 8), dim=-1)
        query = torch.randn(4, 12)
        mask = torch.ones(1, 9, dtype=torch.bool)
        predicted, _ = network.episode_read(hidden[None], query[None], values[None], mask)
        network.write(hidden[:4], values[:4])
        network.write(hidden[4:], values[4:])
        torch.testing.assert_close(network(query), predicted[0], atol=2e-5, rtol=2e-5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.safetensors"
            save_file(network.state_dict(), path)
            restored = ActivationFastMemory(12, 8, 16, ridge=0.02)
            restored.load_state_dict(load_file(path))
        torch.testing.assert_close(restored(query), network(query))
        restored.reset()
        assert not restored(query).any()

    def test_outer_learning_reaches_both_projections_without_mutating_fast_state(self):
        torch.manual_seed(41)
        network = ActivationFastMemory(12, 8, 16)
        initial = network.fast_weight.clone()
        predicted, _ = network.episode_read(
            torch.randn(2, 6, 12),
            torch.randn(2, 3, 12),
            torch.randn(2, 6, 8),
            torch.ones(2, 6, dtype=torch.bool),
        )
        predicted.square().mean().backward()
        for projection in (network.to_key, network.to_query):
            for parameter in projection.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
            assert (
                sum(float(parameter.grad.abs().sum()) for parameter in projection.parameters()) > 0
            )
        torch.testing.assert_close(network.fast_weight, initial, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
