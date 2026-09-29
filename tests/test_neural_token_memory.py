import unittest

import torch
from torch.nn import functional

from src.llm.modules.neural_token_memory import NeuralTokenMemory

torch.set_num_threads(1)


class NeuralTokenMemoryTests(unittest.TestCase):
    def test_first_write_equals_negative_associative_loss_gradient(self):
        torch.manual_seed(81)
        key = functional.normalize(torch.randn(8), dim=-1)
        value = functional.normalize(torch.randn(8), dim=-1)
        weights = torch.zeros(8, 8, requires_grad=True)
        (gradient,) = torch.autograd.grad(0.5 * ((key @ weights - value) ** 2).sum(), weights)
        memory = NeuralTokenMemory(8)
        memory.write(key[None], value[None])
        torch.testing.assert_close(memory.weight, -gradient)

    def test_new_values_and_correction_live_in_fixed_network_state(self):
        torch.manual_seed(84)
        keys = torch.randn(4, 16)
        targets = functional.normalize(torch.randn(4, 16), dim=-1)
        memory = NeuralTokenMemory(16)
        shapes = {name: tensor.shape for name, tensor in memory.state_dict().items()}
        memory.write(keys, targets)
        torch.testing.assert_close(memory(keys), targets, atol=0.002, rtol=0.002)
        replacement = functional.normalize(torch.randn(1, 16), dim=-1)
        memory.write(keys[:1], replacement)
        targets[0] = replacement[0]
        torch.testing.assert_close(memory(keys), targets, atol=0.004, rtol=0.004)
        restored = NeuralTokenMemory(16)
        restored.load_state_dict(memory.state_dict())
        torch.testing.assert_close(restored(keys), memory(keys))
        assert {name: tensor.shape for name, tensor in memory.state_dict().items()} == shapes
        assert set(shapes) == {"weight", "inverse_covariance"}
        assert not memory.weight.requires_grad


if __name__ == "__main__":
    unittest.main()
