from pathlib import Path
import tempfile
import unittest

from safetensors.torch import load_file, save_file
import torch

from src.llm.modules.neural_statement_memory import NeuralStatementMemory

torch.set_num_threads(1)


class NeuralStatementMemoryTests(unittest.TestCase):
    def network(self):
        return NeuralStatementMemory(
            input_width=24,
            feature_width=256,
            max_bytes=80,
            code_width=48,
            latent_width=256,
            latent_sparsity=16,
            bandwidth=2,
        )

    def test_novel_utf8_sentences_survive_noise_and_tensor_only_reload(self):
        torch.manual_seed(82)
        keys = torch.randn(8, 24)
        statements = [f"Repository v{i}: run café-{i} --flag=λ." for i in range(8)]
        network = self.network()
        shapes = {key: value.shape for key, value in network.state_dict().items()}
        assert network.recall(keys[0])[0] == ""
        for key, statement in zip(keys, statements, strict=True):
            network.write(key, statement)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.safetensors"
            save_file(network.state_dict(), path)
            restored = self.network()
            restored.load_state_dict(load_file(path))
        del network
        for key, statement in zip(keys, statements, strict=True):
            assert restored.recall(key + 0.08 * torch.randn(24))[0] == statement
        assert {key: value.shape for key, value in restored.state_dict().items()} == shapes
        assert set(shapes) == {
            "projection",
            "phase",
            "byte_embedding",
            "weight",
            "inverse_covariance",
            "decoder",
            "latent_covariance",
            "write_count",
        }

    def test_aliases_share_a_write_and_correction_preserves_other_facts(self):
        torch.manual_seed(91)
        keys = torch.randn(3, 24)
        aliases = torch.stack([keys[0], keys[0] + 0.05 * torch.randn(24)])
        network = self.network()
        network.write(aliases, "The build tool is Ninja.")
        network.write(keys[1], "The maintainer is Orva.")
        network.write(keys[2], "The configuration is config/étoile.toml.")
        network.write(aliases, "Correction: the build tool is Meson.")
        assert int(network.write_count) == 4
        assert network.recall(aliases[1])[0] == "Correction: the build tool is Meson."
        assert network.recall(keys[1])[0] == "The maintainer is Orva."
        assert network.recall(keys[2])[0] == "The configuration is config/étoile.toml."

    def test_invalid_or_oversized_write_cannot_partially_change_network(self):
        network = self.network()
        before = {key: value.clone() for key, value in network.state_dict().items()}
        for hidden, statement in (
            (torch.ones(24), "é" * 40),
            (torch.zeros(24), "fact"),
            (torch.full((24,), float("nan")), "fact"),
            (torch.ones(3), "fact"),
        ):
            try:
                network.write(hidden, statement)
            except ValueError:
                pass
            else:
                msg = "Invalid writes must be rejected"
                raise AssertionError(msg)
        for key, value in network.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
