"""Run with: .venv/bin/python -m unittest discover -s tests -v."""

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from src.llm.modules.qwen import Qwen35Titans
from src.llm.modules.titans import TitansMemory
from src.llm.modules.training import chunked_lm_loss, token_batches, tokenize_windows


def reference_memory(module, inputs):
    """Independent autograd implementation of the paper's chunk-start gradient rule."""
    x = module.norm(inputs)
    q, k, v = module.to_q(x), module.to_k(x), module.to_v(x)
    if module.normalize_qk:
        q, k = functional.normalize(q, dim=-1), functional.normalize(k, dim=-1)
    alpha = module.to_alpha(x).sigmoid()
    eta = module.to_eta(x).sigmoid()
    theta = module.max_lr * module.to_theta(x).sigmoid()
    outputs = []
    for batch in range(inputs.shape[0]):
        params = {name: p.clone() for name, p in module.memory.named_parameters()}
        surprise = {name: torch.zeros_like(p) for name, p in params.items()}
        sequence = []
        for t in range(inputs.shape[1]):
            if t % module.chunk_size == 0:
                start_params = params
            prediction = torch.func.functional_call(module.memory, start_params, (k[batch, t],))
            grads = torch.autograd.grad(
                (prediction - v[batch, t]).square().sum(),
                tuple(start_params.values()),
                create_graph=True,
                retain_graph=True,
            )
            surprise = {
                name: eta[batch, t] * surprise[name] - theta[batch, t] * grad
                for name, grad in zip(start_params, grads, strict=True)
            }
            params = {
                name: (1 - alpha[batch, t]) * p + surprise[name] for name, p in params.items()
            }
            sequence.append(torch.func.functional_call(module.memory, params, (q[batch, t],)))
        outputs.append(torch.stack(sequence))
    return torch.stack(outputs)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[int(word) for word in text.split()] for text in texts]}


def tiny_wrapper(mixed=False, dtype=torch.float32):
    layer_types = (
        ["full_attention", "linear_attention", "full_attention"] if mixed else ["full_attention"]
    )
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=layer_types,
        max_position_embeddings=64,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [1, 0, 0],
        },
    )
    wrapper = Qwen35Titans.__new__(Qwen35Titans)
    wrapper.model = Qwen3_5ForCausalLM(config).to(dtype=dtype)
    wrapper.tokenizer = TinyTokenizer()
    wrapper._titans_attn = []
    wrapper._install_titans(memory_hidden_size=8, memory_chunk_size=2)
    wrapper.model.eval()
    return wrapper


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_values_and_meta_gradients_match_autograd_reference(self):
        for depth in (1, 2, 3):
            for chunk_size in (1, 3):
                optimized = TitansMemory(4, 7, memory_depth=depth, chunk_size=chunk_size).double()
                reference = copy.deepcopy(optimized)
                x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
                other_x = x.detach().clone().requires_grad_()
                actual = optimized(x)
                expected = reference_memory(reference, other_x)
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(x.grad, other_x.grad, atol=1e-10, rtol=1e-7)
                for p, other in zip(optimized.parameters(), reference.parameters(), strict=True):
                    assert p.grad is not None
                    assert other.grad is not None
                    torch.testing.assert_close(p.grad, other.grad, atol=1e-10, rtol=1e-7)

    def test_causality_batch_isolation_and_padding_state(self):
        memory = TitansMemory(4, 7, chunk_size=3).eval()
        x = torch.randn(2, 7, 4)
        with torch.inference_mode():
            actual = memory(x)
            changed = x.clone()
            changed[0, 4:] += 4
            changed[1] *= 3
            torch.testing.assert_close(actual[0, :4], memory(changed)[0, :4])
            mask = torch.arange(7)[None, :] < torch.tensor([[5], [7]])
            padded, states = memory(x, token_mask=mask, return_state=True)
            solo, solo_state = memory(x[:1, :5], return_state=True)
            torch.testing.assert_close(padded[:1, :5], solo)
            assert torch.count_nonzero(padded[0, 5:]) == 0
            for kind in ("params", "surprise"):
                for name in states[0][kind]:
                    torch.testing.assert_close(states[0][kind][name], solo_state[0][kind][name])

    def test_checkpoint_matches_eager(self):
        memory = TitansMemory(4, 7, chunk_size=3)
        eager = copy.deepcopy(memory)
        eager.checkpoint_chunks = False
        x = torch.randn(2, 7, 4)
        actual, expected = memory(x), eager(x)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected)
        for p, other in zip(memory.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(p.grad, other.grad)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_windows_keep_every_document_target_once(self):
        windows = tokenize_windows(TinyTokenizer(), ["2 3 4 5 6 7 8", "9 10 11"], 4)
        targets = [token for window in windows for token in window.tolist()[1:]]
        assert targets == [3, 4, 5, 6, 7, 8, 10, 11]

    def test_bfloat16_backbone_keeps_memory_float32(self):
        wrapper = tiny_wrapper(dtype=torch.bfloat16)
        before = wrapper.model.model.embed_tokens.weight.detach().clone()
        losses = wrapper.train(["2 3 4 5", "6 7 8"], batch_size=2)
        assert torch.isfinite(torch.tensor(losses)).all()
        assert torch.equal(before, wrapper.model.model.embed_tokens.weight)
        assert all(p.dtype == torch.float32 for p in wrapper._titans_attn[0].memory.parameters())

    def test_chunked_loss_and_gradient_match_full_logits(self):
        head = nn.Linear(8, 31).requires_grad_(False)
        x = torch.randn(2, 5, 8, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        labels = torch.randint(0, 31, (2, 5))
        labels[0, 2:] = -100
        actual = chunked_lm_loss(head, x, labels, 3)
        expected = functional.cross_entropy(
            head(other).reshape(-1, 31), labels.flatten(), reduction="sum"
        )
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, other.grad)

    def test_loss_alignment_matches_qwen_labels(self):
        wrapper = tiny_wrapper()
        windows = tokenize_windows(wrapper.tokenizer, ["2 3 4 5", "6 7 8"], 8)
        inputs, mask, targets, count = next(token_batches(windows, 2, 0, None))
        hidden = wrapper.model.model(
            input_ids=inputs, attention_mask=mask, memory_mask=mask, use_cache=False
        ).last_hidden_state
        actual = chunked_lm_loss(wrapper.model.lm_head, hidden, targets, 2) / count
        full_ids = nn.utils.rnn.pad_sequence(windows, batch_first=True)
        full_mask = full_ids != 0
        labels = full_ids.masked_fill(~full_mask, -100)
        expected = wrapper.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            memory_mask=full_mask,
            labels=labels,
            use_cache=False,
        ).loss
        torch.testing.assert_close(actual, expected)

    def test_accumulation_matches_batch_and_only_adapters_change(self):
        wrapper = tiny_wrapper(mixed=True)
        batched = copy.deepcopy(wrapper)
        before = {name: p.detach().clone() for name, p in wrapper.model.named_parameters()}
        texts = ["2 3 4", "5 6 7 8 9", "10 11 12 13"]
        losses = wrapper.train(
            texts,
            batch_size=1,
            gradient_accumulation_steps=3,
            shuffle=False,
            max_length=8,
            loss_chunk_size=2,
        )
        other_losses = batched.train(
            texts, batch_size=3, shuffle=False, max_length=8, loss_chunk_size=2
        )
        torch.testing.assert_close(torch.tensor(losses), torch.tensor(other_losses))
        changed = []
        for name, p in wrapper.model.named_parameters():
            other = dict(batched.model.named_parameters())[name]
            torch.testing.assert_close(p, other, atol=2e-6, rtol=2e-5)
            if not torch.equal(p, before[name]):
                changed.append(name)
                assert ".memory." in name or name.endswith("memory_gate")
        assert any("to_k.weight" in name for name in changed)
        assert any("to_v.weight" in name for name in changed)
        assert any(name.endswith("memory_gate") for name in changed)

    def test_decoder_checkpoint_and_partial_accumulation(self):
        wrapper = tiny_wrapper(mixed=True)
        eager = copy.deepcopy(wrapper)
        texts = ["2 3 4", "5 6 7 8 9", "10 11 12 13"]
        losses = wrapper.train(
            texts, gradient_accumulation_steps=2, checkpoint_decoder=True, shuffle=False
        )
        expected = eager.train(texts, batch_size=2, shuffle=False)
        assert len(losses) == 2
        torch.testing.assert_close(torch.tensor(losses), torch.tensor(expected))
        for p, other in zip(wrapper.model.parameters(), eager.model.parameters(), strict=True):
            torch.testing.assert_close(p, other, atol=2e-6, rtol=2e-5)
        assert not wrapper.model.training


if __name__ == "__main__":
    unittest.main()
