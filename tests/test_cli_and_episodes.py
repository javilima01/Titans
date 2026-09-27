"""CPU integration checks for state carry, episode loss, checkpoints and CLI."""

# The repository uses unittest rather than pytest for exception assertions.
# ruff: noqa: PT027

from contextlib import redirect_stderr, redirect_stdout
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers
import torch
from transformers import PreTrainedTokenizerFast

from main import build_parser, load_episodes, main
from src.llm.helpers.dataset_generation import MemoryEpisode, write_episodes
from src.llm.helpers.state_io import load_fast_memory_state, save_fast_memory_state
from src.llm.modules.evaluation import answer_metrics, evaluate_episodes
from src.llm.modules.qwen import Qwen35Titans, Qwen35Wrapper
from src.llm.modules.titans import TitansMemory
from src.llm.modules.training import (
    counterfactual_pair_loss,
    detach_memory_states,
    first_counterfactual_positions,
    tokenize_episodes,
)
from tests.test_titans_training import reference_memory, tiny_wrapper


def tokenizer():
    vocab = {
        "[PAD]": 0,
        "[EOS]": 1,
        "[UNK]": 2,
        **{str(i): i for i in range(3, 26)},
        "Question": 26,
        "Answer": 27,
        ":": 28,
        "where": 29,
        "station": 30,
        "code": 31,
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    result.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}Answer:"
    )
    return result


def episode(index=0, split="train"):
    return MemoryEpisode(
        str(index),
        "3 4 5 6 7 " * (index + 1),
        "where",
        ("8 9",),
        "fixture",
        split,
        {"task": "recall"},
    )


def wrapper():
    result = tiny_wrapper(mixed=True)
    result.tokenizer = tokenizer()
    result.model.generation_config.eos_token_id = 1
    result.model.generation_config.pad_token_id = 0
    return result


class EpisodeTrainingTests(unittest.TestCase):
    def test_fast_memory_state_roundtrip_and_adapter_binding(self):
        states = {
            11: {
                "params": {"net.0.weight": torch.arange(4).reshape(1, 2, 2).float()},
                "surprise": {"net.0.weight": torch.ones(1, 2, 2)},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user.safetensors"
            save_fast_memory_state(path, states, adapter_digest="first")
            loaded = load_fast_memory_state(path, adapter_digest="first", device="cpu")
            for kind in ("params", "surprise"):
                torch.testing.assert_close(
                    loaded[11][kind]["net.0.weight"], states[11][kind]["net.0.weight"]
                )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_fast_memory_state(path, adapter_digest="second", device="cpu")

    def test_memorized_text_can_initialize_a_later_generation(self):
        model = wrapper()
        state = model.memorize_text("3 4 5", window_size=4)
        assert state
        answer, later_state = model.generate_text(
            "6 7",
            window_size=4,
            max_new_tokens=2,
            memory_state=state,
            return_memory_state=True,
        )
        assert isinstance(answer, str)
        assert set(later_state) == set(state)
        with self.assertRaisesRegex(ValueError, "normal memory mode"):
            model.generate_text("6 7", memory_state=state, memory_mode="disabled", window_size=4)

    def test_counterfactual_loss_cancels_shared_answer_prior(self):
        hidden = torch.zeros(2, 1, 2, requires_grad=True)
        targets = torch.tensor([[0], [1]])
        loss = counterfactual_pair_loss(torch.nn.Identity(), hidden, targets)
        torch.testing.assert_close(loss, torch.tensor(2.0).log())
        loss.backward()
        torch.testing.assert_close(hidden.grad[:, 0], torch.tensor([[-0.5, 0.5], [0.5, -0.5]]))
        assert (
            counterfactual_pair_loss(torch.nn.Identity(), hidden.detach(), torch.tensor([[0], [0]]))
            == 0
        )
        two_digits = torch.zeros(2, 2, 2, requires_grad=True)
        two_digit_targets = torch.tensor([[0, 0], [1, 1]])
        two_digit_loss = counterfactual_pair_loss(
            torch.nn.Identity(), two_digits, two_digit_targets
        )
        torch.testing.assert_close(two_digit_loss, torch.tensor(2.0).log())
        two_digit_loss.backward()
        torch.testing.assert_close(two_digits.grad[:, 0], hidden.grad[:, 0])
        torch.testing.assert_close(two_digits.grad[:, 1], torch.zeros_like(hidden.grad[:, 0]))

    def test_counterfactual_loss_aligns_shifted_answer_spans(self):
        hidden = torch.zeros(2, 4, 2, requires_grad=True)
        targets = torch.tensor([[-100, -100, 0, 1], [-100, 1, 0, -100]])
        assert first_counterfactual_positions(targets) == (2, 1)
        loss = counterfactual_pair_loss(torch.nn.Identity(), hidden, targets)
        torch.testing.assert_close(loss, torch.tensor(2.0).log())
        loss.backward()
        torch.testing.assert_close(hidden.grad[0, 2], torch.tensor([-0.5, 0.5]))
        torch.testing.assert_close(hidden.grad[1, 1], torch.tensor([0.5, -0.5]))
        torch.testing.assert_close(hidden.grad[0, 3], torch.zeros(2))
        torch.testing.assert_close(hidden.grad[1, 2], torch.zeros(2))

    def test_pair_contrastive_training_requires_adjacent_variants(self):
        first = replace(
            episode(),
            metadata={"pair_id": "one", "pair_variant": 0},
        )
        second = replace(
            episode(),
            context="4 4 5 6 7 ",
            answers=("10 9",),
            metadata={"pair_id": "one", "pair_variant": 1},
        )
        model = wrapper()
        losses = model.train(
            episodes=[first, second],
            max_length=4,
            bptt_windows=0,
            batch_size=2,
            shuffle=False,
            pair_contrastive_weight=2.0,
        )
        assert len(losses) == 1
        assert torch.isfinite(torch.tensor(losses)).all()
        with self.assertRaisesRegex(ValueError, "shuffle=False"):
            model.train(
                episodes=[first, second],
                max_length=4,
                batch_size=2,
                pair_contrastive_weight=2.0,
            )

    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)

    def test_clipped_inner_updates_match_autograd_reference(self):
        memory = TitansMemory(4, 7, chunk_size=2, max_inner_grad_norm=0.05).double()
        reference = copy.deepcopy(memory)
        x = torch.randn(2, 7, 4, dtype=torch.float64)
        actual = memory(x)
        expected = reference_memory(reference, x)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for left, right in zip(memory.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-10, rtol=1e-7)

    def test_memory_continuation_matches_full_values_and_gradients(self):
        memory = TitansMemory(4, 7, chunk_size=2).double()
        other = copy.deepcopy(memory)
        x = torch.randn(2, 8, 4, dtype=torch.float64)
        full = memory(x)
        first, state = other(x[:, :4], return_state=True, batched_state=True)
        second = other(x[:, 4:], state=state)
        actual = torch.cat([first, second], dim=1)
        torch.testing.assert_close(full, actual)
        full[:, -1].square().sum().backward()
        actual[:, -1].square().sum().backward()
        for left, right in zip(memory.parameters(), other.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad)
        detached = detach_memory_states({0: state})
        assert all(
            not value.requires_grad
            for tensors in detached[0].values()
            for value in tensors.values()
        )

    def test_answer_labels_include_eos_and_exclude_context(self):
        tok = tokenizer()
        example = episode()
        ids, labels = tokenize_episodes(tok, [example])
        selected = labels[0][labels[0] != -100].tolist()
        assert selected == [8, 9, tok.eos_token_id]
        assert ids[0][-1] == tok.eos_token_id
        boundary = int((labels[0] != -100).nonzero()[0])
        assert (
            ids[0][:boundary].tolist() == tok(example.prompt, add_special_tokens=False)["input_ids"]
        )
        no_eos = replace(example, supervise_eos=False)
        other_ids, other_labels = tokenize_episodes(tok, [no_eos])
        assert other_ids[0][-1] == tok.eos_token_id
        assert other_labels[0][-1] == -100
        assert other_labels[0][other_labels[0] != -100].tolist() == [8, 9]

    def test_episode_checkpoint_and_batch_accumulation_equivalence(self):
        eager = wrapper()
        checkpointed = copy.deepcopy(eager)
        accumulated = copy.deepcopy(eager)
        before = {name: value.detach().clone() for name, value in eager.model.named_parameters()}
        episodes = [episode(), episode(1)]
        common = {
            "episodes": episodes,
            "max_length": 4,
            "bptt_windows": 0,
            "shuffle": False,
        }
        expected = eager.train(**common, batch_size=2)
        actual = checkpointed.train(**common, batch_size=2, checkpoint_decoder=True)
        separate = accumulated.train(**common, gradient_accumulation_steps=2)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
        torch.testing.assert_close(torch.tensor(separate), torch.tensor(expected))
        changed = []
        for name, parameter in eager.model.named_parameters():
            if not torch.equal(parameter, before[name]):
                assert ".memory." in name or name.endswith("memory_gate")
                changed.append(name)
            torch.testing.assert_close(
                parameter,
                dict(checkpointed.model.named_parameters())[name],
                atol=2e-6,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                parameter,
                dict(accumulated.model.named_parameters())[name],
                atol=2e-6,
                rtol=2e-5,
            )
        assert any("to_k.weight" in name for name in changed)
        assert any("memory.net.0.weight" in name for name in changed)

    def test_truncated_training_carries_detached_state_and_stops(self):
        model = wrapper()
        original = model._stream_forward
        calls = []

        def track(inputs, mask, states=None, **kwargs):
            calls.append(
                (
                    inputs.shape[1],
                    torch.is_grad_enabled(),
                    bool(states),
                    any(
                        value.requires_grad
                        for state in (states or {}).values()
                        for tensors in state.values()
                        for value in tensors.values()
                    ),
                )
            )
            return original(inputs, mask, states, **kwargs)

        events = []
        with patch.object(model, "_stream_forward", side_effect=track):
            losses = model.train(
                episodes=[episode(5), episode(6)],
                max_length=4,
                bptt_windows=2,
                max_steps=1,
                on_step=events.append,
                checkpoint_decoder=True,
                shuffle=False,
            )
        assert len(losses) == len(events) == 1
        assert torch.isfinite(torch.tensor(losses)).all()
        assert all(length <= 4 for length, *_ in calls)
        assert any(not grad and state for _, grad, state, _ in calls)
        assert any(grad and state and not history for _, grad, state, history in calls)

    def test_validation_loss_reported_after_each_epoch(self):
        model = wrapper()
        events = []
        validation = [episode(split="validation"), episode(1, split="validation")]
        losses = model.train(
            episodes=[episode(), episode(1)],
            max_length=4,
            bptt_windows=0,
            batch_size=2,
            epochs=2,
            shuffle=False,
            validation_episodes=validation,
            on_validation=events.append,
        )
        assert len(losses) == 2
        assert [event["epoch"] for event in events] == [1, 2]
        assert [event["step"] for event in events] == [1, 2]
        windows, labels = tokenize_episodes(model.tokenizer, validation)
        expected, tokens = model._episodes_loss(
            windows, labels, 2, 4, 128, model.tokenizer.pad_token_id
        )
        assert all(event["target_tokens"] == tokens for event in events)
        assert abs(events[-1]["val_loss"] - expected) < 1e-6

    def test_train_interrupt_restores_model_state(self):
        model = wrapper()

        def interrupt(event):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            model.train(
                episodes=[episode(), episode(1)],
                max_length=4,
                bptt_windows=0,
                batch_size=2,
                on_step=interrupt,
                checkpoint_decoder=True,
            )
        assert not model.model.training
        assert all(layer.memory.checkpoint_chunks for layer in model._titans_attn)
        assert len(model.train(episodes=[episode()], max_length=4, bptt_windows=0)) == 1

    def test_generation_recomputes_partial_windows_without_double_writes(self):
        model = wrapper()
        prompt = "3 4 5 6 7 8 9"
        actual = model.generate_text(prompt, max_new_tokens=6, window_size=4)
        # Independent reference: recompute the entire segmented prefix every step.
        ids = model.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        generated = []
        with torch.no_grad():
            for _ in range(6):
                states = {}
                for start in range(0, len(ids), 4):
                    inputs = torch.tensor([ids[start : start + 4]])
                    hidden, states = model._stream_forward(
                        inputs, torch.ones_like(inputs, dtype=torch.bool), states
                    )
                token = int(model.model.lm_head(hidden[:, -1]).argmax())
                if token == 1:
                    break
                generated.append(token)
                ids.append(token)
        expected = model.tokenizer.decode(generated, skip_special_tokens=True).strip()
        assert actual == expected
        assert actual == model.generate_text(prompt, max_new_tokens=6, window_size=4)


class CheckpointAndCLITests(unittest.TestCase):
    def test_explicit_mps_reports_sandbox_visibility(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "execution sandbox has GPU access"),
        ):
            Qwen35Wrapper(device="mps")

    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)
        base = tiny_wrapper(mixed=True).model
        for layer in base.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn = layer.self_attn.attention
        base.generation_config.eos_token_id = 1
        base.generation_config.pad_token_id = 0

        def fake_load(instance):
            instance.model = copy.deepcopy(base).to(device=instance.device, dtype=instance.dtype)
            instance.model.requires_grad_(False).eval()
            instance.tokenizer = tokenizer()

        self.loader = patch.object(Qwen35Wrapper, "_load", fake_load)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_checkpoint_restores_weights_config_and_predictions(self):
        model = Qwen35Titans(
            device="cpu",
            layer_indices=[0, 2],
            memory_hidden_size=8,
            memory_chunk_size=2,
        )
        model.train(episodes=[episode()], max_length=4, bptt_windows=0)
        expected = model.generate_text(episode().prompt.rstrip(), max_new_tokens=3)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint"
            model.save_pretrained(checkpoint_path)
            loaded = Qwen35Titans.from_pretrained(checkpoint_path, device="cpu")
            assert loaded.training_config == model.training_config
            for name, value in model._adapter_tensors().items():
                torch.testing.assert_close(value, loaded._adapter_tensors()[name], rtol=0, atol=0)
            assert loaded.generate_text(episode().prompt.rstrip(), max_new_tokens=3) == expected
            with self.assertRaises(FileExistsError):
                model.save_pretrained(checkpoint_path)
            config_path = checkpoint_path / "adapter_config.json"
            config = json.loads(config_path.read_text())
            for setting in (
                "memory_qk_scale",
                "aligned_qk_init",
                "memory_delta_read",
                "memory_gate_init",
            ):
                config["adapter"].pop(setting)
            config_path.write_text(json.dumps(config))
            legacy = Qwen35Titans.from_pretrained(checkpoint_path, device="cpu")
            assert legacy.generate_text(episode().prompt.rstrip(), max_new_tokens=3) == expected
            config["version"] = 999
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                Qwen35Titans.from_pretrained(checkpoint_path, device="cpu")

    def test_linear_attention_adapter_streams_trains_and_roundtrips(self):
        model = Qwen35Titans(
            device="cpu", layer_indices=[1], memory_hidden_size=8, memory_chunk_size=2
        )
        ids = torch.tensor([[3, 4, 5, 6]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        normal, state = model._stream_forward(ids, valid)
        disabled, disabled_state = model._stream_forward(ids, valid, memory_mode="disabled")
        assert set(state) == {1}
        assert disabled_state == {}
        assert (normal - disabled).abs().max() > 1e-7
        assert len(model.train(episodes=[episode()], max_length=4, bptt_windows=0)) == 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear-checkpoint"
            model.save_pretrained(path)
            loaded = Qwen35Titans.from_pretrained(path, device="cpu")
            assert set(loaded._stream_forward(ids, valid)[1]) == {1}
            assert loaded.generate_text("3 4 5", max_new_tokens=2, window_size=4) == (
                model.generate_text("3 4 5", max_new_tokens=2, window_size=4)
            )
            for name, value in model._adapter_tensors().items():
                torch.testing.assert_close(value, loaded._adapter_tensors()[name], rtol=0, atol=0)

        combined = Qwen35Titans(
            device="cpu", layer_indices=[1, 2], memory_hidden_size=8, memory_chunk_size=2
        )
        assert set(combined._stream_forward(ids, valid)[1]) == {1, 2}

    def test_shared_bank_reads_all_layers_causally_and_persists_one_state(self):
        model = Qwen35Titans(
            device="cpu",
            shared_across_layers=True,
            layer_indices=[0, 1, 2],
            memory_hidden_size=8,
            memory_chunk_size=2,
            aligned_qk_init=True,
            memory_qk_scale=2.0,
        )
        ids = torch.tensor([[3, 4, 5, 6]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        _, state = model._stream_forward(ids, valid)
        assert set(state) == {-1}
        assert len(model._shared_bank.layer_indices) == 3
        assert len(model._titans_attn) == 1
        first, _ = model._stream_forward(ids, valid, state)
        changed = ids.clone()
        changed[0, -1] = 7
        second, _ = model._stream_forward(changed, valid, state)
        torch.testing.assert_close(first[:, :-1], second[:, :-1], atol=1e-5, rtol=1e-5)
        disabled, disabled_state = model._stream_forward(ids, valid, memory_mode="disabled")
        assert disabled_state == {}
        assert (first - disabled).abs().max() > 1e-7
        assert len(model.train(episodes=[episode()], max_length=4, bptt_windows=0)) == 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared-checkpoint"
            model.save_pretrained(path)
            loaded = Qwen35Titans.from_pretrained(path, device="cpu")
            assert loaded._shared_bank is not None
            assert set(loaded._stream_forward(ids, valid)[1]) == {-1}
            for name, value in model._adapter_tensors().items():
                torch.testing.assert_close(value, loaded._adapter_tensors()[name], rtol=0, atol=0)
            _, saved_state = loaded._stream_forward(ids, valid)
            state_path = Path(directory) / "one-state.safetensors"
            loaded.save_memory_state(state_path, saved_state)
            restored = loaded.load_memory_state(state_path)
            assert set(restored) == {-1}
            assert loaded.generate_text("3 4 5", max_new_tokens=2, window_size=4) == (
                model.generate_text("3 4 5", max_new_tokens=2, window_size=4)
            )

    def test_scaled_query_key_configuration_roundtrips(self):
        model = Qwen35Titans(
            device="cpu",
            layer_indices=[0],
            memory_hidden_size=8,
            memory_chunk_size=2,
            memory_qk_scale=2.0,
            aligned_qk_init=True,
            memory_delta_read=True,
            memory_gate_init=-4.0,
        )
        memory = model._titans_attn[0].memory
        assert memory.qk_scale == 2.0
        assert memory.aligned_qk_init
        assert memory.delta_read
        assert model._titans_attn[0].memory_gate.item() == -4.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            model.save_pretrained(path)
            loaded = Qwen35Titans.from_pretrained(path, device="cpu")
            assert loaded._titans_attn[0].memory.qk_scale == 2.0
            assert loaded._titans_attn[0].memory.aligned_qk_init
            assert loaded._titans_attn[0].memory.delta_read
            assert loaded._titans_attn[0].memory_gate.item() == -4.0
            for name, value in model._adapter_tensors().items():
                torch.testing.assert_close(value, loaded._adapter_tensors()[name], rtol=0, atol=0)

    def test_cli_train_validate_test_and_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for split in ("train", "validation", "test"):
                write_episodes(data / f"{split}.jsonl", [episode(split=split)])
            checkpoint_path = root / "checkpoint"
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                main(
                    [
                        "train",
                        "--data",
                        str(data),
                        "--output",
                        str(checkpoint_path),
                        "--layers",
                        "0",
                        "--memory-hidden-size",
                        "8",
                        "--memory-chunk-size",
                        "2",
                        "--window-size",
                        "4",
                        "--max-steps",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
                for command in ("validate", "test"):
                    main(
                        [
                            command,
                            "--data",
                            str(data),
                            "--checkpoint",
                            str(checkpoint_path),
                            "--device",
                            "cpu",
                            "--max-new-tokens",
                            "2",
                            "--ablations",
                            "--report",
                            str(root / f"{command}.json"),
                        ]
                    )
                main(
                    [
                        "chat",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--device",
                        "cpu",
                        "--prompt",
                        "3 4 5",
                        "--max-new-tokens",
                        "2",
                    ]
                )
                main(
                    [
                        "chat",
                        "--device",
                        "cpu",
                        "--prompt",
                        "3 4 5",
                        "--max-new-tokens",
                        "2",
                    ]
                )
                session_file = root / "chat-session.json"
                seen = []

                def reply(_model, messages, **_kwargs):
                    seen.append([message.content for message in messages])
                    return "Ari"

                with patch.object(Qwen35Titans, "generate", autospec=True, side_effect=reply):
                    for prompt in ("My name is Ari", "What is my name?"):
                        main(
                            [
                                "chat",
                                "--checkpoint",
                                str(checkpoint_path),
                                "--device",
                                "cpu",
                                "--session-file",
                                str(session_file),
                                "--prompt",
                                prompt,
                            ]
                        )
                assert seen[1][1:4] == ["My name is Ari", "Ari", "What is my name?"]
                assert len(json.loads(session_file.read_text())) == 5
                memory_file = root / "fast-memory.safetensors"
                for prompt in ("Remember 3 4 5", "What was 3 4 5?"):
                    main(
                        [
                            "chat",
                            "--checkpoint",
                            str(checkpoint_path),
                            "--device",
                            "cpu",
                            "--memory-state-file",
                            str(memory_file),
                            "--prompt",
                            prompt,
                            "--max-new-tokens",
                            "2",
                        ]
                    )
                    assert memory_file.exists()
                main(
                    [
                        "chat",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--device",
                        "cpu",
                        "--memory-state-file",
                        str(memory_file),
                        "--prompt",
                        "/reset",
                    ]
                )
                assert not memory_file.exists()
                main(
                    [
                        "chat",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--device",
                        "cpu",
                        "--session-file",
                        str(session_file),
                        "--prompt",
                        "/reset",
                    ]
                )
                assert len(json.loads(session_file.read_text())) == 1
                evidence_file = root / "episodic-memory.json"
                evidence_seen = []

                def evidence_reply(_model, messages, **_kwargs):
                    evidence_seen.append([message.content for message in messages])
                    return "ok"

                with patch.object(
                    Qwen35Titans, "generate", autospec=True, side_effect=evidence_reply
                ):
                    for prompt in (
                        "The build tool for repository alpha is Bazel.",
                        "Correction: the build tool for repository alpha is Ninja.",
                        "Which build tool does repository alpha use?",
                    ):
                        main(
                            [
                                "chat",
                                "--checkpoint",
                                str(checkpoint_path),
                                "--device",
                                "cpu",
                                "--episodic-memory-file",
                                str(evidence_file),
                                "--prompt",
                                prompt,
                            ]
                        )
                assert "Ninja" in evidence_seen[-1][1]
                assert "Bazel" not in evidence_seen[-1][1]
                assert evidence_seen[-1][-1] == "Which build tool does repository alpha use?"
                assert len(json.loads(evidence_file.read_text())["records"]) == 2
                main(
                    [
                        "chat",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--device",
                        "cpu",
                        "--episodic-memory-file",
                        str(evidence_file),
                        "--prompt",
                        "/remember The build tool for repository beta is Meson.",
                    ]
                )
                assert len(json.loads(evidence_file.read_text())["records"]) == 3
                main(
                    [
                        "chat",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--device",
                        "cpu",
                        "--episodic-memory-file",
                        str(evidence_file),
                        "--prompt",
                        "/reset",
                    ]
                )
                assert not evidence_file.exists()
            for command, split in (("validate", "validation"), ("test", "test")):
                report = json.loads((root / f"{command}.json").read_text())
                assert report["split"] == split
                assert report["data_source"] == str(data)
                assert set(report["evaluations"]) == {"normal", "disabled", "reset"}
                assert report["evaluations"]["normal"]["metrics"]["count"] == 1
            experiment = json.loads((root / "experiments/checkpoint/record.json").read_text())
            assert experiment["training"]["options"]["device"] == "cpu"
            assert set(experiment["reports"]) == {"validate.json", "test.json"}
            assert (root / "experiments/checkpoint/README.md").exists()

    def test_cli_train_writes_monitor_and_validation_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for split in ("train", "validation", "test"):
                write_episodes(data / f"{split}.jsonl", [episode(split=split)])
            checkpoint_path = root / "checkpoint"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(
                    [
                        "train",
                        "--data",
                        str(data),
                        "--output",
                        str(checkpoint_path),
                        "--layers",
                        "0",
                        "--memory-hidden-size",
                        "8",
                        "--memory-chunk-size",
                        "2",
                        "--window-size",
                        "4",
                        "--device",
                        "cpu",
                    ]
                )
            assert (checkpoint_path / "training.png").read_bytes()[:4] == b"\x89PNG"
            assert not (root / "checkpoint.training.png").exists()
            records = [
                json.loads(line)
                for line in (checkpoint_path / "training_log.jsonl").read_text().splitlines()
            ]
            assert [record["type"] for record in records] == [
                "started",
                "step",
                "validation",
                "finished",
            ]
            config = json.loads((checkpoint_path / "adapter_config.json").read_text())
            assert config["metadata"]["interrupted"] is False
            assert len(config["metadata"]["validation"]) == 1

    def test_cli_train_saves_partial_checkpoint_on_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            write_episodes(data / "train.jsonl", [episode()])
            checkpoint_path = root / "checkpoint"
            output = io.StringIO()

            def two_steps_then_interrupt(*args, **kwargs):
                for step in (1, 2):
                    kwargs["on_step"](
                        {
                            "step": step,
                            "epoch": 1,
                            "loss": 4.0 - step,
                            "target_tokens": 3,
                        }
                    )
                raise KeyboardInterrupt

            with (
                patch.object(Qwen35Titans, "train", side_effect=two_steps_then_interrupt),
                redirect_stdout(output),
                redirect_stderr(io.StringIO()),
            ):
                main(
                    [
                        "train",
                        "--data",
                        str(data),
                        "--output",
                        str(checkpoint_path),
                        "--layers",
                        "0",
                        "--memory-hidden-size",
                        "8",
                        "--memory-chunk-size",
                        "2",
                        "--window-size",
                        "4",
                        "--device",
                        "cpu",
                    ]
                )
            config = json.loads((checkpoint_path / "adapter_config.json").read_text())
            assert config["metadata"]["interrupted"] is True
            assert config["metadata"]["losses"] == [3.0, 2.0]
            assert config["metadata"]["optimizer_steps"] == 2
            assert (checkpoint_path / "training.png").exists()
            summary = json.loads(output.getvalue().strip().splitlines()[-1])
            assert summary["interrupted"] is True
            assert summary["steps"] == 2

    def test_cli_train_interrupt_before_first_step_saves_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            write_episodes(data / "train.jsonl", [episode()])
            checkpoint_path = root / "checkpoint"
            errors = io.StringIO()
            with (
                patch.object(Qwen35Titans, "train", side_effect=KeyboardInterrupt),
                redirect_stdout(io.StringIO()),
                redirect_stderr(errors),
            ):
                main(
                    [
                        "train",
                        "--data",
                        str(data),
                        "--output",
                        str(checkpoint_path),
                        "--layers",
                        "0",
                        "--memory-hidden-size",
                        "8",
                        "--memory-chunk-size",
                        "2",
                        "--window-size",
                        "4",
                        "--device",
                        "cpu",
                    ]
                )
            assert not checkpoint_path.exists()
            assert (root / "checkpoint.training.png").exists()
            assert "checkpoint not saved" in errors.getvalue()

    def test_cli_refuses_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.jsonl"
            write_episodes(path, [episode(split="test")])
            args = build_parser().parse_args(["train", "--data", str(path), "--output", "unused"])
            with self.assertRaisesRegex(ValueError, "Expected only train"):
                load_episodes(args, "train")


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_reference_aliases(self):
        assert answer_metrics("The RED box.", ("blue box", "the red box"))["exact_match"] == 1.0
        score = answer_metrics("red red", ("red blue",))
        assert score == {"exact_match": 0.0, "precision": 0.5, "recall": 0.5, "f1": 0.5}
        assert answer_metrics("", ("red",))["recall"] == 0.0

    def test_evaluation_uses_prompt_only_and_groups_tasks(self):
        class Predictor:
            def __init__(self):
                self.training_config = {"window_size": 4}

            def generate_text(self, prompt, **kwargs):
                assert prompt.endswith("Answer:\n")
                assert "8 9" not in prompt
                return "8 9"

        result = evaluate_episodes(Predictor(), [episode(), episode(1)])
        assert result["metrics"]["exact_match"] == 1.0
        assert result["by_task"]["recall"]["count"] == 2

    def test_paired_metrics_require_both_counterfactual_answers(self):
        episodes = [
            MemoryEpisode(
                id=f"pair-{variant}",
                context=f"The name is {name}.",
                question="What is the name?",
                answers=(name,),
                source="pair-test",
                split="validation",
                metadata={"pair_id": "one", "pair_variant": variant},
            )
            for variant, name in enumerate(("Alice", "David"))
        ]

        class Predictor:
            def __init__(self, use_fact):
                self.use_fact = use_fact
                self.training_config = {"window_size": 256}

            def generate_text(self, prompt, **kwargs):
                return "David" if self.use_fact and "David" in prompt else "Alice"

        solved = evaluate_episodes(Predictor(True), episodes)["paired_metrics"]
        ignored = evaluate_episodes(Predictor(False), episodes)["paired_metrics"]
        assert solved["pairs"] == 1
        assert solved["both_correct"] == 1
        assert solved["changed_prediction_rate"] == 1.0
        assert ignored["both_correct"] == 0
        assert ignored["changed_prediction_rate"] == 0.0


if __name__ == "__main__":
    unittest.main()
