"""CPU integration checks for state carry, episode loss, checkpoints and CLI."""
# The repository uses unittest rather than pytest for exception assertions.
# ruff: noqa: PT027

from contextlib import redirect_stderr, redirect_stdout
import copy
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
from src.llm.modules.evaluation import answer_metrics, evaluate_episodes
from src.llm.modules.qwen import Qwen35Titans, Qwen35Wrapper
from src.llm.modules.titans import TitansMemory
from src.llm.modules.training import detach_memory_states, tokenize_episodes
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
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
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

    def test_episode_checkpoint_and_batch_accumulation_equivalence(self):
        eager = wrapper()
        checkpointed = copy.deepcopy(eager)
        accumulated = copy.deepcopy(eager)
        before = {name: value.detach().clone() for name, value in eager.model.named_parameters()}
        episodes = [episode(), episode(1)]
        common = {"episodes": episodes, "max_length": 4, "bptt_windows": 0, "shuffle": False}
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
                parameter, dict(checkpointed.model.named_parameters())[name], atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                parameter, dict(accumulated.model.named_parameters())[name], atol=2e-6, rtol=2e-5
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
            device="cpu", layer_indices=[0, 2], memory_hidden_size=8, memory_chunk_size=2
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
            config["version"] = 999
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                Qwen35Titans.from_pretrained(checkpoint_path, device="cpu")

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
                main(["chat", "--device", "cpu", "--prompt", "3 4 5", "--max-new-tokens", "2"])
            for command, split in (("validate", "validation"), ("test", "test")):
                report = json.loads((root / f"{command}.json").read_text())
                assert report["split"] == split
                assert set(report["evaluations"]) == {"normal", "disabled", "reset"}
                assert report["evaluations"]["normal"]["metrics"]["count"] == 1

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


if __name__ == "__main__":
    unittest.main()
