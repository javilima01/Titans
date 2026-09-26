import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

from safetensors.torch import load_file, save_file
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5Tokenizer
from transformers.utils import logging

from src.llm.helpers.utils import ensure_model
from src.llm.helpers.visualize import print_module_tree
from src.llm.modules.titans import AttentionWithTitans, TitansMemory
from src.llm.modules.training import (
    chunked_lm_loss,
    detach_memory_states,
    token_batches,
    tokenize_episodes,
    tokenize_windows,
)
from src.llm.schemas.messages import Message

logging.set_verbosity_error()

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from src.llm.helpers.dataset_generation import MemoryEpisode


class Qwen35Wrapper:
    _model_name: str = "Qwen/Qwen3.5-0.8B"

    def __init__(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.dtype = dtype or (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self._load()

    def _load(self):
        model_path = ensure_model(self._model_name)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_path,
            dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)
        self.tokenizer: Qwen3_5Tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model.requires_grad_(False)
        self.model.eval()

    @staticmethod
    def _msg_to_dict(msgs: list[Message] | Message) -> list[dict]:
        if isinstance(msgs, Message):
            msgs = [msgs]

        return [msg.model_dump(mode="python") for msg in msgs]

    def generate(self, msgs: list[Message] | Message, max_new_tokens: int = 200, **kwargs):
        msgs = self._msg_to_dict(msgs=msgs)

        text = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

    def inspect(self, max_depth: int = 3):
        print_module_tree(self.model, max_depth=max_depth)


class Qwen35Titans(Qwen35Wrapper):
    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        layer_indices: list[int] | None = None,
        memory_hidden_size: int | None = None,
        memory_chunk_size: int = 16,
        checkpoint_memory: bool = True,
        max_inner_grad_norm: float | None = 1.0,
    ):
        super().__init__(device=device, dtype=dtype)
        self.model.requires_grad_(False)
        self._titans_attn: list[AttentionWithTitans] = []
        self._install_titans(
            layer_indices,
            memory_hidden_size,
            memory_chunk_size,
            checkpoint_memory,
            max_inner_grad_norm,
        )
        self._training_config = {}
        # Fast memory is not part of HF's KV cache yet. Recompute the prefix so
        # generation uses the same chunk boundaries and history as training.
        self.model.generation_config.use_cache = False
        self.model.eval()

    def _full_attention_layers(self) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer in enumerate(self.model.model.layers)
            if hasattr(layer, "self_attn")
        ]

    def _install_titans(
        self,
        layer_indices=None,
        memory_hidden_size=None,
        memory_chunk_size=16,
        checkpoint_memory=True,
        max_inner_grad_norm=1.0,
    ):
        ids = self._full_attention_layers() if layer_indices is None else layer_indices
        full_attention = set(self._full_attention_layers())
        if not ids or len(set(ids)) != len(ids) or any(i not in full_attention for i in ids):
            msg = "Select distinct full-attention layer indices"
            raise ValueError(msg)
        for layer_idx in ids:
            self._add_to_existing(
                layer_idx,
                memory_hidden_size,
                memory_chunk_size,
                checkpoint_memory,
                max_inner_grad_norm,
            )

    def _add_to_existing(
        self,
        layer_idx: int = 11,
        memory_hidden_size=None,
        memory_chunk_size=16,
        checkpoint_memory=True,
        max_inner_grad_norm=1.0,
    ):
        layer = self.model.model.layers[layer_idx]

        if not hasattr(layer, "self_attn"):
            error_msg = f"Layer {layer_idx} is not a full-attention layer"
            raise ValueError(error_msg)

        attention = layer.self_attn

        hidden_size = self.model.config.hidden_size
        memory = TitansMemory(
            hidden_size=(memory_hidden_size if memory_hidden_size is not None else hidden_size),
            dim=hidden_size,
            chunk_size=memory_chunk_size,
            checkpoint_chunks=checkpoint_memory,
            max_inner_grad_norm=max_inner_grad_norm,
        )

        param = next(attention.parameters())
        # Fast weights, inner updates and AdamW state stay in FP32.
        memory = memory.to(device=param.device, dtype=torch.float32)
        wrapped_attention = AttentionWithTitans(
            attention=attention,
            memory=memory,
            memory_id=layer_idx,
        )
        wrapped_attention.to(device=param.device)

        layer.self_attn = wrapped_attention

        self._titans_attn.append(wrapped_attention)

    def _adapter_tensors(self):
        tensors = {}
        for layer in self._titans_attn:
            prefix = f"layers.{layer.memory_id}"
            tensors.update(
                {
                    f"{prefix}.memory.{name}": tensor
                    for name, tensor in layer.memory.state_dict().items()
                }
            )
            tensors[f"{prefix}.memory_gate"] = layer.memory_gate
        return tensors

    @property
    def training_config(self):
        return dict(getattr(self, "_training_config", {}))

    def save_pretrained(self, path: str | Path, *, metadata: dict | None = None):
        """Save adapters + tokenizer atomically; the frozen base is reused on load.

        A new directory is required. Fast episode states and optimizer state are
        not saved: loading starts fresh episodes and training uses a new AdamW.
        """
        path = Path(path)
        if path.exists():
            msg = f"Checkpoint already exists: {path}"
            raise FileExistsError(msg)
        memories = [layer.memory for layer in self._titans_attn]
        settings = [
            {
                "memory_hidden_size": memory.memory.net[0].out_features,
                "memory_chunk_size": memory.chunk_size,
                "checkpoint_memory": memory.checkpoint_chunks,
                "max_inner_grad_norm": memory.max_inner_grad_norm,
            }
            for memory in memories
        ]
        if any(setting != settings[0] for setting in settings) or any(
            len(memory.memory.linear_indices) != 2
            or memory.max_lr != 0.1
            or not memory.normalize_qk
            for memory in memories
        ):
            msg = (
                "Checkpoint saving requires the standard uniform Qwen35Titans adapter configuration"
            )
            raise ValueError(msg)
        config = {
            "format": "qwen-titans-adapter",
            "version": 1,
            "base_model": self._model_name,
            "base_dimensions": {
                name: getattr(self.model.config, name)
                for name in ("hidden_size", "vocab_size", "num_hidden_layers")
            },
            "adapter": {
                **settings[0],
                "layer_indices": [layer.memory_id for layer in self._titans_attn],
            },
            "training": getattr(self, "_training_config", {}),
            "metadata": metadata or {},
        }
        serialized = json.dumps(config, indent=2) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
            staged = Path(temporary) / "checkpoint"
            staged.mkdir()
            save_file(
                {
                    name: tensor.detach().cpu().contiguous().clone()
                    for name, tensor in self._adapter_tensors().items()
                },
                staged / "adapter_model.safetensors",
            )
            self.tokenizer.save_pretrained(staged)
            (staged / "adapter_config.json").write_text(serialized, encoding="utf-8")
            if path.exists():
                msg = f"Checkpoint already exists: {path}"
                raise FileExistsError(msg)
            staged.rename(path)

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device=None, dtype=None):
        """Reconstruct the adapter architecture and strictly load its trained weights."""
        path = Path(path)
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        if (config.get("format"), config.get("version"), config.get("base_model")) != (
            "qwen-titans-adapter",
            1,
            cls._model_name,
        ):
            msg = "Unsupported checkpoint format/version or base model"
            raise ValueError(msg)
        model = cls(device=device, dtype=dtype, **config["adapter"])
        if any(
            getattr(model.model.config, name) != value
            for name, value in config["base_dimensions"].items()
        ):
            msg = "Checkpoint dimensions do not match the frozen base model"
            raise ValueError(msg)
        tensors = load_file(path / "adapter_model.safetensors", device="cpu")
        expected = model._adapter_tensors()
        if tensors.keys() != expected.keys() or any(
            tensors[name].shape != tensor.shape for name, tensor in expected.items()
        ):
            msg = "Checkpoint adapter tensors do not match the saved architecture"
            raise ValueError(msg)
        with torch.no_grad():
            for name, tensor in expected.items():
                tensor.copy_(tensors[name])
        model.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if len(model.tokenizer) > model.model.config.vocab_size:
            msg = "Checkpoint tokenizer vocabulary exceeds the base model vocabulary"
            raise ValueError(msg)
        model._training_config = config.get("training", {})
        model.model.eval()
        return model

    def _check_window_size(self, window_size):
        if window_size < 1 or any(
            window_size % layer.memory.chunk_size for layer in self._titans_attn
        ):
            msg = "window_size must be positive and a multiple of every memory_chunk_size"
            raise ValueError(msg)
        if window_size > self.model.config.max_position_embeddings:
            msg = "window_size exceeds the base model's configured position limit"
            raise ValueError(msg)

    def _stream_forward(
        self, inputs, mask, states=None, *, memory_mode="normal", checkpoint_window=False
    ):
        if memory_mode not in ("normal", "disabled", "reset"):
            msg = "memory_mode must be normal, disabled, or reset"
            raise ValueError(msg)

        def run(ids, valid, initial):
            # Local output capture is recreated during checkpoint recomputation.
            # Nothing mutates the input state or a persistent module cache.
            context = {"initial": initial, "final": {}, "mode": memory_mode}
            hidden = self.model.model(
                input_ids=ids,
                attention_mask=valid,
                memory_mask=valid,
                memory_context=context,
                use_cache=False,
            ).last_hidden_state
            return hidden, context["final"]

        if checkpoint_window and torch.is_grad_enabled():
            return checkpoint(
                run, inputs, mask, states or {}, use_reentrant=False, preserve_rng_state=False
            )
        return run(inputs, mask, states or {})

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        window_size: int | None = None,
        memory_mode: str = "normal",
        temperature: float = 0.0,
        top_p: float = 0.9,
        stop_at_newline: bool = False,
    ) -> str:
        """Generate with bounded Qwen windows and per-request Titans state.

        Completed windows commit state once. The current partial window is
        recomputed from its starting state, preserving chunk-start gradients.
        Qwen KV/linear-attention caches and positions restart at each window.
        """
        window_size = window_size or getattr(self, "_training_config", {}).get("window_size", 512)
        self._check_window_size(window_size)
        if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
            msg = "Require max_new_tokens >= 1, temperature >= 0 and 0 < top_p <= 1"
            raise ValueError(msg)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not ids:
            msg = "The generation prompt must contain at least one token"
            raise ValueError(msg)
        self.model.eval()
        device = self.model.model.embed_tokens.weight.device
        states = {}
        offset = 0

        def forward(tokens, initial):
            inputs = torch.tensor([tokens], device=device)
            return self._stream_forward(
                inputs, torch.ones_like(inputs, dtype=torch.bool), initial, memory_mode=memory_mode
            )

        while len(ids) - offset > window_size:
            _, states = forward(ids[offset : offset + window_size], states)
            offset += window_size
        current = ids[offset:]
        generated = []
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        eos.add(self.tokenizer.eos_token_id)
        for _ in range(max_new_tokens):
            hidden, final = forward(current, states)
            logits = self.model.lm_head(hidden[:, -1]).float().squeeze(0)
            if temperature == 0:
                token = int(logits.argmax())
            else:
                probabilities, indices = (logits / temperature).softmax(-1).sort(descending=True)
                probabilities[(probabilities.cumsum(-1) - probabilities) >= top_p] = 0
                token = int(indices[torch.multinomial(probabilities, 1)])
            if token in eos:
                break
            generated.append(token)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if stop_at_newline and "\n" in text:
                return text.split("\n", 1)[0].strip()
            if len(current) == window_size:
                states, current = final, []
            current.append(token)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(self, msgs: list[Message] | Message, max_new_tokens: int = 200, **kwargs):
        prompt = self.tokenizer.apply_chat_template(
            self._msg_to_dict(msgs),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.generate_text(prompt, max_new_tokens=max_new_tokens, **kwargs)

    @torch.no_grad()
    def _episodes_loss(self, windows, labels, batch_size, window_size, loss_chunk_size, pad_id):
        """Token-mean answer loss without gradients; memory carries within episodes."""
        device = self.model.model.embed_tokens.weight.device
        total_loss = torch.zeros((), device=device)
        total_tokens = 0
        for inputs, mask, targets, token_count in token_batches(
            windows, batch_size, pad_id, None, labels=labels
        ):
            states = {}
            episode_loss = torch.zeros((), device=device)
            for start in range(0, inputs.shape[1], window_size):
                end = min(start + window_size, inputs.shape[1])
                hidden, states = self._stream_forward(
                    inputs[:, start:end].to(device),
                    mask[:, start:end].to(device),
                    states,
                )
                window_labels = targets[:, start:end]
                if bool((window_labels != -100).any()):
                    episode_loss = episode_loss + chunked_lm_loss(
                        self.model.lm_head, hidden, window_labels.to(device), loss_chunk_size
                    )
            total_loss += episode_loss
            total_tokens += token_count
        if not total_tokens:
            msg = "Validation data must contain at least one supervised token"
            raise ValueError(msg)
        return float(total_loss) / total_tokens, total_tokens

    def _episode_backward(
        self, inputs, mask, targets, window_size, bptt_windows, loss_chunk_size, checkpoint_window
    ):
        """Backpropagate within fixed spans, carrying detached state between them."""
        states = {}
        device = self.model.model.embed_tokens.weight.device
        length = inputs.shape[1]
        horizon = window_size * bptt_windows if bptt_windows else length
        total_loss = torch.zeros((), device=device)
        for span in range(0, length, horizon):
            stop = min(span + horizon, length)
            # Prefix-only spans cannot affect any loss across the following detach.
            supervised = bool((targets[:, span:stop] != -100).any())
            with torch.set_grad_enabled(supervised):
                loss = torch.zeros((), device=device)
                for start in range(span, stop, window_size):
                    end = min(start + window_size, stop)
                    window_inputs = inputs[:, start:end].to(device)
                    window_mask = mask[:, start:end].to(device)
                    hidden, states = self._stream_forward(
                        window_inputs,
                        window_mask,
                        states,
                        checkpoint_window=checkpoint_window,
                    )
                    labels = targets[:, start:end]
                    if bool((labels != -100).any()):
                        loss = loss + chunked_lm_loss(
                            self.model.lm_head, hidden, labels.to(device), loss_chunk_size
                        )
                if supervised:
                    loss.backward()
                    total_loss += loss.detach()
            states = detach_memory_states(states)
        return total_loss

    def train(
        self,
        train_texts: list[str] | None = None,
        lr: float = 1e-4,
        max_length: int = 128,
        *,
        batch_size: int = 1,
        epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        loss_chunk_size: int = 128,
        checkpoint_decoder: bool = False,
        shuffle: bool = True,
        seed: int = 0,
        episodes: Iterable[MemoryEpisode] | None = None,
        bptt_windows: int = 4,
        max_steps: int | None = None,
        on_step: Callable[[dict], None] | None = None,
        validation_episodes: Iterable[MemoryEpisode] | None = None,
        on_validation: Callable[[dict], None] | None = None,
    ) -> list[float]:
        """Train only memory initializations, projections, update gates and residual gates.

        Returns token-mean losses per optimizer step (including a final partial
        accumulation group). Pass either legacy texts or normalized QA episodes.
        Episodes retain state across max_length-token Qwen windows, supervise
        answers/EOS and detach every bptt_windows windows (0 = full unroll).
        Validation episodes are scored once per completed epoch under no_grad,
        carrying memory across each episode's windows like generation does;
        on_validation receives {"epoch", "step", "val_loss", "target_tokens"}.
        Stopping via max_steps ends training before that epoch's validation.
        KeyboardInterrupt propagates after adapter state is restored, so the
        caller can still save the partially trained adapters.
        """
        if (
            max_length < 2
            or min(batch_size, epochs, gradient_accumulation_steps, loss_chunk_size) < 1
        ):
            msg = "Use max_length >= 2 and positive batch/epoch/accumulation/chunk sizes"
            raise ValueError(msg)
        if lr <= 0 or weight_decay < 0 or max_grad_norm <= 0:
            msg = "Use positive lr/max_grad_norm and nonnegative weight_decay"
            raise ValueError(msg)
        if torch.is_inference_mode_enabled():
            msg = "Training cannot run inside torch.inference_mode()"
            raise RuntimeError(msg)
        if (train_texts is None) == (episodes is None):
            msg = "Pass exactly one of train_texts or episodes"
            raise ValueError(msg)
        if bptt_windows < 0 or (max_steps is not None and max_steps < 1):
            msg = "Require bptt_windows >= 0 and max_steps >= 1"
            raise ValueError(msg)
        episode_mode = episodes is not None
        labels = None
        if episode_mode:
            self._check_window_size(max_length)
            windows, labels = tokenize_episodes(self.tokenizer, episodes)
            self._training_config = {
                "window_size": max_length,
                "bptt_windows": bptt_windows,
                "objective": "answer_ce",
            }
        else:
            windows = tokenize_windows(self.tokenizer, train_texts, max_length)
        if not windows:
            msg = "Training data must contain at least one next-token target"
            raise ValueError(msg)
        validation_windows = validation_labels = None
        if validation_episodes is not None:
            self._check_window_size(max_length)
            validation_windows, validation_labels = tokenize_episodes(
                self.tokenizer, validation_episodes
            )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            msg = "Tokenizer must define a padding or EOS token"
            raise ValueError(msg)

        self.model.requires_grad_(False)
        self.model.eval()
        chunk_checkpoints = [layer.memory.checkpoint_chunks for layer in self._titans_attn]
        for layer in self._titans_attn:
            layer.memory.requires_grad_(True)
            layer.memory.train()
            layer.memory_gate.requires_grad_(True)
            # Avoid nested recomputation if checkpointing the whole decoder.
            if checkpoint_decoder:
                layer.memory.checkpoint_chunks = False
        if checkpoint_decoder and not episode_mode:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Enable the decoder checkpoint wrapper; keep its frozen children in eval mode.
            for layer in self.model.model.layers:
                layer.training = True

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        device = self.model.model.embed_tokens.weight.device
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in trainable if p.ndim >= 2],
                    "weight_decay": weight_decay,
                },
                {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0},
            ],
            lr=lr,
            fused=device.type == "cuda",
        )
        losses = []
        try:
            with torch.enable_grad():
                for epoch in range(epochs):
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_tokens = 0
                    accumulated_loss = torch.zeros((), device=device)
                    microbatches = 0

                    def step(total_tokens, total_loss, current_epoch=epoch + 1):
                        # Normalize by actual target count, including uneven microbatches.
                        for parameter in trainable:
                            if parameter.grad is not None:
                                parameter.grad.div_(total_tokens)
                        torch.nn.utils.clip_grad_norm_(
                            trainable, max_grad_norm, error_if_nonfinite=True
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        losses.append(total_loss.item() / total_tokens)
                        if on_step is not None:
                            on_step(
                                {
                                    "step": len(losses),
                                    "epoch": current_epoch,
                                    "loss": losses[-1],
                                    "target_tokens": total_tokens,
                                }
                            )

                    for inputs, mask, targets, token_count in token_batches(
                        windows,
                        batch_size,
                        pad_id,
                        seed + epoch if shuffle else None,
                        labels=labels,
                    ):
                        if episode_mode:
                            loss = self._episode_backward(
                                inputs,
                                mask,
                                targets,
                                max_length,
                                bptt_windows,
                                loss_chunk_size,
                                checkpoint_decoder,
                            )
                        else:
                            tensors = (inputs, mask, targets)
                            if device.type == "cuda":
                                tensors = tuple(t.pin_memory() for t in tensors)
                            inputs, mask, targets = (
                                t.to(device, non_blocking=True) for t in tensors
                            )
                            hidden = self.model.model(
                                input_ids=inputs,
                                attention_mask=mask,
                                memory_mask=mask,
                                use_cache=False,
                            ).last_hidden_state
                            loss = chunked_lm_loss(
                                self.model.lm_head, hidden, targets, loss_chunk_size
                            )
                            loss.backward()
                        accumulated_tokens += token_count
                        accumulated_loss += loss.detach()
                        microbatches += 1
                        if microbatches == gradient_accumulation_steps:
                            step(accumulated_tokens, accumulated_loss)
                            if max_steps is not None and len(losses) >= max_steps:
                                return losses
                            accumulated_tokens = 0
                            accumulated_loss.zero_()
                            microbatches = 0
                    if microbatches:
                        step(accumulated_tokens, accumulated_loss)
                        if max_steps is not None and len(losses) >= max_steps:
                            return losses
                    if validation_windows is not None:
                        val_loss, val_tokens = self._episodes_loss(
                            validation_windows,
                            validation_labels,
                            batch_size,
                            max_length,
                            loss_chunk_size,
                            pad_id,
                        )
                        if on_validation is not None:
                            on_validation(
                                {
                                    "epoch": epoch + 1,
                                    "step": len(losses),
                                    "val_loss": val_loss,
                                    "target_tokens": val_tokens,
                                }
                            )
        finally:
            optimizer.zero_grad(set_to_none=True)
            if checkpoint_decoder and not episode_mode:
                self.model.gradient_checkpointing_disable()
            for layer, enabled in zip(self._titans_attn, chunk_checkpoints, strict=True):
                layer.memory.checkpoint_chunks = enabled
            self.model.eval()
        return losses
