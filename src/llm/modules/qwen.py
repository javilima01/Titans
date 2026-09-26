import os

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5Tokenizer
from transformers.utils import logging

from src.llm.helpers.utils import ensure_model
from src.llm.helpers.visualize import print_module_tree
from src.llm.modules.titans import AttentionWithTitans, TitansMemory
from src.llm.modules.training import chunked_lm_loss, token_batches, tokenize_windows
from src.llm.schemas.messages import Message

logging.set_verbosity_error()


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
    ):
        super().__init__(device=device, dtype=dtype)
        self.model.requires_grad_(False)
        self._titans_attn: list[AttentionWithTitans] = []
        self._install_titans(
            layer_indices, memory_hidden_size, memory_chunk_size, checkpoint_memory
        )
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
    ):
        ids = self._full_attention_layers() if layer_indices is None else layer_indices
        full_attention = set(self._full_attention_layers())
        if not ids or len(set(ids)) != len(ids) or any(i not in full_attention for i in ids):
            msg = "Select distinct full-attention layer indices"
            raise ValueError(msg)
        for layer_idx in ids:
            self._add_to_existing(
                layer_idx, memory_hidden_size, memory_chunk_size, checkpoint_memory
            )

    def _add_to_existing(
        self,
        layer_idx: int = 11,
        memory_hidden_size=None,
        memory_chunk_size=16,
        checkpoint_memory=True,
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
        )

        param = next(attention.parameters())
        # Fast weights, inner updates and AdamW state stay in FP32.
        memory = memory.to(device=param.device, dtype=torch.float32)
        wrapped_attention = AttentionWithTitans(
            attention=attention,
            memory=memory,
        )
        wrapped_attention.to(device=param.device)

        layer.self_attn = wrapped_attention

        self._titans_attn.append(wrapped_attention)

    def train(
        self,
        train_texts: list[str],
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
    ) -> list[float]:
        """Train only memory initializations, projections, update gates and residual gates.

        Returns token-mean losses per optimizer step (including a final partial
        accumulation group). Windows are independent episodes; all inner memory
        updates remain differentiable. Long documents are windowed, not truncated.
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
        windows = tokenize_windows(self.tokenizer, train_texts, max_length)
        if not windows:
            msg = "Training data must contain at least one next-token target"
            raise ValueError(msg)
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
        if checkpoint_decoder:
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

                    def step(total_tokens, total_loss):
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

                    for inputs, mask, targets, token_count in token_batches(
                        windows, batch_size, pad_id, seed + epoch if shuffle else None
                    ):
                        tensors = (inputs, mask, targets)
                        if device.type == "cuda":
                            tensors = tuple(t.pin_memory() for t in tensors)
                        inputs, mask, targets = (t.to(device, non_blocking=True) for t in tensors)
                        hidden = self.model.model(
                            input_ids=inputs,
                            attention_mask=mask,
                            memory_mask=mask,
                            use_cache=False,
                        ).last_hidden_state
                        loss = chunked_lm_loss(self.model.lm_head, hidden, targets, loss_chunk_size)
                        loss.backward()
                        accumulated_tokens += token_count
                        accumulated_loss += loss.detach()
                        microbatches += 1
                        if microbatches == gradient_accumulation_steps:
                            step(accumulated_tokens, accumulated_loss)
                            accumulated_tokens = 0
                            accumulated_loss.zero_()
                            microbatches = 0
                    if microbatches:
                        step(accumulated_tokens, accumulated_loss)
        finally:
            optimizer.zero_grad(set_to_none=True)
            if checkpoint_decoder:
                self.model.gradient_checkpointing_disable()
            for layer, enabled in zip(self._titans_attn, chunk_checkpoints, strict=True):
                layer.memory.checkpoint_chunks = enabled
            self.model.eval()
        return losses
