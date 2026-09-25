import os

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5Tokenizer
from transformers.utils import logging

from src.llm.helpers.utils import ensure_model
from src.llm.helpers.visualize import print_module_tree
from src.llm.modules.titans import AttentionWithTitans, TitansMemory
from src.llm.schemas.messages import Message

logging.set_verbosity_error()


class Qwen35Wrapper:
    _model_name: str = "Qwen/Qwen3.5-0.8B"

    def __init__(self):
        # Ensure that the model is downloaded locally
        self._load()
        self._install_titans()

    def _load(self):
        model_path = ensure_model(self._model_name)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        self.tokenizer: Qwen3_5Tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

    def _msg_to_dict(self, msgs: list[Message] | Message) -> list[dict]:
        if isinstance(msgs, Message):
            msgs = [msgs]

        return [msg.model_dump(mode="python") for msg in msgs]

    def _full_attention_layers(self) -> list[int]:
        return [
            layer_idx
            for layer_idx, layer in enumerate(self.model.model.layers)
            if hasattr(layer, "self_attn")
        ]

    def _install_titans(self):
        # detect full attention layers
        ids = self._full_attention_layers()
        for id in ids:
            self._add_to_existing(layer_idx=id)

    def _add_to_existing(self, layer_idx: int = 11):
        layer = self.model.model.layers[layer_idx]

        if not hasattr(layer, "self_attn"):
            error_msg = f"Layer {layer_idx} is not a full-attention layer"
            raise ValueError(error_msg)

        attention = layer.self_attn

        hidden_size = self.model.config.hidden_size
        memory = TitansMemory(
            hidden_size=hidden_size,
            dim=hidden_size,
        )

        # Important because the model has already been loaded using
        # device_map="auto".
        param = next(attention.parameters())

        memory = memory.to(
            device=param.device,
            dtype=param.dtype,
        )

        wrapped_attention = AttentionWithTitans(
            attention=attention,
            memory=memory,
        )

        wrapped_attention.memory_gate.data = wrapped_attention.memory_gate.data.to(
            param.device
        )

        layer.self_attn = wrapped_attention

    def generate(
        self, msgs: list[Message] | Message, max_new_tokens: int = 200, **kwargs
    ):
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
