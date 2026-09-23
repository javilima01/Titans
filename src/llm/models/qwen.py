import os

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5Tokenizer
from transformers.utils import logging

from src.llm.helpers.utils import ensure_model
from src.llm.schemas.messages import Message

logging.set_verbosity_error()


class Qwen35Wrapper:
    _model_name: str = "Qwen/Qwen3.5-0.8B"

    def __init__(self):
        # Ensure that the model is downloaded locally
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
