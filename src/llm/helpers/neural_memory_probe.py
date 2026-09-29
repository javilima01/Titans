"""Pure network memory: direct questions, no text/index/retrieval/replay on reads."""

import argparse
import json
from pathlib import Path
import time

from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional

from src.llm.helpers.dataset_generation import read_episodes
from src.llm.modules.neural_token_memory import NeuralTokenMemory
from src.llm.modules.qwen import Qwen35Wrapper

SYSTEM = "Answer with only the requested value."


class NeuralMemoryProbe:
    """Frozen Qwen + one output memory network; no growing data structure."""

    def __init__(self, model):
        self.model = model
        self.memory = NeuralTokenMemory(model.model.lm_head.weight.shape[1])
        self.output_norms = model.model.lm_head.weight.norm(dim=-1).clamp_min(1e-8)

    def prompt_ids(self, question):
        text = self.model.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.model.tokenizer.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def features(self, ids):
        tokens = torch.tensor([ids], device=self.model.device)
        return (
            self.model.model.model(input_ids=tokens, use_cache=False)
            .last_hidden_state[0]
            .cpu()
            .float()
        )

    @torch.no_grad()
    def teach(self, question, answer, *, repeats=1):
        prefix = self.prompt_ids(question)
        labels = [
            *self.model.tokenizer.encode(answer, add_special_tokens=False),
            self.model.tokenizer.eos_token_id,
        ]
        keys = self.features(prefix + labels[:-1])[len(prefix) - 1 :]
        values = (
            self.model.model.lm_head.weight[torch.tensor(labels, device=self.model.device)]
            .cpu()
            .float()
        )
        return self.memory.write(keys, values, repeats=repeats)

    @torch.no_grad()
    def answer(self, question, *, max_tokens=24, memory_scale=None):
        tokens = self.prompt_ids(question)
        generated, confidence = [], []
        for _ in range(max_tokens):
            hidden = self.features(tokens)[-1]
            predicted = self.memory(hidden)
            if predicted.norm() < 1e-8:
                return "", [0.0]
            logits = (
                self.model.model.lm_head(
                    functional.normalize(predicted, dim=-1).to(self.model.device)
                )
                / self.output_norms
            )
            score = logits.max(-1).values
            if memory_scale is not None:
                logits = memory_scale * logits + self.model.model.lm_head(
                    hidden.to(self.model.device)
                )
            selected = logits.argmax(-1)
            token = int(selected)
            confidence.append(float(score))
            if token == self.model.tokenizer.eos_token_id:
                break
            generated.append(token)
            tokens.append(token)
        return self.model.tokenizer.decode(generated, skip_special_tokens=True).strip(), confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists() or args.limit < 2 or args.limit % 2:
        parser.error("Require a new output file and positive even limit")
    torch.set_num_threads(4)
    model = Qwen35Wrapper(device="mps")
    probe = NeuralMemoryProbe(model)
    rows = list(
        read_episodes(Path(".datasets_cache/session-broad-novel-values-v1/validation.jsonl"))
    )[: args.limit]
    started = time.monotonic()
    results, writes = [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.parent / "network.safetensors"
    for variant in (0, 1):
        writes.extend(
            probe.teach(row.question, row.answers[0], repeats=args.repeats)
            for row in rows[variant::2]
        )
        save_file(probe.memory.state_dict(), snapshot)
        probe.memory = NeuralTokenMemory(probe.memory.weight.shape[0])
        probe.memory.load_state_dict(load_file(snapshot))
        for row in rows[variant::2]:
            prediction, scores = probe.answer(row.question)
            result = {
                "id": row.id,
                "variant": variant,
                "question": row.question,
                "expected": row.answers[0],
                "prediction": prediction,
                "correct": prediction == row.answers[0],
                "confidence": scores,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "architecture": "fixed_size_fast_linear_network_to_frozen_token_embeddings",
        "repeats": args.repeats,
        "elapsed_seconds": time.monotonic() - started,
        "network_bytes": snapshot.stat().st_size,
        "state_shapes": {
            key: list(value.shape) for key, value in probe.memory.state_dict().items()
        },
        "count": len(results),
        "correct": sum(row["correct"] for row in results),
        "writes": writes,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
