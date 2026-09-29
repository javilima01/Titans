"""Compare direct memory decoding with a parallel connection to frozen LM logits."""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from src.llm.helpers.neural_fact_probe import READS
from src.llm.helpers.neural_memory_probe import NeuralMemoryProbe
from src.llm.modules.qwen import Qwen35Wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    torch.set_num_threads(4)
    probe = NeuralMemoryProbe(Qwen35Wrapper(device="mps"))
    probe.memory.load_state_dict(load_file(args.state))
    results = []
    for scale in (0, 8, 16, 32):
        for question, expected in READS:
            prediction, confidence = probe.answer(question, memory_scale=scale)
            row = {
                "scale": scale,
                "question": question,
                "expected": expected,
                "prediction": prediction,
                "correct": prediction == expected,
                "confidence": confidence,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"state": str(args.state), "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
