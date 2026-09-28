"""Measure whether a trained memory's read strength limits unseen-value recall."""

import argparse
import json
from pathlib import Path

import torch

from src.llm.helpers.dataset_generation import read_episodes
from src.llm.helpers.experiment_tracking import record_experiment
from src.llm.modules.evaluation import evaluate_episodes
from src.llm.modules.qwen import Qwen35Titans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--multipliers", type=float, nargs="+", default=(1, 4, 8, 16))
    args = parser.parse_args()
    if (
        args.limit < 1
        or args.window_size < 1
        or args.max_new_tokens < 1
        or any(value <= 0 for value in args.multipliers)
        or args.report.exists()
    ):
        parser.error("Require positive settings and a new report path")
    path = args.data / "validation.jsonl" if args.data.is_dir() else args.data
    episodes = list(read_episodes(path))[: args.limit]
    if not episodes or any(episode.split != "validation" for episode in episodes):
        parser.error("Expected nonempty validation episodes")
    model = Qwen35Titans.from_pretrained(args.checkpoint, device=args.device)
    gates = model.memory_gates
    original = tuple(gate.detach().clone() for gate in gates)
    evaluations = {}
    for factor in args.multipliers:
        with torch.no_grad():
            for gate, initial in zip(gates, original, strict=True):
                gate.copy_((initial.sigmoid() * factor).clamp(max=0.95).logit())
        evaluations[str(factor)] = evaluate_episodes(
            model,
            episodes,
            window_size=args.window_size,
            max_new_tokens=args.max_new_tokens,
        )
        metrics = evaluations[str(factor)]
        print(
            json.dumps(
                {
                    "multiplier": factor,
                    "exact_match": metrics["metrics"]["exact_match"],
                    "both_correct": metrics.get("paired_metrics", {}).get("both_correct"),
                }
            )
        )
    with torch.no_grad():
        for gate, initial in zip(gates, original, strict=True):
            gate.copy_(initial)
    report = {
        "checkpoint": str(args.checkpoint),
        "data_source": str(path),
        "evaluation_kind": "read_gate_multiplier_sweep",
        "evaluation_options": {
            "limit": args.limit,
            "window_size": args.window_size,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
        },
        "initial_gate_range": [
            min(value.sigmoid().min().item() for value in original),
            max(value.sigmoid().max().item() for value in original),
        ],
        "evaluations": evaluations,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record_experiment(args.checkpoint, additional_reports=(args.report,))
    print(json.dumps({"report": str(args.report)}))


if __name__ == "__main__":
    main()
