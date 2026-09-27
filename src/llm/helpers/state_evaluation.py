"""Evaluate held-out facts after serializing and restoring Titans fast memory.

Example: python -m src.llm.helpers.state_evaluation --data DATA --checkpoint CHECKPOINT
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import tempfile

from src.llm.helpers.config import ROOT
from src.llm.helpers.dataset_generation import read_episodes
from src.llm.helpers.experiment_tracking import record_experiment
from src.llm.modules.evaluation import evaluate_episodes
from src.llm.modules.qwen import Qwen35Titans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.limit < 1 or args.max_new_tokens < 1:
        parser.error("--limit and --max-new-tokens must be positive")
    path = args.data / f"{args.split}.jsonl" if args.data.is_dir() else args.data
    episodes = list(read_episodes(path))[: args.limit]
    if not episodes or any(episode.split != args.split for episode in episodes):
        parser.error(f"Expected nonempty {args.split} episodes")
    model = Qwen35Titans.from_pretrained(args.checkpoint, device=args.device)
    report_path = args.report or args.checkpoint / (
        f"state-{args.split}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    )
    if report_path.exists():
        parser.error(f"Evaluation report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "user.safetensors"

        def predict_state(episode):
            state = model.memorize_text(episode.context, window_size=args.window_size)
            model.save_memory_state(state_path, state)
            restored = model.load_memory_state(state_path)
            question = episode.prompt[len(episode.context) :]
            return model.generate_text(
                question,
                memory_state=restored,
                window_size=args.window_size,
                max_new_tokens=args.max_new_tokens,
                stop_at_newline=True,
            )

        def predict_reset(episode):
            question = episode.prompt[len(episode.context) :]
            return model.generate_text(
                question,
                window_size=args.window_size,
                max_new_tokens=args.max_new_tokens,
                stop_at_newline=True,
            )

        evaluations = {}
        for mode, predictor in (("state_only", predict_state), ("reset", predict_reset)):
            evaluations[mode] = evaluate_episodes(
                model,
                episodes,
                window_size=args.window_size,
                max_new_tokens=args.max_new_tokens,
                predict=predictor,
                on_example=lambda event, current=mode: print(
                    json.dumps({"mode": current, **event}), file=sys.stderr, flush=True
                ),
            )
            evaluations[mode]["memory_mode"] = mode
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "data_source": str(args.data),
        "evaluation_options": {
            "device": args.device,
            "window_size": args.window_size or model.training_config.get("window_size", 512),
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
        },
        "evaluation_kind": "serialized_fast_state",
        "evaluations": evaluations,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record_experiment(
        args.checkpoint,
        output_root=ROOT / "experiments",
        additional_reports=(report_path,),
    )
    summary = {
        mode: {key: value for key, value in result.items() if key != "predictions"}
        for mode, result in evaluations.items()
    }
    print(json.dumps({"report": str(report_path), "evaluations": summary}, indent=2))


if __name__ == "__main__":
    main()
