"""Evaluate serialized source-text memory on cross-session episodes."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import tempfile

from src.llm.helpers.config import ROOT
from src.llm.helpers.dataset_generation import read_episodes
from src.llm.helpers.experiment_tracking import record_experiment
from src.llm.helpers.retrieval_evaluation import USER_MESSAGE, _prompt
from src.llm.modules.episodic_memory import EpisodicMemory
from src.llm.modules.evaluation import evaluate_episodes


def main() -> None:
    from src.llm.modules.qwen import Qwen35Titans

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--evidence-limit", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.limit, args.window_size, args.max_new_tokens, args.evidence_limit) < 1:
        parser.error("All numeric evaluation options must be positive")
    path = args.data / f"{args.split}.jsonl" if args.data.is_dir() else args.data
    episodes = list(read_episodes(path))[: args.limit]
    if not episodes or any(episode.split != args.split for episode in episodes):
        parser.error(f"Expected nonempty {args.split} episodes")
    model = Qwen35Titans.from_pretrained(args.checkpoint, device=args.device)
    report_path = args.report or args.checkpoint / (
        f"episodic-{args.split}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    )
    if report_path.exists():
        parser.error(f"Evaluation report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    hits = 0

    with tempfile.TemporaryDirectory() as directory:
        memory_path = Path(directory) / "user.json"

        def predict(episode):
            nonlocal hits
            memory = EpisodicMemory()
            for message in USER_MESSAGE.findall(episode.context):
                memory.remember(message)
            memory.save(memory_path)
            restored = EpisodicMemory.load(memory_path)
            evidence = restored.retrieve(episode.question, limit=args.evidence_limit)
            start, end = episode.metadata["support_spans"][-1]
            support = episode.context[start:end]
            hits += any(
                record.text in support and episode.answers[0] in record.text for record in evidence
            )
            return model.generate_text(
                _prompt("\n".join(record.text for record in evidence), episode.question),
                memory_mode="disabled",
                window_size=args.window_size,
                max_new_tokens=args.max_new_tokens,
                stop_at_newline=True,
            )

        evaluation = evaluate_episodes(
            model,
            episodes,
            window_size=args.window_size,
            max_new_tokens=args.max_new_tokens,
            predict=predict,
            on_example=lambda event: print(json.dumps(event), file=sys.stderr, flush=True),
        )
    evaluation["memory_mode"] = "episodic_text"
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "data_source": str(args.data),
        "evaluation_kind": "serialized_episodic_text",
        "evaluation_options": {
            "device": args.device,
            "window_size": args.window_size,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "evidence_limit": args.evidence_limit,
        },
        "retrieval_hit_rate": hits / len(episodes),
        "evaluations": {"episodic": evaluation},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record_experiment(
        args.checkpoint, output_root=ROOT / "experiments", additional_reports=(report_path,)
    )
    summary = {key: value for key, value in evaluation.items() if key != "predictions"}
    print(
        json.dumps(
            {"report": str(report_path), "hit_rate": hits / len(episodes), "evaluation": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
