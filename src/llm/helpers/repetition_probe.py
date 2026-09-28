"""Test fast-memory recall after repeated question/answer demonstrations.

Each example starts with an empty state. A Q/A pair is written at test time,
then the state is serialized and reloaded before a question-only new session.
"""

import argparse
import json
from pathlib import Path
import tempfile

from src.llm.helpers.dataset_generation import read_episodes
from src.llm.helpers.experiment_tracking import record_experiment
from src.llm.modules.evaluation import evaluate_episodes
from src.llm.modules.qwen import Qwen35Titans

SYSTEM = (
    "<|im_start|>system\nAnswer questions with only the requested value, "
    "without explanation.<|im_end|>\n"
)


def demonstration(episode) -> str:
    """Match the non-thinking chat answer format used by episode training."""
    return (
        f"<|im_start|>user\n{episode.question}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        f"{episode.answers[0]}<|im_end|>\n"
    )


def question_only_prompt(episode, *, include_system: bool = False) -> str:
    """Remove the earlier episode context before asking in a new session."""
    prompt = episode.prompt[len(episode.context) :]
    if include_system:
        prompt = SYSTEM + prompt
    return prompt


def evaluate(
    checkpoint: Path,
    episodes,
    *,
    device: str,
    window_size: int,
    max_new_tokens: int,
    repetitions: tuple[int, ...],
    include_system: bool = False,
    direct_context: bool = False,
) -> dict:
    model = Qwen35Titans.from_pretrained(checkpoint, device=device)
    if not episodes:
        msg = "Need at least one held-out episode"
        raise ValueError(msg)
    evaluations = {}
    with tempfile.TemporaryDirectory() as directory:
        state_file = Path(directory) / "one-user.safetensors"
        for count in repetitions:
            if count < 0:
                msg = "Repetition counts must be nonnegative"
                raise ValueError(msg)

            def predict(episode, repetitions=count):
                prompt = question_only_prompt(episode, include_system=include_system)
                if repetitions == 0:
                    return model.generate_text(
                        prompt,
                        window_size=window_size,
                        max_new_tokens=max_new_tokens,
                        stop_at_newline=True,
                    )
                text = demonstration(episode) * repetitions
                state = model.memorize_text(text, window_size=window_size)
                model.save_memory_state(state_file, state)
                restored = model.load_memory_state(state_file)
                return model.generate_text(
                    prompt,
                    memory_state=restored,
                    window_size=window_size,
                    max_new_tokens=max_new_tokens,
                    stop_at_newline=True,
                )

            result = evaluate_episodes(
                model,
                episodes,
                window_size=window_size,
                max_new_tokens=max_new_tokens,
                predict=predict,
            )
            result["memory_mode"] = f"qa_repeated_{count}_serialized_state"
            evaluations[str(count)] = result
            print(
                json.dumps(
                    {
                        "repetitions": count,
                        "exact_match": result["metrics"]["exact_match"],
                        "complete_pairs": result.get("paired_metrics", {}).get("both_correct"),
                    }
                ),
                flush=True,
            )
        if direct_context:

            def predict_direct(episode):
                prompt = (
                    (SYSTEM if include_system else "")
                    + demonstration(episode) * 5
                    + question_only_prompt(episode)
                )
                return model.generate_text(
                    prompt,
                    memory_mode="disabled",
                    window_size=512,
                    max_new_tokens=max_new_tokens,
                    stop_at_newline=True,
                )

            result = evaluate_episodes(
                model,
                episodes,
                window_size=512,
                max_new_tokens=max_new_tokens,
                predict=predict_direct,
            )
            result["memory_mode"] = "direct_context_5_memory_disabled"
            evaluations["direct_context_5"] = result
            print(
                json.dumps(
                    {
                        "mode": "direct_context_5",
                        "exact_match": result["metrics"]["exact_match"],
                        "complete_pairs": result.get("paired_metrics", {}).get("both_correct"),
                    }
                ),
                flush=True,
            )
    return {
        "checkpoint": str(checkpoint),
        "evaluation_kind": "repeated_test_time_qa_into_serialized_fast_state",
        "evaluation_options": {
            "device": device,
            "window_size": window_size,
            "max_new_tokens": max_new_tokens,
            "repetitions": list(repetitions),
            "include_system": include_system,
            "direct_context": direct_context,
        },
        "evaluations": evaluations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--repetitions", type=int, nargs="*", default=(0, 1, 5))
    parser.add_argument("--include-system", action="store_true")
    parser.add_argument("--direct-context", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.limit < 1
        or args.window_size < 1
        or args.max_new_tokens < 1
        or (not args.repetitions and not args.direct_context)
        or any(count < 0 for count in args.repetitions)
        or args.report.exists()
    ):
        parser.error("Require valid positive settings and a new report path")
    path = args.data / f"{args.split}.jsonl" if args.data.is_dir() else args.data
    episodes = list(read_episodes(path))[: args.limit]
    if not episodes or any(episode.split != args.split for episode in episodes):
        parser.error(f"Expected nonempty {args.split} episodes")
    report = evaluate(
        args.checkpoint,
        episodes,
        device=args.device,
        window_size=args.window_size,
        max_new_tokens=args.max_new_tokens,
        repetitions=tuple(args.repetitions),
        include_system=args.include_system,
        direct_context=args.direct_context,
    )
    report["split"] = args.split
    report["data_source"] = str(path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record_experiment(args.checkpoint, additional_reports=(args.report,))
    print(json.dumps({"report": str(args.report)}))


if __name__ == "__main__":
    main()
