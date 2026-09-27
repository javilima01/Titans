"""Evaluate text retrieval as a control for arbitrary-fact neural memory.

The lexical mode ranks prior user messages from an episode. Oracle mode uses
the recorded support span only as an upper bound; it is not deployable retrieval.
"""

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import sys

from src.llm.helpers.config import ROOT
from src.llm.helpers.dataset_generation import read_episodes
from src.llm.helpers.experiment_tracking import record_experiment
from src.llm.modules.evaluation import evaluate_episodes

STOPWORDS = {
    "a",
    "am",
    "and",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "only",
    "reply",
    "session",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
}
USER_MESSAGE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word[:5] for word in words if word not in STOPWORDS and len(word) >= 3}


def lexical_fact(episode) -> str:
    """Rank prior user messages by rare query-term overlap, breaking ties by recency."""
    messages = USER_MESSAGE.findall(episode.context)
    if not messages:
        msg = "Expected at least one prior user message"
        raise ValueError(msg)
    query = _terms(episode.question)
    documents = [_terms(message) for message in messages]
    frequency = Counter(term for document in documents for term in document)
    scores = [
        sum(math.log1p((len(documents) + 1) / (frequency[term] + 1)) for term in query & doc)
        for doc in documents
    ]
    return messages[max(range(len(messages)), key=lambda index: (scores[index], index))]


def _prompt(fact: str, question: str) -> str:
    return (
        "<|im_start|>system\nAnswer questions with only the requested value, "
        "without explanation.<|im_end|>\n"
        f"<|im_start|>user\n{fact}<|im_end|>\n"
        "<|im_start|>assistant\nUnderstood.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def main() -> None:
    from src.llm.modules.qwen import Qwen35Titans

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--modes", nargs="+", choices=("lexical", "oracle"), default=("lexical", "oracle")
    )
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
        f"retrieval-{args.split}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    )
    if report_path.exists():
        parser.error(f"Evaluation report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    hits = 0

    def predict_lexical(episode):
        nonlocal hits
        fact = lexical_fact(episode)
        start, end = episode.metadata["support_spans"][-1]
        hits += fact == episode.context[start:end]
        return model.generate_text(
            _prompt(fact, episode.question),
            memory_mode="disabled",
            window_size=256,
            max_new_tokens=args.max_new_tokens,
            stop_at_newline=True,
        )

    def predict_oracle(episode):
        start, end = episode.metadata["support_spans"][-1]
        return model.generate_text(
            _prompt(episode.context[start:end], episode.question),
            memory_mode="disabled",
            window_size=256,
            max_new_tokens=args.max_new_tokens,
            stop_at_newline=True,
        )

    evaluations = {}
    predictors = {"lexical": predict_lexical, "oracle": predict_oracle}
    for mode in dict.fromkeys(args.modes):
        predictor = predictors[mode]
        evaluations[mode] = evaluate_episodes(
            model,
            episodes,
            window_size=256,
            max_new_tokens=args.max_new_tokens,
            predict=predictor,
            on_example=lambda event, current=mode: print(
                json.dumps({"mode": current, **event}), file=sys.stderr, flush=True
            ),
        )
        evaluations[mode]["memory_mode"] = f"retrieval_{mode}"
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "data_source": str(args.data),
        "evaluation_kind": "text_retrieval_control",
        "evaluation_options": {
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "modes": list(dict.fromkeys(args.modes)),
        },
        "retrieval_hit_rate": hits / len(episodes) if "lexical" in args.modes else None,
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
    print(
        json.dumps(
            {
                "report": str(report_path),
                "hit_rate": hits / len(episodes) if "lexical" in args.modes else None,
                "evaluations": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
