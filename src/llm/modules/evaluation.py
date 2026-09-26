"""Generated-answer QA metrics. No teacher-forced answers enter the prompt."""

from collections import Counter, defaultdict
import re
from typing import TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from src.llm.helpers.dataset_generation import MemoryEpisode


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def answer_metrics(prediction: str, answers: tuple[str, ...]) -> dict[str, float]:
    """Exact match and multiset word overlap, using the best reference by F1.

    Recall here is answer-token recall, not document retrieval Recall@K.
    Articles are retained so one-word answers such as 'a' remain meaningful.
    """
    prediction = normalize_answer(prediction)
    predicted_tokens = prediction.split()
    scores = []
    for answer in answers:
        normalized = normalize_answer(answer)
        tokens = normalized.split()
        overlap = sum((Counter(predicted_tokens) & Counter(tokens)).values())
        exact = float(prediction == normalized)
        precision = overlap / len(predicted_tokens) if predicted_tokens else exact
        recall = overlap / len(tokens) if tokens else exact
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append({"exact_match": exact, "precision": precision, "recall": recall, "f1": f1})
    return max(scores, key=lambda score: (score["f1"], score["exact_match"]))


def evaluate_episodes(
    model,
    episodes: Iterable[MemoryEpisode],
    *,
    window_size: int | None = None,
    max_new_tokens: int = 32,
    memory_mode: str = "normal",
    on_example: Callable[[dict], None] | None = None,
) -> dict:
    """Greedy decoding, macro metrics, per-task breakdown, and predictions."""
    predictions = []
    groups = defaultdict(list)
    for episode in episodes:
        prediction = model.generate_text(
            episode.prompt,
            window_size=window_size,
            max_new_tokens=max_new_tokens,
            memory_mode=memory_mode,
            temperature=0.0,
            stop_at_newline=True,
        )
        scores = answer_metrics(prediction, episode.answers)
        task = episode.metadata.get("task", episode.source)
        result = {
            "id": episode.id,
            "task": task,
            "prediction": prediction,
            "answers": list(episode.answers),
            **scores,
        }
        predictions.append(result)
        groups[task].append(scores)
        if on_example is not None:
            on_example({"completed": len(predictions), "id": episode.id, **scores})
    if not predictions:
        msg = "Evaluation requires at least one episode"
        raise ValueError(msg)

    def average(rows):
        return {
            "count": len(rows),
            **{
                key: sum(row[key] for row in rows) / len(rows)
                for key in ("exact_match", "precision", "recall", "f1")
            },
        }

    return {
        "memory_mode": memory_mode,
        "window_size": window_size or model.training_config.get("window_size", 512),
        "max_new_tokens": max_new_tokens,
        "metrics": average(predictions),
        "by_task": {task: average(rows) for task, rows in sorted(groups.items())},
        "predictions": predictions,
    }
