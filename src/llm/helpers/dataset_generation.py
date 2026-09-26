"""Build ordered memory episodes, without loading Qwen or changing its trainer.

Synthetic generation and JSONL IO use only the standard library. Hugging Face
imports are lazy; install ``datasets`` to use ``load_hf_episodes``. Run
``python -m src.llm.helpers.dataset_generation --help`` for the CLI.
"""

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import string
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
TASKS = ("recall", "update", "multi_hop")
BABILONG_TRAIN = "RMT-team/babilong-train-5k-samples"
BABILONG_TEST = "RMT-team/babilong"
QASPER = "allenai/qasper"


@dataclass(frozen=True)
class MemoryEpisode:
    """One independent memory lifetime; context order must be preserved.

    ``answers`` contains alternative accepted answers, not successive turns.
    Metadata is for evaluation/provenance and must not be fed to the model.
    """

    id: str
    context: str
    question: str
    answers: tuple[str, ...]
    source: str
    split: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.split not in SPLITS:
            msg = f"Unknown episode split: {self.split!r}"
            raise ValueError(msg)
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.id, self.context, self.question, self.source)
        ):
            msg = "Episode id, context, question and source must be nonempty strings"
            raise ValueError(msg)
        if (
            not isinstance(self.answers, tuple)
            or not self.answers
            or any(not isinstance(a, str) or not a.strip() for a in self.answers)
        ):
            msg = "answers must be a nonempty tuple of nonempty strings"
            raise ValueError(msg)
        if not isinstance(self.metadata, dict):
            msg = "metadata must be a dictionary"
            raise ValueError(msg)

    @property
    def prompt(self) -> str:
        return f"{self.context}\n\nQuestion: {self.question}\nAnswer: "

    def training_example(self) -> dict:
        """Render a canonical answer with a half-open CHARACTER loss span.

        Tokenize the complete text with offsets when building token labels;
        separately tokenizing the prompt/answer can change the BPE boundary.
        This is plain text, not a Qwen chat template. No EOS is added here.
        """
        prompt = self.prompt
        text = prompt + self.answers[0]
        return {"text": text, "answer_span": [len(prompt), len(text)]}


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 0
    num_entities: int = 8
    distractor_sentences: int = 128
    min_tail_sentences: int = 32
    tasks: tuple[str, ...] = TASKS

    def __post_init__(self):
        if not 2 <= self.num_entities <= 1000:
            msg = "num_entities must be between 2 and 1000"
            raise ValueError(msg)
        if not 0 <= self.min_tail_sentences <= self.distractor_sentences:
            msg = "Require 0 <= min_tail_sentences <= distractor_sentences"
            raise ValueError(msg)
        if not self.tasks or any(task not in TASKS for task in self.tasks):
            msg = f"tasks must be a nonempty selection from {TASKS}"
            raise ValueError(msg)


def _digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _noise(rng: random.Random) -> str:
    subject = rng.choice(("technician", "visitor", "gardener", "driver", "clerk", "painter"))
    action = rng.choice(("inspected", "cleaned", "photographed", "repaired", "measured", "moved"))
    item = rng.choice(("bench", "window", "shelf", "bicycle", "cabinet", "lamp"))
    place = rng.choice(("courtyard", "workshop", "lobby", "warehouse", "office", "garden"))
    return f"The {subject} {action} the {item} in the {place}."


def _synthetic_episode(index: int, split: str, config: SyntheticConfig) -> MemoryEpisode:
    # Independent RNG per episode: results do not depend on iteration/batch size.
    key = _digest("memory-synthetic-v1", asdict(config), split, index)
    rng = random.Random(key)
    task = rng.choice(config.tasks)
    names = []
    while len(names) < config.num_entities:
        name = "".join(rng.choices(string.ascii_lowercase, k=7)).capitalize()
        if name not in names:
            names.append(name)
    codes = rng.sample(range(1000, 10000), 2 * len(names))
    values = dict(zip(names, codes[: len(names)], strict=True))
    target = rng.choice(names)
    answer = str(values[target])
    facts = [
        (f"The access code for station {name} is {values[name]}.", name == target) for name in names
    ]
    rng.shuffle(facts)
    if task == "update":
        # Update competing entities too, so "pick the only correction" cannot solve it.
        updates = []
        others = [name for name in names if name != target]
        changed = [target, *rng.sample(others, max(2, len(names) // 2) - 1)]
        for name, code in zip(changed, codes[len(names) :], strict=False):
            updates.append(
                (f"The access code for station {name} has changed to {code}.", name == target)
            )
            if name == target:
                answer = str(code)
        rng.shuffle(updates)
        facts.extend(updates)  # Every correction follows its original assignment.
    if task == "multi_hop":
        couriers = rng.sample(range(100, 10000), len(names))
        assignments = [
            (f"Courier {courier} is assigned to station {name}.", name == target)
            for courier, name in zip(couriers, names, strict=True)
        ]
        rng.shuffle(assignments)
        facts.extend(assignments)
        rng.shuffle(facts)
        courier = couriers[names.index(target)]
        question = f"What is the access code for the station assigned to courier {courier}?"
    else:
        question = rng.choice(
            (
                f"What is the current access code for station {target}?",
                f"Which access code should a visitor use for station {target} now?",
                f"Give the latest access code for station {target}.",
            )
        )

    # Leave a configurable gap AFTER all relevant facts, and vary their positions.
    prefix_length = config.distractor_sentences - config.min_tail_sentences + len(facts)
    positions = set(rng.sample(range(prefix_length), len(facts)))
    facts_iter = iter(facts)
    lines, support_spans = [], []
    offset = 0
    for position in range(len(facts) + config.distractor_sentences):
        sentence, support = next(facts_iter) if position in positions else (_noise(rng), False)
        if support:
            support_spans.append([offset, offset + len(sentence)])
        lines.append(sentence)
        offset += len(sentence) + 1
    return MemoryEpisode(
        id=f"synthetic-{key[:24]}",
        context="\n".join(lines),
        question=question,
        answers=(answer,),
        source="synthetic-v1",
        split=split,
        metadata={
            "task": task,
            "seed": config.seed,
            "episode_index": index,
            "num_entities": len(names),
            "distractor_sentences": config.distractor_sentences,
            "min_tail_sentences": config.min_tail_sentences,
            "support_spans": support_spans,
        },
    )


def generate_synthetic(
    count: int,
    *,
    split: str = "train",
    config: SyntheticConfig | None = None,
    start_index: int = 0,
) -> Iterator[MemoryEpisode]:
    """Generate fresh associations with reproducible, separate RNGs for each split.

    Lengths are in sentences, NOT tokenizer tokens. Support spans are character
    offsets into context; a future trainer can measure actual token distances.
    Shared task templates across splits test new associations, not new task types.
    """
    if count < 0 or start_index < 0 or split not in SPLITS:
        msg = "Require nonnegative count/start_index and a train/validation/test split"
        raise ValueError(msg)
    config = config or SyntheticConfig()
    for index in range(start_index, start_index + count):
        yield _synthetic_episode(index, split, config)


def _records(value) -> list[dict]:
    """Accept HF's dict-of-lists representation and ordinary lists of records."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [dict(zip(value, row, strict=True)) for row in zip(*value.values(), strict=True)]
    msg = "Expected a list of records or a dictionary of parallel lists"
    raise ValueError(msg)


def _qasper_episodes(row: dict, split: str, provenance: dict) -> Iterator[MemoryEpisode]:
    parts = [row["title"], row["abstract"]]
    for section in _records(row["full_text"]):
        parts.extend([section["section_name"], *section["paragraphs"]])
    parts.extend(
        figure.get("caption", "") for figure in _records(row.get("figures_and_tables", []))
    )
    context = "\n\n".join(part for part in parts if part)
    for qa in _records(row["qas"]):
        answers = []
        for annotation in _records(qa["answers"]):
            answer = annotation["answer"]
            if answer["unanswerable"]:
                text = "unanswerable"
            elif answer.get("free_form_answer"):
                text = answer["free_form_answer"].strip()
            elif answer.get("extractive_spans"):
                text = ", ".join(answer["extractive_spans"])
            elif isinstance(answer.get("yes_no"), bool):
                text = "yes" if answer["yes_no"] else "no"
            else:
                continue
            if text and text not in answers:
                answers.append(text)
        if answers:
            yield MemoryEpisode(
                id=f"qasper-{split}-{row['id']}-{qa['question_id']}",
                context=context,
                question=qa["question"]
                + " If the paper does not provide an answer, say unanswerable.",
                answers=tuple(answers),
                source=QASPER,
                split=split,
                metadata={**provenance, "document_id": row["id"]},
            )


def load_hf_episodes(
    dataset: str,
    *,
    split: str = "train",
    limit: int = 1000,
    length: str = "1k",
    task: str = "qa1",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Iterator[MemoryEpisode]:
    """Stream a bounded number of episodes from BABILong or QASPER.

    Resolves the requested revision to a commit and records it on every episode.
    BABILong has separate training/evaluation repositories; its HF split is the
    task name, e.g. qa1. Only train/test are supported for BABILong. QASPER keeps
    its official document-level train/validation/test splits. ``limit`` counts
    emitted QA episodes, not source documents or downloaded bytes. QASPER uses
    HF's refs/convert/parquet export by default; no dataset scripts are executed.
    """
    if dataset not in ("babilong", "qasper") or split not in SPLITS or limit < 1:
        msg = "Select babilong/qasper, a train/validation/test split, and limit >= 1"
        raise ValueError(msg)
    if dataset == "babilong" and split == "validation":
        msg = "BABILong has no official validation split; use a separate held-out training subset"
        raise ValueError(msg)
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise
        msg = "Hugging Face loading requires datasets: uv pip install datasets"
        raise ImportError(msg) from exc
    from huggingface_hub import HfApi

    repo = (
        QASPER if dataset == "qasper" else (BABILONG_TRAIN if split == "train" else BABILONG_TEST)
    )
    revision = revision or ("refs/convert/parquet" if dataset == "qasper" else "main")
    info = HfApi().dataset_info(repo, revision=revision)
    commit = info.sha
    if not commit:
        msg = f"Could not resolve dataset revision for {repo}"
        raise RuntimeError(msg)
    config_name = length if dataset == "babilong" else "qasper"
    source_split = task if dataset == "babilong" else split
    available = {file.rfilename for file in info.siblings}
    if dataset == "babilong":
        filename = f"data/{task}/{length}.json"
        files = [filename] if filename in available else []
    else:
        files = sorted(
            filename
            for filename in available
            if filename.startswith(f"qasper/{split}/") and filename.endswith(".parquet")
        )
    if not files:
        msg = (
            f"No files for {repo}, revision={revision}, config={config_name}, split={source_split}"
        )
        raise ValueError(msg)
    # Select only the requested files. BABILong's dataset card also lists missing
    # task files, and QASPER's main branch needs a legacy executable loader.
    rows = load_dataset(
        "json" if dataset == "babilong" else "parquet",
        data_files={source_split: [f"hf://datasets/{repo}@{commit}/{file}" for file in files]},
        split=source_split,
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    provenance = {
        "revision": commit,
        "config": config_name,
        "source_split": source_split,
        "data_files": files,
    }

    def episodes():
        for index, row in enumerate(rows):
            if dataset == "qasper":
                yield from _qasper_episodes(row, split, provenance)
            else:
                yield MemoryEpisode(
                    id=f"babilong-{split}-{length}-{task}-{index}",
                    context=row["input"],
                    question=row["question"],
                    answers=(row["target"],),
                    source=repo,
                    split=split,
                    metadata={**provenance, "row_index": index, "task": task},
                )

    yield from islice(episodes(), limit)


def write_episodes(path: str | Path, episodes: Iterable[MemoryEpisode]) -> int:
    """Stream to JSONL, publishing only a complete file; refuse to overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        msg = f"Dataset already exists: {path}"
        raise FileExistsError(msg)
    count = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for episode in episodes:
                record = {"schema_version": SCHEMA_VERSION, **asdict(episode)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            # Same-filesystem atomic publication, with no race that overwrites a file.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return count


def read_episodes(path: str | Path) -> Iterator[MemoryEpisode]:
    """Read normalized episodes offline without Hugging Face dependencies."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                if record.pop("schema_version") != SCHEMA_VERSION:
                    msg = "Unsupported episode schema version"
                    raise ValueError(msg)
                if not isinstance(record["answers"], list):
                    msg = "JSON answers must be an array"
                    raise ValueError(msg)
                record["answers"] = tuple(record["answers"])
                yield MemoryEpisode(**record)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                msg = f"Invalid episode in {path}, line {line_number}: {exc}"
                raise ValueError(msg) from exc


def write_synthetic_dataset(
    output_dir: str | Path,
    *,
    train_size: int = 10000,
    validation_size: int = 500,
    test_size: int = 500,
    config: SyntheticConfig | None = None,
) -> dict[str, int]:
    """Create three independent splits and a reproducibility manifest."""
    counts = dict(zip(SPLITS, (train_size, validation_size, test_size), strict=True))
    if min(counts.values()) < 0 or not sum(counts.values()):
        msg = "Split sizes must be nonnegative and at least one must be positive"
        raise ValueError(msg)
    config = config or SyntheticConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, count in counts.items():
        write_episodes(
            output_dir / f"{split}.jsonl", generate_synthetic(count, split=split, config=config)
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": "synthetic-v1",
        "config": asdict(config),
        "counts": counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic", help="Generate all three synthetic splits")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--train-size", type=int, default=10000)
    synthetic.add_argument("--validation-size", type=int, default=500)
    synthetic.add_argument("--test-size", type=int, default=500)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--num-entities", type=int, default=8)
    synthetic.add_argument("--distractor-sentences", type=int, default=128)
    synthetic.add_argument("--min-tail-sentences", type=int, default=32)
    synthetic.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    hf = commands.add_parser("huggingface", help="Download and normalize a selected HF split")
    hf.add_argument("--dataset", choices=("babilong", "qasper"), required=True)
    hf.add_argument("--output", type=Path, required=True)
    hf.add_argument("--split", choices=SPLITS, default="train")
    hf.add_argument("--limit", type=int, default=1000)
    hf.add_argument("--length", default="1k", help="BABILong context-length config")
    hf.add_argument("--task", default="qa1", help="BABILong task / HF split")
    hf.add_argument(
        "--revision",
        help="HF commit/tag/branch; defaults to main for BABILong, refs/convert/parquet for QASPER",
    )
    hf.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.command == "synthetic":
        result = write_synthetic_dataset(
            args.output,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            config=SyntheticConfig(
                seed=args.seed,
                num_entities=args.num_entities,
                distractor_sentences=args.distractor_sentences,
                min_tail_sentences=args.min_tail_sentences,
                tasks=tuple(args.tasks),
            ),
        )
    else:
        result = {
            "episodes": write_episodes(
                args.output,
                load_hf_episodes(
                    args.dataset,
                    split=args.split,
                    limit=args.limit,
                    length=args.length,
                    task=args.task,
                    revision=args.revision,
                    cache_dir=args.cache_dir,
                ),
            )
        }
    print(json.dumps({"output": str(args.output), **result}))


if __name__ == "__main__":
    main()
