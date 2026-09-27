"""Replace held-out paired answers with values absent from training pools.

The questions, users, distractors and fact positions stay fixed, isolating
whether an adapter copies a new value rather than selecting familiar answers.
"""

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

from src.llm.helpers.dataset_generation import read_episodes, write_episodes

NOVEL_VALUES = {
    "name": ("Lena", "Omar", "Priya", "Noah", "Mina", "Theo"),
    "age": tuple(str(value) for value in range(81, 100)),
    "age_update": tuple(str(value) for value in range(81, 100)),
    "preference": ("Portuguese", "Swedish", "Hindi", "Japanese", "Korean", "Turkish"),
    "repo_entry": tuple(f"src/{part}/main.py" for part in ("billing", "search", "queue", "cache")),
    "repo_update": tuple(f"src/{part}/main.py" for part in ("billing", "search", "queue", "cache")),
    "timezone": ("Lisbon", "Seoul", "Auckland", "Nairobi", "Vancouver", "Reykjavik"),
    "editor": ("Helix", "Fleet", "Kate", "Micro", "Sublime Text"),
    "current_project": ("Caldera", "Horizon", "Solstice", "Polaris", "Summit"),
    "repo_runtime": ("Kotlin", "Elixir", "Scala", "PHP", "Swift"),
    "repo_test_command": ("npm test", "pnpm test", "ctest", "mvn test", "dotnet test"),
    "repo_test_command_update": (
        "npm test",
        "pnpm test",
        "ctest",
        "mvn test",
        "dotnet test",
    ),
    "repo_config_file": (
        "config/cache.yaml",
        "settings/api.toml",
        "conf/worker.json",
        "config/queue.ini",
        "settings/web.yaml",
    ),
    "repo_build_tool": ("Ninja", "Ant", "Meson", "Buck", "SCons"),
}


def novel_pair(first, second, *, seed: int):
    """Replace one earlier supporting fact in both variants of a held-out pair."""
    if (
        first.metadata.get("pair_id") != second.metadata.get("pair_id")
        or first.metadata.get("pair_variant") != 0
        or second.metadata.get("pair_variant") != 1
        or first.metadata.get("task") != second.metadata.get("task")
    ):
        msg = "Expected adjacent variants of one paired example"
        raise ValueError(msg)
    task = first.metadata["task"]
    digest = hashlib.sha256(
        f"novel-v1:{seed}:{first.split}:{first.metadata['pair_id']}".encode()
    ).hexdigest()
    rng = random.Random(digest)
    values = rng.sample(NOVEL_VALUES[task], 2)
    variants = []
    for variant, value in enumerate(values):
        metadata = copy.deepcopy(first.metadata)
        start, end = metadata["support_spans"][-1]
        support = first.context[start:end]
        old = first.answers[0]
        if support.count(old) != 1:
            msg = "Expected one answer mention in the latest supporting fact"
            raise ValueError(msg)
        replacement = support.replace(old, value, 1)
        delta = len(replacement) - len(support)
        metadata["support_spans"][-1][1] += delta
        metadata["session_boundaries"] = [
            boundary + delta if boundary > start else boundary
            for boundary in metadata["session_boundaries"]
        ]
        metadata.update(pair_id=digest[:24], pair_variant=variant, value_set="novel")
        variants.append(
            replace(
                first,
                id=f"session-novel-{digest[:24]}-{variant}",
                context=first.context[:start] + replacement + first.context[end:],
                answers=(value,),
                source="session-counterfactual-novel-values-v1",
                metadata=metadata,
            )
        )
    return tuple(variants)


def write_novel_value_dataset(source: Path, output: Path, *, seed: int = 71) -> dict:
    """Transform only held-out splits; never synthesize training examples here."""
    output.mkdir(parents=True, exist_ok=False)
    counts = {"train": write_episodes(output / "train.jsonl", ())}
    for split in ("validation", "test"):
        episodes = list(read_episodes(source / f"{split}.jsonl"))
        if len(episodes) % 2:
            msg = f"{split} must contain complete adjacent pairs"
            raise ValueError(msg)
        transformed = (
            episode
            for first, second in zip(episodes[::2], episodes[1::2], strict=True)
            for episode in novel_pair(first, second, seed=seed)
        )
        counts[split] = write_episodes(output / f"{split}.jsonl", transformed)
    manifest = {
        "schema_version": 1,
        "source": "session-counterfactual-novel-values-v1",
        "parent_dataset": str(source),
        "seed": seed,
        "counts": counts,
        "value_pools": {task: list(values) for task, values in NOVEL_VALUES.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    print(json.dumps(write_novel_value_dataset(args.source, args.output, seed=args.seed)))


if __name__ == "__main__":
    main()
