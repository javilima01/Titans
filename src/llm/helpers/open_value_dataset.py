"""Build paired cross-session episodes with many training-only answer values.

Questions and trailing conversation stay paired. Only the earlier supporting
assertion changes, so a model must use prior context to distinguish variants.
"""

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

from src.llm.helpers.dataset_generation import (
    SESSION_TASKS,
    SessionSyntheticConfig,
    generate_session_counterfactual_pairs,
    read_episodes,
    write_episodes,
)
from src.llm.helpers.novel_value_dataset import NOVEL_VALUES

STEMS = (
    "Ari",
    "Bela",
    "Cali",
    "Demi",
    "Eri",
    "Faro",
    "Gali",
    "Havi",
    "Ira",
    "Jori",
    "Kali",
    "Lumi",
    "Meri",
    "Navi",
    "Ori",
    "Pali",
)
ENDS = (
    "den",
    "fina",
    "gora",
    "hira",
    "jora",
    "kavi",
    "lora",
    "miri",
    "nari",
    "pira",
    "ravi",
    "sela",
    "tavi",
    "vori",
    "wena",
    "zari",
)


def _value(task: str, number: int) -> str:
    token = STEMS[number % len(STEMS)] + ENDS[(number // len(STEMS)) % len(ENDS)]
    if task in ("age", "age_update"):
        return str(18 + number % 62)
    if task in ("repo_entry", "repo_update"):
        return f"src/{token.lower()}/main.py"
    if task == "repo_config_file":
        return f"config/{token.lower()}.yaml"
    if task in ("repo_test_command", "repo_test_command_update"):
        return f"python -m {token.lower()}"
    return token


def _replace_pair(first, second, pair_index: int):
    task = first.metadata["task"]
    if first.metadata["pair_id"] != second.metadata["pair_id"]:
        msg = "Expected adjacent variants of one paired episode"
        raise ValueError(msg)
    variants = []
    for variant in range(2):
        value = _value(task, pair_index * 2 + variant)
        if value in NOVEL_VALUES[task]:
            msg = f"Training answer overlaps held-out values: {value}"
            raise ValueError(msg)
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
        metadata.update(pair_variant=variant, value_set="open_train")
        variants.append(
            replace(
                first,
                id=f"open-value-{pair_index}-{variant}",
                context=first.context[:start] + replacement + first.context[end:],
                answers=(value,),
                source="session-open-values-v1",
                metadata=metadata,
            )
        )
    return variants


def write_open_value_dataset(output: Path, heldout: Path, *, pair_count: int = 2000) -> dict:
    if pair_count < 1:
        msg = "pair_count must be positive"
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=False)
    config = SessionSyntheticConfig(seed=61, gap_turns=7, tasks=SESSION_TASKS)
    base = generate_session_counterfactual_pairs(pair_count, split="train", config=config)

    def examples():
        for index in range(pair_count):
            yield from _replace_pair(next(base), next(base), index)

    counts = {"train": write_episodes(output / "train.jsonl", examples())}
    for split in ("validation", "test"):
        counts[split] = write_episodes(
            output / f"{split}.jsonl", read_episodes(heldout / f"{split}.jsonl")
        )
    manifest = {
        "schema_version": 1,
        "source": "session-open-values-v1",
        "pair_count": pair_count,
        "config": asdict(config),
        "heldout_source": str(heldout),
        "heldout_values": {task: list(values) for task, values in NOVEL_VALUES.items()},
        "counts": counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--pair-count", type=int, default=2000)
    args = parser.parse_args()
    print(
        json.dumps(write_open_value_dataset(args.output, args.heldout, pair_count=args.pair_count))
    )


if __name__ == "__main__":
    main()
