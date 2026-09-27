import json
from pathlib import Path
import tempfile
import unittest

from src.llm.helpers.dataset_generation import (
    SESSION_TASKS,
    SessionSyntheticConfig,
    generate_session_counterfactual_pairs,
    read_episodes,
    write_episodes,
)
from src.llm.helpers.novel_value_dataset import NOVEL_VALUES
from src.llm.helpers.open_value_dataset import _replace_pair, write_open_value_dataset


class OpenValueDatasetTests(unittest.TestCase):
    def test_every_task_keeps_pair_context_and_heldout_answers_separate(self):
        for task in SESSION_TASKS:
            base = list(
                generate_session_counterfactual_pairs(
                    1, config=SessionSyntheticConfig(seed=61, gap_turns=7, tasks=(task,))
                )
            )
            first, second = _replace_pair(*base, pair_index=17)
            assert first.question == second.question
            assert first.answers != second.answers
            assert first.answers[0] not in NOVEL_VALUES[task]
            assert second.answers[0] not in NOVEL_VALUES[task]
            for episode in (first, second):
                start, end = episode.metadata["support_spans"][-1]
                assert episode.answers[0] in episode.context[start:end]
                assert (
                    episode.context[end:]
                    == base[0].context[base[0].metadata["support_spans"][-1][1] :]
                )

    def test_write_records_counts_and_preserves_heldout_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heldout = root / "heldout"
            heldout.mkdir()
            for split in ("validation", "test"):
                write_episodes(
                    heldout / f"{split}.jsonl",
                    generate_session_counterfactual_pairs(1, split=split),
                )
            counts = write_open_value_dataset(root / "open", heldout, pair_count=3)
            assert counts == {"train": 6, "validation": 2, "test": 2}
            manifest = json.loads((root / "open" / "manifest.json").read_text())
            assert manifest["counts"] == counts
            for split in ("validation", "test"):
                assert list(read_episodes(root / "open" / f"{split}.jsonl")) == list(
                    read_episodes(heldout / f"{split}.jsonl")
                )


if __name__ == "__main__":
    unittest.main()
