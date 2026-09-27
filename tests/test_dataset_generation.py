# Keep exception checks compatible with the project's unittest runner.
# ruff: noqa: PT027

import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.llm.helpers.dataset_generation import (
    BABILONG_TEST,
    BABILONG_TRAIN,
    HELDOUT_SESSION_TASKS,
    SESSION_TASKS,
    MemoryEpisode,
    SessionSyntheticConfig,
    SyntheticConfig,
    generate_session_counterfactual_pairs,
    generate_session_synthetic,
    generate_synthetic,
    load_hf_episodes,
    read_episodes,
    write_episodes,
    write_session_counterfactual_dataset,
    write_session_synthetic_dataset,
    write_synthetic_dataset,
)
from src.llm.helpers.novel_value_dataset import NOVEL_VALUES, write_novel_value_dataset
from src.llm.helpers.retrieval_evaluation import lexical_fact


class SyntheticDatasetTests(unittest.TestCase):
    def test_answers_follow_the_context_and_corrections(self):
        for task in ("recall", "update", "multi_hop"):
            config = SyntheticConfig(
                seed=24, tasks=(task,), distractor_sentences=12, min_tail_sentences=5
            )
            for episode in generate_synthetic(20, config=config):
                with self.subTest(task=task, episode=episode.id):
                    codes, assignments, previous = {}, {}, {}
                    for sentence in episode.context.splitlines():
                        fact = re.fullmatch(
                            r"The access code for station (\w+) (?:is|has changed to) (\d+)\.",
                            sentence,
                        )
                        assignment = re.fullmatch(
                            r"Courier (\d+) is assigned to station (\w+)\.", sentence
                        )
                        if fact:
                            name, code = fact.groups()
                            if name in codes:
                                previous[name] = codes[name]
                            codes[name] = code
                        if assignment:
                            courier, name = assignment.groups()
                            assignments[courier] = name
                    if task == "multi_hop":
                        courier = re.search(r"courier (\d+)", episode.question).group(1)
                        target = assignments[courier]
                    else:
                        target = re.search(r"station (\w+)", episode.question).group(1)
                    assert episode.answers == (codes[target],)
                    if task == "update":
                        assert episode.answers[0] != previous[target]
                        assert len(previous) >= 2
                    spans = episode.metadata["support_spans"]
                    assert len(spans) == (1 if task == "recall" else 2)
                    for start, end in spans:
                        assert target in episode.context[start:end]
                    tail = episode.context[spans[-1][1] :].strip().splitlines()
                    assert len(tail) >= config.min_tail_sentences

    def test_reproducibility_resuming_and_split_isolation(self):
        config = SyntheticConfig(distractor_sentences=0, min_tail_sentences=0)
        train = list(generate_synthetic(10, config=config))
        assert train == list(generate_synthetic(10, config=config))
        assert train[7:] == list(generate_synthetic(3, start_index=7, config=config))
        validation = list(generate_synthetic(10, split="validation", config=config))
        test = list(generate_synthetic(10, split="test", config=config))
        episodes = [*train, *validation, *test]
        assert len({episode.id for episode in episodes}) == 30
        assert len({episode.context for episode in episodes}) == 30

    def test_configuration_errors(self):
        for kwargs in (
            {"num_entities": 1},
            {"distractor_sentences": -1},
            {"distractor_sentences": 0, "min_tail_sentences": 1},
            {"tasks": ()},
            {"tasks": ("unknown",)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SyntheticConfig(**kwargs)
        with self.assertRaises(ValueError):
            list(generate_synthetic(-1))

    def test_export_manifest_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            config = SyntheticConfig(distractor_sentences=2, min_tail_sentences=1)
            counts = write_synthetic_dataset(
                output, train_size=3, validation_size=2, test_size=1, config=config
            )
            assert json.loads((output / "manifest.json").read_text())["counts"] == counts
            for split, count in counts.items():
                assert list(read_episodes(output / f"{split}.jsonl")) == list(
                    generate_synthetic(count, split=split, config=config)
                )
            with self.assertRaises(FileExistsError):
                write_synthetic_dataset(output, train_size=1)


class SessionSyntheticDatasetTests(unittest.TestCase):
    def test_later_sessions_recall_latest_user_and_repository_facts(self):
        for task in SESSION_TASKS:
            config = SessionSyntheticConfig(seed=12, gap_turns=2, tasks=(task,))
            episode = next(generate_session_synthetic(1, config=config))
            with self.subTest(task=task):
                assert episode.prompt_style == "qwen_chat"
                assert not episode.supervise_eos
                assert episode.prompt.count("<|im_start|>user") == (
                    3 + config.gap_turns + int(task.endswith("update"))
                )
                assert episode.prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
                assert episode.answers[0] not in episode.question
                boundaries = episode.metadata["session_boundaries"]
                assert len(boundaries) == 3
                assert boundaries == sorted(boundaries)
                assert boundaries[-1] == len(episode.context)
                spans = episode.metadata["support_spans"]
                assert len(spans) == (2 if task.endswith("update") else 1)
                assert episode.answers[0] in episode.context[slice(*spans[-1])]
                assert spans[-1][1] < boundaries[-1]

    def test_reproducible_splits_and_roundtrip(self):
        config = SessionSyntheticConfig(seed=31, gap_turns=1)
        train = list(generate_session_synthetic(7, config=config))
        assert train == list(generate_session_synthetic(7, config=config))
        assert train[4:] == list(generate_session_synthetic(3, config=config, start_index=4))
        validation = list(generate_session_synthetic(7, split="validation", config=config))
        assert {episode.id for episode in train}.isdisjoint(episode.id for episode in validation)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sessions"
            counts = write_session_synthetic_dataset(
                output, train_size=7, validation_size=2, test_size=1, config=config
            )
            assert counts == {"train": 7, "validation": 2, "test": 1}
            assert json.loads((output / "manifest.json").read_text())["source"] == (
                "session-synthetic-v1"
            )
            assert list(read_episodes(output / "train.jsonl")) == train

    def test_configuration_errors(self):
        for kwargs in ({"gap_turns": -1}, {"tasks": ()}, {"tasks": ("unknown",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SessionSyntheticConfig(**kwargs)
        with self.assertRaises(ValueError):
            list(generate_session_synthetic(-1))


class SessionCounterfactualDatasetTests(unittest.TestCase):
    def test_lexical_retrieval_finds_latest_support(self):
        for task in ("repo_test_command_update", "age_update", "repo_deploy_region"):
            config = SessionSyntheticConfig(seed=23, gap_turns=7, tasks=(task,))
            first, second = list(generate_session_counterfactual_pairs(1, config=config))
            for episode in (first, second):
                with self.subTest(task=task, variant=episode.metadata["pair_variant"]):
                    start, end = episode.metadata["support_spans"][-1]
                    assert lexical_fact(episode) == episode.context[start:end]

    def test_novel_values_preserve_held_out_pair_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "novel"
            write_session_counterfactual_dataset(
                source,
                train_pairs=0,
                validation_pairs=14,
                test_pairs=2,
                config=SessionSyntheticConfig(seed=19, gap_turns=2),
            )
            assert write_novel_value_dataset(source, output) == {
                "train": 0,
                "validation": 28,
                "test": 4,
            }
            old = list(read_episodes(source / "validation.jsonl"))
            new = list(read_episodes(output / "validation.jsonl"))
            for before, after in zip(old, new, strict=True):
                assert after.answers[0] in NOVEL_VALUES[before.metadata["task"]]
                assert after.answers != before.answers
                assert after.question == before.question
                start, end = before.metadata["support_spans"][-1]
                new_start, new_end = after.metadata["support_spans"][-1]
                assert before.context[:start] == after.context[:new_start]
                assert before.context[end:] == after.context[new_end:]

    def test_unseen_fact_categories_are_available_outside_training_defaults(self):
        assert set(HELDOUT_SESSION_TASKS).isdisjoint(SESSION_TASKS)
        for task in HELDOUT_SESSION_TASKS:
            config = SessionSyntheticConfig(seed=37, gap_turns=2, tasks=(task,))
            first, second = list(generate_session_counterfactual_pairs(1, config=config))
            with self.subTest(task=task):
                assert first.answers != second.answers
                assert first.question == second.question
                for episode in (first, second):
                    start, end = episode.metadata["support_spans"][-1]
                    assert episode.answers[0] in episode.context[start:end]

    def test_pairs_change_only_the_latest_support_and_answer(self):
        for task in SESSION_TASKS:
            config = SessionSyntheticConfig(seed=49, gap_turns=2, tasks=(task,))
            first, second = list(generate_session_counterfactual_pairs(1, config=config))
            with self.subTest(task=task):
                assert first.question == second.question
                assert first.answers != second.answers
                assert first.metadata["pair_id"] == second.metadata["pair_id"]
                assert first.metadata["pair_variant"] == 0
                assert second.metadata["pair_variant"] == 1
                assert not first.supervise_eos
                assert not second.supervise_eos
                first_start, first_end = first.metadata["support_spans"][-1]
                second_start, second_end = second.metadata["support_spans"][-1]
                assert first.context[:first_start] == second.context[:second_start]
                assert first.context[first_end:] == second.context[second_end:]
                assert first.answers[0] in first.context[first_start:first_end]
                assert second.answers[0] in second.context[second_start:second_end]
                assert first.metadata["session_boundaries"][-1] == len(first.context)
                assert second.metadata["session_boundaries"][-1] == len(second.context)

    def test_pair_manifest_and_split_isolation(self):
        config = SessionSyntheticConfig(seed=4, gap_turns=1)
        train = list(generate_session_counterfactual_pairs(3, config=config))
        assert train == list(generate_session_counterfactual_pairs(3, config=config))
        assert train[2:] == list(
            generate_session_counterfactual_pairs(2, start_index=1, config=config)
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairs"
            counts = write_session_counterfactual_dataset(
                output, train_pairs=3, validation_pairs=2, test_pairs=1, config=config
            )
            assert counts == {"train": 6, "validation": 4, "test": 2}
            assert json.loads((output / "manifest.json").read_text())["pair_counts"] == {
                "train": 3,
                "validation": 2,
                "test": 1,
            }
            assert list(read_episodes(output / "train.jsonl")) == train
            validation = list(read_episodes(output / "validation.jsonl"))
            assert {episode.id for episode in train}.isdisjoint(
                episode.id for episode in validation
            )


class EpisodeIOTests(unittest.TestCase):
    def test_answer_span_with_repeated_answer_and_unicode(self):
        episode = MemoryEpisode(
            "id", 'The word is "café".', "Which word?", ('"café"',), "test", "train"
        )
        example = episode.training_example()
        start, end = example["answer_span"]
        assert example["text"][start:end] == episode.answers[0]
        assert example["text"][:start] == episode.prompt
        assert "answers" not in episode.prompt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            assert write_episodes(path, [episode]) == 1
            assert list(read_episodes(path)) == [episode]
            with self.assertRaises(FileExistsError):
                write_episodes(path, [])
            assert list(read_episodes(path)) == [episode]

    def test_failed_export_does_not_publish_a_partial_dataset(self):
        def broken_source():
            yield next(generate_synthetic(1))
            msg = "Source interrupted"
            raise RuntimeError(msg)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            with self.assertRaises(RuntimeError):
                write_episodes(path, broken_source())
            assert list(Path(directory).iterdir()) == []

    def test_invalid_jsonl_reports_the_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_episodes(path, generate_synthetic(1))
            with path.open("a") as handle:
                handle.write('{"schema_version": 99}\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                list(read_episodes(path))


def qasper_row():
    def annotation(**fields):
        answer = {
            "unanswerable": False,
            "extractive_spans": [],
            "free_form_answer": "",
            "yes_no": None,
        }
        return {"answer": {**answer, **fields}}

    return {
        "id": "paper-1",
        "title": "A paper",
        "abstract": "An abstract",
        "full_text": {
            "section_name": ["Results"],
            "paragraphs": [["First paragraph", "Second paragraph"]],
        },
        "figures_and_tables": {"caption": ["A caption"], "file": ["figure.png"]},
        "qas": {
            "question_id": ["q1", "q2", "q3", "q4", "q5"],
            "question": ["What?", "Where?", "Does it?", "Missing?", "No labels?"],
            "answers": [
                {
                    "answer": [
                        annotation(free_form_answer="Result A")["answer"],
                        annotation(free_form_answer="Result B")["answer"],
                    ]
                },
                [annotation(extractive_spans=["First", "Second"])],
                [annotation(yes_no=False)],
                [annotation(unanswerable=True)],
                [],
            ],
        },
    }


class HuggingFaceDatasetTests(unittest.TestCase):
    def _load(self, dataset, rows, **kwargs):
        loader = Mock(return_value=iter(rows))
        api = Mock()
        api.dataset_info.return_value.sha = "fixed-commit"
        api.dataset_info.return_value.siblings = [
            SimpleNamespace(rfilename=name)
            for name in (
                "data/qa1/1k.json",
                "data/qa2/4k.json",
                "qasper/train/0000.parquet",
                "qasper/validation/0000.parquet",
            )
        ]
        # These tests require neither the optional datasets package nor network access.
        with (
            patch.dict("sys.modules", {"datasets": SimpleNamespace(load_dataset=loader)}),
            patch("huggingface_hub.HfApi", return_value=api),
        ):
            result = list(load_hf_episodes(dataset, **kwargs))
        return result, loader, api

    def test_babilong_routes_splits_and_limits_stream_consumption(self):
        def rows():
            yield {
                "input": "Mary moved to the office.",
                "question": "Where is Mary?",
                "target": "office",
            }
            self.fail("Read past the requested episode limit")

        for split, repo in (("train", BABILONG_TRAIN), ("test", BABILONG_TEST)):
            result, loader, api = self._load(
                "babilong", rows(), split=split, length="4k", task="qa2", limit=1, revision="v1"
            )
            assert result[0].answers == ("office",)
            assert result[0].source == repo
            assert result[0].split == split
            assert result[0].metadata["revision"] == "fixed-commit"
            api.dataset_info.assert_called_once_with(repo, revision="v1")
            loader.assert_called_once_with(
                "json",
                data_files={"qa2": [f"hf://datasets/{repo}@fixed-commit/data/qa2/4k.json"]},
                split="qa2",
                streaming=True,
                cache_dir=None,
            )

    def test_qasper_answers_and_document_grouping(self):
        result, loader, _ = self._load("qasper", [qasper_row()], split="validation", limit=10)
        assert [episode.answers for episode in result] == [
            ("Result A", "Result B"),
            ("First, Second",),
            ("no",),
            ("unanswerable",),
        ]
        assert loader.call_args.kwargs["split"] == "validation"
        for episode in result:
            assert episode.metadata["document_id"] == "paper-1"
            assert episode.split == "validation"
            assert "First paragraph\n\nSecond paragraph" in episode.context
            assert "A caption" in episode.context
            assert "Result A" not in episode.context

    def test_hf_validation_before_network_access(self):
        for kwargs in ({"limit": 0}, {"split": "validation"}, {"split": "invalid"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                list(load_hf_episodes("babilong", **kwargs))


if __name__ == "__main__":
    unittest.main()
