import unittest

from src.llm.helpers.dataset_generation import MemoryEpisode
from src.llm.helpers.repetition_probe import demonstration, question_only_prompt


class RepetitionProbeTests(unittest.TestCase):
    def test_question_session_contains_no_demonstrated_answer(self):
        episode = MemoryEpisode(
            id="one",
            context="<|im_start|>user\nMy code is ZXQ42.<|im_end|>\n",
            question="What is my code?",
            answers=("ZXQ42",),
            source="unit",
            split="validation",
            prompt_style="qwen_chat",
        )
        shown = demonstration(episode) * 5
        assert shown.count("ZXQ42") == 5
        for include_system in (False, True):
            prompt = question_only_prompt(episode, include_system=include_system)
            assert "ZXQ42" not in prompt
            assert "What is my code?" in prompt


if __name__ == "__main__":
    unittest.main()
