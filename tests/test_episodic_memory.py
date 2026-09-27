from pathlib import Path
import tempfile
import unittest

from src.llm.modules.episodic_memory import EpisodicMemory, prompt_with_evidence


class EpisodicMemoryTests(unittest.TestCase):
    def test_retrieves_separate_facts_and_later_correction_after_reload(self):
        memory = EpisodicMemory()
        assert (
            memory.remember(
                "The build tool for repository alpha is Bazel. "
                "The build tool for repository beta is Meson."
            )
            == 2
        )
        assert memory.remember("Correction: the build tool for repository alpha is Ninja.") == 1
        assert memory.remember("Which build tool does repository alpha use?") == 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-memory.json"
            memory.save(path)
            assert path.stat().st_mode & 0o777 == 0o600
            restored = EpisodicMemory.load(path)
            assert restored.records == memory.records
            alpha = restored.retrieve("Which build tool does repository alpha use?", limit=2)
            latest_alpha = restored.retrieve("Which build tool does repository alpha use?", limit=1)
            beta = restored.retrieve("Which build tool does repository beta use?", limit=1)
            assert all("Bazel" not in record.text for record in alpha)
            assert any("Ninja" in record.text for record in alpha)
            assert [record.text for record in beta] == [
                "The build tool for repository beta is Meson."
            ]
            assert [record.text for record in latest_alpha] == [
                "Correction: the build tool for repository alpha is Ninja."
            ]
            prompt = prompt_with_evidence("Which build tool does repository alpha use?", alpha)
            assert "Bazel" not in prompt
            assert prompt.index("Ninja") < prompt.index("Current message")


if __name__ == "__main__":
    unittest.main()
