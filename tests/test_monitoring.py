"""Unit checks for the live training monitor (PNG charts + JSONL event log)."""

import json
from pathlib import Path
import tempfile
import unittest

from src.llm.helpers.monitoring import TrainingMonitor


class TrainingMonitorTests(unittest.TestCase):
    def test_events_render_charts_and_move_into_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            monitor = TrainingMonitor(root / "run.training.png", title="run", min_render_interval=0)
            assert (root / "run.training.png").read_bytes()[:4] == b"\x89PNG"
            monitor.record_step({"step": 1, "epoch": 1, "loss": 2.5, "target_tokens": 7})
            monitor.record_step({"step": 2, "epoch": 1, "loss": 1.5, "target_tokens": 9})
            monitor.record_validation(
                {"epoch": 1, "step": 2, "val_loss": 1.75, "target_tokens": 11}
            )
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            monitor.finish("interrupted", move_into=checkpoint)
            image = checkpoint / "training.png"
            log = checkpoint / "training_log.jsonl"
            assert image.read_bytes()[:4] == b"\x89PNG"
            assert not (root / "run.training.png").exists()
            assert monitor.image_path == image
            assert monitor.log_path == log
            records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            assert [record["type"] for record in records] == [
                "started",
                "step",
                "step",
                "validation",
                "interrupted",
            ]
            assert records[1]["loss"] == 2.5
            assert records[3]["val_loss"] == 1.75
            assert all("time" in record for record in records)

    def test_render_tolerates_empty_and_nonfinite_history(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMonitor(Path(directory) / "run.png", title="run")
            monitor.render()  # No data yet.
            monitor.record_step({"step": 1, "epoch": 1, "loss": float("inf")})
            monitor.record_step({"step": 2, "epoch": 1, "loss": 1.0})
            monitor.finish()
            assert Path(monitor.image_path).stat().st_size > 0


if __name__ == "__main__":
    unittest.main()
