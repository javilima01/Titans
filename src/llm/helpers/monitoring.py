"""Live training monitor: a JSONL event log plus periodically regenerated loss charts."""

import json
import math
from pathlib import Path
import time


def _format_seconds(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"


def _finite(points):
    return [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]


class TrainingMonitor:
    """Track step/validation losses, appending JSONL events and re-rendering PNG charts.

    The PNG is rewritten atomically as training progresses, so opening it in an
    image viewer that reloads changed files (Preview, browsers) gives live
    monitoring. Renders are time-throttled; finish() always renders and can
    move the report files into the checkpoint directory.
    """

    def __init__(self, image_path, *, title: str, min_render_interval: float = 2.0):
        self.image_path = Path(image_path)
        self.log_path = self.image_path.with_suffix(".jsonl")
        self.title = title
        self.min_render_interval = min_render_interval
        self.steps = []  # {"step", "epoch", "loss"}
        self.validations = []  # {"epoch", "step", "val_loss"}
        self.status = "running"
        self.started = time.time()
        self._last_render = 0.0
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self._append_log({"type": "started", "title": title})
        self.render()

    def _append_log(self, record):
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": round(time.time(), 3), **record}) + "\n")

    def record_step(self, event):
        """Event keys: step, epoch, loss."""
        self.steps.append({"step": event["step"], "epoch": event["epoch"], "loss": event["loss"]})
        self._append_log({"type": "step", **event})
        if time.monotonic() - self._last_render >= self.min_render_interval:
            self.render()

    def record_validation(self, event):
        """Event keys: epoch, step, val_loss."""
        self.validations.append(
            {"epoch": event["epoch"], "step": event["step"], "val_loss": event["val_loss"]}
        )
        self._append_log({"type": "validation", **event})
        self.render()

    def finish(self, status: str = "finished", move_into=None):
        self.status = status
        self._append_log({"type": status})
        self.render()
        if move_into is not None:
            target = Path(move_into)
            self.image_path.replace(target / "training.png")
            self.log_path.replace(target / "training_log.jsonl")
            self.image_path = target / "training.png"
            self.log_path = target / "training_log.jsonl"

    def render(self):
        import matplotlib as mpl  # Deferred: keeps CLI startup and --help fast.

        mpl.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, (step_axis, epoch_axis) = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle(f"{self.title} — {self.status}")

        train_points = _finite((s["step"], s["loss"]) for s in self.steps)
        validation_points = _finite((v["step"], v["val_loss"]) for v in self.validations)
        if train_points:
            step_axis.plot(
                *zip(*train_points, strict=True), color="#2563eb", linewidth=1.2, label="train"
            )
        if validation_points:
            step_axis.plot(
                *zip(*validation_points, strict=True),
                "o--",
                color="#dc2626",
                label="validation",
            )
        step_axis.set(
            xlabel="optimizer step", ylabel="loss", title="Training and validation loss per step"
        )

        epoch_means = []
        for epoch in sorted({s["epoch"] for s in self.steps}):
            values = [
                s["loss"] for s in self.steps if s["epoch"] == epoch and math.isfinite(s["loss"])
            ]
            if values:
                epoch_means.append((epoch, sum(values) / len(values)))
        validation_epochs = _finite((v["epoch"], v["val_loss"]) for v in self.validations)
        if epoch_means:
            epoch_axis.plot(
                *zip(*epoch_means, strict=True), "o-", color="#2563eb", label="train (epoch mean)"
            )
        if validation_epochs:
            epoch_axis.plot(
                *zip(*validation_epochs, strict=True), "o-", color="#dc2626", label="validation"
            )
        epoch_axis.set(xlabel="epoch", ylabel="loss", title="Training vs validation loss per epoch")
        epoch_axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        for axis in (step_axis, epoch_axis):
            axis.grid(True, alpha=0.3)
            if axis.lines:
                axis.legend()
            else:
                axis.text(0.5, 0.5, "waiting for data", transform=axis.transAxes, ha="center")

        latest = self.steps[-1] if self.steps else None
        latest_validation = self.validations[-1] if self.validations else None
        finite_validations = [
            v["val_loss"] for v in self.validations if math.isfinite(v["val_loss"])
        ]
        summary = "  ".join(
            [
                f"elapsed {_format_seconds(time.time() - self.started)}",
                f"step {latest['step']} (epoch {latest['epoch']})" if latest else "step 0",
                f"train loss {latest['loss']:.6g}" if latest else "train loss —",
                (
                    f"validation loss {latest_validation['val_loss']:.6g}"
                    if latest_validation
                    else "validation loss —"
                ),
                f"best validation {min(finite_validations):.6g}" if finite_validations else "",
            ]
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.text(0.02, 0.015, summary, fontsize=8, family="monospace")

        temporary = self.image_path.with_suffix(".tmp.png")
        try:
            fig.savefig(temporary, dpi=110)
        finally:
            plt.close(fig)
        temporary.replace(self.image_path)
        self._last_render = time.monotonic()
