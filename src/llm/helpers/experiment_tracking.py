"""Summarize saved training/evaluation artifacts under experiments/.

Run ``python -m src.llm.helpers.experiment_tracking --checkpoint PATH`` after
training or evaluation. Re-running updates the same record with newly saved
reports while preserving its note, command, and manually entered result.
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "experiments"
MODEL_JSON_FILES = {
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
}


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_events(path: Path) -> dict:
    if not path.exists():
        return {}
    events = [_read_json_line(line) for line in path.read_text(encoding="utf-8").splitlines()]
    steps = [event for event in events if event.get("type") == "step"]
    validation = [event for event in events if event.get("type") == "validation"]
    result = {"steps": len(steps)}
    if steps:
        for label, sample in (("first_50", steps[:50]), ("last_50", steps[-50:])):
            count = sum(event["target_tokens"] for event in sample)
            result[f"loss_{label}"] = (
                sum(event["loss"] * event["target_tokens"] for event in sample) / count
            )
    if validation:
        result["last_validation_loss"] = validation[-1]["val_loss"]
    times = [event["time"] for event in events if "time" in event]
    if times:
        result["started_at"] = datetime.fromtimestamp(min(times), UTC).isoformat()
        result["ended_at"] = datetime.fromtimestamp(max(times), UTC).isoformat()
    return result


def _read_json_line(line: str) -> dict:
    return json.loads(line)


def _reports(
    checkpoint: Path, additional_reports: tuple[Path, ...], *, include_checkpoint: bool = True
) -> tuple[dict, list[str]]:
    results = {}
    supplementary = []
    checkpoint_reports = set(checkpoint.glob("*.json")) if include_checkpoint else set()
    for path in sorted(checkpoint_reports | set(additional_reports)):
        if path.name in MODEL_JSON_FILES:
            continue
        try:
            report = _read_json(path)
        except json.JSONDecodeError, UnicodeDecodeError:
            continue
        if "evaluations" not in report:
            supplementary.append(_display_path(path))
            continue
        modes = {}
        for mode, evaluation in report["evaluations"].items():
            modes[mode] = {
                "metrics": evaluation.get("metrics"),
                "paired_metrics": evaluation.get("paired_metrics"),
                "by_task": evaluation.get("by_task"),
            }
        data_source = report.get("data_source")
        manifest_path = ROOT / data_source / "manifest.json" if data_source else None
        results[path.name] = {
            "path": _display_path(path),
            "split": report.get("split"),
            "data_source": data_source,
            "data_manifest": _read_json(manifest_path)
            if manifest_path and manifest_path.exists()
            else None,
            "evaluation_kind": report.get("evaluation_kind", "full_prompt"),
            "evaluation_options": report.get("evaluation_options"),
            "evaluated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            "window_size": report["evaluations"].get("normal", {}).get("window_size"),
            "max_new_tokens": report["evaluations"].get("normal", {}).get("max_new_tokens"),
            "evaluations": modes,
        }
    return results, supplementary


def _write_readme(folder: Path, record: dict) -> None:
    checkpoint = record["checkpoint"]
    training = record["training"]
    lines = [f"# {record['id']}", ""]
    if record["note"]:
        lines.extend([record["note"], ""])
    if record["result"]:
        lines.extend([f"Result: {record['result']}", ""])
    lines.extend(
        [
            f"- Checkpoint: `{checkpoint}`",
            f"- Parent checkpoint: `{record['parent_checkpoint'] or 'none'}`",
            f"- Source data: `{record['source']}`",
            f"- Adapter: `{json.dumps(record['adapter'], sort_keys=True)}`",
            f"- Training: `{json.dumps(training, sort_keys=True)}`",
            "",
        ]
    )
    if record["command"]:
        lines.extend(["## Command", "", "```sh", record["command"], "```", ""])
    if record["reports"]:
        lines.extend(["## Evaluations", ""])
        for name, report in record["reports"].items():
            details = f"{report['split']}; {report.get('evaluation_kind', 'full_prompt')}"
            if report.get("data_source"):
                details += f"; data `{report['data_source']}`"
            lines.append(f"- `{name}` ({details}):")
            for mode, values in report["evaluations"].items():
                metrics = values["metrics"] or {}
                pair = values["paired_metrics"] or {}
                lines.append(
                    f"  - {mode}: EM {metrics.get('exact_match', 'n/a')}, "
                    f"F1 {metrics.get('f1', 'n/a')}, "
                    f"both pairs {pair.get('both_correct', 'n/a')}/{pair.get('pairs', 'n/a')}"
                )
            primary = report["evaluations"].get("normal") or report["evaluations"].get("state_only")
            if primary and primary.get("by_task"):
                task_results = ", ".join(
                    f"{task} {round(values['exact_match'] * values['count'])}/{values['count']}"
                    for task, values in sorted(primary["by_task"].items())
                )
                lines.append(f"  - By task: {task_results}")
        lines.append("")
    if record["supplementary_reports"]:
        lines.extend(["Other reports:", ""])
        lines.extend(f"- `{path}`" for path in record["supplementary_reports"])
        lines.append("")
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_index(output_root: Path) -> None:
    records = [_read_json(path) for path in output_root.glob("*/record.json")]
    records.sort(key=lambda record: (record["training"].get("started_at", ""), record["id"]))
    lines = [
        "# Memory experiments",
        "",
        "Each folder records the configuration, dataset, checkpoint, and available evaluations.",
        "Re-run `python -m src.llm.helpers.experiment_tracking --checkpoint PATH` after",
        "adding an evaluation report. The checkpoint and dataset artifacts are kept at their",
        "listed paths; compact metrics are copied into each `record.json`.",
        "Older checkpoints predate complete CLI option logging, so some original flags",
        "cannot be recovered from their saved artifacts.",
        "See [research notes](RESEARCH.md) for the benchmark interpretation and next checks.",
        "",
        "| Experiment | Steps | Evaluation | EM | Solved pairs | Finding |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for record in records:
        candidates = (
            [
                report
                for report in record["reports"].values()
                if report["split"] == "validation"
                and "normal" in report["evaluations"]
                and report.get("data_source") in (None, record["source"])
            ]
            if record.get("kind") == "checkpoint"
            else list(record["reports"].values())
        )
        latest = (
            max(candidates, key=lambda report: report.get("evaluated_at", "")) if candidates else {}
        )
        modes = latest.get("evaluations", {})
        mode = next(
            (
                choice
                for choice in ("normal", "state_only", "lexical", "disabled", "reset", "oracle")
                if choice in modes
            ),
            None,
        )
        result = modes.get(mode, {})
        evaluation = f"{latest['split']}/{mode}" if mode else "n/a"
        metrics = result.get("metrics") or {}
        paired = result.get("paired_metrics") or {}
        count = metrics.get("count")
        exact_match = metrics.get("exact_match")
        accuracy = (
            f"{round(count * exact_match)}/{count} ({exact_match:.1%})"
            if count is not None and exact_match is not None
            else "n/a"
        )
        solved_pairs = f"{paired['both_correct']}/{paired['pairs']}" if paired else "n/a"
        finding_text = record["result"] if not metrics and record["result"] else record["note"]
        finding = finding_text.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| [{record['id']}]({record['id']}/README.md) | "
            f"{record['training'].get('steps', 'n/a') if record.get('kind') == 'checkpoint' else 'n/a'} | "
            f"{evaluation} | "
            f"{accuracy} | "
            f"{solved_pairs} | "
            f"{finding} |"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_experiment(
    checkpoint: str | Path,
    *,
    experiment_id: str | None = None,
    note: str | None = None,
    command: str | None = None,
    result: str | None = None,
    parent_checkpoint: str | None = None,
    kind: str | None = None,
    output_root: str | Path = EXPERIMENTS,
    additional_reports: tuple[Path, ...] = (),
) -> Path:
    checkpoint = Path(checkpoint)
    config = _read_json(checkpoint / "adapter_config.json")
    experiment_id = experiment_id or checkpoint.name
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", experiment_id):
        msg = "Experiment id must be a lowercase path-safe slug"
        raise ValueError(msg)
    output_root = Path(output_root)
    folder = output_root / experiment_id
    folder.mkdir(parents=True, exist_ok=True)
    existing = _read_json(folder / "record.json") if (folder / "record.json").exists() else {}
    resolved_kind = kind or existing.get("kind", "checkpoint")
    reports, supplementary = _reports(
        checkpoint, additional_reports, include_checkpoint=resolved_kind == "checkpoint"
    )
    reports = {**existing.get("reports", {}), **reports}
    supplementary = sorted(
        path
        for path in set(existing.get("supplementary_reports", [])) | set(supplementary)
        if Path(path).name not in MODEL_JSON_FILES
    )
    metadata = config.get("metadata", {})
    source = metadata.get("source", "unknown")
    source_path = ROOT / source
    manifest = (
        _read_json(source_path / "manifest.json")
        if source_path.is_dir() and (source_path / "manifest.json").exists()
        else None
    )
    training = {
        **_training_events(checkpoint / "training_log.jsonl"),
        "optimizer_steps": metadata.get("optimizer_steps"),
        "episodes": metadata.get("episodes"),
        "seed": metadata.get("seed"),
        "interrupted": metadata.get("interrupted"),
        "last_validation_loss": metadata["validation"][-1]["val_loss"]
        if metadata.get("validation")
        else None,
        "options": metadata.get("training_options"),
        "supervise_eos": metadata.get("supervise_eos"),
        **config.get("training", {}),
    }
    record = {
        "id": experiment_id,
        "kind": resolved_kind,
        "checkpoint": _display_path(checkpoint),
        "base_model": config.get("base_model"),
        "adapter": config.get("adapter", {}),
        "source": source,
        "parent_checkpoint": (
            parent_checkpoint
            if parent_checkpoint is not None
            else metadata.get("parent_checkpoint") or existing.get("parent_checkpoint")
        ),
        "dataset_manifest": manifest,
        "training": training,
        "reports": reports,
        "supplementary_reports": supplementary,
        "note": note if note is not None else existing.get("note", ""),
        "command": command if command is not None else existing.get("command", ""),
        "result": result if result is not None else existing.get("result", ""),
    }
    (folder / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _write_readme(folder, record)
    _write_index(output_root)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--id", dest="experiment_id")
    parser.add_argument("--note")
    parser.add_argument("--command")
    parser.add_argument("--result")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--kind", choices=("checkpoint", "evaluation"))
    parser.add_argument("--output-root", type=Path, default=EXPERIMENTS)
    args = parser.parse_args()
    print(record_experiment(**vars(args)))


if __name__ == "__main__":
    main()
