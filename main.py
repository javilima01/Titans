"""Train memory adapters, evaluate checkpoints, or chat with Qwen."""

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from itertools import islice
import json
from pathlib import Path
import shlex
import sys
import tempfile

from src.llm.helpers.config import ROOT
from src.llm.helpers.dataset_generation import load_hf_episodes, read_episodes


def _positive(value):
    value = int(value)
    if value < 1:
        msg = "must be positive"
        raise argparse.ArgumentTypeError(msg)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def runtime(command):
        command.add_argument(
            "--device", choices=("cpu", "mps", "cuda"), help="Default: auto-select"
        )
        command.add_argument(
            "--dtype", choices=("float32", "bfloat16"), help="Default: auto-select"
        )
        command.add_argument("--seed", type=int, default=42)

    def dataset(command):
        source = command.add_mutually_exclusive_group()
        source.add_argument(
            "--data",
            type=Path,
            help="Normalized JSONL file or split directory (default: .datasets_cache/memory-v1)",
        )
        source.add_argument(
            "--hf-dataset", choices=("babilong", "qasper"), help="Stream directly from Hugging Face"
        )
        command.add_argument("--hf-task", default="qa1")
        command.add_argument("--hf-length", default="1k")
        command.add_argument("--hf-revision")
        command.add_argument("--limit", type=_positive, help="Maximum episodes (HF default: 1000)")

    train = commands.add_parser("train", help="Train memory adapters with answer-only supervision")
    runtime(train)
    dataset(train)
    train.add_argument("--output", type=Path, required=True, help="New checkpoint directory")
    train.add_argument(
        "--checkpoint", type=Path, help="Continue adapter training with a new optimizer"
    )
    train.add_argument("--layers", type=int, nargs="+", help="New model only; default: 11")
    train.add_argument(
        "--shared-memory",
        action="store_true",
        help="New model only; one fast memory read by every selected layer (default: all 24)",
    )
    train.add_argument("--memory-hidden-size", type=_positive, help="New model only; default: 256")
    train.add_argument("--memory-chunk-size", type=_positive, help="New model only; default: 16")
    train.add_argument("--memory-qk-scale", type=float, help="New model only; default: 1.0")
    train.add_argument("--max-inner-grad-norm", type=float, help="New model only; default: 1.0")
    train.add_argument(
        "--aligned-qk-init",
        action="store_true",
        help="New model only; initialize query/key as identity",
    )
    train.add_argument(
        "--memory-delta-read",
        action="store_true",
        help="New model only; expose only fast-memory changes to Qwen",
    )
    train.add_argument(
        "--memory-gate-init",
        type=float,
        help="New model only; initial logit of each adapter output gate",
    )
    train.add_argument("--window-size", type=_positive, help="Default: checkpoint setting or 512")
    train.add_argument(
        "--bptt-windows",
        type=int,
        default=4,
        help="Detach interval in windows; 0 means full backpropagation",
    )
    train.add_argument("--epochs", type=_positive, default=1)
    train.add_argument("--batch-size", type=_positive, default=1)
    train.add_argument("--gradient-accumulation-steps", type=_positive, default=1)
    train.add_argument(
        "--pair-contrastive-weight",
        type=float,
        default=0.0,
        help="Additional paired answer-ranking loss; requires --batch-size 2 --no-shuffle",
    )
    train.add_argument(
        "--supervise-eos",
        action="store_true",
        help="Train an end-of-answer token after the answer in each episode",
    )
    train.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--loss-chunk-size", type=_positive, default=128)
    train.add_argument("--max-steps", type=_positive)
    train.add_argument("--checkpoint-decoder", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--log-every", type=_positive, default=10)
    train.add_argument(
        "--validation-data",
        type=Path,
        help="Validation JSONL file or split directory for per-epoch validation loss "
        "(default: the validation split of the training source when available)",
    )
    train.add_argument("--validation-limit", type=_positive, help="Maximum validation episodes")
    train.add_argument("--experiment-dir", type=Path, help="Experiment records directory")

    for name, split in (("validate", "validation"), ("test", "test")):
        command = commands.add_parser(name, help=f"Evaluate generated answers on the {split} split")
        runtime(command)
        dataset(command)
        command.set_defaults(split=split)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--window-size", type=_positive, help="Default: saved training window")
        command.add_argument("--max-new-tokens", type=_positive, default=32)
        command.add_argument(
            "--ablations",
            action="store_true",
            help="Also disable memory and reset memory at each window",
        )
        command.add_argument(
            "--memory-modes",
            nargs="+",
            choices=("normal", "disabled", "reset"),
            help="Evaluate selected memory modes (overrides --ablations)",
        )
        command.add_argument(
            "--report", type=Path, help="Write JSON metrics and individual predictions"
        )
        command.add_argument("--experiment-dir", type=Path, help="Experiment records directory")

    chat = commands.add_parser("chat", help="Chat with the base model or trained memory adapters")
    runtime(chat)
    chat.add_argument("--checkpoint", type=Path, help="Omit to chat with the original Qwen model")
    chat.add_argument("--prompt", help="Generate one response and exit")
    chat.add_argument("--system", default="You are a helpful assistant.")
    chat.add_argument(
        "--session-file", type=Path, help="Save and reload this conversation across chat runs"
    )
    chat.add_argument(
        "--memory-state-file",
        type=Path,
        help="Save and reload one user's compact Titans fast-memory state across chat runs",
    )
    chat.add_argument(
        "--episodic-memory-file",
        type=Path,
        help="Save source statements and retrieve relevant evidence across chat runs",
    )
    chat.add_argument(
        "--episodic-hits",
        type=_positive,
        default=3,
        help="Maximum retrieved source statements per question (default: 3)",
    )
    chat.add_argument("--max-new-tokens", type=_positive, default=200)
    chat.add_argument(
        "--window-size", type=_positive, help="Adapter model only; default: saved training window"
    )
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--top-p", type=float, default=0.9)
    return parser


def load_episodes(args, split):
    """Use the same normalized schema for local synthetic and downloaded episodes."""
    if args.hf_dataset:
        episodes = load_hf_episodes(
            args.hf_dataset,
            split=split,
            limit=args.limit or 1000,
            length=args.hf_length,
            task=args.hf_task,
            revision=args.hf_revision,
            cache_dir=ROOT / ".datasets_cache" / "hf",
        )
    else:
        path = args.data
        if path is None:
            path = ROOT / ".datasets_cache" / "memory-v1"
            if not path.exists():
                path = ROOT / "datasets_cache" / "memory-v1"
        if path.is_dir():
            path = path / f"{split}.jsonl"
        episodes = read_episodes(path)
    result = list(islice(episodes, args.limit)) if args.limit else list(episodes)
    if not result:
        msg = f"No episodes in the {split} dataset"
        raise ValueError(msg)
    if any(episode.split != split for episode in result):
        msg = f"Expected only {split} episodes; refusing to mix training and evaluation splits"
        raise ValueError(msg)
    return result


def load_validation_episodes(args):
    """Resolve episodes for per-epoch validation loss; None when no validation split exists."""
    limit = args.validation_limit
    if args.validation_data is not None:
        path = args.validation_data
        if path.is_dir():
            path = path / "validation.jsonl"
        episodes = read_episodes(path)
    elif args.hf_dataset:
        try:
            episodes = load_hf_episodes(
                args.hf_dataset,
                split="validation",
                limit=limit or 1000,
                length=args.hf_length,
                task=args.hf_task,
                revision=args.hf_revision,
                cache_dir=ROOT / ".datasets_cache" / "hf",
            )
        except ValueError as exc:
            print(f"Monitoring training loss only: {exc}", file=sys.stderr)
            return None
    else:
        path = args.data
        if path is None:
            path = ROOT / ".datasets_cache" / "memory-v1"
            if not path.exists():
                path = ROOT / "datasets_cache" / "memory-v1"
        if not path.is_dir():
            print(
                "Monitoring training loss only: pass --validation-data for a file dataset",
                file=sys.stderr,
            )
            return None
        path = path / "validation.jsonl"
        if not path.exists():
            print(f"Monitoring training loss only: no validation data at {path}", file=sys.stderr)
            return None
        episodes = read_episodes(path)
    result = list(islice(episodes, limit)) if limit else list(episodes)
    if not result:
        print("Monitoring training loss only: validation data is empty", file=sys.stderr)
        return None
    if any(episode.split != "validation" for episode in result):
        msg = "Validation monitoring expects episodes labeled with the validation split"
        raise ValueError(msg)
    return result


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _load_chat_session(path, message_type):
    if not path.exists():
        return None
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid chat session JSON: {path}"
        raise ValueError(msg) from exc
    if not isinstance(records, list) or not records:
        msg = "Chat session must contain a nonempty message list"
        raise ValueError(msg)
    history = [message_type.model_validate(record) for record in records]
    if history[0].role.value != "system" or history[-1].role.value not in (
        "system",
        "assistant",
    ):
        msg = "Chat session must start with a system message and end after an assistant turn"
        raise ValueError(msg)
    return history


def _save_chat_session(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump([message.model_dump(mode="json") for message in history], handle)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _experiment_root(checkpoint: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    try:
        checkpoint.resolve().relative_to(ROOT)
    except ValueError:
        return checkpoint.parent / "experiments"
    return ROOT / "experiments"


def run(args):
    # Keep argparse/help and dataset inspection independent of model imports.
    import torch

    from src.llm.helpers.experiment_tracking import record_experiment
    from src.llm.helpers.monitoring import TrainingMonitor
    from src.llm.modules.episodic_memory import EpisodicMemory, evidence_message
    from src.llm.modules.evaluation import evaluate_episodes
    from src.llm.modules.qwen import Qwen35Titans, Qwen35Wrapper
    from src.llm.schemas.messages import Message

    torch.manual_seed(args.seed)
    runtime = {"device": args.device, "dtype": getattr(torch, args.dtype) if args.dtype else None}
    if args.command == "train":
        if args.output.exists():
            msg = f"Checkpoint output already exists: {args.output}"
            raise FileExistsError(msg)
        if args.bptt_windows < 0:
            msg = "--bptt-windows must be nonnegative"
            raise ValueError(msg)
        if args.checkpoint and any(
            value is not None
            for value in (
                args.layers,
                args.memory_hidden_size,
                args.memory_chunk_size,
                args.memory_qk_scale,
                args.max_inner_grad_norm,
                args.aligned_qk_init or None,
                args.memory_delta_read or None,
                args.memory_gate_init,
                args.shared_memory or None,
            )
        ):
            msg = "Adapter settings come from --checkpoint; omit new-adapter options"
            raise ValueError(msg)
        episodes = load_episodes(args, "train")
        if args.supervise_eos:
            episodes = [replace(episode, supervise_eos=True) for episode in episodes]
        validation_episodes = load_validation_episodes(args)
        model = (
            Qwen35Titans.from_pretrained(args.checkpoint, **runtime)
            if args.checkpoint
            else Qwen35Titans(
                **runtime,
                layer_indices=args.layers if args.shared_memory else args.layers or [11],
                memory_hidden_size=args.memory_hidden_size or 256,
                memory_chunk_size=args.memory_chunk_size or 16,
                memory_qk_scale=args.memory_qk_scale if args.memory_qk_scale is not None else 1.0,
                max_inner_grad_norm=(
                    args.max_inner_grad_norm if args.max_inner_grad_norm is not None else 1.0
                ),
                aligned_qk_init=args.aligned_qk_init,
                memory_delta_read=args.memory_delta_read,
                memory_gate_init=args.memory_gate_init,
                shared_across_layers=args.shared_memory,
            )
        )
        window_size = args.window_size or model.training_config.get("window_size", 512)
        # The output directory must stay absent until saving, so the live
        # report is written next to it and moved into the checkpoint at the end.
        monitor = TrainingMonitor(
            args.output.parent / f"{args.output.name}.training.png",
            title=args.output.name,
        )
        print(
            json.dumps(
                {
                    "event": "training_started",
                    "episodes": len(episodes),
                    "validation_episodes": len(validation_episodes or []),
                    "window_size": window_size,
                    "bptt_windows": args.bptt_windows,
                    "monitor": str(monitor.image_path),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

        step_losses, validation_history = [], []

        def progress(event):
            step_losses.append(event["loss"])
            monitor.record_step(event)
            if event["step"] == 1 or event["step"] % args.log_every == 0:
                print(json.dumps(event), file=sys.stderr, flush=True)

        def validation(event):
            validation_history.append(event)
            monitor.record_validation(event)
            print(json.dumps(event), file=sys.stderr, flush=True)

        interrupted = False
        try:
            losses = model.train(
                episodes=episodes,
                max_length=window_size,
                bptt_windows=args.bptt_windows,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                loss_chunk_size=args.loss_chunk_size,
                checkpoint_decoder=args.checkpoint_decoder,
                max_steps=args.max_steps,
                seed=args.seed,
                shuffle=args.shuffle,
                pair_contrastive_weight=args.pair_contrastive_weight,
                on_step=progress,
                validation_episodes=validation_episodes,
                on_validation=validation,
            )
        except KeyboardInterrupt:
            interrupted = True
            losses = step_losses
            print(
                "\nInterrupted; saving a checkpoint from the completed steps...",
                file=sys.stderr,
                flush=True,
            )
        if not losses:
            monitor.finish("interrupted" if interrupted else "finished")
            print("No optimizer steps completed; checkpoint not saved.", file=sys.stderr)
            return
        model.save_pretrained(
            args.output,
            metadata={
                "episodes": len(episodes),
                "seed": args.seed,
                "optimizer_steps": len(losses),
                "losses": losses,
                "validation": validation_history,
                "interrupted": interrupted,
                "source": str(args.data) if args.data else args.hf_dataset or "memory-v1",
                "supervise_eos": args.supervise_eos,
                "parent_checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "training_options": {
                    "device": args.device,
                    "dtype": args.dtype,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "max_grad_norm": args.max_grad_norm,
                    "loss_chunk_size": args.loss_chunk_size,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "epochs": args.epochs,
                    "max_steps": args.max_steps,
                    "shuffle": args.shuffle,
                    "checkpoint_decoder": args.checkpoint_decoder,
                    "pair_contrastive_weight": args.pair_contrastive_weight,
                    "supervise_eos": args.supervise_eos,
                    "validation_limit": args.validation_limit,
                    "limit": args.limit,
                },
            },
        )
        monitor.finish("interrupted" if interrupted else "finished", move_into=args.output)
        record_experiment(
            args.output,
            output_root=_experiment_root(args.output, args.experiment_dir),
            command=args.command_line,
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "interrupted": interrupted,
                    "episodes": len(episodes),
                    "steps": len(losses),
                    "final_loss": losses[-1],
                    "final_validation_loss": (
                        validation_history[-1]["val_loss"] if validation_history else None
                    ),
                    "monitor": str(args.output / "training.png"),
                }
            )
        )
        return

    if args.command in ("validate", "test"):
        if args.report and args.report.exists():
            msg = f"Evaluation report already exists: {args.report}"
            raise FileExistsError(msg)
        episodes = load_episodes(args, args.split)
        model = Qwen35Titans.from_pretrained(args.checkpoint, **runtime)
        reports = {}
        modes = args.memory_modes or (
            ("normal", "disabled", "reset") if args.ablations else ("normal",)
        )
        for mode in modes:
            reports[mode] = evaluate_episodes(
                model,
                episodes,
                window_size=args.window_size,
                max_new_tokens=args.max_new_tokens,
                memory_mode=mode,
                on_example=lambda event, current_mode=mode: print(
                    json.dumps({"memory_mode": current_mode, **event}),
                    file=sys.stderr,
                    flush=True,
                ),
            )
        report = {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "data_source": str(args.data) if args.data else args.hf_dataset or "memory-v1",
            "evaluation_options": {
                "device": args.device,
                "window_size": args.window_size or model.training_config.get("window_size", 512),
                "max_new_tokens": args.max_new_tokens,
                "limit": args.limit,
                "ablations": args.ablations,
                "memory_modes": list(modes),
            },
            "evaluations": reports,
        }
        report_path = args.report or (
            args.checkpoint / f"{args.split}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
        )
        _write_report(report_path, report)
        record_experiment(
            args.checkpoint,
            output_root=_experiment_root(args.checkpoint, args.experiment_dir),
            additional_reports=(report_path,),
        )
        summary = {
            mode: {key: value for key, value in result.items() if key != "predictions"}
            for mode, result in reports.items()
        }
        print(
            json.dumps(
                {"split": args.split, "report": str(report_path), "evaluations": summary}, indent=2
            )
        )
        return

    if args.temperature < 0 or not 0 < args.top_p <= 1:
        msg = "Require temperature >= 0 and 0 < top_p <= 1"
        raise ValueError(msg)
    if args.window_size and not args.checkpoint:
        msg = "--window-size is available for adapter checkpoints only"
        raise ValueError(msg)
    if args.memory_state_file and not args.checkpoint:
        msg = "--memory-state-file requires --checkpoint"
        raise ValueError(msg)
    if args.memory_state_file and args.session_file:
        msg = "Use either --memory-state-file or --session-file for a chat"
        raise ValueError(msg)
    if args.episodic_memory_file and (args.memory_state_file or args.session_file):
        msg = "Use --episodic-memory-file separately from other persistent chat modes"
        raise ValueError(msg)
    model = (
        Qwen35Titans.from_pretrained(args.checkpoint, **runtime)
        if args.checkpoint
        else Qwen35Wrapper(**runtime)
    )
    history = (_load_chat_session(args.session_file, Message) if args.session_file else None) or [
        Message.system_msg(args.system)
    ]
    memory_state = (
        model.load_memory_state(args.memory_state_file)
        if args.memory_state_file and args.memory_state_file.exists()
        else None
    )
    episodic_memory = (
        EpisodicMemory.load(args.episodic_memory_file) if args.episodic_memory_file else None
    )
    if args.prompt is None:
        instructions = "Type /exit to quit or /reset to clear the conversation."
        if episodic_memory is not None:
            instructions += " Use /remember TEXT to save a fact without asking a question."
        print(instructions)
    while True:
        try:
            text = args.prompt if args.prompt is not None else input("You: ")
        except EOFError:
            break
        if text.strip() in ("/exit", "/quit"):
            break
        if text.strip() == "/reset":
            history = [Message.system_msg(args.system)]
            memory_state = None
            if args.session_file:
                _save_chat_session(args.session_file, history)
            if args.memory_state_file:
                args.memory_state_file.unlink(missing_ok=True)
            if args.episodic_memory_file:
                args.episodic_memory_file.unlink(missing_ok=True)
                episodic_memory = EpisodicMemory()
            if args.prompt is not None:
                break
            continue
        if episodic_memory is not None and (
            text.strip() == "/remember" or text.strip().startswith("/remember ")
        ):
            fact = text.strip().removeprefix("/remember").strip()
            if fact:
                count = episodic_memory.remember(fact)
                episodic_memory.save(args.episodic_memory_file)
                print(f"Remembered {count} statement(s).")
            else:
                print("Usage: /remember TEXT")
            if args.prompt is not None:
                break
            continue
        recalled = (
            episodic_memory.retrieve(text, limit=args.episodic_hits)
            if episodic_memory is not None
            else []
        )
        if recalled:
            history.append(Message.user_msg(evidence_message(recalled)))
            history.append(Message.assistant_msg("Understood."))
        history.append(Message.user_msg(text))
        if isinstance(model, Qwen35Titans):
            response = model.generate(
                history,
                max_new_tokens=args.max_new_tokens,
                window_size=args.window_size,
                temperature=args.temperature,
                top_p=args.top_p,
                memory_mode="disabled" if episodic_memory is not None else "normal",
                memory_state=memory_state,
                return_memory_state=bool(args.memory_state_file),
            )
            if args.memory_state_file:
                answer, memory_state = response
                model.save_memory_state(args.memory_state_file, memory_state)
            else:
                answer = response
        else:
            options = {"do_sample": args.temperature > 0}
            if args.temperature > 0:
                options.update(temperature=args.temperature, top_p=args.top_p)
            answer = model.generate(history, max_new_tokens=args.max_new_tokens, **options)
        history.append(Message.assistant_msg(answer))
        if args.session_file:
            _save_chat_session(args.session_file, history)
        if args.memory_state_file:
            history = [Message.system_msg(args.system)]
        if episodic_memory is not None:
            episodic_memory.remember(text)
            episodic_memory.save(args.episodic_memory_file)
        print(answer if args.prompt is not None else f"Assistant: {answer}")
        if args.prompt is not None:
            break


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        args.command_line = shlex.join([".venv/bin/python", "main.py", *(argv or sys.argv[1:])])
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
