"""Train memory adapters, evaluate checkpoints, or chat with Qwen."""

import argparse
from itertools import islice
import json
from pathlib import Path
import sys

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
    train.add_argument("--memory-hidden-size", type=_positive, help="New model only; default: 256")
    train.add_argument("--memory-chunk-size", type=_positive, help="New model only; default: 16")
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
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--loss-chunk-size", type=_positive, default=128)
    train.add_argument("--max-steps", type=_positive)
    train.add_argument("--checkpoint-decoder", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--log-every", type=_positive, default=10)

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
            "--report", type=Path, help="Write JSON metrics and individual predictions"
        )

    chat = commands.add_parser("chat", help="Chat with the base model or trained memory adapters")
    runtime(chat)
    chat.add_argument("--checkpoint", type=Path, help="Omit to chat with the original Qwen model")
    chat.add_argument("--prompt", help="Generate one response and exit")
    chat.add_argument("--system", default="You are a helpful assistant.")
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


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run(args):
    # Keep argparse/help and dataset inspection independent of model imports.
    import torch

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
            for value in (args.layers, args.memory_hidden_size, args.memory_chunk_size)
        ):
            msg = "Adapter architecture comes from --checkpoint; omit --layers/--memory-*-size"
            raise ValueError(msg)
        episodes = load_episodes(args, "train")
        model = (
            Qwen35Titans.from_pretrained(args.checkpoint, **runtime)
            if args.checkpoint
            else Qwen35Titans(
                **runtime,
                layer_indices=args.layers or [11],
                memory_hidden_size=args.memory_hidden_size or 256,
                memory_chunk_size=args.memory_chunk_size or 16,
            )
        )
        window_size = args.window_size or model.training_config.get("window_size", 512)
        print(
            json.dumps(
                {
                    "event": "training_started",
                    "episodes": len(episodes),
                    "window_size": window_size,
                    "bptt_windows": args.bptt_windows,
                }
            ),
            file=sys.stderr,
            flush=True,
        )

        def progress(event):
            if event["step"] == 1 or event["step"] % args.log_every == 0:
                print(json.dumps(event), file=sys.stderr, flush=True)

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
            on_step=progress,
        )
        model.save_pretrained(
            args.output,
            metadata={
                "episodes": len(episodes),
                "seed": args.seed,
                "optimizer_steps": len(losses),
                "losses": losses,
                "source": str(args.data) if args.data else args.hf_dataset or "memory-v1",
            },
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "episodes": len(episodes),
                    "steps": len(losses),
                    "final_loss": losses[-1],
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
        for mode in ("normal", "disabled", "reset") if args.ablations else ("normal",):
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
        report = {"checkpoint": str(args.checkpoint), "split": args.split, "evaluations": reports}
        if args.report:
            _write_report(args.report, report)
        summary = {
            mode: {key: value for key, value in result.items() if key != "predictions"}
            for mode, result in reports.items()
        }
        print(json.dumps({"split": args.split, "evaluations": summary}, indent=2))
        return

    if args.temperature < 0 or not 0 < args.top_p <= 1:
        msg = "Require temperature >= 0 and 0 < top_p <= 1"
        raise ValueError(msg)
    if args.window_size and not args.checkpoint:
        msg = "--window-size is available for adapter checkpoints only"
        raise ValueError(msg)
    model = (
        Qwen35Titans.from_pretrained(args.checkpoint, **runtime)
        if args.checkpoint
        else Qwen35Wrapper(**runtime)
    )
    history = [Message.system_msg(args.system)]
    if args.prompt is None:
        print("Type /exit to quit or /reset to clear the conversation.")
    while True:
        try:
            text = args.prompt if args.prompt is not None else input("You: ")
        except EOFError:
            break
        if text.strip() in ("/exit", "/quit"):
            break
        if text.strip() == "/reset":
            history = [Message.system_msg(args.system)]
            if args.prompt is not None:
                break
            continue
        history.append(Message.user_msg(text))
        if isinstance(model, Qwen35Titans):
            answer = model.generate(
                history,
                max_new_tokens=args.max_new_tokens,
                window_size=args.window_size,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            options = {"do_sample": args.temperature > 0}
            if args.temperature > 0:
                options.update(temperature=args.temperature, top_p=args.top_p)
            answer = model.generate(history, max_new_tokens=args.max_new_tokens, **options)
        print(answer if args.prompt is not None else f"Assistant: {answer}")
        history.append(Message.assistant_msg(answer))
        if args.prompt is not None:
            break


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
