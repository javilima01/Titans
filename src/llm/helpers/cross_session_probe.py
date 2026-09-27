"""Five-process probe for two repository facts and a later correction."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from src.llm.helpers.config import ROOT
from src.llm.helpers.experiment_tracking import record_experiment

CASES = (
    (
        "write_two",
        (
            "Session 1. My account is jbtest2. The build tool for repository alpha is Bazel. "
            "The build tool for repository beta is Meson."
        ),
        None,
    ),
    (
        "ask_alpha_before",
        "Session 2. My account is jbtest2. Which build tool does repository alpha use?",
        "Bazel",
    ),
    (
        "correct_alpha",
        "Session 3. My account is jbtest2. Correction: the build tool for repository alpha is Ninja.",
        None,
    ),
    (
        "ask_alpha_after",
        "Session 4. My account is jbtest2. Which build tool does repository alpha use?",
        "Ninja",
    ),
    (
        "ask_beta_after",
        "Session 5. My account is jbtest2. Which build tool does repository beta use?",
        "Meson",
    ),
)


def evaluate(checkpoint: Path, *, device: str, window_size: int = 256) -> dict:
    """Start a fresh CLI process for every turn, carrying only serialized state."""
    cases = []
    with tempfile.TemporaryDirectory() as directory:
        state_file = Path(directory) / "user.safetensors"
        for name, prompt, expected in CASES:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "chat",
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    device,
                    "--memory-state-file",
                    str(state_file),
                    "--window-size",
                    str(window_size),
                    "--max-new-tokens",
                    "12",
                    "--prompt",
                    prompt,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                msg = f"Cross-session process {name} failed: {result.stderr[-1000:]}"
                raise RuntimeError(msg)
            answer = result.stdout.strip()
            cases.append(
                {
                    "case": name,
                    "prompt": prompt,
                    "answer": answer,
                    "expected": expected,
                    "exact_match": answer.casefold() == expected.casefold() if expected else None,
                    "state_bytes": state_file.stat().st_size,
                }
            )
    scored = [case for case in cases if case["expected"] is not None]
    return {
        "evaluation_kind": "five_separate_processes_fast_state",
        "checkpoint": str(checkpoint),
        "device": device,
        "window_size": window_size,
        "cases": cases,
        "scored": {
            "correct": sum(case["exact_match"] for case in scored),
            "questions": len(scored),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.window_size < 1 or args.report.exists():
        parser.error("Require positive window size and a new report path")
    report = evaluate(args.checkpoint, device=args.device, window_size=args.window_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record_experiment(args.checkpoint, additional_reports=(args.report,))
    print(json.dumps({"report": str(args.report), "scored": report["scored"]}))


if __name__ == "__main__":
    main()
