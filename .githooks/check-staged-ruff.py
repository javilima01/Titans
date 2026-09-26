"""Run Ruff against the Python files in Git's index, not the working tree."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command: list[str], *, cwd: Path) -> bool:
    return subprocess.run(command, cwd=cwd, check=False).returncode == 0


def tool(name: str, *, repo: Path) -> str:
    vendored = repo / ".venv" / "bin" / name
    return str(vendored) if vendored.exists() else name


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    uv = tool("uv", repo=repo)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    python_files = [Path(os.fsdecode(name)) for name in staged if name.endswith(b".py")]

    if not python_files:
        return 0

    with tempfile.TemporaryDirectory(prefix="staged-ruff-") as temp_name:
        temp = Path(temp_name)
        # Materialize the whole index so Ruff sees the same package layout as the commit.
        subprocess.run(
            ["git", "checkout-index", "--all", f"--prefix={temp}{os.sep}"],
            cwd=repo,
            check=True,
        )

        # Lint from the temp dir so per-file-ignores patterns match the copied files.
        config_path = temp / "pyproject.toml"
        files = [path.as_posix() for path in python_files]
        print("Checking staged Python files with Ruff formatter...")
        if not run(
            [
                uv,
                "run",
                "--project",
                str(repo),
                "ruff",
                "format",
                "--check",
                "--config",
                str(config_path),
                *files,
            ],
            cwd=temp,
        ):
            print(
                "Staged Python code needs formatting. Run `uv run ruff format`, stage the fixes, "
                "and retry the commit.",
                file=sys.stderr,
            )
            return 1

        print("Checking staged Python files with Ruff...")
        if not run(
            [
                uv,
                "run",
                "--project",
                str(repo),
                "ruff",
                "check",
                "--config",
                str(config_path),
                *files,
            ],
            cwd=temp,
        ):
            print(
                "Ruff found issues in the staged code. Fix them, stage the fixes, "
                "and retry the commit.",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
