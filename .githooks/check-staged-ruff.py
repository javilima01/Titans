"""Run Ruff against the Python files in Git's index, not the working tree."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command: list[str], *, cwd: Path) -> bool:
    return subprocess.run(command, cwd=cwd, check=False).returncode == 0


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
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
        staged_paths: list[Path] = []
        for relative_path in python_files:
            target = temp / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            blob = subprocess.run(
                ["git", "show", f":{relative_path.as_posix()}"],
                cwd=repo,
                check=True,
                capture_output=True,
            ).stdout
            target.write_bytes(blob)
            staged_paths.append(target)

        staged_config = temp / "pyproject.toml"
        config_path = repo / "pyproject.toml"
        if b"pyproject.toml" in staged:
            staged_config.write_bytes(
                subprocess.run(
                    ["git", "show", ":pyproject.toml"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                ).stdout
            )
            config_path = staged_config

        files = [str(path) for path in staged_paths]
        print("Checking staged Python files with Ruff formatter...")
        if not run(
            [
                "uv",
                "run",
                "ruff",
                "format",
                "--check",
                "--config",
                str(config_path),
                *files,
            ],
            cwd=repo,
        ):
            print(
                "Staged Python code needs formatting. Run `uv run ruff format`, stage the fixes, "
                "and retry the commit.",
                file=sys.stderr,
            )
            return 1

        print("Checking staged Python files with Ruff...")
        if not run(
            ["uv", "run", "ruff", "check", "--config", str(config_path), *files],
            cwd=repo,
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
