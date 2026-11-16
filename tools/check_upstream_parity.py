"""Verify that a transformed upstream delta is already carried by this fork."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from regenerate_upstream_library import regenerate


ROOT = Path(__file__).resolve().parents[1]


def verify(baseline: Path, upstream: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkout = Path(directory) / "fork"
        subprocess.run(["git", "-C", str(destination), "worktree", "add", "--detach", str(checkout), "HEAD"], check=True)
        try:
            regenerate(baseline, upstream, checkout)
            changes = subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"], text=True
            ).splitlines()
            if changes:
                names = subprocess.check_output(
                    ["git", "-C", str(checkout), "status", "--porcelain"], text=True
                )
                raise SystemExit(f"transformed upstream delta is missing:\n{names}")
        finally:
            subprocess.run(
                ["git", "-C", str(destination), "worktree", "remove", "--force", str(checkout)],
                check=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    verify(args.baseline, args.upstream, args.destination)


if __name__ == "__main__":
    main()
