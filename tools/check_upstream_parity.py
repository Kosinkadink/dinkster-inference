"""Verify that a transformed upstream delta is already carried by this fork."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from regenerate_upstream_library import MAPPED_FILES, MAPPED_ROOTS, regenerate, transform


ROOT = Path(__file__).resolve().parents[1]
SCOPE = tuple(f"{target}/" for _, target in MAPPED_ROOTS) + tuple(target for _, target in MAPPED_FILES)


def verify(baseline: Path, upstream: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkout = Path(directory) / "fork"
        subprocess.run(["git", "-C", str(destination), "worktree", "add", "--detach", str(checkout), "HEAD"], check=True)
        try:
            regenerate(baseline, upstream, checkout)
            changes = subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"], text=True
            )
            if changes:
                raise SystemExit(f"transformed upstream delta is missing:\n{changes}")
        finally:
            subprocess.run(
                ["git", "-C", str(destination), "worktree", "remove", "--force", str(checkout)],
                check=False,
            )


def differences(upstream: Path, destination: Path) -> list[str]:
    """Every place this fork differs from the transformed upstream revision."""
    with tempfile.TemporaryDirectory() as directory:
        transformed = Path(directory)
        transform(upstream, transformed)
        expected = {path.relative_to(transformed).as_posix()
                    for path in transformed.rglob("*") if path.is_file()}
        tracked = subprocess.check_output(["git", "-C", str(destination), "ls-files"], text=True).splitlines()
        rows = [f"fork-only {path}" for path in tracked if path.startswith(SCOPE) and path not in expected]
        for path in sorted(expected):
            if not (destination / path).exists():
                rows.append(f"removed {path}")
            elif (destination / path).read_bytes() != (transformed / path).read_bytes():
                rows.append(f"modified {path}")
    return sorted(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--differences", action="store_true",
                        help="list fork differences from --upstream instead of checking a delta")
    parser.add_argument("--declared", type=Path,
                        help="with --differences: fail when a changed upstream file is not listed in this file")
    args = parser.parse_args()
    if args.differences and args.declared:
        declared = {line.strip() for line in args.declared.read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
        changed = {row.split(" ", 1)[1] for row in differences(args.upstream, args.destination)
                   if not row.startswith("fork-only ")}
        for path in sorted(declared - changed):
            sys.stdout.write(f"Declared file no longer differs from upstream: {path}\n")
        if changed - declared:
            raise SystemExit("\n".join(f"Undeclared change to an upstream file: {path}"
                                       for path in sorted(changed - declared)))
    elif args.differences:
        sys.stdout.write("".join(row + "\n" for row in differences(args.upstream, args.destination)))
    elif args.baseline is None:
        parser.error("--baseline is required unless --differences is given")
    else:
        verify(args.baseline, args.upstream, args.destination)


if __name__ == "__main__":
    main()
