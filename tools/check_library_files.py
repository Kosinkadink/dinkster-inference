"""Reject repository files that have not been classified for the library."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"library", "packaging", "tooling", "tests", "LICENSE", "pending-library", "pending-tests"}
PENDING_CLASSES = {
    "pending-library": "library",
    "pending-tests": "tests",
}


def activate_pending_files(root: Path) -> None:
    tracked = set(subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"], text=True,
    ).rstrip("\0").split("\0"))
    manifest = root / "tools/library_files.txt"
    lines = []
    for line in manifest.read_text().splitlines():
        category, path = line.split(" ", 1)
        if category in PENDING_CLASSES and path in tracked:
            category = PENDING_CLASSES[category]
        lines.append(f"{category} {path}")
    manifest.write_text("\n".join(lines) + "\n")


def check(root):
    inventory = {}
    errors = []
    for line in (root / "tools/library_files.txt").read_text().splitlines():
        category, path = line.split(" ", 1)
        if category not in CLASSES or path in inventory:
            errors.append(f"Invalid library file classification: {line}")
        inventory[path] = category
    tracked = set(subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"], text=True,
    ).rstrip("\0").split("\0"))
    errors.extend(f"Unclassified repository file: {path}" for path in sorted(tracked - inventory.keys()))
    errors.extend(
        f"Classified file is not tracked: {path}"
        for path, category in sorted(inventory.items())
        if path not in tracked and category not in PENDING_CLASSES
    )
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--activate-pending", action="store_true")
    args = parser.parse_args()
    if args.activate_pending:
        activate_pending_files(args.root)
    errors = check(args.root)
    for error in errors:
        sys.stderr.write(error + "\n")
    raise SystemExit(bool(errors))
