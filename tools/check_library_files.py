"""Reject repository files that have not been classified for the library."""

import argparse
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"library", "packaging", "tooling", "tests", "documentation", "LICENSE"}


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
    errors.extend(f"Classified file is not tracked: {path}" for path in sorted(inventory.keys() - tracked))
    project = root / "pyproject.toml"
    globs = tomllib.loads(project.read_text()).get("tool", {}).get("setuptools", {}).get(
        "package-data", {}).get("dinkster_inference") if project.exists() else None
    if globs:
        # A data file the wheel would not ship is application material or a packaging gap.
        shipped = {".py"} | {Path(pattern).suffix for pattern in globs}
        errors.extend(
            f"File type is not packaged: {path}" for path in sorted(tracked)
            if path.startswith("dinkster_inference/") and Path(path).suffix not in shipped
        )
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    errors = check(parser.parse_args().root)
    for error in errors:
        sys.stderr.write(error + "\n")
    raise SystemExit(bool(errors))
