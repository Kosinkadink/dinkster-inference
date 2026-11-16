from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from pathlib import Path


REWRITES = (
    (re.compile(r"\bcomfy_extras\.mediapipe\b"), "dinkster_inference.mediapipe"),
    (re.compile(r"\bcomfy\.(?!gguf\.orig_shape\.)"), "dinkster_inference."),
    (re.compile(r"\bfrom comfy\b"), "from dinkster_inference"),
    (re.compile(r"\bimport comfy\b"), "import dinkster_inference"),
)


def ascii_source(source: str) -> str:
    """Escape non-ASCII tokens where they stand; every other character is kept as written.

    `tokenize.untokenize` is not used: its spacing differs between Python releases.
    """
    if source.isascii():
        return source
    lines = io.StringIO(source).readlines()
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    pieces = []
    position = 0
    for token in tokenize.generate_tokens(iter(lines).__next__):
        if token.string.isascii():
            continue
        if token.type == tokenize.STRING:
            value = ascii(ast.literal_eval(token.string))
        else:
            value = token.string.encode("ascii", "backslashreplace").decode("ascii")
        start = starts[token.start[0] - 1] + token.start[1]
        end = starts[token.end[0] - 1] + token.end[1]
        if source[start:end] != token.string:
            raise SystemExit(f"token position does not match its text at line {token.start[0]}")
        pieces += [source[position:start], value]
        position = end
    return "".join(pieces) + source[position:]


def rewrite(path: Path) -> bool:
    original = path.read_text()
    updated = original
    for pattern, replacement in REWRITES:
        updated = pattern.sub(replacement, updated)
    updated = ascii_source(updated)
    if updated == original:
        return False
    path.write_text(updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite upstream ComfyUI Python imports for dinkster_inference."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    changed = 0
    for root in args.paths:
        files = (root.rglob("*.py") if root.is_dir() else (root,))
        for path in files:
            # This test's fixtures must retain upstream names as rewrite inputs.
            if path.name == "rewrite_upstream_imports_test.py":
                continue
            changed += rewrite(path)
    print(f"rewrote {changed} files")  # noqa: T201


if __name__ == "__main__":
    main()
