"""Build the inference library from a ComfyUI checkout."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

if __package__:
    from tools.rewrite_upstream_imports import rewrite
else:
    from rewrite_upstream_imports import rewrite


ROOT = Path(__file__).resolve().parents[1]
OMITTED_TESTS = {
    # Every other comfy_test file is retained whole. Exclusions are explicit
    # because they are application tests rather than library behavior.
    "folder_path_test.py",
    "image_invert_test.py",
    "test_seedvr2_model.py",
    "test_seedvr2_vae_decode.py",
    "test_sensenova.py",
    "test_vae_decode_tiled_nested.py",
}
INFRASTRUCTURE_TESTS = {
    "mediapipe_import_test.py",
    "rewrite_upstream_imports_test.py",
    "upstream_sync_test.py",
}


def rewrite_tree(root: Path) -> None:
    for path in root.rglob("*.py"):
        rewrite(path)


def transform(upstream: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (upstream / "comfy", destination / "dinkster_inference"),
        (upstream / "comfy_extras" / "mediapipe", destination / "dinkster_inference" / "mediapipe"),
        (upstream / "tests-unit" / "comfy_test", destination / "tests-unit" / "comfy_test"),
        (upstream / "tests-unit" / "comfy_quant", destination / "tests-unit" / "comfy_quant"),
    ):
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if path.is_file() and "__pycache__" not in path.parts:
                if target.name == "dinkster_inference" and (
                    relative == Path("comfy_types/node_typing.py")
                    or relative == Path("comfy_types/README.md")
                    or Path("comfy_types/examples") in relative.parents
                ):
                    continue
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(path.read_bytes())
    for name in OMITTED_TESTS:
        (destination / "tests-unit" / "comfy_test" / name).unlink(missing_ok=True)
    rewrite_tree(destination)
    output = destination / "node_helpers.py"
    output.write_bytes((upstream / "node_helpers.py").read_bytes())
    rewrite(output)


def merge_file(current: Path, baseline: Path | None, target: Path | None, relative: Path) -> None:
    if baseline is None:
        if current.exists():
            raise SystemExit(f"fork-only file conflicts with upstream addition: {relative}")
        if target is not None:
            current.parent.mkdir(parents=True, exist_ok=True)
            current.write_bytes(target.read_bytes())
        return
    if target is None:
        if current.exists() and current.read_bytes() != baseline.read_bytes():
            raise SystemExit(f"cannot carry fork change through upstream deletion: {relative}")
        current.unlink(missing_ok=True)
        return
    if baseline.read_bytes() == target.read_bytes():
        return
    if not current.exists():
        raise SystemExit(f"fork removed upstream file changed by target: {relative}")
    if current.read_bytes() == baseline.read_bytes():
        current.write_bytes(target.read_bytes())
        return
    with tempfile.TemporaryDirectory() as temporary:
        merged = Path(temporary) / "merged"
        result = subprocess.run(
            ["git", "merge-file", "-p", str(current), str(baseline), str(target)],
            text=True,
            capture_output=True,
        )
        if result.returncode > 1:
            raise SystemExit(f"could not merge {relative}: {result.stderr.strip()}")
        if result.returncode:
            raise SystemExit(f"cannot carry fork change through upstream change: {relative}")
        merged.write_text(result.stdout)
        current.write_bytes(merged.read_bytes())


def regenerate(baseline: Path, upstream: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        transformed_baseline = Path(temporary) / "baseline"
        transformed_target = Path(temporary) / "target"
        transform(baseline, transformed_baseline)
        transform(upstream, transformed_target)
        paths = {
            path.relative_to(transformed_baseline)
            for path in transformed_baseline.rglob("*") if path.is_file()
        } | {
            path.relative_to(transformed_target)
            for path in transformed_target.rglob("*") if path.is_file()
        }
        for relative in sorted(paths):
            source = transformed_baseline / relative
            target = transformed_target / relative
            merge_file(destination / relative, source if source.exists() else None,
                       target if target.exists() else None, relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    regenerate(args.baseline, args.upstream, args.destination)


if __name__ == "__main__":
    main()
