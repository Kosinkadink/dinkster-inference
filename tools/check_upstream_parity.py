"""Compare the library and retained tests with a mechanically renamed upstream tree."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from rewrite_upstream_imports import rewrite


FORK_POINT = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
OMITTED_TESTS = {
    "folder_path_test.py",
    "image_invert_test.py",
    "test_seedvr2_model.py",
    "test_seedvr2_vae_decode.py",
    "test_sensenova.py",
    "test_vae_decode_tiled_nested.py",
}
INFRASTRUCTURE_TESTS = {
    "comfy_test/rewrite_upstream_imports_test.py",
    "comfy_test/upstream_sync_test.py",
}
ROOT = Path(__file__).resolve().parents[1]


def compare(upstream: Path) -> int:
    differences = []
    for source, target in (
        ("comfy", "dinkster_inference"),
        ("tests-unit/comfy_test", "tests-unit/comfy_test"),
        ("tests-unit/comfy_quant", "tests-unit/comfy_quant"),
    ):
        expected = {}
        for path in (upstream / source).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(upstream / source).as_posix()
            if source == "tests-unit/comfy_test" and relative in OMITTED_TESTS:
                continue
            if path.suffix == ".py":
                rewrite(path)
            expected[relative] = path.read_bytes()
        actual = {
            path.relative_to(ROOT / target).as_posix(): path.read_bytes()
            for path in (ROOT / target).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        if target.startswith("tests-unit/"):
            for path in INFRASTRUCTURE_TESTS:
                prefix, name = path.split("/", 1)
                if target == f"tests-unit/{prefix}":
                    actual.pop(name, None)
        changed = sorted(name for name in expected.keys() | actual.keys()
                         if expected.get(name) != actual.get(name))
        differences.extend(f"{target}/{name}" for name in changed)
        print(f"{target}: {len(expected)} upstream files, {len(changed)} differences")  # noqa: T201
    rewrite(upstream / "node_helpers.py")
    if (upstream / "node_helpers.py").read_bytes() != (ROOT / "node_helpers.py").read_bytes():
        differences.append("node_helpers.py")
    for path in differences:
        print(path)  # noqa: T201
    return bool(differences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, help="Existing clean detached upstream checkout")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        checkout = Path(directory) / "ComfyUI"
        subprocess.run(["git", "clone", "--quiet", "--no-checkout",
                        str(args.upstream) if args.upstream else "https://github.com/comfyanonymous/ComfyUI.git",
                        str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", FORK_POINT], check=True)
        raise SystemExit(compare(checkout))


if __name__ == "__main__":
    main()
