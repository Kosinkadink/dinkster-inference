"""Build the inference library from a ComfyUI checkout."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__:
    from tools.rewrite_upstream_imports import rewrite
else:
    from rewrite_upstream_imports import rewrite


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = Path("tools/library_files.txt")
# Upstream directory -> library directory. Everything the sync may write comes
# from these roots, MAPPED_FILES, docs/ledger.md and the inventory itself.
MAPPED_ROOTS = (
    ("comfy", "dinkster_inference"),
    ("comfy_extras/mediapipe", "dinkster_inference/mediapipe"),
    ("tests-unit/comfy_test", "tests-unit/comfy_test"),
    ("tests-unit/comfy_quant", "tests-unit/comfy_quant"),
)
MAPPED_FILES = (
    ("node_helpers.py", "node_helpers.py"),
    ("QUANTIZATION.md", "docs/quantization.md"),
)
# Node-authoring material inside comfy/. Markdown inside the mapped roots is
# upstream application documentation and is never carried.
EXCLUDED = ("comfy/comfy_types/node_typing.py", "comfy/comfy_types/examples/")
# cli_args names that Dinkster packages use and this library does not.
CONSUMER_NAMES = {"disable_metadata"}  # read by dinkster-nodes save nodes
TEST_ROOTS = ("tests-unit/comfy_test", "tests-unit/comfy_quant")


def classify(path: str) -> str:
    if path.startswith("tests-unit/"):
        return "tests"
    if path.startswith("docs/"):
        return "documentation"
    return "library"


def imported_modules(tree: ast.AST, root: Path) -> set[str]:
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            package = root / node.module.replace(".", "/")
            modules.update(
                f"{node.module}.{alias.name}" for alias in node.names
                if (package / f"{alias.name}.py").exists() or (package / alias.name).is_dir()
            )
    return modules


def imports_application(path: Path, upstream: Path) -> bool:
    """A test that imports upstream code outside the mapped roots tests the application."""
    for module in imported_modules(ast.parse(path.read_text()), upstream):
        top = module.split(".")[0]
        if not ((upstream / top).is_dir() or (upstream / f"{top}.py").exists()):
            continue
        if top in ("comfy", "node_helpers") or module == "comfy_extras":
            continue
        if module == "comfy_extras.mediapipe" or module.startswith("comfy_extras.mediapipe."):
            continue
        return True
    return False


def option_reads(tree: ast.AST) -> set[str]:
    """Option names a module reads or sets on the parsed arguments, under any import alias."""
    aliases = {"args"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "cli_args":
            aliases.update(alias.asname for alias in node.names if alias.name == "args" and alias.asname)

    def is_args(node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id in aliases) or (
            isinstance(node, ast.Attribute) and node.attr == "args")

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and is_args(node.value):
            names.add(node.attr)
        elif (isinstance(node, ast.Call) and len(node.args) >= 2 and is_args(node.args[0])
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            names.add(node.args[1].value)  # getattr, setattr, monkeypatch.setattr
    return names


def bound_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
        return [target.id for target in node.targets]
    if isinstance(node, ast.Import):
        return [(alias.asname or alias.name).split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [alias.asname or alias.name for alias in node.names]
    return []


def references(nodes) -> set[str]:
    found = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                found.add(child.id)
            elif isinstance(child, ast.Attribute):
                found.add(child.attr)
    return found


def cli_args_names(tree: ast.AST) -> set[str]:
    """Names another module takes from cli_args."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "cli_args":
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute) and ast.unparse(node.value).split(".")[-1] == "cli_args":
            found.add(node.attr)
    return found


def option_destination(node: ast.stmt) -> str | None:
    call = node.value if isinstance(node, ast.Expr) else None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"):
        return None
    for keyword in call.keywords:
        if keyword.arg == "dest":
            return keyword.value.value
    flags = [argument.value for argument in call.args if isinstance(argument, ast.Constant)]
    flag = next((flag for flag in flags if flag.startswith("--")), flags[0])
    return flag.lstrip("-").replace("-", "_")


def strip_application_options(source: str, kept_options: set[str], external_names: set[str]) -> str:
    """Remove command-line options, and code that exists only for them, that nothing here reads.

    An option stays when `kept_options` names its destination. The same rule runs on
    every upstream revision, so removed options never appear in a three-way merge.
    """
    module = ast.parse(source)
    destinations = {index: option_destination(node) for index, node in enumerate(module.body)}
    dropped = {name for name in destinations.values() if name} - kept_options
    while True:
        removed = {index for index, name in destinations.items() if name in dropped}
        lines_removed: set[int] = set()
        still_read: set[str] = set()
        for index, node in enumerate(module.body):
            if index in removed:
                continue
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if option_reads(node) & dropped:
                    removed.add(index)  # external_names keeps the options of helpers in use
            elif not isinstance(node, ast.If):
                still_read |= option_reads(node) & dropped
            elif option_reads(node.test) & dropped:
                removed.add(index)
            else:
                stale = [child for child in node.body if option_reads(child) & dropped]
                if len(stale) == len(node.body) and not node.orelse:
                    removed.add(index)
                else:
                    for child in stale:
                        lines_removed.update(range(child.lineno, child.end_lineno + 1))
                    still_read |= {name for child in node.orelse for name in option_reads(child)} & dropped
        if not still_read:
            break
        dropped -= still_read  # an option that remaining code reads is not removable
    changed = True
    while changed:
        changed = False
        remaining = [node for index, node in enumerate(module.body) if index not in removed]
        for index, node in enumerate(module.body):
            names = bound_names(node)
            if index in removed or not names or {"args", "parser"} & set(names):
                continue
            used = references(other for other in remaining if other is not node)
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                used |= external_names
            if not set(names) & used:
                removed.add(index)
                changed = True
    remaining = [node for index, node in enumerate(module.body) if index not in removed]
    gone = {name for index in removed for name in bound_names(module.body[index])}
    gone -= {name for node in remaining for name in bound_names(node)}
    if gone & references(remaining):
        raise SystemExit(f"cli_args.py still uses removed names: {sorted(gone & references(remaining))}")
    lines = source.splitlines(keepends=True)
    for index in removed:
        node = module.body[index]
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while first > 1 and lines[first - 2].lstrip().startswith("#"):
            first -= 1  # a comment directly above describes the removed statement
        lines_removed.update(range(first, node.end_lineno + 1))
    kept = "".join(line for number, line in enumerate(lines, start=1) if number not in lines_removed)
    return re.sub(r"\n{4,}", "\n\n\n", kept)


def strip_excluded_imports(path: Path, upstream: Path) -> bytes:
    """Remove imports of excluded modules and the `__all__` entries they fed."""
    source = path.read_text()
    module = ast.parse(source)
    package = path.parent.relative_to(upstream).as_posix()
    lines_removed: set[int] = set()
    names = set()
    for node in module.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        base = package if node.level else ""
        for _ in range(max(node.level - 1, 0)):
            base = base.rpartition("/")[0]
        target = "/".join(part for part in (base, (node.module or "").replace(".", "/")) if part)
        if f"{target}.py".startswith(EXCLUDED) or f"{target}/".startswith(EXCLUDED):
            names.update(alias.asname or alias.name for alias in node.names)
            lines_removed.update(range(node.lineno, node.end_lineno + 1))
    if not names:
        return path.read_bytes()
    for node in module.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "__all__":
            for element in node.value.elts:
                if references([element]) & names:
                    lines_removed.update(range(element.lineno, element.end_lineno + 1))
    lines = source.splitlines(keepends=True)
    kept = "".join(line for number, line in enumerate(lines, start=1) if number not in lines_removed)
    if references([ast.parse(kept)]) & names:
        raise SystemExit(f"{path.relative_to(upstream)} uses an excluded module")
    return kept.encode()


def copy_tree(source: Path, target: Path, upstream: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(upstream).as_posix()
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".md":
            continue
        if relative.startswith(EXCLUDED):
            continue
        if relative.startswith(TEST_ROOTS) and path.suffix == ".py" and imports_application(path, upstream):
            continue
        output = target / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(strip_excluded_imports(path, upstream) if path.suffix == ".py" else path.read_bytes())


def transform(upstream: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source, target in MAPPED_ROOTS:
        copy_tree(upstream / source, destination / target, upstream)
    for path in destination.rglob("*.py"):
        rewrite(path)
    for source, target in MAPPED_FILES:
        if (upstream / source).exists():
            output = destination / target
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes((upstream / source).read_bytes())
    rewrite(destination / "node_helpers.py")
    specification = destination / "docs/quantization.md"
    if specification.exists():
        # Its implementation links are relative to the upstream repository root.
        specification.write_text(specification.read_text().replace("](comfy/", "](../dinkster_inference/"))
    options = destination / "dinkster_inference/cli_args.py"
    if not options.exists():
        return
    reads, names = set(CONSUMER_NAMES), set(CONSUMER_NAMES)
    for path in destination.rglob("*.py"):
        if path != options:
            tree = ast.parse(path.read_text())
            reads |= option_reads(tree)
            names |= cli_args_names(tree)
    for node in ast.parse(options.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            reads |= option_reads(node)
    options.write_text(strip_application_options(options.read_text(), reads, names))


def merge_file(current: Path, baseline: Path | None, target: Path | None, relative: Path) -> None:
    if baseline is None:
        if target is None:
            return
        if current.exists():
            if current.read_bytes() == target.read_bytes():
                return  # this upstream addition is already carried
            raise SystemExit(f"fork-only file conflicts with upstream addition: {relative}")
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
    result = subprocess.run(
        ["git", "merge-file", "-p", str(current), str(baseline), str(target)],
        capture_output=True,
    )
    if result.returncode < 0 or result.returncode > 127:
        raise SystemExit(f"could not merge {relative}: {result.stderr.decode().strip()}")
    if result.returncode:
        raise SystemExit(f"cannot carry fork change through upstream change: {relative}")
    current.write_bytes(result.stdout)


def requirement_specifiers(upstream: Path) -> dict[str, str]:
    found = {}
    path = upstream / "requirements.txt"
    for line in path.read_text().splitlines() if path.exists() else ():
        match = re.match(r"([A-Za-z0-9][\w.-]*)\s*([^#]*)", line.strip())
        if match:
            found[match[1].lower().replace("_", "-")] = match[2].strip()
    return found


def carry_dependency_specifiers(baseline: Path, upstream: Path, destination: Path) -> None:
    """Move a declared dependency's version specifier when upstream moves its own."""
    project = destination / "pyproject.toml"
    if not project.exists():
        return
    before, after = requirement_specifiers(baseline), requirement_specifiers(upstream)

    def carry(match: re.Match) -> str:
        name, current = match[1], match[2]
        key = name.lower().replace("_", "-")
        old, new = before.get(key), after.get(key)
        if old is None or new is None or old == new or current == new:
            return match[0]
        if current != old:
            raise SystemExit(f"cannot carry fork dependency specifier through upstream change: {name}")
        return f'"{name}{new}"'

    text = project.read_text()
    block = re.search(r"^dependencies = \[\n(.*?)^\]", text, re.DOTALL | re.MULTILINE)
    if block is None:
        return
    updated = re.sub(r'"([A-Za-z0-9][\w.-]*)([^"]*)"', carry, block[1])
    if updated != block[1]:
        project.write_text(text[:block.start(1)] + updated + text[block.end(1):])


def update_inventory(destination: Path, added: list[str], deleted: list[str]) -> None:
    """List files the transformed upstream delta added, and unlist the ones it deleted."""
    inventory = destination / INVENTORY
    if not inventory.exists():
        return
    entries = dict(reversed(line.split(" ", 1)) for line in inventory.read_text().splitlines())
    for path in deleted:
        if not (destination / path).exists():
            entries.pop(path, None)
    for path in added:
        entries.setdefault(path, classify(path))
    inventory.write_text("".join(f"{category} {path}\n" for path, category in sorted(entries.items())))


def regenerate(baseline: Path, upstream: Path, destination: Path) -> tuple[list[str], list[str]]:
    with tempfile.TemporaryDirectory() as temporary:
        transformed_baseline = Path(temporary) / "baseline"
        transformed_target = Path(temporary) / "target"
        transform(baseline, transformed_baseline)
        transform(upstream, transformed_target)
        before = {path.relative_to(transformed_baseline)
                  for path in transformed_baseline.rglob("*") if path.is_file()}
        after = {path.relative_to(transformed_target)
                 for path in transformed_target.rglob("*") if path.is_file()}
        for relative in sorted(before | after):
            source = transformed_baseline / relative
            target = transformed_target / relative
            merge_file(destination / relative, source if source.exists() else None,
                       target if target.exists() else None, relative)
    added = sorted(path.as_posix() for path in after - before)
    deleted = sorted(path.as_posix() for path in before - after)
    update_inventory(destination, added, deleted)
    carry_dependency_specifiers(baseline, upstream, destination)
    return added, deleted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    added, deleted = regenerate(args.baseline, args.upstream, args.destination)
    for sign, paths in (("+", added), ("-", deleted)):
        for path in paths:
            sys.stdout.write(f"{sign} {path}\n")


if __name__ == "__main__":
    main()
