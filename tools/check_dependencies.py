"""Check installed library imports against its runtime dependency declarations."""

import ast
from pathlib import Path
import re
import sys
import tomllib


# An import inside a `try` body is a backend probe with a fallback, not a requirement.
# xformers is also imported behind the availability flag its probe sets.
OPTIONAL_IMPORTS = {"xformers"}
DISTRIBUTIONS = {"PIL": "pillow", "yaml": "pyyaml"}


def check(root):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    declared = {
        re.match(r"[\w.-]+", requirement)[0].lower().replace("_", "-")
        for requirement in project["dependencies"]
    }
    imports = {}
    probed = set()
    for path in sorted(root.glob("dinkster_inference/**/*.py")) + [root / "node_helpers.py"]:
        tree = ast.parse(path.read_text(), filename=str(path))
        dynamic_imports = {"importlib.import_module", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dynamic_imports.update(
                    f"{alias.asname or alias.name}.import_module"
                    for alias in node.names if alias.name == "importlib"
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "importlib":
                dynamic_imports.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "import_module"
                )
        probes = {id(child) for node in ast.walk(tree) if isinstance(node, ast.Try)
                  for statement in node.body for child in ast.walk(statement)}
        # Older stable sources load DirectML only when the caller selects it.
        directml = {id(child) for node in ast.walk(tree)
                    if isinstance(node, ast.If) and ast.unparse(node.test) == "args.directml is not None"
                    for statement in node.body for child in ast.walk(statement)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.Call) and ast.unparse(node.func) in dynamic_imports:
                name = node.args[0] if node.args else next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "name"), None,
                )
                if not isinstance(name, ast.Constant) or not isinstance(name.value, str) or name.value.startswith("."):
                    continue
                names = [name.value]
            else:
                continue
            for name in names:
                module = name.split(".")[0]
                if module in sys.stdlib_module_names or module in {"comfy", "dinkster_inference", "node_helpers"}:
                    continue
                distribution = DISTRIBUTIONS.get(module, module.lower().replace("_", "-"))
                if id(node) in probes or module in OPTIONAL_IMPORTS or (module == "torch_directml" and id(node) in directml):
                    probed.add(distribution)
                else:
                    imports.setdefault(distribution, f"{path.relative_to(root)}:{node.lineno}")
    errors = [f"Undeclared dependency {name}: {imports[name]}" for name in sorted(imports.keys() - declared)]
    errors += [f"Unused dependency {name}" for name in sorted(declared - imports.keys() - probed)]
    return errors


if __name__ == "__main__":
    errors = check(Path(__file__).resolve().parents[1])
    for error in errors:
        sys.stderr.write(error + "\n")
    raise SystemExit(bool(errors))
