"""Check installed library imports against its runtime dependency declarations."""

import ast
from pathlib import Path
import re
import sys
import tomllib


# These backends are probed with a working fallback, not required by the library.
OPTIONAL_IMPORTS = {
    "cv2",  # SAM3 uses scipy.ndimage when OpenCV is absent.
    "cublas_ops", "flash_attn", "sageattention", "sageattn3", "xformers",
    "torch_mlu", "torch_npu", "triton",
}
DISTRIBUTIONS = {"PIL": "pillow", "yaml": "pyyaml"}


def check(root):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    declared = {
        re.match(r"[\w.-]+", requirement)[0].lower().replace("_", "-")
        for requirement in project["dependencies"]
    }
    imports = {}
    for path in sorted(root.glob("dinkster_inference/**/*.py")) + [root / "node_helpers.py"]:
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) in {"importlib.import_module", "__import__"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and not node.args[0].value.startswith(".")
            ):
                names = [node.args[0].value]
            else:
                continue
            for name in names:
                module = name.split(".")[0]
                if module in sys.stdlib_module_names or module in {"comfy", "dinkster_inference", "node_helpers"}:
                    continue
                if module not in OPTIONAL_IMPORTS:
                    distribution = DISTRIBUTIONS.get(module, module.lower().replace("_", "-"))
                    imports.setdefault(distribution, f"{path.relative_to(root)}:{node.lineno}")
    errors = [f"Undeclared dependency {name}: {imports[name]}" for name in sorted(imports.keys() - declared)]
    errors += [f"Unused dependency {name}" for name in sorted(declared - imports.keys())]
    return errors


if __name__ == "__main__":
    errors = check(Path(__file__).resolve().parents[1])
    for error in errors:
        sys.stderr.write(error + "\n")
    raise SystemExit(bool(errors))
