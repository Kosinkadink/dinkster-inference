from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("dependencies,source,expected", [
    ('"Pillow>=10", "PyYAML", "typing-extensions"',
     'from PIL import Image\nimport yaml\nfrom typing_extensions import TypedDict\nimport os\nfrom . import local\n', None),
    ('"Pillow"', 'import PIL\nimport importlib\nimportlib.import_module("requests")\n', "Undeclared dependency requests"),
    ('"Pillow"', 'import PIL\nimport importlib as imp\nimp.import_module("requests")\n', "Undeclared dependency requests"),
    ('"Pillow"', 'import PIL\nfrom importlib import import_module as load\nload("requests")\n', "Undeclared dependency requests"),
    ('"requests"', 'import importlib as imp\nimp.import_module("requests")\n', None),
    ('"requests"', 'from importlib import import_module as load\nload("requests")\n', None),
    ('"Pillow"', 'import PIL\nfrom pathlib import Path as load\nload("requests")\n', None),
    ('"Pillow"', 'import PIL\nimport importlib\nimportlib.import_module(name="requests")\n', "Undeclared dependency requests"),
    ('"Pillow"', 'import PIL\nimport importlib as imp\nimp.import_module(name="requests")\n', "Undeclared dependency requests"),
    ('"Pillow"', 'import PIL\nfrom importlib import import_module as load\nload(name="requests")\n', "Undeclared dependency requests"),
    ('"Pillow"', 'import PIL\n__import__(name="requests")\n', "Undeclared dependency requests"),
    ('"requests"', 'import importlib\nimportlib.import_module(name="requests")\n', None),
    ('"requests"', 'import importlib as imp\nimp.import_module(name="requests")\n', None),
    ('"requests"', 'from importlib import import_module as load\nload(name="requests")\n', None),
    ('"requests"', '__import__(name="requests")\n', None),
    ('"Pillow"', 'import PIL\nimport importlib as imp\nimp.import_module(package="local", name=".requests")\n', None),
    ('"Pillow"', 'import PIL\nfrom requests import Session\n', "Undeclared dependency requests"),
    ('"Pillow", "requests"', 'import PIL\n', "Unused dependency requests"),
    ('"Pillow"', 'import PIL\nif args.directml is not None:\n    import torch_directml\n', None),
    ('"Pillow"', 'import PIL\nimport torch_directml\n', "Undeclared dependency torch-directml"),
    ('"Pillow"', 'import PIL\nif args.directml is not None:\n    import requests\n', "Undeclared dependency requests"),
])
def test_dependency_check_rejects_missing_and_unused_runtime_packages(tmp_path, dependencies, source, expected):
    (tmp_path / "tools").mkdir()
    script = tmp_path / "tools/check_dependencies.py"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "tools/check_dependencies.py", script)
    (tmp_path / "pyproject.toml").write_text(f"[project]\ndependencies = [{dependencies}]\n")
    (tmp_path / "dinkster_inference").mkdir()
    (tmp_path / "dinkster_inference/model.py").write_text(source)
    (tmp_path / "node_helpers.py").write_text("")
    result = subprocess.run([sys.executable, str(script)], text=True, capture_output=True)
    if expected is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 1
        assert expected in result.stderr
