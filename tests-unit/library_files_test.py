from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("restored", [
    "CODEOWNERS",
    "dinkster_inference/comfy_types/README.md",
    "dinkster_inference/comfy_types/examples/example_nodes.py",
    "dinkster_inference/new_application.py",
])
def test_file_policy_rejects_unclassified_additions(tmp_path, restored):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/library_files.txt").write_text("tooling tools/library_files.txt\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    command = [sys.executable, str(ROOT / "tools/check_library_files.py"), "--root", str(tmp_path)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    path = tmp_path / restored
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("restored application file\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 1
    assert f"Unclassified repository file: {restored}" in result.stderr
