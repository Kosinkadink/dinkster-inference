import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize("tag", ["v1.2.3", "v1.2.4.dev7", "v1.2.4rc1", ""])
def test_release_builds_source_distribution_only_for_stable_tags(tmp_path, tag):
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / ".github/workflows/release.yml").read_text()
    )
    [command] = [
        step["run"] for step in workflow["jobs"]["build"]["steps"]
        if "python -m build" in step.get("run", "")
    ]
    version = tag.removeprefix("v") or "1.2.3"
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=77"]\n'
        'build-backend = "setuptools.build_meta"\n'
        f'[project]\nname = "release-fixture"\nversion = "{version}"\n'
    )
    (tmp_path / "release_fixture.py").write_text("VALUE = 19\n")
    environment = {**os.environ, "RELEASE_TAG": tag}
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    subprocess.run(["bash", "-eu", "-c", command], cwd=tmp_path, env=environment, check=True)
    assert [path.name for path in (tmp_path / "dist").glob("*.whl")] == [
        f"release_fixture-{version}-py3-none-any.whl"
    ]
    assert [path.name for path in (tmp_path / "dist").glob("*.tar.gz")] == (
        [f"release_fixture-{version}.tar.gz"] if tag == "v1.2.3" else []
    )
