import json
import os
import shutil
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


@pytest.mark.parametrize("ci_branch,ci_source,expected", [("main", "selected", True), ("wrong", "selected", False), ("main", "wrong", False), (None, None, False)])
def test_release_tag_waits_for_exact_source_push_ci(tmp_path, ci_branch, ci_source, expected):
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/inference-release-tags.yml").read_text())
    [command] = [step["run"] for step in workflow["jobs"]["tag"]["steps"]
                 if step.get("name") == "Create immutable source tags"]
    (tmp_path / "tools").mkdir()
    release = {"version": "1.2.3.dev9", "sourceCommit": "selected", "upstreamCommit": "upstream"}
    (tmp_path / "tools/inference_release.py").write_text(f"print({json.dumps(json.dumps([release]))})\n")
    runs = [] if ci_branch is None else [{"head_sha": ci_source, "head_branch": ci_branch, "event": "push"}]
    (tmp_path / "runs.json").write_text(json.dumps({"workflow_runs": runs}))
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/bash\nset -eu\nif [[ "$*" == *"/runs?"* ]]; then\ncat runs.json\nelse\necho "$*" >> mutations\necho object\nfi\n')
    gh.chmod(0o755)
    environment = {**os.environ, "PATH": os.pathsep.join((str(tmp_path), str(Path(sys.executable).parent), os.environ["PATH"])),
                   "SOURCE": "source", "REPOSITORY": "owner/repo", "RUNNER_TEMP": str(tmp_path),
                   "GITHUB_OUTPUT": str(tmp_path / "outputs"), "REPOSITORY_TOKEN": "fixture"}
    subprocess.run(["bash", "-eu", "-c", command], cwd=tmp_path, env=environment, check=True)
    mutations = tmp_path / "mutations"
    assert mutations.exists() == expected
    if expected:
        assert len(mutations.read_text().splitlines()) == 2
        assert "refs/tags/v1.2.3.dev9" in mutations.read_text()


@pytest.mark.parametrize("changes", ["library", "application", "historical"])
def test_stable_release_uses_tag_scope_and_retains_fork_changes(tmp_path, changes, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    # Disposable histories inherit source identity rather than requiring host Git configuration.
    for name, field in (("AUTHOR_NAME", "%an"), ("AUTHOR_EMAIL", "%ae"),
                        ("COMMITTER_NAME", "%cn"), ("COMMITTER_EMAIL", "%ce")):
        identity = subprocess.check_output(
            ["git", "-C", str(root), "show", "-s", f"--format={field}", "HEAD"], text=True,
        ).strip()
        monkeypatch.setenv(f"GIT_{name}", identity)
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    def commit(message):
        git("add", "-A")
        git("commit", "-m", message)
        return git("rev-parse", "HEAD")

    git("init", "-b", "upstream")
    (repo / "comfy").mkdir()
    model = "TAG = 0\n" + "\n" * 20 + "FORK = 0\n"
    (repo / "comfy/model.py").write_text(model)
    (repo / "node_helpers.py").write_text("")
    (repo / "nodes.py").write_text("APPLICATION = True\n")
    original = "tests-unit/comfy_test/application_test.py"
    adaptation = "tests-unit/comfy_test/adapted_test.py"
    (repo / original).parent.mkdir(parents=True)
    (repo / original).write_text("import nodes\n")
    base = commit("Upstream stable baseline")
    git("update-ref", "refs/upstream-tags/v0.45.0", base)
    git("switch", "-c", "upstream-stable")
    if changes == "library":
        (repo / "comfy/model.py").write_text(model.replace("TAG = 0", "TAG = 3"))
        (repo / original).unlink()
    (repo / "partner_nodes.py").write_text("EXCLUDED = True\n")
    target = commit("Stable tag branch")
    git("update-ref", "refs/upstream-tags/v0.45.1", target)
    git("switch", "-c", "main", base)
    shutil.rmtree(repo / "comfy")
    (repo / original).unlink()
    (repo / "nodes.py").unlink()
    (repo / "dinkster_inference").mkdir()
    (repo / "dinkster_inference/model.py").write_text(model.replace("FORK = 0", "FORK = 19"))
    (repo / "dinkster_inference/fork_only.py").write_text("OWNED = 23\n")
    (repo / "docs").mkdir()
    (repo / "docs/ledger.md").write_text(f"Upstream fork point: `{base}`.\n")
    (repo / "pyproject.toml").write_text('[project]\nversion = "0.1.2"\ndependencies = []\n')
    tools = Path(__file__).resolve().parents[1] / "tools"
    shutil.copytree(tools, repo / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "tools/fork_differences.txt").write_text("dinkster_inference/model.py\n")
    (repo / "tools/adapted_upstream_tests.json").write_text(json.dumps({adaptation: original}))
    (repo / adaptation).write_text("def test_adapted():\n    assert False, 'retained adaptation blocks CI'\n")
    fork_test = "tests-unit/comfy_test/fork_only_test.py"
    (repo / fork_test).write_text("def test_fork_only():\n    assert True\n")
    inventory = repo / "tools/library_files.txt"
    inventory.write_text("".join(f"tooling {p.relative_to(repo).as_posix()}\n"
                                 for p in sorted(repo.rglob("*"))
                                 if p.is_file() and ".git" not in p.parts))
    baseline = commit("Fork stable baseline")
    git("tag", "v0.1.2")
    if changes != "application":
        git("switch", "-c", "upstream-master", base)
        (repo / "comfy/model.py").write_text(model.replace("TAG = 0", "TAG = 7"))
        (repo / "comfy/master_only.py").write_text("MASTER_ONLY = 11\n")
        master = commit("Master-only inference changes")
        git("switch", "main")
        (repo / "dinkster_inference/model.py").write_text(
            model.replace("TAG = 0", "TAG = 7").replace("FORK = 0", "FORK = 19"))
        (repo / "dinkster_inference/master_only.py").write_text("MASTER_ONLY = 11\n")
        inventory.write_text(inventory.read_text() + "library dinkster_inference/master_only.py\n")
        (repo / "docs/ledger.md").write_text(f"Upstream fork point: `{master}`.\n")
        source = commit("Synced master")
    else:
        source = baseline
    previous = None
    if changes == "library":
        (repo / "docs/ledger.md").write_text(f"Upstream fork point: `{target}`.\n")
        git("add", "docs/ledger.md")
        previous_identity = {"upstreamTag": "v0.45.1", "upstreamTagCommit": target,
                             "version": "0.1.3", "mainSourceCommit": source}
        previous = git("commit-tree", git("write-tree"), "-p", source, "-p", target,
                       "-m", "Old preparation\n\n" + json.dumps(previous_identity))
        git("restore", "--source=HEAD", "--staged", "--worktree", "docs/ledger.md")
        git("update-ref", "refs/remotes/origin/release/upstream-v0.45.1", previous)
        git("update-ref", "refs/heads/release/upstream-v0.45.1", source)
    command = [sys.executable, str(repo / "tools/inference_release.py"), "plan", "--source", source]
    releases = json.loads(subprocess.check_output(command, cwd=repo, text=True))
    stable = next(row for row in releases if row.get("upstreamTag") == "v0.45.1")
    assert stable["upstreamTagCommit"] == target
    assert stable["version"] == "0.1.3"
    expected_model = model.replace("FORK = 0", "FORK = 19")
    if changes == "library":
        assert stable["releaseSource"] == "upstream-tag-branch"
        assert stable["branch"] == "release/upstream-v0.45.1"
        expected_model = expected_model.replace("TAG = 0", "TAG = 3")
        assert git("show", f"{stable['sourceCommit']}^1:docs/ledger.md") == git("show", f"{source}:docs/ledger.md")
    else:
        assert stable["releaseSource"] == "synced-main"
        assert stable["sourceCommit"] == baseline
        assert "branch" not in stable
    selected = stable["sourceCommit"]
    assert git("show", f"{selected}:dinkster_inference/model.py") == expected_model.strip()
    assert git("show", f"{selected}:dinkster_inference/fork_only.py") == "OWNED = 23"
    files = git("ls-tree", "-r", "--name-only", selected).splitlines()
    assert "partner_nodes.py" not in files
    assert "dinkster_inference/master_only.py" not in files
    assert fork_test in files
    assert (adaptation in files) == (changes != "library")
    assert stable["adaptedTestSources"] == {adaptation: original}
    assert stable["excludedAdaptedTests"] == ({adaptation: original} if changes == "library" else {})
    if previous:
        git("merge-base", "--is-ancestor", previous, selected)
        assert adaptation not in git("show", f"{selected}:tools/library_files.txt")
    assert git("rev-parse", "HEAD") == source
    assert json.loads(subprocess.check_output(command, cwd=repo, text=True)) == releases
    git("tag", "-a", "v0.1.3", selected, "-m", json.dumps(stable))
    if changes == "historical":
        workflow = yaml.safe_load((tools.parent / ".github/workflows/release.yml").read_text())
        [command] = [step["run"] for step in workflow["jobs"]["build"]["steps"]
                     if step.get("name") == "Select and validate immutable release source"]
        manifest = tmp_path / "stable-main-release"
        manifest.mkdir()
        (manifest / "inference-main-release.json").write_text(json.dumps([stable]))
        environment = {**os.environ, "HISTORICAL": "true", "RELEASE_TAG": "",
                       "RUNNER_TEMP": str(tmp_path), "GITHUB_OUTPUT": str(tmp_path / "outputs")}
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
        subprocess.run(["bash", "-eu", "-c", command], cwd=repo, env=environment, check=True)
        assert git("rev-parse", "HEAD") == selected
        assert (tmp_path / "outputs").read_text() == "version=0.1.3\n"
    else:
        git("clone", "--bare", str(repo), str(tmp_path / "origin"))
        git("remote", "add", "origin", str(tmp_path / "origin"))
        git("switch", "--detach", selected)
        # actions/checkout replaces the fetched annotated tag with its event commit.
        git("update-ref", "refs/tags/v0.1.3", selected)
        workflow = yaml.safe_load((tools.parent / ".github/workflows/release.yml").read_text())
        [command] = [step["run"] for step in workflow["jobs"]["build"]["steps"]
                     if step.get("name") == "Select and validate immutable release source"]
        environment = {**os.environ, "HISTORICAL": "false", "RELEASE_TAG": "v0.1.3",
                       "GITHUB_OUTPUT": str(tmp_path / "outputs")}
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
        subprocess.run(["bash", "-eu", "-c", command], cwd=repo, env=environment, check=True)
    assert json.loads((repo / "dinkster_inference/_release.json").read_text()) == stable
    assert json.loads((repo / "release-provenance.json").read_text()) == stable
    if changes != "library":
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", adaptation], cwd=repo,
                                text=True, capture_output=True)
        assert result.returncode == 1
        assert "retained adaptation blocks CI" in result.stdout
