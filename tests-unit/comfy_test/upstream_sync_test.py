import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
STEPS = yaml.safe_load((ROOT / ".github/workflows/upstream-sync.yml").read_text())["jobs"]["sync"]["steps"]


def git(repo, *args, input=None):
    return subprocess.check_output(["git", *args], cwd=repo, input=input, text=True).strip()


@pytest.fixture
def sync_repo(tmp_path, request):
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    remote.mkdir()
    repo.mkdir()
    git(remote, "init", "--bare")
    git(repo, "init")
    git(repo, "remote", "add", "origin", str(remote))
    # Reuse real commit headers for disposable history without setting Git identity.
    headers = [line for line in git(ROOT, "cat-file", "commit", "HEAD").splitlines()
               if line.startswith(("author ", "committer "))]

    def commit(parent, files, subject):
        if parent:
            git(repo, "read-tree", parent)
        else:
            git(repo, "read-tree", "--empty")
        for name, value in files.items():
            if value is None:
                git(repo, "update-index", "--force-remove", name)
            else:
                blob = git(repo, "hash-object", "-w", "--stdin", input=value)
                git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{name}")
        tree = git(repo, "write-tree")
        lines = [f"tree {tree}"] + ([f"parent {parent}"] if parent else []) + headers
        return git(repo, "hash-object", "-t", "commit", "-w", "--stdin",
                   input="\n".join(lines) + "\n\n" + subject + "\n")

    recorded = commit(None, {"conflict.txt": "base\n"}, "Upstream base")
    old = commit(recorded, {"conflict.txt": "old upstream\n"}, "Upstream update")
    target = commit(old, {"conflict.txt": "new upstream\n"}, "New upstream update")
    ledger = f"Upstream fork point: `{recorded}`\n"
    base = commit(recorded, {
        "conflict.txt": "base\n" if getattr(request, "param", "conflict") == "clean" else "fork\n",
        "docs/ledger.md": ledger,
        "UPSTREAM_SYNC_GUARDED_PATHS": "docs/ledger.md\n",
        "tools/rewrite_upstream_imports.py": (ROOT / "tools/rewrite_upstream_imports.py").read_text(),
        "dinkster_inference/stub.py": "",
        "tests-unit/comfy_test/stub.py": "",
        "tests-unit/comfy_quant/stub.py": "",
        "tests-unit/deploy_environment_test.py": "",
    }, "Fork base")
    marker = commit(base, {"UPSTREAM_SYNC_CONFLICTS.md": f"Upstream commit: `{old}`\n"},
                    f"Record upstream sync conflicts through {old[:8]}")
    git(repo, "update-ref", "refs/heads/main", base)
    git(repo, "update-ref", "refs/remotes/upstream/master", target)
    git(repo, "push", "origin", "main", f"{marker}:refs/heads/upstream-sync")
    git(repo, "checkout", "-f", "main")

    state_path = tmp_path / "github.json"
    state_path.write_text(json.dumps({"number": "91", "draft": False, "comments": [],
                                     "headers": headers, "git": shutil.which("git"), "remote": str(remote)}))
    output = tmp_path / "output"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
p = Path(os.environ["FAKE_GITHUB"])
s = json.loads(p.read_text())
a = sys.argv[1:]
if a[:2] == ["pr", "list"]:
    print(s["number"])
elif a[0] == "api":
    print("\\n".join(s["comments"]))
elif a[:2] == ["pr", "comment"]:
    s["comments"].append(a[a.index("--body") + 1])
elif a[:2] == ["pr", "ready"]:
    s["draft"] = "--undo" in a
elif a[:2] == ["pr", "create"]:
    s["draft"] = "--draft" in a
    s["number"] = "91"
    print("https://github.com/example/fork/pull/91")
elif a[:2] != ["pr", "edit"]:
    raise SystemExit(a)
p.write_text(json.dumps(s))
''')
    gh.chmod(0o755)
    git_cli = bin_dir / "git"
    git_cli.write_text(f"#!{sys.executable}\n" + '''
import json, os, subprocess, sys
from pathlib import Path
s = json.loads(Path(os.environ["FAKE_GITHUB"]).read_text())
a = sys.argv[1:]
def real(*args, input=None):
    return subprocess.check_output([s["git"], *args], input=input, text=True).strip()
if a[0] == "merge-recursive" and "concurrent_head" in s:
    real("--git-dir=" + s["remote"], "update-ref", "refs/heads/upstream-sync", s["concurrent_head"])
if a[0] in ("commit", "commit-tree"):
    tree = real("write-tree") if a[0] == "commit" else a[1]
    parents = [real("rev-parse", "HEAD")] if a[0] == "commit" else [a[i+1] for i, v in enumerate(a) if v == "-p"]
    message = a[a.index("-m") + 1] if a[0] == "commit" else sys.stdin.read().strip()
    lines = ["tree " + tree] + ["parent " + p for p in parents] + s["headers"]
    commit = real("hash-object", "-t", "commit", "-w", "--stdin", input="\\n".join(lines) + "\\n\\n" + message + "\\n")
    if a[0] == "commit":
        real("update-ref", "HEAD", commit)
    print(commit)
else:
    os.execv(s["git"], [s["git"], *a])
''')
    git_cli.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "FAKE_GITHUB": str(state_path), "GITHUB_OUTPUT": str(output),
           "DRY_RUN": "false", "REPOSITORY": "example/fork"}

    def run(step_id, **values):
        git(repo, "reset", "--hard", "main")
        step = next(step for step in STEPS if step.get("id") == step_id)
        return subprocess.run(["bash", "-c", step["run"]], cwd=repo,
                              env={**env, **values}, text=True, capture_output=True)

    return repo, remote, commit, base, old, target, marker, state_path, output, run


@pytest.mark.parametrize("owner_change", ["library", "ledger", "marker-removal", "reverted-library"])
def test_sync_preserves_owner_history_and_reports_upstream(sync_repo, owner_change):
    repo, remote, commit, base, old, target, marker, state, output, run = sync_repo
    if owner_change == "ledger":
        files = {"docs/ledger.md": f"Upstream fork point: `{old}`\nOwner notes\n"}
    elif owner_change == "marker-removal":
        marker = commit(marker, {"dinkster_inference/owner.py": "owner work\n"}, "Owner library update")
        files = {"UPSTREAM_SYNC_CONFLICTS.md": None}
    else:
        files = {"dinkster_inference/owner.py": "owner work\n"}
    head = commit(marker, files, "Owner resolution")
    if owner_change == "reverted-library":
        head = commit(head, {"dinkster_inference/owner.py": None}, "Revert owner experiment")
    git(repo, "push", "origin", f"{head}:refs/heads/upstream-sync")

    for _ in range(2):
        result = run("upstream")
        assert result.returncode == 0, result.stderr
        assert output.read_text().splitlines()[-1] == "changed=false"
        assert git(remote, "rev-parse", "refs/heads/upstream-sync") == head
    github = json.loads(state.read_text())
    assert github["draft"] is False
    assert len(github["comments"]) == 1
    assert target in github["comments"][0]


@pytest.mark.parametrize("replacement", ["no-open-pr", "squash-merged", "squash-merged-before-main-tip"])
def test_sync_replaces_inactive_or_already_merged_owner_history(sync_repo, replacement):
    repo, remote, commit, base, old, target, marker, state, output, run = sync_repo
    files = {"docs/ledger.md": f"Upstream fork point: `{old}`\n", "UPSTREAM_SYNC_CONFLICTS.md": None,
             "dinkster_inference/owner.py": "owner work\n"}
    head = commit(marker, files, "Owner resolution")
    git(repo, "push", "origin", f"{head}:refs/heads/upstream-sync")
    if replacement == "no-open-pr":
        github = json.loads(state.read_text())
        github["number"] = ""
        state.write_text(json.dumps(github))
    else:
        main = commit(base, files, "Squash upstream sync")
        assert git(repo, "rev-parse", f"{main}^{{tree}}") == git(repo, "rev-parse", f"{head}^{{tree}}")
        if replacement == "squash-merged-before-main-tip":
            main = commit(main, {"dinkster_inference/later.py": "later main work\n"}, "Later main work")
        git(repo, "update-ref", "refs/heads/main", main)
        git(repo, "push", "origin", "main")
    result = run("upstream")
    assert result.returncode == 0, result.stderr
    new_head = git(remote, "rev-parse", "refs/heads/upstream-sync")
    assert new_head != head
    path = "UPSTREAM_SYNC_CONFLICTS.md" if "conflict=true" in output.read_text() else "docs/ledger.md"
    assert target in git(remote, "show", f"{new_head}:{path}")
    assert json.loads(state.read_text())["comments"] == []


@pytest.mark.parametrize("kind", ["marker", "ledger"])
@pytest.mark.parametrize("sync_repo", ["conflict", "clean"], indirect=True)
def test_sync_replaces_only_workflow_marker_or_ledger_history(sync_repo, kind):
    repo, remote, commit, base, old, target, marker, state, output, run = sync_repo
    if kind == "ledger":
        ledger = commit(base, {"docs/ledger.md": f"Upstream fork point: `{old}`\n"},
                        f"Sync ComfyUI upstream through {old[:8]}")
        git(repo, "push", "--force", "origin", f"{ledger}:refs/heads/upstream-sync")
    result = run("upstream")
    assert result.returncode == 0, result.stderr
    head = git(remote, "rev-parse", "refs/heads/upstream-sync")
    assert head != marker
    path = "UPSTREAM_SYNC_CONFLICTS.md" if "conflict=true" in output.read_text() else "docs/ledger.md"
    assert target in git(remote, "show", f"{head}:{path}")
    assert json.loads(state.read_text())["comments"] == []
    result = run("upstream_pr", TARGET=target, COMMITS="", CONFLICT="true",
                 CONFLICTS="conflict.txt", GUARDED="")
    assert result.returncode == 0, result.stderr
    assert json.loads(state.read_text())["draft"] is False


@pytest.mark.parametrize("sync_repo", ["conflict", "clean"], indirect=True)
def test_sync_rejects_an_owner_push_after_inspection(sync_repo):
    repo, remote, commit, base, old, target, marker, state, output, run = sync_repo
    owner = commit(marker, {"dinkster_inference/owner.py": "concurrent work\n"}, "Owner resolution")
    git(repo, "push", "origin", f"{owner}:refs/heads/concurrent-owner")
    github = json.loads(state.read_text())
    github["concurrent_head"] = owner
    state.write_text(json.dumps(github))
    result = run("upstream")
    assert result.returncode != 0
    assert git(remote, "rev-parse", "refs/heads/upstream-sync") == owner


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("conflict", [True, False])
def test_guarded_pr_draft_state_is_owned_by_the_owner(sync_repo, existing, conflict):
    repo, remote, commit, base, old, target, marker, state, output, run = sync_repo
    github = json.loads(state.read_text())
    github["number"] = "91" if existing else ""
    state.write_text(json.dumps(github))
    result = run("upstream_pr", TARGET=target, COMMITS="", CONFLICT=str(conflict).lower(),
                 CONFLICTS="conflict.txt", GUARDED="docs/ledger.md" if not conflict else "")
    assert result.returncode == 0, result.stderr
    assert json.loads(state.read_text())["draft"] is (not existing)
