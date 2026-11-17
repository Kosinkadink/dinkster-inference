"""Plan immutable inference tags and embed their source identity in distributions."""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

if __package__:
    from tools.regenerate_upstream_library import MAPPED_FILES, MAPPED_ROOTS, regenerate, update_inventory
else:
    from regenerate_upstream_library import MAPPED_FILES, MAPPED_ROOTS, regenerate, update_inventory


ROOT = Path(__file__).resolve().parents[1]
SCOPE = tuple(target for _, target in MAPPED_ROOTS + MAPPED_FILES)


def git(*arguments):
    return subprocess.check_output(["git", *arguments], text=True).strip()


def upstream_at(commit):
    ledger = git("show", f"{commit}:docs/ledger.md")
    match = re.search(r"Upstream fork point:.*?`([0-9a-f]{40})`", ledger, re.DOTALL)
    if match is None:
        raise ValueError(f"No upstream commit recorded at {commit}")
    return match.group(1)


def contains(descendant, ancestor):
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], check=False
    )
    if result.returncode not in (0, 1):
        raise ValueError("Cannot establish release ancestry")
    return result.returncode == 0


def library_identity(commit):
    files = git("ls-tree", "-r", commit, "--", *SCOPE)
    dependencies = tomllib.loads(git("show", f"{commit}:pyproject.toml"))["project"]["dependencies"]
    return files, sorted(dependencies)


def stable_source(source, baseline, tag, upstream, version):
    branch = f"release/upstream-{tag}"
    adapted = json.loads(git("show", f"{source}:tools/adapted_upstream_tests.json"))
    local = git("for-each-ref", f"refs/heads/{branch}", "--format=%(objectname)")
    remote = git("for-each-ref", f"refs/remotes/origin/{branch}", "--format=%(objectname)")
    existing = local if local and remote and contains(local, remote) else remote or local
    if existing:
        commit = existing
        identity = json.loads(git("show", "-s", "--format=%B", commit).split("\n\n", 1)[1])
        if (identity["upstreamTag"] != tag or identity["upstreamTagCommit"] != upstream
                or identity["version"] != version or not contains(source, identity["mainSourceCommit"])
                or upstream_at(commit) != upstream):
            raise ValueError(f"Release branch does not match stable source: {branch}")
        if identity.get("adaptedTestSources") == adapted:
            return {**identity, "sourceCommit": commit, "upstreamCommit": upstream, "branch": branch}

    recorded = upstream_at(source)
    merge_base = git("merge-base", recorded, upstream)
    identity = {
        "version": version,
        "upstreamTag": tag,
        "upstreamTagCommit": upstream,
        "upstreamMergeBase": merge_base,
        "mainSourceCommit": source,
        "releaseSource": "upstream-tag-branch",
        "adaptedTestSources": adapted,
    }
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        checkouts = []
        try:
            for name, commit in (("fork", source), ("recorded", recorded),
                                 ("base", merge_base), ("tag", upstream)):
                path = directory / name
                git("worktree", "add", "--detach", str(path), commit)
                checkouts.append(path)
            fork, recorded_tree, base_tree, tag_tree = checkouts
            # Remove master-only upstream changes before applying the stable branch.
            # Both passes preserve fork changes with the sync's same three-way rules.
            regenerate(recorded_tree, base_tree, fork)
            regenerate(base_tree, tag_tree, fork)
            excluded = {path: original for path, original in adapted.items()
                        if not (tag_tree / original).is_file()}
            for path in excluded:
                (fork / path).unlink(missing_ok=True)
            update_inventory(fork, [], list(excluded))
            identity["excludedAdaptedTests"] = excluded
            ledger = fork / "docs/ledger.md"
            updated, count = re.subn(
                r"(Upstream fork point:.*?`)[0-9a-f]{40}(`)",
                rf"\g<1>{upstream}\2", ledger.read_text(), count=1, flags=re.DOTALL,
            )
            if count != 1:
                raise ValueError("Cannot update stable source ledger")
            ledger.write_text(updated)
            git("-C", str(fork), "add", "-A")
            for tool in ("check_library_files.py", "check_dependencies.py"):
                subprocess.run([sys.executable, str(fork / "tools" / tool)], check=True, stdout=sys.stderr)
            tree = git("-C", str(fork), "write-tree")
            target_identity = library_identity(tree)
            for main in [source] + git("rev-list", "--first-parent", f"{baseline}..{source}").splitlines() + [baseline]:
                if library_identity(main) == target_identity:
                    main = git("rev-parse", f"{main}^{{commit}}")
                    return {**identity, "releaseSource": "synced-main", "sourceCommit": main,
                            "upstreamCommit": upstream_at(main)}
            message = f"Prepare inference stable source for {tag}\n\n{json.dumps(identity, sort_keys=True)}\n"
            parents = ["-p", source, "-p", upstream]
            if existing:
                parents += ["-p", existing]
            commit = subprocess.check_output(
                ["git", "commit-tree", tree, *parents], input=message, text=True,
            ).strip()
            git("-C", str(fork), "reset", "--hard", commit)
            subprocess.run(
                [sys.executable, str(fork / "tools/check_upstream_parity.py"),
                 "--baseline", str(base_tree), "--upstream", str(tag_tree), "--destination", str(fork)],
                check=True, stdout=sys.stderr,
            )
            subprocess.run(
                [sys.executable, str(fork / "tools/check_upstream_parity.py"), "--differences",
                 "--declared", str(fork / "tools/fork_differences.txt"),
                 "--upstream", str(tag_tree), "--destination", str(fork)], check=True, stdout=sys.stderr,
            )
            git("update-ref", f"refs/heads/{branch}", commit, local or "0" * 40)
            return {**identity, "sourceCommit": commit, "upstreamCommit": upstream, "branch": branch}
        finally:
            for path in reversed(checkouts):
                git("worktree", "remove", "--force", str(path))


def release_plan(source):
    source = git("rev-parse", f"{source}^{{commit}}")
    stable = []
    manual_stable = []
    published_upstream = set()
    for tag in git("tag", "--list", "v*").splitlines():
        match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
        if match:
            stable.append((tuple(map(int, match.groups())), tag))
            annotation = git("for-each-ref", f"refs/tags/{tag}", "--format=%(contents)")
            try:
                identity = json.loads(annotation)
            except json.JSONDecodeError:
                identity = {}
            if isinstance(identity, dict) and identity.get("upstreamTag"):
                published_upstream.add(identity["upstreamTag"])
            else:
                manual_stable.append((tuple(map(int, match.groups())), tag))
    if not manual_stable:
        raise ValueError("An initial stable inference tag is required")
    latest = max(stable)[0]
    # Manual releases before automation already represent their upstream history.
    baseline = max(row for row in manual_stable if contains(source, row[1]))[1]
    previous_upstream = upstream_at(baseline)
    current_upstream = upstream_at(source)
    if not contains(current_upstream, previous_upstream):
        raise ValueError("Current upstream does not contain the last stable release")
    releases = []
    # Recover every pending sync even if its CI event was superseded by newer main.
    previous = previous_upstream
    for commit in git(
        "rev-list", "--first-parent", "--reverse", f"{baseline}..{source}"
    ).splitlines():
        upstream = upstream_at(commit)
        changed, previous = upstream != previous, upstream
        prereleases = git("tag", "--points-at", commit, "--list", "v*.dev*").splitlines()
        if (
            not changed
            or prereleases
            or not git("ls-tree", "--name-only", commit, "tools/inference_release.py")
        ):
            continue
        version = f"{latest[0]}.{latest[1]}.{latest[2] + 1}.dev{git('rev-list', '--count', commit)}"
        if git("tag", "--list", f"v{version}"):
            raise ValueError(f"Prerelease version already identifies another source: {version}")
        releases.append(
            {
                "version": version,
                "sourceCommit": commit,
                "upstreamCommit": upstream,
            }
        )
    upstream_tags = []
    represented = []
    for ref in git(
        "for-each-ref", "refs/upstream-tags/", "--format=%(refname)"
    ).splitlines():
        tag = ref.removeprefix("refs/upstream-tags/")
        match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
        if match and tag not in published_upstream:
            commit = git("rev-parse", f"{ref}^{{commit}}")
            number = tuple(map(int, match.groups()))
            if contains(previous_upstream, commit):
                represented.append(number)
            else:
                upstream_tags.append((number, tag, commit))
    # The initial manual stable also predates older maintenance-branch tags.
    upstream_tags = [row for row in upstream_tags if not represented or row[0] > max(represented)]
    # Publish in upstream version order so a pending branch keeps its reserved version.
    if upstream_tags:
        _, tag, commit = min(upstream_tags)
        version = f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
        releases.append(stable_source(source, baseline, tag, commit, version))
    return releases


def prepare(version):
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:\.dev\d+)?", version):
        raise ValueError("Release version must be stable or a development prerelease")
    source = git("rev-parse", "HEAD")
    tag_source = git("rev-parse", f"v{version}^{{commit}}")
    if tag_source != source:
        raise ValueError("Release tag does not identify the checked-out source")
    identity = {
        "version": version,
        "sourceCommit": source,
        "upstreamCommit": upstream_at(source),
    }
    annotation = git("for-each-ref", f"refs/tags/v{version}", "--format=%(contents)")
    try:
        recorded = json.loads(annotation)
    except json.JSONDecodeError:
        recorded = {}
    if recorded:
        if not isinstance(recorded, dict) or any(recorded.get(key) != value for key, value in identity.items()):
            raise ValueError("Release annotation does not match source identity")
        identity = recorded
    project = Path("pyproject.toml")
    updated, count = re.subn(
        r'^version = "[^"]+"$',
        f'version = "{version}"',
        project.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Expected one project version")
    project.write_text(updated)
    Path("dinkster_inference/_release.json").write_text(
        json.dumps(identity, sort_keys=True) + "\n"
    )
    Path("release-provenance.json").write_text(
        json.dumps(identity, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--source", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--version", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(release_plan(args.source), sort_keys=True))  # noqa: T201
    else:
        prepare(args.version)


if __name__ == "__main__":
    main()
