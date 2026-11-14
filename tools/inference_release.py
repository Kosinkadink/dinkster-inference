"""Plan immutable inference tags and embed their source identity in distributions."""

import argparse
import json
import re
import subprocess
from pathlib import Path


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
    for ref in git(
        "for-each-ref", "refs/upstream-tags/", "--format=%(refname)"
    ).splitlines():
        tag = ref.removeprefix("refs/upstream-tags/")
        match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
        if match and tag not in published_upstream:
            commit = git("rev-parse", f"{ref}^{{commit}}")
            if contains(current_upstream, commit) and not contains(
                previous_upstream, commit
            ):
                upstream_tags.append((tuple(map(int, match.groups())), tag, commit))
    for offset, (_, tag, commit) in enumerate(sorted(upstream_tags), start=1):
        version = f"{latest[0]}.{latest[1]}.{latest[2] + offset}"
        releases.append(
            {
                "version": version,
                "sourceCommit": source,
                "upstreamCommit": current_upstream,
                "upstreamTag": tag,
                "upstreamTagCommit": commit,
            }
        )
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
