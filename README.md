# dinkster-inference

Dinkster's inference library provides model loading, sampling, conditioning,
text/image/audio encoders, device management and inference runtime resources.
The `comfy` import name aliases `dinkster_inference` modules without duplicating
model-management state. The distribution does not include an application.
A module this library does not provide still resolves from a ComfyUI checkout's
`comfy/` directory on `sys.path`; library modules always take precedence.

```bash
python -m pip install .
```

CPU callers select CPU execution before importing model management or loaders:

```python
import dinkster_inference.cli_args as cli_args
cli_args.args.cpu = True
import comfy.sd
import dinkster_inference.sd
assert comfy.sd is dinkster_inference.sd
```

## Development and upstream sync

Runtime dependencies are declared only in `pyproject.toml`. Install test tools
with `python -m pip install -r tests-unit/requirements.txt ruff build PyYAML`.
Run the repository checks with:

```bash
python tools/check_library_files.py
python tools/check_dependencies.py
python -m ruff check dinkster_inference comfy node_helpers.py tools
python -m pytest -q tests-unit
python -m build
```

The library is generated from ComfyUI; [AGENTS.md](AGENTS.md) has the rules for
changing it and [docs/quantization.md](docs/quantization.md) specifies the
quantized checkpoint format. The [sync ledger](docs/ledger.md) records the
upstream commit and what the transform keeps.

`tools/regenerate_upstream_library.py --baseline /path/to/recorded-upstream
--upstream /path/to/new-upstream --destination .` transforms both revisions and
merges their difference into this tree, keeping fork changes.
`tools/check_upstream_parity.py` takes the same arguments and verifies that the
difference is present. With `--differences --upstream /path/to/recorded-upstream`
it lists every file that differs from upstream; CI runs that check with
`--declared tools/fork_differences.txt` and rejects a changed upstream file
that is not declared there.

`tools/library_files.txt` lists every tracked file. The scheduled sync adds and
removes the entries of files that upstream added or removed and names them in
its pull request. Any other new file needs an entry in the pull request that
adds it. Packaging includes only Python modules and runtime JSON, tokenizer
model and text resources.

`tools/compare_workflows.py` compares the installed wheel against a clean
ComfyUI checkout at the ledger's commit, in separate processes with the same
interpreter, models, seed and deterministic settings. `--help` lists its
inputs. It downloads nothing; a missing model file, a worker error or a
non-finite value is a failure, and the exit status is zero only when every
selected case is identical. `comparison.json` in the output directory records
input hashes, package provenance and per-case differences.

## Releases

Consumers pin exact versions and PyPI wheel hashes. After a landed upstream
sync passes main CI, automation publishes an immutable `vX.Y.Z.devN` release.
`X.Y.Z` is the next patch after the latest stable inference release; `N` is the
source commit count. Tags are kept even when a package is removed from PyPI,
so a version is never reused. An hourly check detects newly contained upstream stable
tags and publishes separate stable inference patch releases. Superseded CI
completions do not publish; the next successful current-main check recovers
unpublished syncs in source order.

Release tags set the built version without changing the source commit.
Distributions contain `dinkster_inference/_release.json` with the version,
inference source commit and upstream commit. Annotated tags record the same
identity. Development/prerelease publication is wheel-only; stable releases
also include source distributions. Failed publication never moves a tag.

Trusted PyPI publication uses repository `Kosinkadink/dinkster-inference`,
workflow `release.yml`, environment `pypi`; no PyPI token is stored. Branches
and pull requests build without publishing. Source licensing is GPL-3.0; see [LICENSE](LICENSE).
