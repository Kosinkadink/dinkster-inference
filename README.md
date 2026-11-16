# dinkster-inference

Dinkster's inference library provides model loading, sampling, conditioning,
text/image/audio encoders, device management and inference runtime resources.
The `comfy` import name aliases `dinkster_inference` modules without duplicating
model-management state. The distribution does not include an application.

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

`tools/library_files.txt` classifies every tracked file as library code/runtime
data, packaging, release/upstream/dependency tooling, tests or LICENSE. Classify
new files there only when they serve one of those purposes. CI rejects missing
or unclassified entries. Packaging includes only Python modules and runtime
JSON, tokenizer model and text resources.

The [sync ledger](docs/ledger.md) supplies the upstream fork point.
`tools/regenerate_upstream_library.py --baseline /path/to/recorded-upstream
--upstream /path/to/new-upstream --destination .` transforms both revisions and
merges their library delta while preserving fork changes. Node-authoring
examples and node types are excluded. `tools/check_upstream_parity.py` accepts
the same arguments and verifies that the transformed delta is already present.
The sync workflow checks the classified inventory before pushing; previously
unseen upstream files require classification before they can land.

`tools/compare_workflows.py --help` describes the isolated installed-wheel
comparison harness. It uses caller-supplied local models and a reference
checkout; it does not download model artifacts.

## Releases

Consumers pin exact versions and PyPI wheel hashes. After a landed upstream
sync passes main CI, automation publishes an immutable `vX.Y.Z.devN` release.
`X.Y.Z` is the next patch after the latest stable inference release; `N` is the
source commit count. An hourly check detects newly contained upstream stable
tags and publishes separate stable inference patch releases. Superseded CI
completions do not publish; the next successful current-main check recovers
unpublished syncs in source order.

Release tags set the built version without changing the source commit.
Distributions contain `dinkster_inference/_release.json` with the version,
inference source commit and upstream commit. Annotated tags record the same
identity. Development/prerelease publication is wheel-only; stable releases
also include source distributions. Failed publication never moves a tag.

Trusted PyPI publication uses repository `Kosinkadink/dinkster-inference`,
workflow `release.yml`, environment `pypi`. Branches and pull requests build
without publishing. Source licensing is GPL-3.0; see [LICENSE](LICENSE).
