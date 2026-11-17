# Upstream sync provenance

Upstream fork point: `comfyanonymous/ComfyUI` commit
`5071348042489b20e1a9b64cc178357df67ad08b`.

The sync and release tools read this exact upstream identity.

## What the transform keeps

`tools/regenerate_upstream_library.py` is the definition; this is its summary.

| Upstream path | Path here |
| --- | --- |
| `comfy/` | `dinkster_inference/` |
| `comfy_extras/mediapipe/` | `dinkster_inference/mediapipe/` |
| `node_helpers.py` | `node_helpers.py` |
| `tests-unit/comfy_test/`, `tests-unit/comfy_quant/` | same paths |
| `QUANTIZATION.md` | `docs/quantization.md` |

Imports are renamed and non-ASCII source text is escaped. Left out:

- every other upstream path;
- Markdown files inside the copied directories;
- node-authoring types and examples (`comfy_types/node_typing.py`,
  `comfy_types/examples/`) and the names `comfy_types/__init__.py` imported
  from them;
- a test that imports upstream code outside the copied paths, because that
  code is not in this library;
- a command-line option in `cli_args.py` that nothing in the library or its
  tests reads, except the options Dinkster packages read
  (`CONSUMER_NAMES` in the tool).

The MediaPipe modules are PyTorch implementations; neither needs Google's
MediaPipe runtime package.

## What the sync writes

The transformed difference between the recorded commit and the new one, the
commit above, the entries of added and removed upstream files in
`tools/library_files.txt`, and the version specifier of a dependency in
`pyproject.toml` when upstream's `requirements.txt` changes that dependency's
specifier. Everything else is fork-owned: the `comfy` alias, packaging, tools,
workflows and documents.

## Adapted upstream tests in stable releases

`tools/adapted_upstream_tests.json` records each fork-adapted test's upstream
source path. A sync that adds an adaptation adds its mapping in the same
change. Stable preparation keeps an adaptation only when that source path
exists in the target tag, and records excluded test/source pairs in the
release metadata. Tests without a mapping remain fork-only and are always
carried. Main keeps every adaptation and runs the same CI suite unchanged.
An adaptation whose source exists in a tag is kept even if it fails; that
failure blocks publication.

## Fork changes to upstream files

`tools/fork_differences.txt` lists them with the reason for each, and CI
rejects one that is not listed. `python tools/check_upstream_parity.py
--differences --upstream /path/to/ComfyUI-at-the-commit-above` prints the
current list, including fork-only files inside the copied directories. A
listed file is merged with each upstream change; a change to the same lines
stops the sync until a person resolves it.
