# Upstream sync provenance

Upstream fork point: `comfyanonymous/ComfyUI` commit
`c844cb672186f187bf9455e2808821417a10d296`.

The sync and release tools read this exact upstream identity. The transform
maps upstream `comfy/` to `dinkster_inference/`, includes
`comfy_extras/mediapipe/` and `node_helpers.py`, and rewrites inference imports.
Only retained library tests under `tests-unit/comfy_test/` and
`tests-unit/comfy_quant/` are copied; application-dependent exclusions are
specified in `tools/regenerate_upstream_library.py`.

Node-authoring types, examples and documentation are excluded. Other upstream
paths outside the transformed library/test scope are not imported. The
classified file inventory is enforced before a sync push and by CI.
Fork-owned source changes are carried by a three-way merge of transformed
baseline and target files; overlapping changes stop regeneration for review.
The `comfy` alias, packaging, dependency declarations, tooling and release
workflows remain fork-owned.
