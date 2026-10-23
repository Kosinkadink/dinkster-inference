# Upstream provenance

Upstream fork point: `comfyanonymous/ComfyUI` commit
`9fe7ef764d0452dcc9dcb925efd076d4e68eece1`.

The library is upstream `comfy/` under the mechanical `dinkster_inference`
import rename. `node_helpers.py` is packaged from the same revision because
upstream `hooks.py` imports it. The `comfy` package aliases the installed library.

Omitted application paths: ComfyUI's server, application, nodes, extras,
execution, middleware, frontend, assets, API and model directories, and tests
outside `tests-unit/comfy_test/` and `tests-unit/comfy_quant/`.

The console-tip implementation in `app/product_tips.py`, its
`tests-unit/app_test/product_tips_test.py` coverage, and changes in `main.py`
are omitted with the application. The upstream `--disable-product-tips`
parser option is retained for library parity but no console tips are shipped.
MOSS edit node controls in `comfy_extras/nodes_moss_image.py`, the preview
changes in `execution.py`, and `tests/execution/test_execution.py` are also
application-only; MOSS attention-bias conditioning in the library is retained.

The following upstream tests within those two directories are omitted because
they import application modules not distributed by this library:

| Path under `tests-unit/comfy_test/` | Application imports |
| --- | --- |
| `folder_path_test.py` | `folder_paths` |
| `image_invert_test.py` | `nodes` |
| `test_seedvr2_model.py` | `nodes` |
| `test_seedvr2_vae_decode.py` | `comfy_extras.nodes_seedvr` |
| `test_sensenova.py` | `nodes`, `comfy_extras.nodes_hidream_o1`, `comfy_extras.nodes_sensenova` |
| `test_vae_decode_tiled_nested.py` | `nodes` |
