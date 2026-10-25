# Upstream provenance

Upstream fork point: `comfyanonymous/ComfyUI` commit
`fe298e4e20f1ff26d30e8639267bbd0b57811390`.

The library is upstream `comfy/` under the mechanical `dinkster_inference`
import rename. `node_helpers.py` is packaged from the same revision because
upstream `hooks.py` imports it. The `comfy` package aliases the installed library.

Deliberate library additions in `utils.load_torch_file`: state dictionaries
retain storage provenance through `storage.annotate_state_dict`, and files
without a `.safetensors` or `.sft` suffix use a bounded nine-byte safetensors
header probe before parser dispatch. Safetensors stored under digest paths or
misnamed `.ckpt`, `.pt`, or `.bin` therefore load where upstream calls
`torch.load` and fails. Existing suffix dispatch and parser policies are unchanged.

Omitted application paths: ComfyUI's server, application, nodes, extras,
execution, middleware, frontend, assets, API and model directories, and tests
outside `tests-unit/comfy_test/` and `tests-unit/comfy_quant/`.

The MOSS model merge node in `comfy_extras/nodes_model_merging_model_specific.py`
is omitted with the application nodes; it does not change the inference library.

Cloud deployment in `app/comfy_api_env.py`, `app/deploy_to_cloud.py`,
`comfy_api/feature_flags.py`, `main.py`, `openapi.yaml`, `server.py`, and
`tests-unit/deploy_to_cloud_test/` is omitted with the application.

The Branch node in `comfy_extras/nodes_logic.py` is omitted with the
application nodes; it does not change the inference library.

The console-tip implementation in `app/product_tips.py`, its
`tests-unit/app_test/product_tips_test.py` coverage, and changes in `main.py`
are omitted with the application. The upstream `--disable-product-tips`
parser option is retained for library parity but no console tips are shipped.
MOSS edit node controls in `comfy_extras/nodes_moss_image.py`, failed-stage
status in `execution.py`, and `tests/execution/test_execution.py` are also
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
