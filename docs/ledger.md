# Upstream provenance

Upstream fork point: `comfyanonymous/ComfyUI` commit
`6ea48ffc21d2a0c9be0df79f08d72b4ecf2d678f`.

The library is upstream `comfy/` under the mechanical `dinkster_inference`
import rename. `node_helpers.py` is packaged from the same revision because
upstream `hooks.py` imports it. The `comfy` package aliases the installed library.

Upstream's ComfyUI 0.43.0 application version bump does not change the
fork-owned package metadata or the retained `comfyui_version.py` file.

The application-only `comfyui-workflow-templates` dependency is omitted;
its upstream bump to 0.11.100 does not change the library requirements.

The application-only `comfyui-embedded-docs` dependency is omitted;
its upstream bump to 0.5.15 does not change the library requirements.

The application-only `comfyui-frontend-package` dependency is omitted;
its upstream bump to 1.57.6 does not change the library requirements.

The fork's library README replaces upstream application installation guidance;
upstream Huawei Ascend branding changes do not apply to that document.

Deliberate library additions in `utils.load_torch_file`: state dictionaries
retain storage provenance through `storage.annotate_state_dict`, and files
without a `.safetensors` or `.sft` suffix use a bounded nine-byte safetensors
header probe before parser dispatch. Safetensors stored under digest paths or
misnamed `.ckpt`, `.pt`, or `.bin` therefore load where upstream calls
`torch.load` and fails. Existing suffix dispatch and parser policies are unchanged.

Omitted application paths: ComfyUI's server, application, nodes, extras,
execution, middleware, frontend, assets, API and model directories, and tests
outside `tests-unit/comfy_test/` and `tests-unit/comfy_quant/`.

Cached pinned frontend selection and release tag lookups in
`app/frontend_management.py` and their
`tests-unit/app_test/frontend_manager_test.py` coverage are omitted with the
application frontend manager.

Proxy environment handling in `comfy_api_nodes/` and its
`tests-unit/comfy_api_nodes_test/comfy_cloud_test.py` coverage are omitted
with the partner-node application clients.

Kling 4.0 API schemas and video nodes in `comfy_api_nodes/apis/kling.py` and
`comfy_api_nodes/nodes_kling.py` are omitted with those partner-node clients.

BFL Flux Video Upscale V2 resolution targets and legacy-node handling in
`comfy_api_nodes/nodes_bfl.py` are omitted with those partner-node clients.

Veed API schemas and video nodes in `comfy_api_nodes/apis/veed.py` and
`comfy_api_nodes/nodes_veed.py` are omitted with those partner-node clients.

Pruna P-Video Upscale schemas and node, plus shared request submission changes,
in `comfy_api_nodes/apis/pruna.py` and `comfy_api_nodes/nodes_pruna.py` are
omitted with those partner-node clients.

WaveSpeed FlashVSR and image upscale router migration, schemas, and shared
router submit/poll/cancel helpers in `comfy_api_nodes/` are omitted with
those partner-node clients.

Ideogram 4.5 Precise Edit region editing in `comfy_api_nodes/apis/ideogram.py`
and `comfy_api_nodes/nodes_ideogram.py` is omitted with those partner-node clients.

MiniMax-H3 context-window support is retained in the library. Its node in
`comfy_extras/nodes_context_windows.py` is omitted with the application nodes.

VAE cloning is retained in the library. The VAE compute-dtype node in
`comfy_extras/nodes_model_advanced.py` and VAE device-selection changes in
`comfy_extras/nodes_multigpu.py` are omitted with the application nodes.

The MOSS model merge node in `comfy_extras/nodes_model_merging_model_specific.py`
is omitted with the application nodes; it does not change the inference library.

The MiniMax-H3 latent upscaler model is retained in the library. Its loader in
`comfy_extras/nodes_hunyuan.py` and node in `comfy_extras/nodes_minimax_h3.py`
are omitted with the application nodes.

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
