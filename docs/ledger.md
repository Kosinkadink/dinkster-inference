# Upstream provenance

Upstream fork point: `comfyanonymous/ComfyUI` commit
`7b6c412a2a24d5298a3f39e8c54ac413e9335091`.

The library is upstream `comfy/` under the mechanical `dinkster_inference`
import rename. `node_helpers.py` is packaged from the same revision because
upstream `hooks.py` imports it. The `comfy` package aliases the installed library.

RelayVSR model detection, keyframe model, latent format and Flash decoder are
retained, including data-copy streams and the SeedVR/pose-cache callers.
HostWriter allocates both staging buffers before registering either one so
a host-allocation failure cannot strand a pinned buffer during construction.
RelayVSR conditioning and VSR Decode nodes in `comfy_extras/nodes_wan.py`
are omitted with the application nodes.

Qwen combined linear projections, residency-scoped weight combining, and
prefetch relayout changes are retained with their library tests.

SeedVR2 adaptive VAE slicing, per-pass compiler cache scopes, and VAE patcher
wiring are retained with their library regression coverage.

`dinkster_inference/mediapipe/` retains `comfy_extras/mediapipe/` from
ComfyUI `bc5f880dbdd43e426077f42ccec53bb577ffee9f`, mechanically normalized
to ASCII. SAM3D Body imports this packaged PyTorch face landmarker instead of
the omitted application extras. Face geometry is also available to library
callers; neither module requires Google's MediaPipe runtime package.

| Carried library path | Upstream path | Reference F |
| --- | --- | --- |
| `dinkster_inference/mediapipe/face_landmarker.py` | `comfy_extras/mediapipe/face_landmarker.py` | `bc5f880dbdd43e426077f42ccec53bb577ffee9f` |
| `dinkster_inference/mediapipe/face_geometry.py` | `comfy_extras/mediapipe/face_geometry.py` | `bc5f880dbdd43e426077f42ccec53bb577ffee9f` |

Upstream's ComfyUI 0.44.0 application version bump does not change the
fork-owned package metadata or the retained `comfyui_version.py` file.

The application-only `comfyui-workflow-templates` dependency is omitted;
its upstream bump to 0.11.102 does not change the library requirements.

The retained `comfy-kitchen` dependency is pinned to 0.2.41 in both library
dependency manifests.

The application-only `comfyui-embedded-docs` dependency is omitted;
its upstream bump to 0.5.16 does not change the library requirements.

The application-only `comfyui-frontend-package` dependency is omitted;
its upstream bump to 1.58.5 does not change the library requirements.

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

Imported-mesh remeshing bounds fitting and invalid-vertex cleanup in
`comfy_extras/nodes_mesh_postprocess.py` are omitted with the application nodes.

The unpinned agent requirement and upgrade install hints in
`agent_requirements.txt` and `app/agent_host.py`, plus their
`tests-unit/agent_requirements_test.py` and
`tests-unit/app_test/agent_host_test.py` coverage, are omitted with the
application agent host.

The pixel-space VAE selection tip for Pillow dtype errors in `execution.py`
is omitted with the application execution layer.

Local browser SSO and partner-token feature flags, server feature
initialization, and CLI override rules in `comfy_api/feature_flags.py` and
`tests-unit/feature_flags_test.py` are omitted with the application API.
Their enabled-by-default registry and server initialization changes are also
omitted; the library does not ship application authentication features.

PH3 Image VAE, text connector/tokenizer, and loader detection are retained.
Conditioning and latent helpers in `comfy_extras/nodes_minimax_h3.py` are
omitted with the application nodes.

TextGenerateV2 Autogrow inputs and legacy-node replacements in
`comfy_extras/nodes_textgen.py`, `comfy_extras/nodes_replacements.py`, and
`app/node_replace_manager.py` are omitted with the application nodes and
replacement manager. The Gemma3/Gemma4 multi-input tokenizer support is retained.

Cached pinned frontend selection, release tag lookups, and lazy release
pagination that stops at the first prerelease in `app/frontend_management.py`
and their
`tests-unit/app_test/frontend_manager_test.py` coverage are omitted with the
application frontend manager.

Proxy environment handling in `comfy_api_nodes/` and its
`tests-unit/comfy_api_nodes_test/comfy_cloud_test.py` coverage are omitted
with the partner-node application clients.

Pika Soundtrack, SFX and Music API schemas and nodes in
`comfy_api_nodes/apis/pika.py` and `comfy_api_nodes/nodes_pika.py` are omitted
with those partner-node clients.

OpenAI GPT-6.1 Sol/Terra choices, reasoning options, retired model removal,
image defaults, and pricing updates in `comfy_api_nodes/nodes_openai.py` are
omitted with those partner-node clients.

Kling 4.0 API schemas and video nodes in `comfy_api_nodes/apis/kling.py` and
`comfy_api_nodes/nodes_kling.py` are omitted with those partner-node clients.
Kling 4.0 Flash model choices, duration/resolution limits, pricing, and the
original-audio option are also omitted with those clients.

Seedance 1.x video-node deprecation flags in `comfy_api_nodes/nodes_bytedance.py`
are omitted with those partner-node clients.

BFL Flux Video Upscale V2 resolution targets and legacy-node handling in
`comfy_api_nodes/nodes_bfl.py` are omitted with those partner-node clients.

Topaz Wonder 4 image enhancement, pricing, and image sizing limits in
`comfy_api_nodes/nodes_topaz.py` are omitted with those partner-node clients.

Sonilo Music 1.5 request/task schemas, remote music generation, and pricing
updates in `comfy_api_nodes/apis/sonilo.py` and `comfy_api_nodes/nodes_sonilo.py`
are omitted with those partner-node clients.

Veed API schemas and video nodes in `comfy_api_nodes/apis/veed.py` and
`comfy_api_nodes/nodes_veed.py` are omitted with those partner-node clients.

Pruna P-Video Upscale schemas and node, plus shared request submission changes,
in `comfy_api_nodes/apis/pruna.py` and `comfy_api_nodes/nodes_pruna.py` are
omitted with those partner-node clients.

MiniMax H3 Max Recast, Extend, Reframe, Insert, Lip Sync and Camera Control
schemas and nodes, plus related request/reference handling, in
`comfy_api_nodes/apis/minimax.py` and `comfy_api_nodes/nodes_minimax.py` are
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

Job-detail execution metadata and current-node metadata for cached outputs in
`comfy_execution/asset_enrichment.py`, `comfy_execution/jobs.py`, and
`execution.py`, plus their `tests-unit/execution_test/test_enrich_output.py`,
`tests/execution/test_execution.py`, and `tests/execution/test_jobs.py` coverage,
are omitted with the application execution layer.

ProRes/MOV encoding and chroma-subsampling controls in
`comfy_api/latest/_input/video_types.py`,
`comfy_api/latest/_input_impl/video_types.py`,
`comfy_api/latest/_util/video_types.py`, and `comfy_extras/nodes_video.py`,
plus their `tests-unit/comfy_api_test/video_accumulation_test.py` and
`tests-unit/comfy_api_test/video_bit_depth_test.py` coverage, are omitted
with the application video API and nodes.

Chroma-subsampling input validation in `comfy_extras/nodes_video.py`
is also omitted: values other than `444`, `422`, and `420` select automatic mode.

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
