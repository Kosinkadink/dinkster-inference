# Fork additions reference

This branch preserves the fork implementation before the upstream parity reset.
It is a reference, not a merge target. Re-additions require their own design,
contract tests and upstream comparison; this index does not approve any design.
Paths below are relative to `dinkster_inference/` unless otherwise stated.
Tests are relative to `tests-unit/comfy_test/`.

| Addition | Implementation files | Tests | Known consumer and purpose |
| --- | --- | --- | --- |
| PatchProgram and structural insertions | `patch_program.py`, `model_patcher.py`, `model_base.py`, `patcher_extension.py` | `test_patch_program.py` | `dinkster-training/lora_program.py` and `trainer.py` use content-identified patches and bypass adapters; no Dinkster structural-insertion consumer was found by the audit. |
| Training execution settings and gradient checkpoints | `model_management.py`, `model_patcher.py`, `patch_program.py` | `training_api_test.py`, `test_patch_program.py` | dinkster-training selects training execution and reversible checkpoint entries. Its pin stays on the preserved implementation until a separately designed re-add. |
| Scheduled bypass materialization and dynamic hook patches | `weight_adapter/bypass.py`, `model_patcher.py`, `patch_program.py` | `test_patch_program.py`, `training_api_test.py` | dinkster-training uses bypass resources. No Dinkster scheduled-bypass consumer was identified. Hook weight patches extend the upstream dynamic patcher. |
| ContributionGain | `contribution_gain.py`, `hooks.py`, `controlnet.py`, `samplers.py`, `model_patcher.py`, `sd.py`, `weight_adapter/bypass.py` | `test_contribution_gain.py`, mask cases in `window_execution_test.py` | No Dinkster consumer identified. Replaces upstream keyframe, conditioning and control semantics with compiled timeline/site/lane gains. This is not a re-add candidate. |
| Process-owned ModelManager and CPU selection | `model_management.py`, `model_patcher.py`, `pinned_memory.py` | `model_management_test.py`, `test_model_management_cpu.py`, `model_loading_test.py`, `qwen21_cache_admission_test.py` | Dinkster `dinkster_native/devices.py` reads manager-owned pinned-memory totals. The manager also owns loaded models, streams, interrupts, offload and pin eviction. |
| Joint window planning and execution | `window_plan.py`, `window_execution.py`, `context_windows.py`, `samplers.py`, `patcher_extension.py` | `window_plan_test.py`, `window_execution_test.py` | Dinkster `dinkster_native/fork_nodes.py`, `dinkster_nodes_generation/nodes.py`, `tools/verify_pinned_inference.py` and `tools/flux_window_multigpu_receipt.py` use temporal plans/executors. Compiles layered media axes and merges joint-window outputs. |
| Caller-owned attention registry and Sol changes | `ldm/modules/attention.py` | `attention_registry_test.py` | Dinkster `dinkster_native/multigpu.py` attaches isolated registries. Sol accepts rewritten options and rejects masked calls instead of upstream fallback. |
| Native GGUF loader and operations | `gguf.py`, `gguf_ops.py`, `sd.py`, `utils.py` | `gguf_test.py`, `model_loading_test.py` | Loader/sampler users; no specific Dinkster consumer identified by the audit. Adds GGUF model/CLIP loading. |
| RES4LYF solvers and typed sampler assembly | `res4lyf_rk.py`, `res4lyf_sampler.py`, `sampler_assembly.py`, `samplers.py` | `res4lyf_sampler_test.py`, `goldens/res4lyf_rk_goldens.json` | Sampler users; no specific Dinkster consumer identified by the audit. Adds ordered RK substeps and separate noise streams. |
| MiniMax H3 Fun ControlNet | `minimax_control.py` | `minimax_control_test.py` | No specific Dinkster consumer identified by the audit. Compiles full-domain control inputs for semantic-window model patches. |
| Relocated environment helpers | `deploy_environment.py`, `comfy_api_env.py` | `tests-unit/deploy_environment_test.py` | Library deployment/environment imports preserve modules that upstream moved to the application. Not part of upstream `comfy/` at reset. |
| Relocated SAM 3D face helper | `ldm/sam3d_body/face_landmarker.py`, `ldm/sam3d_body/model/model.py` | No dedicated retained test | SAM 3D Body loader imports an application helper copied into the library. The reset restores the upstream import instead. |
| Inlined application conditioning helper and CLI differences | `hooks.py`, `cli_args.py` | Upstream hook callers; no dedicated fork test | `hooks.py` owns a helper upstream imports from `node_helpers`; CLI assets/agent options differ because the application is absent. The reset restores upstream behavior. |

The twelve fork-only top-level modules are all indexed above:
`patch_program.py`, `contribution_gain.py`, `window_plan.py`,
`window_execution.py`, `gguf.py`, `gguf_ops.py`, `res4lyf_rk.py`,
`res4lyf_sampler.py`, `sampler_assembly.py`, `minimax_control.py`,
`deploy_environment.py`, and `comfy_api_env.py`.

Import rewriting, wheel packaging and automatic sync infrastructure are not
inference additions. Their tests (`rewrite_upstream_imports_test.py` and
`upstream_sync_test.py`) stay on the reset branch. See the pre-reset ledger and
the [audit](https://github.com/Kosinkadink/comfy-vibe-station/issues/503#issuecomment-6281721796)
for measured API differences and the limits of consumer discovery.
