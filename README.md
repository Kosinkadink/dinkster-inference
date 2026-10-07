# dinkster-inference

`dinkster-inference` is Dinkster's inference library. It starts from ComfyUI's
`comfy/` package while excluding the ComfyUI application, server, nodes, and
web assets.

The distribution name is `dinkster-inference`; Python imports use
`dinkster_inference`:

```bash
python -m pip install .
python -c "import dinkster_inference.sd, dinkster_inference.samplers, dinkster_inference.model_management"
```

The upstream fork point and maintained differences are recorded in
[`docs/ledger.md`](docs/ledger.md). Run `tools/rewrite_upstream_imports.py` on
Python files brought in from upstream before committing a sync.

## Training API

The following eight symbols are the contract consumed by `dinkster-training`.
The inference CI contract test imports them and exercises trainable LoRA
construction, so a rename fails in inference before training updates its pin.

| Symbol | Contract |
| --- | --- |
| `cli_args.args` | Set `cpu` before model-management imports when selecting a CPU host. |
| `model_base.BaseModel` | Owns `diffusion_model`, `model_sampling`, and `process_latent_in`. |
| `sd.load_checkpoint_guess_config` | Loads model/CLIP/VAE/CLIP-vision as a four-item tuple; output flags select components. |
| `lora.model_lora_keys_unet` | Populates an export-key to model-weight-key map. |
| `model_patcher.ModelPatcher` | Owns adapter attachment through `set_bypass_adapters`, patching and unpatching. |
| `weight_adapter.LoRAAdapter` | `create_train(weight, rank, alpha)` constructs trainable adapter resources. |
| `weight_adapter.lora.LoraDiff` | Exposes `lora_down`, `lora_up`, and `alpha` for training and export. |
| `model_management.unload_all_models` | Releases models held by the host's model manager. |

Before `patch_model`, call `patcher.set_training_settings(True,
fp8_backward=False)` to select a model's training execution policy. Inference
patchers default to both settings false. Never assign
`model_management.in_training` or `training_fp8_bwd`: these are compatibility reads
of the active forward's context, not stored process-wide settings.

`patcher.set_gradient_checkpoints(["diffusion_model.input_blocks.1.1", ...])`
adds content-identified checkpoint entries for module paths; `""` selects the
root. An empty list removes checkpoint entries. Checkpointing requires training
mode and uses PyTorch non-reentrant checkpointing with RNG preservation and the
same policy on recomputation. Settings and checkpoint entries survive cloning
without changing the original program. Repatch after changing either setting.

Patching installs reversible scoped forwards on the root, its `diffusion_model`
when present, and explicit checkpoint targets. Invoke one of these boundaries
to execute under the policy; arbitrary direct child forwards are not training
entry points. Nested inference patchers temporarily select inference policy,
including inside a training forward. Unpatching restores original forwards.
Choose a target whose forward is called: SD UNet execution iterates block
containers directly, so select a child such as the spatial transformer above,
not its enclosing `input_blocks.1` container.
Shared-model clones select a policy when materialized, like weight patches;
concurrent training and inference require separate model instances. This API
does not freeze parameters, select an optimizer, or provide activation offload.
