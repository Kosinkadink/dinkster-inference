# dinkster-inference

Rules for changing this repository. They keep what applies to an inference
library from ComfyUI's contributor guidance and add the rules for living with
the upstream sync.

## Scope

- `dinkster_inference/` is ComfyUI's `comfy/` package under another name: model
  detection, loading and conversion; encoders, tokenizers and latent formats;
  conditioning, sampling and model patching; tensor operations, quantization,
  and device, memory and offload management. `comfy/` is an import alias for it.
- Nothing else belongs here: no server, user interface, workflow or queue code,
  node definitions, persistence, telemetry, or clients for remote services.
- Library code makes no network requests. Callers supply local files.

## Upstream files

- `dinkster_inference/`, `node_helpers.py`, `docs/quantization.md` and the
  upstream tests in `tests-unit/comfy_test/` and `tests-unit/comfy_quant/` are
  generated from ComfyUI. `docs/ledger.md` says how and from which commit. The
  scheduled sync rewrites them.
- Do not change a generated file when the change can live elsewhere. A change
  that also applies to ComfyUI belongs upstream. A change that must live here
  is listed in `tools/fork_differences.txt` with its reason; it is merged with
  every later upstream change and stops the sync when both touch the same lines.
- Never edit `dinkster_inference/cli_args.py`, `docs/quantization.md` or
  `dinkster_inference/comfy_types/__init__.py` by hand. Their content is decided
  by `tools/regenerate_upstream_library.py`; change the rule there and
  regenerate.
- Fork tests go in their own files, never inside an upstream test file.
- A new fork-owned file needs an entry in `tools/library_files.txt`. The sync
  maintains the entries of upstream files.

## Compatibility

- Keep how callers invoke a function: arguments, order, return type, side
  effects and errors. Do not return extra values or alternate shapes from a
  shared method for one implementation.
- Keep state-dict keys, module order and checkpoint loading. When removing a
  module would change keys, replace it with `nn.Identity`.
- `comfy_quant`, `comfy_attention` and the other names in
  `docs/quantization.md` are stored in model files. They are never renamed.
- Unsupported model formats and invalid quantization metadata fail with a clear
  error. They never fall back silently to lower quality.
- Do not reject input only because the result may be poor. Adapt input only
  when passing it unchanged would make the operation fail.

## Model code

- dtype, device placement, VRAM use and offloading are correctness. Consider
  CPU, CUDA, ROCm, MPS, XPU, NPU and low-VRAM operation when touching shared
  loading or execution code.
- Model code does no memory management. Loading, unloading, offloading, device
  movement and cache lifetime belong to `model_management`, `model_patcher`,
  `memory_management` and `pinned_memory`.
- No global, module-level, class-level or model-owned store of tensors outlives
  a call. A temporary cache is created by the top-level encode, decode or
  forward call, passed down explicitly, and dropped when that call returns.
- Create tensors on the target device in the target dtype. Do not allocate on
  CPU and move, or allocate in one dtype and convert. Inputs to a model forward
  are already in the compute dtype; do not add casts.
- A raw parameter that no operation owns is cast where it is used, with
  `dinkster_inference.ops.cast_to_input` or
  `dinkster_inference.model_management.cast_to`.
- Parameters and buffers that the state dict fills are created with
  `torch.empty`. Model `__init__` describes structure and never invents
  checkpoint contents.
- The `operations` argument of a model class is never `None`.
- Use an existing optimized operation from comfy-kitchen, `quant_ops` or `ops`
  before writing local math. Attention goes through `AttentionTensorContainer`
  and a per-module `ComfyAttention()` passed as `preferred_attention`. Block
  loops use the helpers in `model_prefetch`. Treat a selected backend function
  as opaque: never branch on its identity or name.
- No `torch.no_grad`, `torch.inference_mode`, or freeze and unfreeze switches
  in library code.
- Model detection reads only the first dimension of a linear weight (the second
  is halved by 4-bit formats), guards every key it reads, and tests specific
  signatures before broad ones.
- DiT models accept sizes that are not patch multiples: pad with
  `ldm.common_dit.pad_to_patch_size` and crop the output.
- Callers change model behavior through `ModelPatcher` and its hooks, never by
  reaching into model internals.
- Keep sizes, offsets, counts and flags as Python values, not tensors.

## Checks

```bash
python tools/check_library_files.py
python tools/check_dependencies.py
python -m ruff check dinkster_inference comfy node_helpers.py tools
python -m build --wheel
python -m pytest -q tests-unit
```

The tests that import the installed package need the built wheel installed.
