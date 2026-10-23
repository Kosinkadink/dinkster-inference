# dinkster-inference

This distribution packages upstream ComfyUI's inference library as
`dinkster_inference`. The `comfy` import name aliases the same modules, not a
second instance of model-management state. ComfyUI's application, server and
node packages are not distributed here.

```bash
python -m pip install .
```

As in upstream ComfyUI, CPU callers select CPU execution before importing model
management or loaders:

```python
import dinkster_inference.cli_args as cli_args
cli_args.args.cpu = True
import comfy.sd
import dinkster_inference.sd
assert comfy.sd is dinkster_inference.sd
```

See [upstream provenance](docs/ledger.md) for the fork point and omitted
application tests. `python tools/check_upstream_parity.py` checks the library
and retained test files against that upstream commit. New upstream files use
`tools/rewrite_upstream_imports.py` for import renaming and ASCII source spelling;
escaped string values and non-ASCII tokenizer resource data retain their meaning.

## Workflow comparison

Install a built wheel and its CUDA dependencies in a separate environment, then
run the harness with that environment's Python. The fork worker uses the installed
package with isolated imports; the reference worker imports a clean, detached
ComfyUI checkout at the recorded fork point. Both use the same interpreter,
models, seed and deterministic PyTorch settings in separate processes.

```bash
CUDA_VISIBLE_DEVICES=1 /path/to/venv/bin/python -I tools/compare_workflows.py \
  --upstream /path/to/ComfyUI --output-dir /path/to/results \
  --checkpoint /path/to/v1-5-pruned-emaonly-fp16.safetensors \
  --lora /path/to/lcm-lora-sdv1-5.safetensors \
  --controlnet /path/to/control_v11p_sd15_canny.safetensors \
  --h3-model /path/to/minimax_h3_fl2va_int8_convrot.safetensors \
  --h3-clip /path/to/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --h3-vae /path/to/minimax_h3_video_vae_fp16.safetensors
```

The five SD1.5 cases use 512x512, seed 459, Euler/normal, 20 steps and CFG 7:
baseline; LoRA strength 0.8; five linearly interpolated hook LoRA keyframes from
strength 0 to 1 over the first half of sampling (first guarantee 1, subsequent
guarantees 0, matching upstream's interpolated-keyframe node); ControlNet strength
1 over timestep range 0.0-0.6 using a fixed asymmetric edge map; and overlapping
16-frame context windows over a 32-frame batch. H3 uses 1344x768, 124 frames,
seed 1064, Euler/normal, 20 steps and CFG 6. If H3 weights are absent, supply a
bundled Flux dev checkpoint via `--flux-checkpoint` instead (512x512, four steps,
seed 459, CFG 1, guidance 3.5). `--cases` selects individual cases for diagnosis.

`comparison.json` records input hashes, installed-package provenance, environment,
per-case raw latent and PNG hashes, and maximum absolute differences. The result
directory also contains tensors, PNGs and worker logs. Exit status is zero only
when every selected case is identical; missing model files, worker errors and
non-finite values never count as parity. No model downloads happen in the harness.
