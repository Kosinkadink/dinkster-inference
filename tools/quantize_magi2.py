"""Streaming converter and quantizer for MAGI-2 Preview, Refiner, and Qwen3.5-27B checkpoints.

Converts upstream `sand-ai/MAGI-2-preview` safetensors shards into normalized Dinkster
single-file or sharded safetensors checkpoints in `bf16`, `nvfp4`, `int8` (ConvRot),
or `fp8` (`float8_e4m3fn` with optional calibrated `input_scale`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import shutil
import struct
from typing import Optional

from safetensors import safe_open
from safetensors.torch import save_file
import torch

if not torch.cuda.is_available():
    import dinkster_inference.cli_args as _cli_args

    _cli_args.args.cpu = True

from dinkster_inference.ldm.magi2.common import _is_stacked_mm_linear_key, default_rope_bands
from dinkster_inference.quant_ops import (
    TensorCoreFP8E4M3Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)

LOGGER = logging.getLogger("quantize_magi2")

PREVIEW_MM_LAYERS = {0, 1, 38, 39}
REFINER_MM_LAYERS = {0, 1, 28, 29}


def _largest_pow4_divisor(k: int, cap: int = 256) -> int:
    g = cap
    while g > 1:
        if k % g == 0:
            return g
        g //= 4
    return 1


def _should_keep_high_precision(
    key: str,
    tensor: torch.Tensor,
    keep_boundary_blocks: bool = False,
    mm_layers: set[int] = PREVIEW_MM_LAYERS,
    fmt: str = "bf16",
) -> bool:
    if not key.endswith(".weight"):
        return True
    if tensor.ndim not in (2, 3):
        return True
    if fmt in ("fp8", "nvfp4") and (tensor.shape[-1] % 16 != 0 or tensor.shape[-2] % 16 != 0):
        return True
    if key.startswith(("pre_adapter.", "post_adapter.", "model.embed_tokens.", "embed_tokens.")):
        return True
    if any(
        tok in key
        for tok in (
            "norm",
            ".sinks",
            ".linear_g.",
            "mhc_",
            ".moe_mlp.gate",
            ".moe_mlp.router.",
            ".in_proj_a.",
            ".in_proj_b.",
            ".conv1d.",
            "mtp.",
            "lm_head.",
        )
    ):
        return True
    if fmt == "nvfp4":
        if any(
            tok in key
            for tok in (
                ".split_linear.",
                ".merge_linear.",
                ".shared_expert_fc1.",
                ".shared_expert_fc2.",
                ".modality_specific_shared_expert_",
            )
        ):
            return True
        if key.startswith("block.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) in mm_layers:
                return True
    if keep_boundary_blocks and key.startswith("block.layers."):
        parts = key.split(".")
        if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) in mm_layers:
            return True
    return False


def _quant_meta_tensor(conf: dict) -> torch.Tensor:
    raw = json.dumps(conf, separators=(",", ":")).encode("utf-8")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def _quantize_2d_weight(
    stem: str,
    weight: torch.Tensor,
    fmt: str,
    device: torch.device,
    input_scales: Optional[dict[str, float]] = None,
) -> dict[str, torch.Tensor]:
    w_dev = weight.to(device=device)
    out: dict[str, torch.Tensor] = {}

    if fmt == "nvfp4":
        qdata, params = TensorCoreNVFP4Layout.quantize(w_dev, scale="recalculate")
        out[f"{stem}.weight"] = qdata.cpu().contiguous()
        out[f"{stem}.weight_scale"] = params.block_scale.cpu().contiguous()
        out[f"{stem}.weight_scale_2"] = params.scale.cpu().float().contiguous()
        if input_scales and stem in input_scales:
            out[f"{stem}.input_scale"] = torch.tensor(input_scales[stem], dtype=torch.float32)
        out[f"{stem}.comfy_quant"] = _quantize_meta("nvfp4")
        return out

    if fmt == "int8":
        gs = _largest_pow4_divisor(w_dev.shape[1], cap=256)
        use_convrot = gs >= 4
        qdata, params = TensorWiseINT8Layout.quantize(
            w_dev,
            is_weight=True,
            per_channel=True,
            convrot=use_convrot,
            convrot_groupsize=gs if use_convrot else 256,
        )
        out[f"{stem}.weight"] = qdata.cpu().contiguous()
        out[f"{stem}.weight_scale"] = params.scale.cpu().float().contiguous()
        meta = {"format": "int8_tensorwise"}
        if use_convrot:
            meta["convrot"] = True
            meta["convrot_groupsize"] = gs
        out[f"{stem}.comfy_quant"] = _quant_meta_tensor(meta)
        return out

    if fmt == "fp8":
        qdata, params = TensorCoreFP8E4M3Layout.quantize(w_dev, scale="recalculate")
        out[f"{stem}.weight"] = qdata.cpu().contiguous()
        out[f"{stem}.weight_scale"] = params.scale.cpu().float().contiguous()
        if input_scales and stem in input_scales:
            out[f"{stem}.input_scale"] = torch.tensor(input_scales[stem], dtype=torch.float32)
        out[f"{stem}.comfy_quant"] = _quantize_meta("float8_e4m3fn")
        return out

    raise ValueError(f"Unsupported format: {fmt}")


def _quantize_meta(fmt_name: str, extra: Optional[dict] = None) -> torch.Tensor:
    d = {"format": fmt_name}
    if extra:
        d.update(extra)
    return _quant_meta_tensor(d)


def _quantize_3d_moe_bank(
    stem: str,
    bank: torch.Tensor,
    fmt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Quantize a `[E, out_features, in_features]` MoE weight bank on GPU in chunks."""
    num_experts, out_f, in_f = bank.shape
    out: dict[str, torch.Tensor] = {}

    if fmt == "nvfp4":
        # When out_f % 128 == 0 and in_f % 64 == 0, cuBLAS block_scale tiles align per expert!
        qdata_list = []
        bscale_list = []
        scale2_list = []
        chunk_e = 256
        for s in range(0, num_experts, chunk_e):
            e = min(num_experts, s + chunk_e)
            sub = bank[s:e].to(device=device)
            for idx in range(e - s):
                qd, prm = TensorCoreNVFP4Layout.quantize(sub[idx], scale="recalculate")
                qdata_list.append(qd.cpu())
                bscale_list.append(prm.block_scale.cpu())
                scale2_list.append(prm.scale.cpu().float())
        out[f"{stem}.weight"] = torch.stack(qdata_list, dim=0).contiguous()
        out[f"{stem}.weight_scale"] = torch.stack(bscale_list, dim=0).contiguous()
        out[f"{stem}.weight_scale_2"] = torch.stack(scale2_list, dim=0).contiguous()
        out[f"{stem}.comfy_quant"] = _quantize_meta("nvfp4", {"num_experts": num_experts})
        return out

    if fmt == "int8":
        gs = _largest_pow4_divisor(in_f, cap=256)
        use_convrot = gs >= 4
        qdata_chunks = []
        scale_chunks = []
        chunk_e = 256
        for s in range(0, num_experts, chunk_e):
            e = min(num_experts, s + chunk_e)
            flat_sub = bank[s:e].to(device=device).reshape((e - s) * out_f, in_f)
            qd, prm = TensorWiseINT8Layout.quantize(
                flat_sub,
                is_weight=True,
                per_channel=True,
                convrot=use_convrot,
                convrot_groupsize=gs if use_convrot else 256,
            )
            qdata_chunks.append(qd.view(e - s, out_f, in_f).cpu())
            scale_chunks.append(prm.scale.view(e - s, out_f, 1).cpu().float())
        out[f"{stem}.weight"] = torch.cat(qdata_chunks, dim=0).contiguous()
        out[f"{stem}.weight_scale"] = torch.cat(scale_chunks, dim=0).contiguous()
        meta = {"format": "int8_tensorwise", "num_experts": num_experts}
        if use_convrot:
            meta["convrot"] = True
            meta["convrot_groupsize"] = gs
        out[f"{stem}.comfy_quant"] = _quant_meta_tensor(meta)
        return out

    if fmt == "fp8":
        qdata_chunks = []
        scale_chunks = []
        chunk_e = 256
        fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
        for s in range(0, num_experts, chunk_e):
            e = min(num_experts, s + chunk_e)
            sub = bank[s:e].to(device=device)
            scales = sub.abs().amax(dim=(1, 2)).float().clamp(min=1e-12) / fp8_max
            inv = (1.0 / scales).view(-1, 1, 1)
            q_sub = (sub.float() * inv).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            qdata_chunks.append(q_sub.cpu())
            scale_chunks.append(scales.cpu())
        out[f"{stem}.weight"] = torch.cat(qdata_chunks, dim=0).contiguous()
        out[f"{stem}.weight_scale"] = torch.cat(scale_chunks, dim=0).contiguous()
        out[f"{stem}.comfy_quant"] = _quantize_meta("float8_e4m3fn", {"num_experts": num_experts})
        return out

    raise ValueError(f"Unsupported format: {fmt}")


def _read_weight_map(src_path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    if src_path.is_file():
        with safe_open(str(src_path), framework="pt", device="cpu") as f:
            meta = dict(f.metadata() or {})
            return {k: src_path for k in f.keys()}, meta
    index_file = src_path / "model.safetensors.index.json"
    if index_file.exists():
        data = json.loads(index_file.read_text())
        return {k: src_path / v for k, v in data["weight_map"].items()}, {}
    single = src_path / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            meta = dict(f.metadata() or {})
            return {k: single for k in f.keys()}, meta
    raise FileNotFoundError(f"No safetensors checkpoint found at {src_path}")


def _merge_safetensors_parts_streaming(
    part_files: list[Path],
    output_file: Path,
    base_metadata: dict[str, str],
) -> None:
    """Concatenate temporary `.safetensors` part files into one `.safetensors` file in O(1) RAM."""
    merged_header: dict[str, object] = {}
    if base_metadata:
        merged_header["__metadata__"] = dict(base_metadata)

    part_data_spans: list[tuple[Path, int, int]] = []
    running_offset = 0

    for part in part_files:
        file_size = part.stat().st_size
        with part.open("rb") as f:
            (hdr_len,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
        max_end = 0
        # Sort entries by original offset so merged offsets are strictly contiguous
        items = [(k, v) for k, v in hdr.items() if k != "__metadata__"]
        items.sort(key=lambda kv: kv[1]["data_offsets"][0])
        for key, info in items:
            s_off, e_off = info["data_offsets"]
            merged_header[key] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [running_offset + s_off, running_offset + e_off],
            }
            if e_off > max_end:
                max_end = e_off
        if data_start + max_end > file_size:
            raise ValueError(f"Corrupt temporary part {part}")
        part_data_spans.append((part, data_start, max_end))
        running_offset += max_end

    hdr_bytes = json.dumps(merged_header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    if pad:
        hdr_bytes += b" " * pad

    with output_file.open("wb") as out_f:
        out_f.write(struct.pack("<Q", len(hdr_bytes)))
        out_f.write(hdr_bytes)
        for part, data_start, data_len in part_data_spans:
            with part.open("rb") as in_f:
                in_f.seek(data_start)
                remaining = data_len
                while remaining > 0:
                    chunk = in_f.read(min(16 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    out_f.write(chunk)
                    remaining -= len(chunk)
            part.unlink()


class _ShardWriter:
    def __init__(
        self,
        output_path: Path,
        max_shard_bytes: int,
        base_metadata: Optional[dict[str, str]] = None,
    ) -> None:
        self.output_path = output_path
        self.base_metadata = dict(base_metadata or {})
        self.is_dir = output_path.suffix != ".safetensors"
        if self.is_dir:
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.flush_bytes = max_shard_bytes
        else:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Spill 4 GiB temporary parts so single-file assembly never buffers >4 GiB in RAM
            self.flush_bytes = 4 * (1024 ** 3)
            self._tmp_dir = self.output_path.parent / f".tmp-{self.output_path.stem}-parts"
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self.current_tensors: dict[str, torch.Tensor] = {}
        self.current_bytes = 0
        self.shard_paths: list[tuple[str, dict[str, str]]] = []
        self.total_bytes = 0

    def add(self, tensors: dict[str, torch.Tensor]) -> None:
        batch_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if (
            self.flush_bytes > 0
            and self.current_tensors
            and self.current_bytes + batch_bytes > self.flush_bytes
        ):
            self._flush_partial()
        self.current_tensors.update(tensors)
        self.current_bytes += batch_bytes
        self.total_bytes += batch_bytes

    def _flush_partial(self) -> None:
        idx = len(self.shard_paths) + 1
        tmp_name = f"model-part-{idx:05d}.safetensors"
        target_dir = self.output_path if self.is_dir else self._tmp_dir
        save_file(
            self.current_tensors,
            str(target_dir / tmp_name),
            metadata=self.base_metadata if (self.is_dir and idx == 1 and self.base_metadata) else None,
        )
        self.shard_paths.append((tmp_name, {k: tmp_name for k in self.current_tensors}))
        self.current_tensors = {}
        self.current_bytes = 0

    def finalize(self) -> None:
        if not self.is_dir:
            if not self.shard_paths:
                save_file(self.current_tensors, str(self.output_path), metadata=self.base_metadata or None)
                self.current_tensors = {}
                if self._tmp_dir.exists():
                    self._tmp_dir.rmdir()
                return
            if self.current_tensors:
                self._flush_partial()
            part_files = [self._tmp_dir / name for name, _ in self.shard_paths]
            _merge_safetensors_parts_streaming(part_files, self.output_path, self.base_metadata)
            if self._tmp_dir.exists():
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
            return

        if self.current_tensors:
            self._flush_partial()

        total_shards = len(self.shard_paths)
        weight_map: dict[str, str] = {}
        for idx, (tmp_name, keys_map) in enumerate(self.shard_paths, start=1):
            final_name = (
                "model.safetensors"
                if total_shards == 1
                else f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
            )
            src = self.output_path / tmp_name
            dst = self.output_path / final_name
            if src != dst:
                src.rename(dst)
            for k in keys_map:
                weight_map[k] = final_name

        if total_shards > 1:
            index_payload = {
                "metadata": {"total_size": self.total_bytes, **self.base_metadata},
                "weight_map": weight_map,
            }
            (self.output_path / "model.safetensors.index.json").write_text(
                json.dumps(index_payload, indent=2, sort_keys=True) + "\n"
            )


def convert_and_quantize(
    src_path: Path,
    dst_path: Path,
    model_kind: str,
    fmt: str,
    device_str: str = "cuda",
    max_shard_gb: float = 0.0,
    keep_boundary_blocks: bool = False,
    input_scales_path: Optional[Path] = None,
    tokenizer_json_path: Optional[Path] = None,
) -> dict:
    if fmt not in ("bf16", "nvfp4", "int8", "fp8"):
        raise ValueError(f"Unsupported format: {fmt}")
    if model_kind not in ("preview", "refiner", "qwen35"):
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    device = torch.device(device_str if (fmt != "bf16" and torch.cuda.is_available()) else "cpu")
    mm_layers = PREVIEW_MM_LAYERS if model_kind == "preview" else REFINER_MM_LAYERS
    input_scales = (
        json.loads(input_scales_path.read_text()) if input_scales_path is not None else None
    )

    weight_map, meta = _read_weight_map(src_path)
    if tokenizer_json_path is not None:
        meta["qwen35_27b_tokenizer_json"] = tokenizer_json_path.read_text()

    by_shard: dict[Path, list[str]] = defaultdict(list)
    for k, p in weight_map.items():
        by_shard[p].append(k)

    # Pre-read small router.expert_bias_ema tensors and auto-detect mm_layers from pre_norm shapes
    ema_override: dict[str, torch.Tensor] = {}
    detected_mm_layers: set[int] = set()
    inferred_hidden: Optional[int] = None
    if model_kind in ("preview", "refiner"):
        ve_key = "pre_adapter.video_embedder.weight"
        mhc_key = "block.layers.0.mhc_bias_pre_attn"
        if ve_key in weight_map:
            with safe_open(str(weight_map[ve_key]), framework="pt", device="cpu") as f:
                ve_rows = f.get_slice(ve_key).get_shape()[0]
            n_stream = 1
            if mhc_key in weight_map:
                with safe_open(str(weight_map[mhc_key]), framework="pt", device="cpu") as f:
                    n_stream = f.get_slice(mhc_key).get_shape()[0]
            inferred_hidden = ve_rows // max(1, n_stream)

    for shard_file, keys in by_shard.items():
        ema_keys = [k for k in keys if k.endswith(".moe_mlp.router.expert_bias_ema")]
        pnorm_keys = [
            k for k in keys
            if inferred_hidden is not None and k.startswith("block.layers.") and k.endswith(".attention.pre_norm.weight")
        ]
        if not ema_keys and not pnorm_keys:
            continue
        with safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for ek in ema_keys:
                ema_override[ek[: -len("_ema")]] = f.get_tensor(ek).float().contiguous()
            for pk in pnorm_keys:
                parts = pk.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    if f.get_slice(pk).get_shape()[0] == 3 * inferred_hidden:
                        detected_mm_layers.add(int(parts[2]))

    if detected_mm_layers:
        mm_layers = detected_mm_layers

    max_shard_bytes = int(max_shard_gb * (1024 ** 3)) if max_shard_gb > 0 else 0
    writer = _ShardWriter(dst_path, max_shard_bytes=max_shard_bytes, base_metadata=meta)

    seen_rope_bands = False
    quantized_count = 0
    kept_count = 0

    for shard_idx, (shard_file, keys) in enumerate(sorted(by_shard.items()), start=1):
        LOGGER.info("Processing shard %d/%d: %s (%d keys)", shard_idx, len(by_shard), shard_file.name, len(keys))
        with safe_open(str(shard_file), framework="pt", device="cpu") as handle:
            for key in keys:
                if model_kind == "qwen35" and key.startswith(("model.visual.", "visual.", "mtp.", "lm_head.")):
                    continue
                if key.endswith(".moe_mlp.router.expert_bias_ema"):
                    continue

                if key in ema_override:
                    writer.add({key: ema_override.pop(key)})
                    kept_count += 1
                    continue

                tensor = handle.get_tensor(key)
                if key == "pre_adapter.rope.bands":
                    seen_rope_bands = True

                # Normalize MAGI-2 MoE 3D bank [3072, in, out] -> <key>.weight [3072, out, in]
                if model_kind in ("preview", "refiner") and key.endswith(
                    (".moe_mlp.W_gate", ".moe_mlp.W_up", ".moe_mlp.W_down")
                ):
                    bank = tensor.transpose(1, 2).contiguous()
                    del tensor
                    if fmt == "bf16":
                        writer.add({f"{key}.weight": bank.to(torch.bfloat16)})
                        kept_count += 1
                    else:
                        writer.add(_quantize_3d_moe_bank(key, bank, fmt, device))
                        quantized_count += 1
                    del bank
                    continue

                # Split 3-modality stacked linears along dim=0 so each stream gets independent quant scales
                if model_kind in ("preview", "refiner") and _is_stacked_mm_linear_key(key, mm_layers):
                    stem, param_name = key.rsplit(".", 1)
                    chunks = tensor.chunk(3, dim=0)
                    for idx, chunk in enumerate(chunks):
                        c_contig = chunk.contiguous()
                        sub_key = f"{stem}.linears.{idx}.{param_name}"
                        if fmt == "bf16" or _should_keep_high_precision(
                            sub_key, c_contig, keep_boundary_blocks=keep_boundary_blocks, mm_layers=mm_layers, fmt=fmt
                        ):
                            writer.add({sub_key: c_contig})
                            kept_count += 1
                        else:
                            writer.add(
                                _quantize_2d_weight(
                                    f"{stem}.linears.{idx}", c_contig, fmt, device, input_scales=input_scales
                                )
                            )
                            quantized_count += 1
                    del tensor
                    continue

                # Standard 2D / 1D tensors
                if fmt == "bf16" or _should_keep_high_precision(
                    key, tensor, keep_boundary_blocks=keep_boundary_blocks, mm_layers=mm_layers, fmt=fmt
                ):
                    writer.add({key: tensor.contiguous()})
                    kept_count += 1
                else:
                    stem = key[: -len(".weight")]
                    writer.add(_quantize_2d_weight(stem, tensor, fmt, device, input_scales=input_scales))
                    quantized_count += 1
                del tensor

    for rem_key, rem_val in ema_override.items():
        writer.add({rem_key: rem_val})
        kept_count += 1

    if model_kind in ("preview", "refiner") and not seen_rope_bands:
        writer.add({"pre_adapter.rope.bands": default_rope_bands(128)})

    writer.finalize()
    return {
        "model_kind": model_kind,
        "format": fmt,
        "output": str(dst_path),
        "total_bytes": writer.total_bytes,
        "quantized_tensors": quantized_count,
        "kept_tensors": kept_count,
    }


def calibrate_fp8_input_scales(
    model_patcher,
    sample_inputs: list[tuple[object, torch.Tensor, torch.Tensor, dict]],
) -> dict[str, float]:
    """Capture max(|activation|) / 448.0 across all 2D Linear layers over reference forward passes."""
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    amax_by_name: dict[str, float] = {}
    handles = []

    diff_model = model_patcher.model.diffusion_model
    for name, mod in diff_model.named_modules():
        if hasattr(mod, "in_features") and hasattr(mod, "out_features") and not hasattr(mod, "num_experts"):
            def _make_hook(layer_name: str):
                def _hook(_module, inputs):
                    if not inputs:
                        return
                    x = inputs[0]
                    if torch.is_tensor(x) and x.numel() > 0:
                        val = float(x.detach().abs().amax().float().cpu().item())
                        if val > amax_by_name.get(layer_name, 0.0):
                            amax_by_name[layer_name] = val

                return _hook

            handles.append(mod.register_forward_pre_hook(_make_hook(name)))

    try:
        for x_in, t_in, ctx_in, payload in sample_inputs:
            diff_model(x_in, t_in, context=ctx_in, magi2_payload=payload)
    finally:
        for h in handles:
            h.remove()

    return {
        name: max(amax / fp8_max, 1e-12)
        for name, amax in sorted(amax_by_name.items())
        if amax > 0.0
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source safetensors file or directory")
    parser.add_argument("--dst", type=Path, required=True, help="Destination .safetensors file or directory")
    parser.add_argument("--kind", choices=("preview", "refiner", "qwen35"), required=True)
    parser.add_argument("--format", choices=("bf16", "nvfp4", "int8", "fp8"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-shard-gb", type=float, default=0.0)
    parser.add_argument("--keep-boundary-blocks", action="store_true")
    parser.add_argument("--input-scales", type=Path, default=None)
    parser.add_argument("--tokenizer-json", type=Path, default=None)
    args = parser.parse_args()

    summary = convert_and_quantize(
        src_path=args.src,
        dst_path=args.dst,
        model_kind=args.kind,
        fmt=args.format,
        device_str=args.device,
        max_shard_gb=args.max_shard_gb,
        keep_boundary_blocks=args.keep_boundary_blocks,
        input_scales_path=args.input_scales,
        tokenizer_json_path=args.tokenizer_json,
    )
    LOGGER.info("Completed: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
