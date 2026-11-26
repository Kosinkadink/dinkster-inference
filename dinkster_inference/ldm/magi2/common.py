from __future__ import annotations

from enum import IntEnum
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy_kitchen
import dinkster_inference.conds
import dinkster_inference.model_management
import dinkster_inference.ops


class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    TIME = 3


def swiglu7(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    target_dtype = x.dtype if out_dtype is None else out_dtype
    x_fp32 = x.float()
    x_glu = x_fp32[..., ::2].clamp(max=limit)
    x_lin = x_fp32[..., 1::2].clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_lin + 1.0)).to(target_dtype)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    pos = position.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, dtype=torch.float32, device=position.device)
        / half
    )
    args = pos[:, None] * freqs[None, :]
    emb = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def default_rope_bands(
    dim: int,
    temperature: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    num_bands = dim // 8
    exp = torch.arange(0, num_bands, dtype=torch.float32, device=device) / float(num_bands)
    return (1.0 / (temperature ** exp)).to(dtype=dtype)


def get_coords(
    shape: tuple[int, int, int],
    ref_feat_shape: tuple[int, int, int],
    offset_thw: tuple[int, int, int] = (0, 0, 0),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape
    off_t, off_h, off_w = offset_thw

    if ori_t == 0 or ori_h == 0 or ori_w == 0:
        return torch.empty((0, 9), device=device, dtype=dtype)

    t_rng = torch.arange(ori_t, device=device, dtype=dtype) + float(off_t)
    h_rng = torch.arange(ori_h, device=device, dtype=dtype) + float(off_h)
    w_rng = torch.arange(ori_w, device=device, dtype=dtype) + float(off_w)

    t_grid, h_grid, w_grid = torch.meshgrid(t_rng, h_rng, w_rng, indexing="ij")
    coords_flat = torch.stack((t_grid, h_grid, w_grid), dim=-1).reshape(-1, 3)
    meta = torch.tensor(
        (ori_t, ori_h, ori_w, ref_t, ref_h, ref_w), device=device, dtype=dtype
    ).expand(coords_flat.shape[0], -1)
    return torch.cat((coords_flat, meta), dim=-1)


class ElementWiseFourierEmbed(nn.Module):
    def __init__(self, dim: int, device=None, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer(
            "bands", torch.empty(dim // 8, device=device, dtype=dtype)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_f32 = coords.float()
        coords_xyz = coords_f32[:, :3]
        sizes = coords_f32[:, 3:6]
        refs = coords_f32[:, 6:9]

        denom = torch.where(sizes == 1.0, torch.ones_like(sizes), sizes - 1.0)
        scales = (refs - 1.0) / denom
        scales = torch.where((refs == 1.0) & (sizes == 1.0), torch.ones_like(scales), scales)

        centers = (sizes - 1.0) * 0.5
        centers = centers.clone()
        centers[:, 0] = 0.0
        coords_xyz = coords_xyz - centers

        bands = dinkster_inference.model_management.cast_to(
            self.bands, dtype=torch.float32, device=coords.device
        )
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * bands
        return torch.cat((proj.sin(), proj.cos()), dim=1).flatten(1)


def apply_magi2_rope(
    xq: torch.Tensor, xk: torch.Tensor, rope: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply split-half partial RoPE on the leading rotary dimensions.

    Inputs xq and xk have shape [B, L, H, D]; rope has shape [L, rotary_dim]
    or [B, L, rotary_dim] containing concatenated (sin, cos) in float32.
    """
    sin_emb, cos_emb = rope.float().tensor_split(2, dim=-1)
    half = sin_emb.shape[-1]
    rot_dim = half * 2

    if sin_emb.ndim == 2:
        sin_emb = sin_emb.unsqueeze(0)
        cos_emb = cos_emb.unsqueeze(0)

    # xq, xk -> [B, H, L, D] for comfy_kitchen.apply_rope_split_half
    q_bhld = xq.transpose(1, 2)
    k_bhld = xk.transpose(1, 2)

    q_rot = q_bhld[..., :rot_dim].float()
    k_rot = k_bhld[..., :rot_dim].float()

    matrix = torch.stack((cos_emb, -sin_emb, sin_emb, cos_emb), dim=-1)
    matrix = matrix.reshape(matrix.shape[0], 1, matrix.shape[1], half, 2, 2)

    q_rot, k_rot = comfy_kitchen.apply_rope_split_half(q_rot, k_rot, matrix)
    q_out = torch.cat((q_rot.to(xq.dtype), q_bhld[..., rot_dim:]), dim=-1).transpose(1, 2)
    k_out = torch.cat((k_rot.to(xk.dtype), k_bhld[..., rot_dim:]), dim=-1).transpose(1, 2)
    return q_out, k_out


class ModalityDispatcher:
    """Call-scoped token permutation and split counts by modality."""

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int = 3) -> None:
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping, stable=True)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        permuted = modality_mapping.index_select(0, self.permute_mapping)
        counts = torch.bincount(permuted.to(torch.int64), minlength=num_modalities)
        self.group_sizes: list[int] = [int(c) for c in counts.cpu().tolist()]

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, self.permute_mapping)

    def inv_permute(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, self.inv_permute_mapping)

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_sizes, dim=0))

    @staticmethod
    def undispatch(groups: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(groups, dim=0)


class MultiModalityRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        num_modality: int = 1,
        out_dtype: Optional[torch.dtype] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(
            torch.empty(dim * num_modality, dtype=torch.float32, device=device)
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        target_dtype = x.dtype if self.out_dtype is None else self.out_dtype
        t = x.float()
        t = t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        w = dinkster_inference.model_management.cast_to(
            self.weight, dtype=torch.float32, device=x.device
        )
        if self.num_modality == 1:
            return (t * (w + 1.0)).to(target_dtype)
        if modality_dispatcher is None:
            raise ValueError("modality_dispatcher is required when num_modality > 1")
        w_chunks = w.chunk(self.num_modality, dim=0)
        parts = modality_dispatcher.dispatch(t)
        out_parts = [
            part * (w_chunks[i] + 1.0) if part.shape[0] > 0 else part
            for i, part in enumerate(parts)
        ]
        return modality_dispatcher.undispatch(out_parts).to(target_dtype)


class MultiModalityLinear(nn.Module):
    """Three independent per-modality Linears so each stream keeps its own quant scale."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_modality: int = 3,
        bias: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.num_modality = num_modality
        self.out_features = out_features
        self.linears = nn.ModuleList(
            [
                operations.Linear(
                    in_features, out_features, bias=bias, dtype=dtype, device=device
                )
                for _ in range(num_modality)
            ]
        )

    def forward(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        parts = modality_dispatcher.dispatch(x)
        outputs = []
        for idx, part in enumerate(parts):
            if part.shape[0] == 0:
                outputs.append(part.new_empty((0, self.out_features)))
            else:
                outputs.append(self.linears[idx](part))
        return modality_dispatcher.undispatch(outputs)


def make_modality_linear(
    in_features: int,
    out_features: int,
    num_modality: int = 1,
    bias: bool = False,
    dtype=None,
    device=None,
    operations=None,
) -> nn.Module:
    if num_modality == 1:
        return operations.Linear(
            in_features, out_features, bias=bias, dtype=dtype, device=device
        )
    return MultiModalityLinear(
        in_features,
        out_features,
        num_modality=num_modality,
        bias=bias,
        dtype=dtype,
        device=device,
        operations=operations,
    )


def apply_modality_linear(
    module: nn.Module,
    x: torch.Tensor,
    modality_dispatcher: Optional[ModalityDispatcher] = None,
) -> torch.Tensor:
    if isinstance(module, MultiModalityLinear):
        if modality_dispatcher is None:
            raise ValueError("modality_dispatcher is required for MultiModalityLinear")
        return module(x, modality_dispatcher)
    return module(x)


def _is_stacked_mm_linear_key(key: str, mm_layers: set[int]) -> bool:
    if not (key.endswith(".weight") or key.endswith(".bias")):
        return False
    if ".linears." in key:
        return False
    if ".mlp.modality_specific_shared_expert_fc1." in key or ".mlp.modality_specific_shared_expert_fc2." in key:
        return True
    if not key.startswith("block.layers."):
        return False
    parts = key.split(".")
    if len(parts) < 5 or not parts[2].isdigit():
        return False
    layer_idx = int(parts[2])
    if layer_idx not in mm_layers:
        return False
    subpath = ".".join(parts[3:-1])
    return subpath in {
        "attention.linear_g",
        "attention.linear_qkv",
        "attention.linear_proj",
        "mlp.up_gate_proj",
        "mlp.down_proj",
    }


def normalize_magi2_state_dict(
    state_dict: dict[str, torch.Tensor],
    mm_layers: tuple[int, ...] = (0, 1, 38, 39),
    rope_dim: int = 128,
) -> dict[str, torch.Tensor]:
    """Normalize raw SandAI MAGI-2 state dicts in-place when unsplit keys are present.

    Already-converted checkpoints (with `.linears.0.weight` and `.W_gate.weight`) pass
    through untouched so lazy TensorFileSlice mmap handles remain unmaterialized.
    """
    mm_set = set(mm_layers)

    if "pre_adapter.rope.bands" not in state_dict:
        state_dict["pre_adapter.rope.bands"] = default_rope_bands(rope_dim)

    # Fast path for pre-converted / quantized checkpoints: zero tensor reads!
    if (
        "block.layers.0.attention.linear_qkv.linears.0.weight" in state_dict
        and "block.layers.2.mlp.moe_mlp.router.expert_bias_ema" not in state_dict
    ):
        return state_dict

    out: dict[str, torch.Tensor] = {}
    ema_biases: dict[str, torch.Tensor] = {}

    for key, val in state_dict.items():
        if key.endswith(".moe_mlp.router.expert_bias_ema"):
            ema_biases[key[: -len("_ema")]] = val
            continue
        if key.endswith((".moe_mlp.W_gate", ".moe_mlp.W_up", ".moe_mlp.W_down")):
            # Raw SandAI MoE parameter has shape [3072, in_dim, out_dim];
            # MoEExperts expects `<name>.weight` with shape [3072, out_dim, in_dim].
            out[f"{key}.weight"] = val.transpose(1, 2).contiguous()
            continue
        if _is_stacked_mm_linear_key(key, mm_set):
            stem, param_name = key.rsplit(".", 1)
            chunks = val.chunk(3, dim=0)
            for idx, chunk in enumerate(chunks):
                out[f"{stem}.linears.{idx}.{param_name}"] = chunk.contiguous()
            continue
        out[key] = val

    for target_key, ema_val in ema_biases.items():
        out[target_key] = ema_val

    return out


class CONDMagi2CrossAttn(dinkster_inference.conds.CONDRegular):
    """Variable-length text conditioning that zero-pads to `max(L)` so cond+uncond always batch."""

    def can_concat(self, other):
        if self.cond.ndim != 3 or other.cond.ndim != 3:
            return False
        if self.cond.shape[0] != other.cond.shape[0] or self.cond.shape[2] != other.cond.shape[2]:
            return False
        return self.cond.device == other.cond.device

    def concat(self, others):
        if len(others) == 0:
            return self.cond
        tensors = [self.cond] + [o.cond for o in others]
        max_len = max(t.shape[1] for t in tensors)
        padded = [
            F.pad(t, (0, 0, 0, max_len - t.shape[1])) if t.shape[1] < max_len else t
            for t in tensors
        ]
        return torch.cat(padded, dim=0)


class CONDMagi2Payload(dinkster_inference.conds.CONDConstant):
    """Per-item MAGI-2 metadata (`txt_lens`, `ref_images`) that merges across batched conds."""

    def process_cond(self, batch_size, **kwargs):
        data = dict(self.cond)
        if "txt_lens" in data and len(data["txt_lens"]) == 1 and batch_size > 1:
            data["txt_lens"] = data["txt_lens"] * batch_size
        if "ref_images" in data and len(data["ref_images"]) == 1 and batch_size > 1:
            data["ref_images"] = data["ref_images"] * batch_size
        return self._copy_with(data)

    def can_concat(self, other):
        return self.cond.get("refiner_include_audio") == other.cond.get("refiner_include_audio")

    def concat(self, others):
        if len(others) == 0:
            return self.cond
        all_conds = [self.cond] + [o.cond for o in others]
        merged = dict(self.cond)
        merged["txt_lens"] = [l for c in all_conds for l in c.get("txt_lens", [0])]
        merged["ref_images"] = [r for c in all_conds for r in c.get("ref_images", [[]])]
        return merged
