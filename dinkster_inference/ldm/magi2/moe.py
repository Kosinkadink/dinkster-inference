from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy_kitchen
import dinkster_inference.model_management
import dinkster_inference.ops
from dinkster_inference.quant_ops import QuantizedTensor

from .common import (
    ModalityDispatcher,
    MultiModalityRMSNorm,
    apply_modality_linear,
    make_modality_linear,
    swiglu7,
)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _swiglu7_triton(gate, up, out_dtype: tl.constexpr):
        gate_clamped = tl.minimum(gate, 7.0)
        up_clamped = tl.maximum(tl.minimum(up, 7.0), -7.0)
        swish = gate_clamped * tl.sigmoid(1.702 * gate_clamped)
        return (swish * (up_clamped + 1.0)).to(out_dtype)

    @triton.jit
    def _bsearch_expert_id(
        cu_tiles_ptr,
        tile_id,
        NUM_EXPERTS: tl.constexpr,
        LOG2_NUM_EXPERTS: tl.constexpr,
    ):
        lo = 0
        hi = NUM_EXPERTS
        for _ in tl.static_range(0, LOG2_NUM_EXPERTS + 1):
            mid = (lo + hi + 1) // 2
            below = tl.load(cu_tiles_ptr + mid) <= tile_id
            lo = tl.where(below, mid, lo)
            hi = tl.where(below, hi, mid - 1)
        return lo

    @triton.jit
    def _mh_moe_fwd_kernel(
        x_ptr,
        wg_ptr,
        wu_ptr,
        wd_ptr,
        wg_scale_ptr,
        wu_scale_ptr,
        wd_scale_ptr,
        y_ptr,
        gather_ids_ptr,
        probs_ptr,
        expert_offsets_ptr,
        cu_tiles_ptr,
        stride_x_s,
        stride_x_h,
        stride_x_dh,
        stride_wg_ne,
        stride_wg_dh,
        stride_wg_de,
        stride_wu_ne,
        stride_wu_dh,
        stride_wu_de,
        stride_wd_ne,
        stride_wd_de,
        stride_wd_dh,
        stride_y_s,
        stride_y_h,
        stride_y_dh,
        D_HEAD: tl.constexpr,
        D_EXPERT: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        LOG2_NUM_EXPERTS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_DH: tl.constexpr,
        BLOCK_DE: tl.constexpr,
        HAS_EXPERT_SCALES: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        total_tiles = tl.load(cu_tiles_ptr + NUM_EXPERTS)
        if tile_id >= total_tiles:
            return

        expert_id = _bsearch_expert_id(
            cu_tiles_ptr, tile_id, NUM_EXPERTS, LOG2_NUM_EXPERTS
        )
        expert_id_i64 = expert_id.to(tl.int64)
        head_id = expert_id // (NUM_EXPERTS // NUM_HEADS)

        tile_in_expert = tile_id - tl.load(cu_tiles_ptr + expert_id)
        token_start = tl.load(expert_offsets_ptr + expert_id) + tile_in_expert * BLOCK_T
        expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
        n = tl.minimum(token_start + BLOCK_T, expert_end) - token_start

        block_dh_offs = tl.arange(0, BLOCK_DH)
        block_de_offs = tl.arange(0, BLOCK_DE)
        block_t_offs = tl.arange(0, BLOCK_T)
        dh_offs = tl.arange(0, D_HEAD)

        block_token_offs = token_start + block_t_offs
        block_token_mask = block_t_offs < n

        gather_ids = tl.load(
            gather_ids_ptr + block_token_offs, mask=block_token_mask, other=0
        )
        probs = tl.load(probs_ptr + block_token_offs, mask=block_token_mask, other=0.0)
        x_offs = gather_ids * stride_x_s + head_id * stride_x_h

        if HAS_EXPERT_SCALES:
            wg_scale = tl.load(wg_scale_ptr + expert_id).to(tl.float32)
            wu_scale = tl.load(wu_scale_ptr + expert_id).to(tl.float32)
            wd_scale = tl.load(wd_scale_ptr + expert_id).to(tl.float32)

        mm_dtype: tl.constexpr = x_ptr.dtype.element_ty
        y_acc = tl.zeros([BLOCK_T, D_HEAD], dtype=tl.float32)
        for dexpert_start in tl.range(0, D_EXPERT, BLOCK_DE):
            dexpert_offs = dexpert_start + block_de_offs
            gate_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=tl.float32)
            up_acc = tl.zeros([BLOCK_T, BLOCK_DE], dtype=tl.float32)
            for dhead_start in tl.static_range(0, D_HEAD, BLOCK_DH):
                dhead_offs = dhead_start + block_dh_offs
                x_block = tl.load(
                    x_ptr + x_offs[:, None] + dhead_offs[None, :] * stride_x_dh,
                    mask=block_token_mask[:, None],
                    other=0.0,
                )
                w_gate = tl.load(
                    wg_ptr
                    + expert_id_i64 * stride_wg_ne
                    + dhead_offs[:, None] * stride_wg_dh
                    + dexpert_offs[None, :] * stride_wg_de
                ).to(mm_dtype)
                w_up = tl.load(
                    wu_ptr
                    + expert_id_i64 * stride_wu_ne
                    + dhead_offs[:, None] * stride_wu_dh
                    + dexpert_offs[None, :] * stride_wu_de
                ).to(mm_dtype)
                gate_acc += tl.dot(x_block, w_gate)
                up_acc += tl.dot(x_block, w_up)

            if HAS_EXPERT_SCALES:
                gate_acc = gate_acc * wg_scale
                up_acc = up_acc * wu_scale

            hidden = _swiglu7_triton(gate_acc, up_acc, mm_dtype)
            w_down = tl.load(
                wd_ptr
                + expert_id_i64 * stride_wd_ne
                + dexpert_offs[:, None] * stride_wd_de
                + dh_offs[None, :] * stride_wd_dh
            ).to(mm_dtype)
            down_dot = tl.dot(hidden, w_down)
            if HAS_EXPERT_SCALES:
                down_dot = down_dot * wd_scale
            y_acc += down_dot

        y_acc = y_acc * probs[:, None]
        y_offs = gather_ids * stride_y_s + head_id * stride_y_h
        y_ptrs = y_ptr + y_offs[:, None] + dh_offs[None, :] * stride_y_dh
        tl.atomic_add(
            y_ptrs,
            y_acc.to(y_ptr.dtype.element_ty),
            mask=block_token_mask[:, None],
        )

    @triton.jit
    def _mh_moe_proj_kernel(
        inp_ptr,
        w_ptr,
        out_ptr,
        expert_offsets_ptr,
        cu_tiles_ptr,
        stride_in_t,
        stride_in_k,
        stride_w_ne,
        stride_w_k,
        stride_w_n,
        stride_out_t,
        stride_out_n,
        K_DIM: tl.constexpr,
        N_DIM: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        LOG2_NUM_EXPERTS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        n_block_id = tl.program_id(1)
        total_tiles = tl.load(cu_tiles_ptr + NUM_EXPERTS)
        if tile_id >= total_tiles:
            return

        expert_id = _bsearch_expert_id(cu_tiles_ptr, tile_id, NUM_EXPERTS, LOG2_NUM_EXPERTS)
        expert_i64 = expert_id.to(tl.int64)
        tile_in_exp = tile_id - tl.load(cu_tiles_ptr + expert_id)
        t_start = tl.load(expert_offsets_ptr + expert_id) + tile_in_exp * BLOCK_T
        t_end = tl.load(expert_offsets_ptr + expert_id + 1)
        n_tok = tl.minimum(t_start + BLOCK_T, t_end) - t_start

        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = tl.arange(0, BLOCK_T) < n_tok
        n_offs = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N_DIM
        k_base = tl.arange(0, BLOCK_K)

        acc = tl.zeros([BLOCK_T, BLOCK_N], dtype=tl.float32)
        mm_dtype: tl.constexpr = inp_ptr.dtype.element_ty
        for k_start in tl.static_range(0, K_DIM, BLOCK_K):
            k_offs = k_start + k_base
            a = tl.load(
                inp_ptr + t_offs[:, None] * stride_in_t + k_offs[None, :] * stride_in_k,
                mask=t_mask[:, None],
                other=0.0,
            )
            b = tl.load(
                w_ptr + expert_i64 * stride_w_ne + k_offs[:, None] * stride_w_k + n_offs[None, :] * stride_w_n,
                mask=n_mask[None, :],
                other=0.0,
            ).to(mm_dtype)
            acc += tl.dot(a, b)

        tl.store(
            out_ptr + t_offs[:, None] * stride_out_t + n_offs[None, :] * stride_out_n,
            acc.to(out_ptr.dtype.element_ty),
            mask=t_mask[:, None] & n_mask[None, :],
        )


def _compute_topk_routing(
    router_logits: torch.Tensor,
    top_k: int,
    expert_bias: Optional[torch.Tensor],
    score_func: str = "sigmoid",
    route_norm: bool = True,
    route_scale: float = 4.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_func == "sigmoid":
        scores = torch.sigmoid(router_logits)
    elif score_func == "softmax":
        scores = F.softmax(router_logits, dim=-1)
    else:
        raise ValueError(f"Unsupported MoE score_func: {score_func}")

    choice_scores = scores
    if expert_bias is not None:
        choice_scores = choice_scores + expert_bias.view(scores.shape[0], 1, scores.shape[2])

    k_eff = min(top_k, choice_scores.shape[-1])
    _, topk_indices = torch.topk(choice_scores, k_eff, dim=-1)
    topk_probs = scores.gather(-1, topk_indices)
    if route_norm:
        topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=1e-12)
    return topk_probs * route_scale, topk_indices


def _global_sort_assignments(
    topk_probs: torch.Tensor, topk_indices: torch.Tensor, num_experts_per_head: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_heads, seq_len, top_k = topk_indices.shape
    device = topk_indices.device

    head_offset = torch.arange(num_heads, device=device).view(num_heads, 1, 1) * num_experts_per_head
    global_indices = (topk_indices + head_offset).reshape(-1)
    flat_probs = topk_probs.reshape(-1)
    flat_tokens = (
        torch.arange(seq_len, device=device).view(1, seq_len, 1).expand(num_heads, seq_len, top_k).reshape(-1)
    )

    sorted_order = global_indices.argsort(stable=True)
    gather_ids = flat_tokens.index_select(0, sorted_order).to(torch.int32)
    probs_sorted = flat_probs.index_select(0, sorted_order).float()

    counts = torch.bincount(global_indices.index_select(0, sorted_order), minlength=num_heads * num_experts_per_head)
    expert_offsets = F.pad(counts.cumsum(0), (1, 0))
    return gather_ids, probs_sorted, expert_offsets


def _launch_triton_mh_moe(
    x_heads: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    w_gate_t: torch.Tensor,
    w_up_t: torch.Tensor,
    w_down_t: torch.Tensor,
    wg_scale: Optional[torch.Tensor] = None,
    wu_scale: Optional[torch.Tensor] = None,
    wd_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total_assign = gather_ids.shape[0]
    if total_assign == 0:
        return torch.zeros_like(x_heads)

    num_heads = x_heads.shape[1]
    d_head = x_heads.shape[2]
    d_expert = w_down_t.shape[1]
    block_t, block_dh, block_de = 64, 64, 32

    y = torch.zeros_like(x_heads, dtype=torch.float32)
    num_flat_experts = expert_offsets.shape[0] - 1
    log2_experts = max(1, math.ceil(math.log2(max(num_flat_experts, 1) + 1)))

    expert_counts = expert_offsets.diff()
    tile_counts = (expert_counts + block_t - 1) // block_t
    cu_tiles = F.pad(torch.cumsum(tile_counts, dim=0, dtype=torch.int32), (1, 0))
    grid_size = (total_assign + block_t - 1) // block_t + num_flat_experts

    has_scales = wg_scale is not None
    dummy = cu_tiles
    _mh_moe_fwd_kernel[(grid_size,)](
        x_ptr=x_heads,
        wg_ptr=w_gate_t,
        wu_ptr=w_up_t,
        wd_ptr=w_down_t,
        wg_scale_ptr=wg_scale if has_scales else dummy,
        wu_scale_ptr=wu_scale if has_scales else dummy,
        wd_scale_ptr=wd_scale if has_scales else dummy,
        y_ptr=y,
        gather_ids_ptr=gather_ids,
        probs_ptr=probs,
        expert_offsets_ptr=expert_offsets,
        cu_tiles_ptr=cu_tiles,
        stride_x_s=x_heads.stride(0),
        stride_x_h=x_heads.stride(1),
        stride_x_dh=x_heads.stride(2),
        stride_wg_ne=w_gate_t.stride(0),
        stride_wg_dh=w_gate_t.stride(1),
        stride_wg_de=w_gate_t.stride(2),
        stride_wu_ne=w_up_t.stride(0),
        stride_wu_dh=w_up_t.stride(1),
        stride_wu_de=w_up_t.stride(2),
        stride_wd_ne=w_down_t.stride(0),
        stride_wd_de=w_down_t.stride(1),
        stride_wd_dh=w_down_t.stride(2),
        stride_y_s=y.stride(0),
        stride_y_h=y.stride(1),
        stride_y_dh=y.stride(2),
        D_HEAD=d_head,
        D_EXPERT=d_expert,
        NUM_HEADS=num_heads,
        NUM_EXPERTS=num_flat_experts,
        LOG2_NUM_EXPERTS=log2_experts,
        BLOCK_T=block_t,
        BLOCK_DH=block_dh,
        BLOCK_DE=block_de,
        HAS_EXPERT_SCALES=has_scales,
        num_stages=2,
        num_warps=4,
    )
    return y.to(x_heads.dtype)


def _launch_triton_single_proj(
    sorted_inp: torch.Tensor,
    w_t: torch.Tensor,
    expert_offsets: torch.Tensor,
    cu_tiles: torch.Tensor,
    grid_t: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    total_assign, k_dim = sorted_inp.shape
    num_flat_experts, _, n_dim = w_t.shape
    log2_experts = max(1, math.ceil(math.log2(max(num_flat_experts, 1) + 1)))
    block_t, block_k, block_n = 128, 64, 64
    out = torch.empty((total_assign, n_dim), device=sorted_inp.device, dtype=out_dtype)
    grid = (grid_t, (n_dim + block_n - 1) // block_n)
    _mh_moe_proj_kernel[grid](
        inp_ptr=sorted_inp,
        w_ptr=w_t,
        out_ptr=out,
        expert_offsets_ptr=expert_offsets,
        cu_tiles_ptr=cu_tiles,
        stride_in_t=sorted_inp.stride(0),
        stride_in_k=sorted_inp.stride(1),
        stride_w_ne=w_t.stride(0),
        stride_w_k=w_t.stride(1),
        stride_w_n=w_t.stride(2),
        stride_out_t=out.stride(0),
        stride_out_n=out.stride(1),
        K_DIM=k_dim,
        N_DIM=n_dim,
        NUM_EXPERTS=num_flat_experts,
        LOG2_NUM_EXPERTS=log2_experts,
        BLOCK_T=block_t,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        num_stages=2,
        num_warps=4,
    )
    return out


def _dequantize_expert_slice(
    bank,
    weight: QuantizedTensor,
    first: int,
    last: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    count = last - first
    out_f, in_f = bank.out_features, bank.in_features
    if weight._qdata.ndim == 2:
        return bank._bank_rows(weight, first, last).dequantize().to(dtype).view(count, out_f, in_f)

    params = weight._params
    # Fast batched NVFP4 dequant when expert dimensions align with cuBLAS 128x64 tiles
    if (
        hasattr(params, "block_scale")
        and out_f % 128 == 0
        and in_f % 64 == 0
    ):
        q_slice = weight._qdata[first:last].reshape(count * out_f, in_f // 2)
        bs_slice = params.block_scale[first:last].reshape(count * out_f, in_f // 16)
        one = torch.ones((), device=q_slice.device, dtype=torch.float32)
        deq = comfy_kitchen.dequantize_nvfp4(q_slice, one, bs_slice, dtype).view(count, out_f, in_f)
        s2 = params.scale[first:last] if params.scale.ndim > 0 else params.scale
        return (deq * s2.view(-1, 1, 1).to(dtype)).to(dtype)

    out = torch.empty((count, out_f, in_f), device=weight.device, dtype=dtype)
    for idx in range(count):
        out[idx] = bank._expert_qt_from(weight, first + idx).dequantize().to(dtype)
    return out


class CoreMultiHeadMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        num_heads: int = 12,
        num_experts: int = 256,
        top_k: int = 6,
        expert_intermediate_size: int = 1280,
        score_func: str = "sigmoid",
        route_norm: bool = True,
        route_scale: float = 4.9,
        dtype=None,
        device=None,
        moe_ops=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_head = hidden_size // num_heads
        self.d_expert = expert_intermediate_size
        self.flatten_num_experts = num_heads * num_experts
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

        self.gate = nn.Parameter(
            torch.empty(self.flatten_num_experts, self.d_head, dtype=torch.float32, device=device)
        )
        self.W_gate = moe_ops.MoEExperts(
            self.flatten_num_experts, self.d_head, self.d_expert, bias=False, device=device, dtype=dtype
        )
        self.W_up = moe_ops.MoEExperts(
            self.flatten_num_experts, self.d_head, self.d_expert, bias=False, device=device, dtype=dtype
        )
        self.W_down = moe_ops.MoEExperts(
            self.flatten_num_experts, self.d_expert, self.d_head, bias=False, device=device, dtype=dtype
        )
        self.router = nn.Module()
        self.router.register_buffer(
            "expert_bias",
            torch.empty(self.flatten_num_experts, dtype=torch.float32, device=device),
        )

    def _route(self, x_heads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = dinkster_inference.model_management.cast_to(
            self.gate, dtype=torch.float32, device=x_heads.device
        ).view(self.num_heads, self.num_experts, self.d_head)
        bias = dinkster_inference.model_management.cast_to(
            self.router.expert_bias, dtype=torch.float32, device=x_heads.device
        ).view(self.num_heads, self.num_experts)
        logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
        return _compute_topk_routing(
            logits,
            top_k=self.top_k,
            expert_bias=bias,
            score_func=self.score_func,
            route_norm=self.route_norm,
            route_scale=self.route_scale,
        )

    def _forward_eager(
        self,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
        wg_bank,
        wu_bank,
        wd_bank,
    ) -> torch.Tensor:
        gather_ids, probs_sorted, expert_offsets = _global_sort_assignments(
            topk_probs, topk_indices, self.num_experts
        )
        counts = expert_offsets.diff().cpu().tolist()
        out = torch.zeros_like(x_heads, dtype=torch.float32)

        start = 0
        for expert_idx, n in enumerate(counts):
            if n == 0:
                continue
            head_idx = expert_idx // self.num_experts
            tok_idx = gather_ids[start : start + n].long()
            x_tok = x_heads.index_select(0, tok_idx)[:, head_idx, :]
            g_proj = wg_bank.expert_linear(x_tok, expert_idx).float()
            u_proj = wu_bank.expert_linear(x_tok, expert_idx).float()
            # Interleave gate and up to reuse exact swiglu7 math
            interleaved = torch.stack((g_proj, u_proj), dim=-1).flatten(-2)
            hidden = swiglu7(interleaved, out_dtype=x_tok.dtype)
            d_proj = wd_bank.expert_linear(hidden, expert_idx).float()
            weighted = d_proj * probs_sorted[start : start + n].unsqueeze(-1)
            out[:, head_idx, :].index_add_(0, tok_idx, weighted)
            start += n
        return out.to(x_heads.dtype)

    def _forward_sequential_banks_triton(
        self,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Low-VRAM BF16 MoE path: pins only ONE 1.875 GiB expert bank in VBAR at a time."""
        gather_ids, probs_sorted, expert_offsets = _global_sort_assignments(
            topk_probs, topk_indices, self.num_experts
        )
        total_assign = gather_ids.shape[0]
        if total_assign == 0:
            return torch.zeros_like(x_heads)

        counts = expert_offsets.diff()
        head_ids = torch.repeat_interleave(
            torch.arange(self.flatten_num_experts, device=x_heads.device) // self.num_experts,
            counts,
        )
        tok_long = gather_ids.long()
        sorted_x = x_heads[tok_long, head_ids, :].contiguous()

        block_t = 128
        tile_counts = (counts + block_t - 1) // block_t
        cu_tiles = F.pad(torch.cumsum(tile_counts, dim=0, dtype=torch.int32), (1, 0))
        grid_t = (total_assign + block_t - 1) // block_t + self.flatten_num_experts

        with self.W_gate.bank_resident(x_heads) as wg_bank:
            wg_t = wg_bank._resident_bank[0].to(x_heads.dtype).transpose(1, 2)
            gate_out = _launch_triton_single_proj(
                sorted_x, wg_t, expert_offsets, cu_tiles, grid_t, torch.float32
            )

        with self.W_up.bank_resident(x_heads) as wu_bank:
            wu_t = wu_bank._resident_bank[0].to(x_heads.dtype).transpose(1, 2)
            up_out = _launch_triton_single_proj(
                sorted_x, wu_t, expert_offsets, cu_tiles, grid_t, torch.float32
            )

        del sorted_x
        gate_c = gate_out.clamp(max=7.0)
        up_c = up_out.clamp(min=-7.0, max=7.0)
        hidden = (gate_c * torch.sigmoid(1.702 * gate_c) * (up_c + 1.0)).to(x_heads.dtype).contiguous()
        del gate_out, up_out, gate_c, up_c

        with self.W_down.bank_resident(x_heads) as wd_bank:
            wd_t = wd_bank._resident_bank[0].to(x_heads.dtype).transpose(1, 2)
            down_out = _launch_triton_single_proj(
                hidden, wd_t, expert_offsets, cu_tiles, grid_t, torch.float32
            )

        del hidden
        weighted = down_out * probs_sorted.unsqueeze(-1)
        flat_out = torch.zeros((x_heads.shape[0] * self.num_heads, self.d_head), device=x_heads.device, dtype=torch.float32)
        scatter_idx = (tok_long * self.num_heads + head_ids).unsqueeze(-1).expand_as(weighted)
        flat_out.scatter_add_(0, scatter_idx, weighted)
        return flat_out.view_as(x_heads).to(x_heads.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[0]
        if seq_len == 0:
            return x
        x_heads = x.view(seq_len, self.num_heads, self.d_head).contiguous()
        topk_probs, topk_indices = self._route(x_heads)

        use_triton = (
            _TRITON_AVAILABLE
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and self.d_head % 64 == 0
            and self.d_expert % 64 == 0
        )

        # When unquantized BF16 MoE is dynamically VBAR-paged inside tight VRAM (<14 GiB free),
        # stream W_gate -> W_up -> W_down one bank at a time (1.88 GiB peak instead of 5.63 GiB).
        raw_wg = self.W_gate.weight
        if (
            use_triton
            and raw_wg is not None
            and not isinstance(raw_wg, QuantizedTensor)
            and getattr(self.W_gate, "_v", None) is not None
        ):
            bank_bytes = self.flatten_num_experts * self.d_head * self.d_expert * 2
            free_bytes = dinkster_inference.model_management.get_free_memory(x.device)
            if 3 * bank_bytes > int(free_bytes * 0.45):
                out = self._forward_sequential_banks_triton(x_heads, topk_probs, topk_indices)
                return out.reshape(seq_len, self.num_heads * self.d_head)

        with (
            self.W_gate.bank_resident(x) as wg_bank,
            self.W_up.bank_resident(x) as wu_bank,
            self.W_down.bank_resident(x) as wd_bank,
        ):
            wg_w = wg_bank._resident_bank[0]
            wu_w = wu_bank._resident_bank[0]
            wd_w = wd_bank._resident_bank[0]

            use_triton = (
                _TRITON_AVAILABLE
                and x.is_cuda
                and x.dtype in (torch.bfloat16, torch.float16)
                and self.d_head % 64 == 0
                and self.d_expert % 32 == 0
            )
            if not use_triton:
                out = self._forward_eager(x_heads, topk_probs, topk_indices, wg_bank, wu_bank, wd_bank)
                return out.reshape(seq_len, self.num_heads * self.d_head)

            # Case 1: unquantized resident 3D bank [3072, out, in] -> single Triton launch
            if not isinstance(wg_w, QuantizedTensor):
                gather_ids, probs_sorted, expert_offsets = _global_sort_assignments(
                    topk_probs, topk_indices, self.num_experts
                )
                out = _launch_triton_mh_moe(
                    x_heads,
                    gather_ids,
                    probs_sorted,
                    expert_offsets,
                    wg_w.to(x.dtype).transpose(1, 2),
                    wu_w.to(x.dtype).transpose(1, 2),
                    wd_w.to(x.dtype).transpose(1, 2),
                )
                return out.reshape(seq_len, self.num_heads * self.d_head)

            # Case 2: FP8 3D bank [3072, out, in] -> single zero-dequant Triton launch
            if (
                wg_w._qdata.dtype == torch.float8_e4m3fn
                and wg_w._qdata.ndim == 3
                and wu_w._qdata.ndim == 3
                and wd_w._qdata.ndim == 3
            ):
                gather_ids, probs_sorted, expert_offsets = _global_sort_assignments(
                    topk_probs, topk_indices, self.num_experts
                )
                n_exp = self.flatten_num_experts
                s_wg = wg_w._params.scale.expand(n_exp).contiguous()
                s_wu = wu_w._params.scale.expand(n_exp).contiguous()
                s_wd = wd_w._params.scale.expand(n_exp).contiguous()
                out = _launch_triton_mh_moe(
                    x_heads,
                    gather_ids,
                    probs_sorted,
                    expert_offsets,
                    wg_w._qdata.transpose(1, 2),
                    wu_w._qdata.transpose(1, 2),
                    wd_w._qdata.transpose(1, 2),
                    wg_scale=s_wg,
                    wu_scale=s_wu,
                    wd_scale=s_wd,
                )
                return out.reshape(seq_len, self.num_heads * self.d_head)

            # Case 3: NVFP4 / INT8 / other QuantizedTensor -> 12 independent per-head batches
            # (<480 MiB transient dequant per head, reusing L2 cache)
            out_heads = torch.empty_like(x_heads)
            for h in range(self.num_heads):
                first = h * self.num_experts
                last = first + self.num_experts
                gather_h, probs_h, offsets_h = _global_sort_assignments(
                    topk_probs[h : h + 1], topk_indices[h : h + 1], self.num_experts
                )
                wg_h = _dequantize_expert_slice(wg_bank, wg_w, first, last, x.dtype).transpose(1, 2)
                wu_h = _dequantize_expert_slice(wu_bank, wu_w, first, last, x.dtype).transpose(1, 2)
                wd_h = _dequantize_expert_slice(wd_bank, wd_w, first, last, x.dtype).transpose(1, 2)
                out_heads[:, h : h + 1, :] = _launch_triton_mh_moe(
                    x_heads[:, h : h + 1, :],
                    gather_h,
                    probs_h,
                    offsets_h,
                    wg_h,
                    wu_h,
                    wd_h,
                )
            return out_heads.reshape(seq_len, self.num_heads * self.d_head)


class MultiHeadMoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        num_modality: int = 3,
        num_heads: int = 12,
        num_experts: int = 256,
        top_k: int = 6,
        expert_intermediate_size: int = 1280,
        shared_expert_intermediate_size: int = 1280,
        modality_specific_expert_intermediate_size: int = 1280,
        score_func: str = "sigmoid",
        route_norm: bool = True,
        route_scale: float = 4.9,
        dtype=None,
        device=None,
        operations=None,
        moe_ops=None,
    ) -> None:
        super().__init__()
        self.shared_inter = shared_expert_intermediate_size
        self.modality_inter = modality_specific_expert_intermediate_size

        self.pre_norm = MultiModalityRMSNorm(
            hidden_size, eps=1e-6, num_modality=num_modality, dtype=dtype, device=device
        )
        self.split_linear = operations.Linear(
            hidden_size, hidden_size, bias=False, dtype=dtype, device=device
        )
        self.merge_linear = operations.Linear(
            hidden_size, hidden_size, bias=False, dtype=dtype, device=device
        )
        self.moe_mlp = CoreMultiHeadMoE(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
            expert_intermediate_size=expert_intermediate_size,
            score_func=score_func,
            route_norm=route_norm,
            route_scale=route_scale,
            dtype=dtype,
            device=device,
            moe_ops=moe_ops,
        )
        self.shared_expert_fc1 = operations.Linear(
            hidden_size,
            shared_expert_intermediate_size * 2,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.shared_expert_fc2 = operations.Linear(
            shared_expert_intermediate_size,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.modality_specific_shared_expert_fc1 = make_modality_linear(
            hidden_size,
            modality_specific_expert_intermediate_size * 2,
            num_modality=num_modality,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.modality_specific_shared_expert_fc2 = make_modality_linear(
            modality_specific_expert_intermediate_size,
            hidden_size,
            num_modality=num_modality,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(
        self, hidden_states: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        norm_out = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher)
        moe_out = self.merge_linear(self.moe_mlp(self.split_linear(norm_out)))

        x1 = self.shared_expert_fc1(norm_out)
        x2 = apply_modality_linear(
            self.modality_specific_shared_expert_fc1, norm_out, modality_dispatcher
        )
        act = swiglu7(torch.cat((x1, x2), dim=-1), out_dtype=norm_out.dtype)
        s1, s2 = act.split((self.shared_inter, self.modality_inter), dim=-1)
        shared_out = self.shared_expert_fc2(s1.contiguous()) + apply_modality_linear(
            self.modality_specific_shared_expert_fc2, s2.contiguous(), modality_dispatcher
        )
        return moe_out + shared_out
