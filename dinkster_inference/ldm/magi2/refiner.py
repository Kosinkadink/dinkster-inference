from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import dinkster_inference.ldm.common_dit
import dinkster_inference.model_prefetch
from dinkster_inference.ldm.modules.attention import (
    AttentionTensorContainer,
    ComfyAttention,
    optimized_attention,
)

from .common import (
    ElementWiseFourierEmbed,
    Modality,
    ModalityDispatcher,
    MultiModalityRMSNorm,
    apply_magi2_rope,
    apply_modality_linear,
    get_coords,
    make_modality_linear,
    swiglu7,
)

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    _FLEX_COMPILED = torch.compile(flex_attention)
    _FLEX_AVAILABLE = True
except Exception:
    BlockMask = None
    create_block_mask = None
    flex_attention = None
    _FLEX_COMPILED = None
    _FLEX_AVAILABLE = False


def _build_refiner_block_order(
    patch_nums: tuple[int, int, int],
    block_sizes: tuple[int, int, int],
    total_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    t_num, h_num, w_num = patch_nums
    bs_t, bs_h, bs_w = block_sizes
    video_tokens = t_num * h_num * w_num

    nt = math.ceil(t_num / bs_t)
    nh = math.ceil(h_num / bs_h)
    nw = math.ceil(w_num / bs_w)
    padded_nums = (nt * bs_t, nh * bs_h, nw * bs_w)

    grid = torch.full(padded_nums, -1, dtype=torch.int64, device=device)
    grid[:t_num, :h_num, :w_num] = torch.arange(video_tokens, device=device, dtype=torch.int64).view(
        t_num, h_num, w_num
    )
    flat_blocks = (
        grid.view(nt, bs_t, nh, bs_h, nw, bs_w)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1)
    )
    video_order = flat_blocks[flat_blocks >= 0]
    tail_order = torch.arange(video_tokens, total_tokens, device=device, dtype=torch.int64)
    token_order = torch.cat((video_order, tail_order), dim=0)

    restore_order = torch.empty_like(token_order)
    restore_order[token_order] = torch.arange(token_order.numel(), device=device, dtype=torch.int64)

    t_sizes = torch.full((nt,), bs_t, device=device, dtype=torch.int64)
    t_sizes[-1] = t_num - bs_t * (nt - 1)
    h_sizes = torch.full((nh,), bs_h, device=device, dtype=torch.int64)
    h_sizes[-1] = h_num - bs_h * (nh - 1)
    w_sizes = torch.full((nw,), bs_w, device=device, dtype=torch.int64)
    w_sizes[-1] = w_num - bs_w * (nw - 1)
    block_counts = (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)
    ends = torch.cumsum(block_counts, dim=0)
    starts = ends - block_counts
    block_ranges = torch.stack((starts, ends), dim=1)
    return token_order, restore_order, block_ranges, (nt, nh, nw)


def _prepare_flex_cube_plan(
    patch_nums: tuple[int, int, int],
    block_sizes: tuple[int, int, int],
    radius: tuple[int, int, int],
    tail_len: int,
    device: torch.device,
) -> dict:
    """Build 128-token-aligned cube scatter/gather indices and a flex_attention BlockMask."""
    t_num, h_num, w_num = patch_nums
    bs_t, bs_h, bs_w = block_sizes
    cube_vol = bs_t * bs_h * bs_w  # 8 * 4 * 4 = 128
    nt = math.ceil(t_num / bs_t)
    nh = math.ceil(h_num / bs_h)
    nw = math.ceil(w_num / bs_w)
    num_cubes = nt * nh * nw
    padded_vid = num_cubes * cube_vol
    padded_tail = math.ceil(max(tail_len, 1) / cube_vol) * cube_vol if tail_len > 0 else 0
    padded_total = padded_vid + padded_tail

    # Map valid block-ordered video tokens + tail tokens into 128-aligned cube slots
    grid = torch.full((nt * bs_t, nh * bs_h, nw * bs_w), -1, dtype=torch.int64, device=device)
    grid[:t_num, :h_num, :w_num] = 1
    valid_cube_mask = (
        grid.view(nt, bs_t, nh, bs_h, nw, bs_w)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1)
        >= 0
    )
    vid_slots = torch.nonzero(valid_cube_mask, as_tuple=False).squeeze(1)
    tail_slots = torch.arange(padded_vid, padded_vid + tail_len, device=device, dtype=torch.int64)
    scatter_slots = torch.cat((vid_slots, tail_slots), dim=0)

    valid_slot_mask = torch.zeros((padded_total,), dtype=torch.bool, device=device)
    valid_slot_mask[scatter_slots] = True

    rt, rh, rw = radius

    def mask_mod(b, h, q_idx, kv_idx):
        valid = valid_slot_mask[q_idx] & valid_slot_mask[kv_idx]
        q_is_tail = q_idx >= padded_vid
        kv_is_tail = kv_idx >= padded_vid
        q_cube = q_idx // cube_vol
        k_cube = kv_idx // cube_vol
        qt = q_cube // (nh * nw)
        qh = (q_cube // nw) % nh
        qw = q_cube % nw
        kt = k_cube // (nh * nw)
        kh = (k_cube // nw) % nh
        kw = k_cube % nw
        in_window = (
            ((qt - kt).abs() <= rt)
            & ((qh - kh).abs() <= rh)
            & ((qw - kw).abs() <= rw)
        )
        return valid & (q_is_tail | kv_is_tail | in_window)

    num_tail_blocks = padded_tail // cube_vol
    num_blocks = num_cubes + num_tail_blocks
    block_valid_grid = valid_slot_mask.view(num_blocks, cube_vol)
    block_any_valid = block_valid_grid.any(dim=1)
    block_all_valid = block_valid_grid.all(dim=1)

    blk_idx = torch.arange(num_blocks, device=device, dtype=torch.int64)
    is_tail_blk = blk_idx >= num_cubes
    blk_t = (blk_idx // (nh * nw)).clamp(max=max(nt - 1, 0))
    blk_h = ((blk_idx // nw) % nh).clamp(max=max(nh - 1, 0))
    blk_w = (blk_idx % nw).clamp(max=max(nw - 1, 0))

    in_win_blk = (
        is_tail_blk[:, None]
        | is_tail_blk[None, :]
        | (
            ((blk_t[:, None] - blk_t[None, :]).abs() <= rt)
            & ((blk_h[:, None] - blk_h[None, :]).abs() <= rh)
            & ((blk_w[:, None] - blk_w[None, :]).abs() <= rw)
        )
    )
    active_blk = in_win_blk & block_any_valid[:, None] & block_any_valid[None, :]
    full_blk = active_blk & block_all_valid[:, None] & block_all_valid[None, :]
    partial_blk = active_blk & ~full_blk

    def _pack_bool_blocks(mask_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = mask_2d.sum(dim=-1, dtype=torch.int32)
        max_k = max(int(counts.max().item()), 1)
        # argsort(descending) packs True block indices into the leading slots per row
        sorted_idx = mask_2d.to(torch.int8).argsort(dim=-1, descending=True, stable=True)[:, :max_k].to(torch.int32)
        return counts.view(1, 1, num_blocks).contiguous(), sorted_idx.view(1, 1, num_blocks, max_k).contiguous()

    kv_num_blocks, kv_indices = _pack_bool_blocks(partial_blk)
    full_kv_num_blocks, full_kv_indices = _pack_bool_blocks(full_blk)

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=cube_vol,
        mask_mod=mask_mod,
        seq_lengths=(padded_total, padded_total),
    )
    return {
        "padded_total": padded_total,
        "scatter_slots": scatter_slots,
        "block_mask": block_mask,
    }


def _window_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ranges: torch.Tensor,
    grid_shape: tuple[int, int, int],
    radius: tuple[int, int, int],
    video_tokens: int,
) -> torch.Tensor:
    """Exact portable 3D block-grid window attention [1, H_q, L, D] -> [1, L, H_q, D]."""
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    enable_gqa = num_heads_q != num_heads_kv
    nt, nh, nw = grid_shape
    rt, rh, rw = radius
    ranges_cpu = block_ranges.cpu().tolist()
    total_len = q.shape[2]
    out = torch.empty_like(q)

    k_tail = k[:, :, video_tokens:total_len, :]
    v_tail = v[:, :, video_tokens:total_len, :]

    for bt in range(nt):
        t0, t1 = max(0, bt - rt), min(nt, bt + rt + 1)
        for bh in range(nh):
            h0, h1 = max(0, bh - rh), min(nh, bh + rh + 1)
            for bw in range(nw):
                q_id = bt * (nh * nw) + bh * nw + bw
                qs, qe = ranges_cpu[q_id]
                if qe <= qs:
                    continue
                w0, w1 = max(0, bw - rw), min(nw, bw + rw + 1)
                k_parts = []
                v_parts = []
                for kt in range(t0, t1):
                    for kh in range(h0, h1):
                        # Contiguous w-run [w0, w1) spans contiguous token indices!
                        b_start = kt * (nh * nw) + kh * nw + w0
                        b_end = kt * (nh * nw) + kh * nw + (w1 - 1)
                        ks = ranges_cpu[b_start][0]
                        ke = ranges_cpu[b_end][1]
                        if ke > ks:
                            k_parts.append(k[:, :, ks:ke, :])
                            v_parts.append(v[:, :, ks:ke, :])
                if k_tail.shape[2] > 0:
                    k_parts.append(k_tail)
                    v_parts.append(v_tail)
                k_cat = torch.cat(k_parts, dim=2)
                v_cat = torch.cat(v_parts, dim=2)
                out[:, :, qs:qe, :] = F.scaled_dot_product_attention(
                    q[:, :, qs:qe, :], k_cat, v_cat, enable_gqa=enable_gqa
                )

    if total_len > video_tokens:
        out[:, :, video_tokens:total_len, :] = F.scaled_dot_product_attention(
            q[:, :, video_tokens:total_len, :], k, v, enable_gqa=enable_gqa
        )

    return out.transpose(1, 2).contiguous()


class RefinerAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads_q: int = 32,
        num_heads_kv: int = 8,
        head_dim: int = 128,
        num_modality: int = 1,
        use_local_attn: bool = True,
        enable_attn_gating: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.q_size = num_heads_q * head_dim
        self.kv_size = num_heads_kv * head_dim
        self.use_local_attn = use_local_attn
        self.enable_attn_gating = enable_attn_gating

        self.pre_norm = MultiModalityRMSNorm(
            hidden_size, eps=1e-6, num_modality=num_modality, dtype=dtype, device=device
        )
        self.linear_qkv = make_modality_linear(
            hidden_size,
            self.q_size + 2 * self.kv_size,
            num_modality=num_modality,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        if enable_attn_gating:
            self.linear_g = make_modality_linear(
                hidden_size,
                num_heads_q,
                num_modality=num_modality,
                bias=False,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        else:
            self.linear_g = None
        self.linear_proj = make_modality_linear(
            self.q_size,
            hidden_size,
            num_modality=num_modality,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.q_norm = MultiModalityRMSNorm(
            head_dim, eps=1e-6, num_modality=num_modality, dtype=dtype, device=device
        )
        self.k_norm = MultiModalityRMSNorm(
            head_dim, eps=1e-6, num_modality=num_modality, dtype=dtype, device=device
        )
        self.comfy_attention = ComfyAttention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        attn_plans: list[dict],
        modality_dispatcher: ModalityDispatcher,
        transformer_options: dict,
    ) -> torch.Tensor:
        compute_dtype = hidden_states.dtype
        x = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher).to(compute_dtype)
        g = None
        if self.linear_g is not None:
            g = apply_modality_linear(self.linear_g, x, modality_dispatcher).float().unsqueeze(-1)
        qkv = apply_modality_linear(self.linear_qkv, x, modality_dispatcher).float()
        del x
        q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
        q_normed = self.q_norm(
            q.view(-1, self.num_heads_q, self.head_dim), modality_dispatcher=modality_dispatcher
        )
        k_normed = self.k_norm(
            k.view(-1, self.num_heads_kv, self.head_dim), modality_dispatcher=modality_dispatcher
        )
        v_unperm = modality_dispatcher.inv_permute(
            v.view(-1, self.num_heads_kv, self.head_dim).to(compute_dtype)
        )
        del qkv, q, k, v

        q_unperm = modality_dispatcher.inv_permute(q_normed).unsqueeze(0)
        del q_normed
        k_unperm = modality_dispatcher.inv_permute(k_normed).unsqueeze(0)
        del k_normed

        q_rot, k_rot = apply_magi2_rope(q_unperm, k_unperm, rope)
        del q_unperm, k_unperm
        q_rot = q_rot.squeeze(0).to(compute_dtype)
        k_rot = k_rot.squeeze(0).to(compute_dtype)

        seq_lens = [p["seq_len"] for p in attn_plans]
        q_parts = torch.split(q_rot, seq_lens, dim=0)
        k_parts = torch.split(k_rot, seq_lens, dim=0)
        v_parts = torch.split(v_unperm, seq_lens, dim=0)
        enable_gqa = self.num_heads_q != self.num_heads_kv

        out_parts = []
        for qb, kb, vb, plan in zip(q_parts, k_parts, v_parts, attn_plans):
            q_bhld = qb.transpose(0, 1).unsqueeze(0)
            k_bhld = kb.transpose(0, 1).unsqueeze(0)
            v_bhld = vb.transpose(0, 1).unsqueeze(0)
            if not self.use_local_attn or plan.get("dense", False):
                out_b = optimized_attention(
                    AttentionTensorContainer(q_bhld),
                    AttentionTensorContainer(k_bhld),
                    AttentionTensorContainer(v_bhld),
                    self.num_heads_q,
                    skip_reshape=True,
                    enable_gqa=enable_gqa,
                    transformer_options=transformer_options,
                    preferred_attention=self.comfy_attention,
                ).reshape(1, qb.shape[0], self.num_heads_q, self.head_dim)
            elif plan.get("flex_plan") is not None:
                fp = plan["flex_plan"]
                pad_n = fp["padded_total"]
                slots = fp["scatter_slots"]
                q_pad = q_bhld.new_zeros((1, self.num_heads_q, pad_n, self.head_dim))
                k_pad = k_bhld.new_zeros((1, self.num_heads_kv, pad_n, self.head_dim))
                v_pad = v_bhld.new_zeros((1, self.num_heads_kv, pad_n, self.head_dim))
                q_pad[:, :, slots, :] = q_bhld
                k_pad[:, :, slots, :] = k_bhld
                v_pad[:, :, slots, :] = v_bhld
                try:
                    out_pad = _FLEX_COMPILED(
                        q_pad, k_pad, v_pad, block_mask=fp["block_mask"], enable_gqa=enable_gqa
                    )
                    out_b = out_pad[:, :, slots, :].transpose(1, 2).contiguous()
                    del out_pad
                except Exception:
                    out_b = _window_attention_eager(
                        q_bhld, k_bhld, v_bhld,
                        plan["block_ranges"], plan["grid_shape"], plan["radius"], plan["video_tokens"]
                    )
                del q_pad, k_pad, v_pad
            else:
                out_b = _window_attention_eager(
                    q_bhld, k_bhld, v_bhld,
                    plan["block_ranges"], plan["grid_shape"], plan["radius"], plan["video_tokens"]
                )
            del q_bhld, k_bhld, v_bhld
            out_parts.append(out_b.squeeze(0))
            del out_b

        del q_parts, k_parts, v_parts, q_rot, k_rot, v_unperm
        attn_out = modality_dispatcher.permute(torch.cat(out_parts, dim=0))
        del out_parts
        if g is not None:
            attn_out = attn_out * torch.sigmoid(g).to(attn_out.dtype)
        attn_out = attn_out.reshape(-1, self.q_size).to(compute_dtype)
        return apply_modality_linear(self.linear_proj, attn_out, modality_dispatcher)


class RefinerMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 10920,
        num_modality: int = 1,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(
            hidden_size, eps=1e-6, num_modality=num_modality, dtype=dtype, device=device
        )
        self.up_gate_proj = make_modality_linear(
            hidden_size,
            intermediate_size * 2,
            num_modality=num_modality,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.down_proj = make_modality_linear(
            intermediate_size,
            hidden_size,
            num_modality=num_modality,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        compute_dtype = x.dtype
        h_norm = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(compute_dtype)
        chunk_tokens = 32768
        if h_norm.shape[0] <= chunk_tokens:
            h = apply_modality_linear(self.up_gate_proj, h_norm, modality_dispatcher)
            h = swiglu7(h, out_dtype=compute_dtype)
            return apply_modality_linear(self.down_proj, h, modality_dispatcher)

        out = torch.empty_like(h_norm)
        offset = 0
        for mod_idx, g_size in enumerate(modality_dispatcher.group_sizes):
            if g_size == 0:
                continue
            up_mod = (
                self.up_gate_proj.linears[mod_idx]
                if hasattr(self.up_gate_proj, "linears")
                else self.up_gate_proj
            )
            down_mod = (
                self.down_proj.linears[mod_idx]
                if hasattr(self.down_proj, "linears")
                else self.down_proj
            )
            for start in range(0, g_size, chunk_tokens):
                end = min(g_size, start + chunk_tokens)
                chunk = h_norm[offset + start : offset + end]
                out[offset + start : offset + end] = down_mod(
                    swiglu7(up_mod(chunk), out_dtype=compute_dtype)
                )
            offset += g_size
        return out


class RefinerTransformerLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = 4096,
        num_heads_q: int = 32,
        num_heads_kv: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 10920,
        mm_layers: tuple[int, ...] = (0, 1, 28, 29),
        local_attn_layers: tuple[int, ...] = tuple(range(30)),
        post_norm_layers: tuple[int, ...] = (),
        enable_attn_gating: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        num_modality = 3 if layer_idx in mm_layers else 1
        use_local_attn = layer_idx in local_attn_layers
        self.has_post_norm = layer_idx in post_norm_layers

        self.attention = RefinerAttention(
            hidden_size=hidden_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            num_modality=num_modality,
            use_local_attn=use_local_attn,
            enable_attn_gating=enable_attn_gating,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.mlp = RefinerMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_modality=num_modality,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        if self.has_post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(
                hidden_size, eps=1e-6, num_modality=num_modality, dtype=dtype, device=device
            )
            self.mlp_post_norm = MultiModalityRMSNorm(
                hidden_size, eps=1e-6, num_modality=num_modality, dtype=dtype, device=device
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        attn_plans: list[dict],
        modality_dispatcher: ModalityDispatcher,
        transformer_options: dict,
    ) -> torch.Tensor:
        attn_out = self.attention(
            hidden_states, rope, attn_plans, modality_dispatcher, transformer_options
        )
        if self.has_post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + attn_out.to(hidden_states.dtype)

        mlp_out = self.mlp(hidden_states, modality_dispatcher)
        if self.has_post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        return hidden_states + mlp_out.to(hidden_states.dtype)


class RefinerPreAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        text_in_channels: int = 5120,
        video_in_channels: int = 48,
        audio_in_channels: int = 64,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.video_embedder = operations.Linear(
            video_in_channels, hidden_size, bias=True, dtype=torch.float32, device=device
        )
        self.text_embedder = operations.Linear(
            text_in_channels, hidden_size, bias=True, dtype=torch.float32, device=device
        )
        self.audio_embedder = operations.Linear(
            audio_in_channels, hidden_size, bias=True, dtype=torch.float32, device=device
        )
        self.rope = ElementWiseFourierEmbed(
            hidden_size // num_attention_heads, device=device, dtype=torch.float32
        )

    def forward(
        self,
        video_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        video_idx: torch.Tensor,
        audio_idx: torch.Tensor,
        text_idx: torch.Tensor,
        coords_mapping: torch.Tensor,
        total_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.rope(coords_mapping)
        out = torch.zeros((total_tokens, self.hidden_size), device=coords_mapping.device, dtype=torch.float32)
        if text_idx.numel() > 0:
            out.index_copy_(0, text_idx, self.text_embedder(text_tokens.float()).float())
        if audio_idx.numel() > 0:
            out.index_copy_(0, audio_idx, self.audio_embedder(audio_tokens.float()).float())
        if video_idx.numel() > 0:
            out.index_copy_(0, video_idx, self.video_embedder(video_tokens.float()).float())
        return out, rope


class RefinerPostAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        video_in_channels: int = 48,
        audio_in_channels: int = 64,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.final_norm_video = MultiModalityRMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.final_norm_audio = MultiModalityRMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.final_linear_video = operations.Linear(
            hidden_size, video_in_channels, bias=False, dtype=torch.float32, device=device
        )
        self.final_linear_audio = operations.Linear(
            hidden_size, audio_in_channels, bias=False, dtype=torch.float32, device=device
        )

    def forward(
        self, x: torch.Tensor, video_idx: torch.Tensor, audio_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_idx.numel() > 0:
            v_out = self.final_linear_video(self.final_norm_video(x.index_select(0, video_idx).float())).float()
        else:
            v_out = x.new_empty((0, self.video_in_channels), dtype=torch.float32)
        if audio_idx.numel() > 0:
            a_out = self.final_linear_audio(self.final_norm_audio(x.index_select(0, audio_idx).float())).float()
        else:
            a_out = x.new_empty((0, self.audio_in_channels), dtype=torch.float32)
        return v_out, a_out


class RefinerTransformerBlock(nn.Module):
    def __init__(self, layers: list[RefinerTransformerLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class Magi2RefinerModel(nn.Module):
    """MAGI-2 6.7B dense 1080p super-resolution DiT with 3D block-grid window attention."""

    def __init__(
        self,
        image_model: str = "magi2_refiner",
        num_layers: int = 30,
        hidden_size: int = 4096,
        head_dim: int = 128,
        num_heads_q: int = 32,
        num_heads_kv: int = 8,
        video_in_channels: int = 48,
        audio_in_channels: int = 64,
        text_in_channels: int = 5120,
        intermediate_size: Optional[int] = None,
        mm_layers: tuple[int, ...] = (0, 1, 28, 29),
        local_attn_layers: Optional[tuple[int, ...]] = None,
        post_norm_layers: tuple[int, ...] = (),
        enable_attn_gating: bool = True,
        patch_size: tuple[int, int, int] = (1, 1, 1),
        block_sizes: tuple[int, int, int] = (8, 4, 4),
        block_radius: tuple[int, int, int] = (2, 2, 2),
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.text_in_channels = text_in_channels
        self.patch_size = tuple(patch_size)
        self.block_sizes = tuple(block_sizes)
        self.block_radius = tuple(block_radius)

        if intermediate_size is None:
            intermediate_size = int(hidden_size * 4 * 2 / 3) // 4 * 4
        if local_attn_layers is None:
            local_attn_layers = tuple(range(num_layers))

        self.pre_adapter = RefinerPreAdapter(
            hidden_size=hidden_size,
            num_attention_heads=num_heads_q,
            text_in_channels=text_in_channels,
            video_in_channels=video_in_channels * math.prod(self.patch_size),
            audio_in_channels=audio_in_channels,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.block = RefinerTransformerBlock(
            [
                RefinerTransformerLayer(
                    layer_idx=i,
                    hidden_size=hidden_size,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    mm_layers=tuple(mm_layers),
                    local_attn_layers=tuple(local_attn_layers),
                    post_norm_layers=tuple(post_norm_layers),
                    enable_attn_gating=enable_attn_gating,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for i in range(num_layers)
            ]
        )
        self.post_adapter = RefinerPostAdapter(
            hidden_size=hidden_size,
            video_in_channels=video_in_channels * math.prod(self.patch_size),
            audio_in_channels=audio_in_channels,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def get_dynamic_vram__units(self) -> list[nn.Module]:
        return list(self.block.layers)

    def _run_layers(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        attn_plans: list[dict],
        modality_dispatcher: ModalityDispatcher,
        transformer_options: dict,
    ) -> torch.Tensor:
        device = x.device
        patches_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        prefetch_queue = dinkster_inference.model_prefetch.make_prefetch_queue(
            list(self.block.layers), device, transformer_options
        )
        for idx, layer in enumerate(self.block.layers):
            dinkster_inference.model_prefetch.prefetch_queue_pop(
                prefetch_queue, device, layer, malloc_scope="block"
            )
            transformer_options["block_index"] = idx
            if ("double_block", idx) in patches_replace:
                def block_wrap(args):
                    return {
                        "img": layer(
                            args["img"],
                            args["rope"],
                            args["attn_plans"],
                            args["modality_dispatcher"],
                            args["transformer_options"],
                        )
                    }

                x = patches_replace[("double_block", idx)](
                    {
                        "img": x,
                        "rope": rope,
                        "attn_plans": attn_plans,
                        "modality_dispatcher": modality_dispatcher,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )["img"]
            else:
                x = layer(x, rope, attn_plans, modality_dispatcher, transformer_options)
        dinkster_inference.model_prefetch.prefetch_queue_pop(
            prefetch_queue, device, None, malloc_scope="block"
        )
        return x

    def forward(
        self,
        x,
        timestep: torch.Tensor,
        context: torch.Tensor,
        transformer_options: dict = {},
        magi2_payload: Optional[dict] = None,
        **kwargs,
    ):
        is_av = not torch.is_tensor(x)
        video_in = x[0] if is_av else x
        audio_in = x[1] if is_av and len(x) > 1 else None

        orig_t, orig_h, orig_w = video_in.shape[2], video_in.shape[3], video_in.shape[4]
        video_padded = dinkster_inference.ldm.common_dit.pad_to_patch_size(video_in, self.patch_size)
        compute_dtype = video_padded.dtype
        device = video_padded.device

        batch_size, _, t_vid, h_vid, w_vid = video_padded.shape
        pt, ph, pw = self.patch_size
        t_steps, h_steps, w_steps = t_vid // pt, h_vid // ph, w_vid // pw
        vid_tok_per_item = t_steps * h_steps * w_steps

        vid_tokens = (
            video_padded.view(batch_size, self.video_in_channels, t_steps, pt, h_steps, ph, w_steps, pw)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(batch_size, vid_tok_per_item, pt * ph * pw * self.video_in_channels)
        )

        # Shipping refiner runs video-only (audio_noise_scale = -1 -> 0 audio tokens) unless explicitly enabled
        payload = magi2_payload or {}
        include_audio = bool(payload.get("refiner_include_audio", False))
        if include_audio and audio_in is not None and audio_in.numel() > 0:
            aud_tokens = audio_in.transpose(1, 2).contiguous()
            aud_len = aud_tokens.shape[1]
        else:
            aud_tokens = video_padded.new_empty((batch_size, 0, self.audio_in_channels))
            aud_len = 0

        txt_lens = payload.get("txt_lens")
        vid_coords = get_coords((t_steps, h_steps, w_steps), (t_steps, h_steps, w_steps), device=device)
        aud_coords = (
            get_coords((aud_len, 1, 1), (max(1, ((aud_len - 1) // 8 + 1) // pt), 1, 1), device=device)
            if aud_len > 0
            else torch.empty((0, 9), device=device, dtype=torch.float32)
        )

        use_flex = (
            _FLEX_AVAILABLE
            and device.type == "cuda"
            and compute_dtype in (torch.bfloat16, torch.float16)
            and vid_tok_per_item > 2048
            and not transformer_options.get("magi2_disable_flex_attn", False)
        )

        video_parts: list[torch.Tensor] = []
        audio_parts: list[torch.Tensor] = []
        text_parts: list[torch.Tensor] = []
        video_idx_parts: list[torch.Tensor] = []
        audio_idx_parts: list[torch.Tensor] = []
        text_idx_parts: list[torch.Tensor] = []
        modality_parts: list[torch.Tensor] = []
        coords_parts: list[torch.Tensor] = []
        restore_orders: list[torch.Tensor] = []
        attn_plans: list[dict] = []

        cursor = 0
        for b in range(batch_size):
            txt_len_b = int(txt_lens[b]) if txt_lens is not None and b < len(txt_lens) else context.shape[1]
            total_b = vid_tok_per_item + aud_len + txt_len_b
            tok_order, restore_order, block_ranges, grid_shape = _build_refiner_block_order(
                (t_steps, h_steps, w_steps), self.block_sizes, total_b, device
            )
            vid_order = tok_order[:vid_tok_per_item]
            restore_orders.append(restore_order[:vid_tok_per_item])

            # Video tokens in 3D cube-contiguous order
            video_parts.append(vid_tokens[b].index_select(0, vid_order))
            video_idx_parts.append(torch.arange(cursor, cursor + vid_tok_per_item, device=device, dtype=torch.long))
            modality_parts.append(torch.full((vid_tok_per_item,), int(Modality.VIDEO), device=device, dtype=torch.int32))
            coords_parts.append(vid_coords.index_select(0, vid_order))
            cursor += vid_tok_per_item

            if aud_len > 0:
                audio_parts.append(aud_tokens[b])
                audio_idx_parts.append(torch.arange(cursor, cursor + aud_len, device=device, dtype=torch.long))
                modality_parts.append(torch.full((aud_len,), int(Modality.AUDIO), device=device, dtype=torch.int32))
                coords_parts.append(aud_coords)
                cursor += aud_len

            if txt_len_b > 0:
                text_parts.append(context[b, :txt_len_b])
                text_idx_parts.append(torch.arange(cursor, cursor + txt_len_b, device=device, dtype=torch.long))
                modality_parts.append(torch.full((txt_len_b,), int(Modality.TEXT), device=device, dtype=torch.int32))
                coords_parts.append(get_coords((txt_len_b, 1, 1), (1, 1, 1), offset_thw=(-txt_len_b, 0, 0), device=device))
                cursor += txt_len_b

            # If the grid fits entirely inside one 5x5x5 window, window attention is exact full attention!
            nt, nh, nw = grid_shape
            rt, rh, rw = self.block_radius
            is_globally_covered = (nt <= 2 * rt + 1) and (nh <= 2 * rh + 1) and (nw <= 2 * rw + 1)
            flex_plan = None
            if not is_globally_covered and use_flex:
                flex_plan = _prepare_flex_cube_plan(
                    (t_steps, h_steps, w_steps), self.block_sizes, self.block_radius, aud_len + txt_len_b, device
                )
            attn_plans.append(
                {
                    "seq_len": total_b,
                    "video_tokens": vid_tok_per_item,
                    "block_ranges": block_ranges,
                    "grid_shape": grid_shape,
                    "radius": self.block_radius,
                    "dense": is_globally_covered,
                    "flex_plan": flex_plan,
                }
            )

        video_cat = torch.cat(video_parts, dim=0)
        audio_cat = torch.cat(audio_parts, dim=0) if audio_parts else video_padded.new_empty((0, self.audio_in_channels))
        text_cat = torch.cat(text_parts, dim=0) if text_parts else context.new_empty((0, self.text_in_channels))
        video_idx = torch.cat(video_idx_parts, dim=0)
        audio_idx = torch.cat(audio_idx_parts, dim=0) if audio_idx_parts else torch.empty((0,), device=device, dtype=torch.long)
        text_idx = torch.cat(text_idx_parts, dim=0) if text_idx_parts else torch.empty((0,), device=device, dtype=torch.long)
        modality_mapping = torch.cat(modality_parts, dim=0)
        coords_mapping = torch.cat(coords_parts, dim=0)

        modality_dispatcher = ModalityDispatcher(modality_mapping, num_modalities=3)
        h, rope = self.pre_adapter(
            video_cat, audio_cat, text_cat, video_idx, audio_idx, text_idx, coords_mapping, cursor
        )

        h = modality_dispatcher.permute(h.to(compute_dtype))
        h = self._run_layers(h, rope, attn_plans, modality_dispatcher, transformer_options)
        h = modality_dispatcher.inv_permute(h)

        v_all, a_all = self.post_adapter(h, video_idx, audio_idx)
        v_items = []
        for b in range(batch_size):
            v_cube = v_all[b * vid_tok_per_item : (b + 1) * vid_tok_per_item]
            v_raster = v_cube.index_select(0, restore_orders[b])
            v_b = (
                v_raster.view(t_steps, h_steps, w_steps, pt, ph, pw, self.video_in_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.video_in_channels, t_steps * pt, h_steps * ph, w_steps * pw)
            )
            v_items.append(v_b[:, :orig_t, :orig_h, :orig_w])
        video_out = torch.stack(v_items, dim=0).to(compute_dtype)

        if not is_av:
            return video_out

        if aud_len > 0:
            a_items = [
                a_all[b * aud_len : (b + 1) * aud_len].transpose(0, 1).contiguous()
                for b in range(batch_size)
            ]
            audio_out = torch.stack(a_items, dim=0).to(audio_in.dtype)
        else:
            audio_out = torch.zeros_like(audio_in)
        return [video_out, audio_out]
