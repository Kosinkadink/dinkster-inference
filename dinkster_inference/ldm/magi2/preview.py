from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import dinkster_inference.ldm.common_dit
import dinkster_inference.model_management
import dinkster_inference.model_prefetch
import dinkster_inference.ops
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
    sinusoidal_embedding_1d,
    swiglu7,
)
from .moe import MultiHeadMoELayer


def _sinkhorn_knopp_fp32(h: torch.Tensor, num_iters: int = 20, eps: float = 1e-12) -> torch.Tensor:
    m = torch.exp(h - h.amax(dim=(-2, -1), keepdim=True))
    for _ in range(num_iters):
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m


def _attention_with_sink_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: Optional[torch.Tensor],
    comfy_attn: ComfyAttention,
    transformer_options: dict,
) -> torch.Tensor:
    """Bidirectional attention for a single sequence [1, H_q, L, D] with per-head sink."""
    seq_len = q.shape[2]
    if seq_len == 0:
        return q
    if sinks is None or sinks.numel() == 0:
        out = optimized_attention(
            AttentionTensorContainer(q),
            AttentionTensorContainer(k),
            AttentionTensorContainer(v),
            q.shape[1],
            skip_reshape=True,
            transformer_options=transformer_options,
            preferred_attention=comfy_attn,
        )
        return out.reshape(q.shape[0], seq_len, q.shape[1], q.shape[3])

    sink_lse = torch.logsumexp(sinks.float(), dim=0)  # [H_q]
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    if num_heads_kv != num_heads_q:
        groups = num_heads_q // num_heads_kv
        k_exp = k.repeat_interleave(groups, dim=1)
        v_exp = v.repeat_interleave(groups, dim=1)
    else:
        k_exp = k
        v_exp = v

    # Fast zero-mask O(L) FlashAttention path on CUDA BF16/FP16
    if (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention")
    ):
        out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q.contiguous(), k_exp.contiguous(), v_exp.contiguous()
        )
        rescale = torch.sigmoid(lse[:, :, :seq_len].float() - sink_lse.view(1, num_heads_q, 1))
        out = out * rescale.unsqueeze(-1).to(out.dtype)
        return out.transpose(1, 2).contiguous()

    # Portable chunked online-safe fallback (CPU / MPS / FP32 / ROCm)
    k_sink = torch.cat((k_exp, k_exp.new_zeros((1, num_heads_q, 1, k_exp.shape[-1]))), dim=2)
    v_sink = torch.cat((v_exp, v_exp.new_zeros((1, num_heads_q, 1, v_exp.shape[-1]))), dim=2)
    chunk_q = 1024
    out_chunks = []
    sink_col = sink_lse.to(q.dtype).view(1, num_heads_q, 1, 1)
    for q_start in range(0, seq_len, chunk_q):
        q_slice = q[:, :, q_start : q_start + chunk_q, :]
        q_len = q_slice.shape[2]
        mask = torch.cat(
            (
                q_slice.new_zeros((1, num_heads_q, q_len, seq_len)),
                sink_col.expand(1, num_heads_q, q_len, 1),
            ),
            dim=-1,
        )
        out_chunks.append(F.scaled_dot_product_attention(q_slice, k_sink, v_sink, attn_mask=mask))
    return torch.cat(out_chunks, dim=2).transpose(1, 2).contiguous()


class PreviewAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        num_heads_q: int = 24,
        num_heads_kv: int = 24,
        head_dim: int = 128,
        num_modality: int = 1,
        sink_token_num: int = 1,
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

        self.pre_norm = MultiModalityRMSNorm(
            hidden_size, eps=1e-6, num_modality=num_modality, dtype=dtype, device=device
        )
        self.linear_g = make_modality_linear(
            hidden_size,
            num_heads_q,
            num_modality=num_modality,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
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
        self.linear_proj = make_modality_linear(
            self.q_size,
            hidden_size,
            num_modality=num_modality,
            bias=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.sinks = nn.Parameter(
            torch.empty(max(1, sink_token_num), num_heads_q, dtype=torch.float32, device=device)
        )
        self.q_norm = MultiModalityRMSNorm(
            head_dim,
            eps=1e-6,
            num_modality=num_modality,
            out_dtype=torch.float32,
            dtype=dtype,
            device=device,
        )
        self.k_norm = MultiModalityRMSNorm(
            head_dim,
            eps=1e-6,
            num_modality=num_modality,
            out_dtype=torch.float32,
            dtype=dtype,
            device=device,
        )
        self.comfy_attention = ComfyAttention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        seq_lens: list[int],
        modality_dispatcher: ModalityDispatcher,
        transformer_options: dict,
    ) -> torch.Tensor:
        compute_dtype = hidden_states.dtype
        x = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher)
        g = apply_modality_linear(self.linear_g, x, modality_dispatcher).view(
            hidden_states.shape[0], self.num_heads_q, 1
        )
        qkv = apply_modality_linear(self.linear_qkv, x, modality_dispatcher)
        q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)

        q = self.q_norm(
            q.view(hidden_states.shape[0], self.num_heads_q, self.head_dim),
            modality_dispatcher=modality_dispatcher,
        )
        k = self.k_norm(
            k.view(hidden_states.shape[0], self.num_heads_kv, self.head_dim),
            modality_dispatcher=modality_dispatcher,
        )
        v = v.view(hidden_states.shape[0], self.num_heads_kv, self.head_dim)

        q_unperm = modality_dispatcher.inv_permute(q).unsqueeze(0)
        k_unperm = modality_dispatcher.inv_permute(k).unsqueeze(0)
        v_unperm = modality_dispatcher.inv_permute(v)

        q_rot, k_rot = apply_magi2_rope(q_unperm, k_unperm, rope)
        q_rot = q_rot.squeeze(0).to(compute_dtype)
        k_rot = k_rot.squeeze(0).to(compute_dtype)
        v_unperm = v_unperm.to(compute_dtype)

        sinks = dinkster_inference.model_management.cast_to(
            self.sinks, dtype=torch.float32, device=hidden_states.device
        )

        q_parts = torch.split(q_rot, seq_lens, dim=0)
        k_parts = torch.split(k_rot, seq_lens, dim=0)
        v_parts = torch.split(v_unperm, seq_lens, dim=0)
        attn_parts = []
        for qb, kb, vb in zip(q_parts, k_parts, v_parts):
            out_b = _attention_with_sink_single(
                qb.transpose(0, 1).unsqueeze(0),
                kb.transpose(0, 1).unsqueeze(0),
                vb.transpose(0, 1).unsqueeze(0),
                sinks,
                self.comfy_attention,
                transformer_options,
            )
            attn_parts.append(out_b.squeeze(0))

        out = modality_dispatcher.permute(torch.cat(attn_parts, dim=0))
        out = (out * torch.sigmoid(g)).reshape(-1, self.q_size).to(compute_dtype)
        return apply_modality_linear(self.linear_proj, out, modality_dispatcher)


class PreviewDenseMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_modality: int = 3,
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
        h = self.pre_norm(x, modality_dispatcher=modality_dispatcher)
        h = apply_modality_linear(self.up_gate_proj, h, modality_dispatcher)
        h = swiglu7(h, out_dtype=x.dtype)
        return apply_modality_linear(self.down_proj, h, modality_dispatcher)


class PreviewTransformerLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = 3072,
        num_heads_q: int = 24,
        num_heads_kv: int = 24,
        head_dim: int = 128,
        intermediate_factor: int = 4,
        mm_layers: tuple[int, ...] = (0, 1, 38, 39),
        moe_layers: tuple[int, ...] = tuple(range(2, 38)),
        num_stream: int = 4,
        moe_num_heads: int = 12,
        moe_num_experts: int = 256,
        moe_top_k: int = 6,
        moe_intermediate_size: int = 1280,
        moe_route_scale: float = 4.9,
        sink_token_num: int = 1,
        dtype=None,
        device=None,
        operations=None,
        moe_ops=None,
    ) -> None:
        super().__init__()
        num_modality = 3 if layer_idx in mm_layers else 1
        self.num_stream = num_stream
        self.hidden_size = hidden_size
        self.matmul_scale = 1.0 / math.sqrt(float(num_stream * hidden_size))

        self.attention = PreviewAttention(
            hidden_size=hidden_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            num_modality=num_modality,
            sink_token_num=sink_token_num,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        if layer_idx in moe_layers:
            self.mlp = MultiHeadMoELayer(
                hidden_size=hidden_size,
                num_modality=3,
                num_heads=moe_num_heads,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                expert_intermediate_size=moe_intermediate_size,
                shared_expert_intermediate_size=moe_intermediate_size,
                modality_specific_expert_intermediate_size=moe_intermediate_size,
                route_scale=moe_route_scale,
                dtype=dtype,
                device=device,
                operations=operations,
                moe_ops=moe_ops,
            )
        else:
            inter = int(hidden_size * intermediate_factor * 2 / 3) // 128 * 128
            self.mlp = PreviewDenseMLP(
                hidden_size=hidden_size,
                intermediate_size=inter,
                num_modality=num_modality,
                dtype=dtype,
                device=device,
                operations=operations,
            )

        n = num_stream
        c = hidden_size
        self.mhc_alpha_pre_attn = nn.Parameter(torch.empty((1,), dtype=torch.float32, device=device))
        self.mhc_alpha_post_attn = nn.Parameter(torch.empty((1,), dtype=torch.float32, device=device))
        self.mhc_alpha_res_attn = nn.Parameter(torch.empty((1,), dtype=torch.float32, device=device))
        self.mhc_alpha_pre_mlp = nn.Parameter(torch.empty((1,), dtype=torch.float32, device=device))
        self.mhc_alpha_post_mlp = nn.Parameter(torch.empty((1,), dtype=torch.float32, device=device))
        self.mhc_alpha_res_mlp = nn.Parameter(torch.empty((1,), dtype=torch.float32, device=device))

        self.mhc_bias_pre_attn = nn.Parameter(torch.empty((n,), dtype=torch.float32, device=device))
        self.mhc_bias_post_attn = nn.Parameter(torch.empty((n,), dtype=torch.float32, device=device))
        self.mhc_bias_pre_mlp = nn.Parameter(torch.empty((n,), dtype=torch.float32, device=device))
        self.mhc_bias_post_mlp = nn.Parameter(torch.empty((n,), dtype=torch.float32, device=device))
        self.mhc_bias_res_attn = nn.Parameter(torch.empty((n, n), dtype=torch.float32, device=device))
        self.mhc_bias_res_mlp = nn.Parameter(torch.empty((n, n), dtype=torch.float32, device=device))

        self.mhc_phi_fused_attn = nn.Parameter(
            torch.empty((n * c, 2 * n + n * n), dtype=torch.float32, device=device)
        )
        self.mhc_phi_fused_mlp = nn.Parameter(
            torch.empty((n * c, 2 * n + n * n), dtype=torch.float32, device=device)
        )
        self.mhc_norm = MultiModalityRMSNorm(
            c * n,
            eps=1e-6,
            num_modality=num_modality,
            out_dtype=torch.float32,
            dtype=dtype,
            device=device,
        )

    def _mhc_pre(
        self,
        s: torch.Tensor,
        phi_param: nn.Parameter,
        alpha_pre_param: nn.Parameter,
        bias_pre_param: nn.Parameter,
        modality_dispatcher: ModalityDispatcher,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.num_stream
        dev = s.device
        s_flat = s.view(s.shape[0], n * self.hidden_size)
        normed = self.mhc_norm(s_flat, modality_dispatcher=modality_dispatcher)
        phi = dinkster_inference.model_management.cast_to(phi_param, dtype=torch.float32, device=dev)
        h_fused = torch.matmul(normed, phi)
        h_pre_raw, h_post_raw, h_res_raw = torch.split(h_fused, (n, n, n * n), dim=-1)

        alpha_pre = dinkster_inference.model_management.cast_to(alpha_pre_param, dtype=torch.float32, device=dev)
        bias_pre = dinkster_inference.model_management.cast_to(bias_pre_param, dtype=torch.float32, device=dev)
        h_pre = torch.sigmoid(alpha_pre * self.matmul_scale * h_pre_raw + bias_pre.unsqueeze(0)).to(s.dtype)
        sub_in = torch.einsum("tn,tnc->tc", h_pre, s)
        return h_post_raw, h_res_raw.view(-1, n, n), sub_in

    def _mhc_post(
        self,
        s: torch.Tensor,
        sub_out: torch.Tensor,
        h_post_raw: torch.Tensor,
        h_res_raw: torch.Tensor,
        alpha_post_param: nn.Parameter,
        bias_post_param: nn.Parameter,
        alpha_res_param: nn.Parameter,
        bias_res_param: nn.Parameter,
    ) -> torch.Tensor:
        dev = s.device
        alpha_post = dinkster_inference.model_management.cast_to(alpha_post_param, dtype=torch.float32, device=dev)
        bias_post = dinkster_inference.model_management.cast_to(bias_post_param, dtype=torch.float32, device=dev)
        alpha_res = dinkster_inference.model_management.cast_to(alpha_res_param, dtype=torch.float32, device=dev)
        bias_res = dinkster_inference.model_management.cast_to(bias_res_param, dtype=torch.float32, device=dev)

        h_post = (2.0 * torch.sigmoid(alpha_post * self.matmul_scale * h_post_raw + bias_post.unsqueeze(0))).to(s.dtype)
        h_res_logits = alpha_res * self.matmul_scale * h_res_raw + bias_res.unsqueeze(0)
        h_res = _sinkhorn_knopp_fp32(h_res_logits, num_iters=20, eps=1e-12).to(s.dtype)
        return torch.einsum("tij,tjc->tic", h_res, s) + torch.einsum("tn,tc->tnc", h_post, sub_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        seq_lens: list[int],
        modality_dispatcher: ModalityDispatcher,
        transformer_options: dict,
    ) -> torch.Tensor:
        s = hidden_states.reshape(hidden_states.shape[0], self.num_stream, self.hidden_size)
        h_post_a, h_res_a, attn_in = self._mhc_pre(
            s, self.mhc_phi_fused_attn, self.mhc_alpha_pre_attn, self.mhc_bias_pre_attn, modality_dispatcher
        )
        attn_out = self.attention(attn_in, rope, seq_lens, modality_dispatcher, transformer_options)
        s = self._mhc_post(
            s, attn_out, h_post_a, h_res_a, self.mhc_alpha_post_attn, self.mhc_bias_post_attn,
            self.mhc_alpha_res_attn, self.mhc_bias_res_attn
        )

        h_post_m, h_res_m, mlp_in = self._mhc_pre(
            s, self.mhc_phi_fused_mlp, self.mhc_alpha_pre_mlp, self.mhc_bias_pre_mlp, modality_dispatcher
        )
        mlp_out = self.mlp(mlp_in, modality_dispatcher)
        s = self._mhc_post(
            s, mlp_out, h_post_m, h_res_m, self.mhc_alpha_post_mlp, self.mhc_bias_post_mlp,
            self.mhc_alpha_res_mlp, self.mhc_bias_res_mlp
        )
        return s.reshape(s.shape[0], self.num_stream * self.hidden_size)


class PreviewPreAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        text_in_channels: int = 5120,
        video_in_channels: int = 48,
        audio_in_channels: int = 64,
        virtual_width_factor: int = 4,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.adapter_dim = hidden_size * virtual_width_factor
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.text_in_channels = text_in_channels
        self.video_embedder = operations.Linear(
            video_in_channels, self.adapter_dim, bias=True, dtype=torch.float32, device=device
        )
        self.text_embedder = operations.Linear(
            text_in_channels, self.adapter_dim, bias=True, dtype=torch.float32, device=device
        )
        self.audio_embedder = operations.Linear(
            audio_in_channels, self.adapter_dim, bias=True, dtype=torch.float32, device=device
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
        out = torch.zeros((total_tokens, self.adapter_dim), device=coords_mapping.device, dtype=torch.float32)
        if text_idx.numel() > 0:
            out.index_copy_(0, text_idx, self.text_embedder(text_tokens.float()).float())
        if audio_idx.numel() > 0:
            out.index_copy_(0, audio_idx, self.audio_embedder(audio_tokens.float()).float())
        if video_idx.numel() > 0:
            out.index_copy_(0, video_idx, self.video_embedder(video_tokens.float()).float())
        return out, rope


class PreviewPostAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        video_in_channels: int = 48,
        audio_in_channels: int = 64,
        virtual_width_factor: int = 4,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        adapter_dim = hidden_size * virtual_width_factor
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.final_norm_video = MultiModalityRMSNorm(adapter_dim, eps=1e-6, dtype=dtype, device=device)
        self.final_norm_audio = MultiModalityRMSNorm(adapter_dim, eps=1e-6, dtype=dtype, device=device)
        self.final_linear_video = operations.Linear(
            adapter_dim, video_in_channels, bias=False, dtype=torch.float32, device=device
        )
        self.final_linear_audio = operations.Linear(
            adapter_dim, audio_in_channels, bias=False, dtype=torch.float32, device=device
        )

    def forward(
        self, x: torch.Tensor, video_idx: torch.Tensor, audio_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_idx.numel() > 0:
            v_h = self.final_norm_video(x.index_select(0, video_idx).float())
            v_out = self.final_linear_video(v_h).float()
        else:
            v_out = x.new_empty((0, self.video_in_channels), dtype=torch.float32)
        if audio_idx.numel() > 0:
            a_h = self.final_norm_audio(x.index_select(0, audio_idx).float())
            a_out = self.final_linear_audio(a_h).float()
        else:
            a_out = x.new_empty((0, self.audio_in_channels), dtype=torch.float32)
        return v_out, a_out


class PreviewTransformerBlock(nn.Module):
    def __init__(self, layers: list[PreviewTransformerLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class Magi2PreviewModel(nn.Module):
    """MAGI-2 114B unified audio-video single-stream DiT with 4-stream mHC and Multi-Head MoE."""

    def __init__(
        self,
        image_model: str = "magi2_preview",
        num_layers: int = 40,
        hidden_size: int = 3072,
        head_dim: int = 128,
        num_heads_q: int = 24,
        num_heads_kv: int = 24,
        video_in_channels: int = 48,
        audio_in_channels: int = 64,
        text_in_channels: int = 5120,
        intermediate_factor: int = 4,
        mm_layers: tuple[int, ...] = (0, 1, 38, 39),
        moe_layers: Optional[tuple[int, ...]] = None,
        num_stream: int = 4,
        moe_num_heads: int = 12,
        moe_num_experts: int = 256,
        moe_top_k: int = 6,
        moe_intermediate_size: int = 1280,
        moe_route_scale: float = 4.9,
        sink_token_num: int = 1,
        time_channel_dim: int = 64,
        patch_size: tuple[int, int, int] = (1, 1, 1),
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_stream = num_stream
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.text_in_channels = text_in_channels
        self.time_channel_dim = time_channel_dim
        self.patch_size = tuple(patch_size)

        if moe_layers is None:
            mm_set = set(mm_layers)
            moe_layers = tuple(i for i in range(num_layers) if i not in mm_set)

        moe_ops = (
            operations
            if hasattr(operations, "MoEExperts")
            else dinkster_inference.ops.mixed_precision_ops({}, dtype)
        )

        self.pre_adapter = PreviewPreAdapter(
            hidden_size=hidden_size,
            num_attention_heads=num_heads_q,
            text_in_channels=text_in_channels,
            video_in_channels=video_in_channels * math.prod(self.patch_size),
            audio_in_channels=audio_in_channels,
            virtual_width_factor=num_stream,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.block = PreviewTransformerBlock(
            [
                PreviewTransformerLayer(
                    layer_idx=i,
                    hidden_size=hidden_size,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    intermediate_factor=intermediate_factor,
                    mm_layers=tuple(mm_layers),
                    moe_layers=tuple(moe_layers),
                    num_stream=num_stream,
                    moe_num_heads=moe_num_heads,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                    moe_intermediate_size=moe_intermediate_size,
                    moe_route_scale=moe_route_scale,
                    sink_token_num=sink_token_num,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                    moe_ops=moe_ops,
                )
                for i in range(num_layers)
            ]
        )
        self.post_adapter = PreviewPostAdapter(
            hidden_size=hidden_size,
            video_in_channels=video_in_channels * math.prod(self.patch_size),
            audio_in_channels=audio_in_channels,
            virtual_width_factor=num_stream,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def get_dynamic_vram__units(self) -> list[nn.Module]:
        return list(self.block.layers)

    def _pack_batch(
        self,
        video: torch.Tensor,
        audio: Optional[torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        magi2_payload: Optional[dict],
        denoise_mask: Optional[torch.Tensor] = None,
        audio_denoise_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size, _, t_vid, h_vid, w_vid = video.shape
        pt, ph, pw = self.patch_size
        t_steps, h_steps, w_steps = t_vid // pt, h_vid // ph, w_vid // pw
        vid_tok_per_item = t_steps * h_steps * w_steps

        # [B, C, T, H, W] -> [B, T*H*W, pt*ph*pw*C]
        vid_tokens = (
            video.view(batch_size, self.video_in_channels, t_steps, pt, h_steps, ph, w_steps, pw)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(batch_size, vid_tok_per_item, pt * ph * pw * self.video_in_channels)
        )

        if audio is not None and audio.numel() > 0:
            # audio arrives as [B, 64, T_aud] from Oobleck / NestedTensor
            aud_tokens = audio.transpose(1, 2).contiguous()
            aud_len = aud_tokens.shape[1]
        else:
            aud_tokens = video.new_empty((batch_size, 0, self.audio_in_channels))
            aud_len = 0

        payload = magi2_payload or {}
        txt_lens = payload.get("txt_lens")
        ref_images = payload.get("ref_images")  # list of dicts per batch item or shared
        device = video.device

        vid_mask_tokens: Optional[torch.Tensor] = None
        if denoise_mask is not None:
            v_m = denoise_mask.to(device=device, dtype=torch.float32)
            if v_m.ndim == 4:
                v_m = v_m.unsqueeze(1)
            else:
                v_m = v_m[:, :1]
            if v_m.shape[0] == 1 and batch_size > 1:
                v_m = v_m.expand(batch_size, -1, -1, -1, -1)
            v_m = dinkster_inference.ldm.common_dit.pad_to_patch_size(v_m, self.patch_size)
            vid_mask_tokens = (
                v_m.view(batch_size, 1, t_steps, pt, h_steps, ph, w_steps, pw)
                .amax(dim=(1, 3, 5, 7))
                .reshape(batch_size, vid_tok_per_item)
                .clamp(0.0, 1.0)
            )

        aud_mask_tokens: Optional[torch.Tensor] = None
        if audio_denoise_mask is not None and aud_len > 0:
            a_m = audio_denoise_mask.to(device=device, dtype=torch.float32)
            if a_m.ndim >= 3:
                a_m = a_m.amax(dim=1)
            elif a_m.ndim == 1:
                a_m = a_m.unsqueeze(0)
            if a_m.shape[0] == 1 and batch_size > 1:
                a_m = a_m.expand(batch_size, -1)
            if a_m.shape[1] < aud_len:
                a_m = F.pad(a_m, (0, aud_len - a_m.shape[1]), value=1.0)
            elif a_m.shape[1] > aud_len:
                a_m = a_m[:, :aud_len]
            aud_mask_tokens = a_m.clamp(0.0, 1.0)

        video_parts: list[torch.Tensor] = []
        audio_parts: list[torch.Tensor] = []
        text_parts: list[torch.Tensor] = []
        video_idx_parts: list[torch.Tensor] = []
        audio_idx_parts: list[torch.Tensor] = []
        text_idx_parts: list[torch.Tensor] = []
        gen_video_slices: list[tuple[int, int]] = []
        gen_audio_slices: list[tuple[int, int]] = []
        modality_parts: list[torch.Tensor] = []
        coords_parts: list[torch.Tensor] = []
        raw_t_parts: list[torch.Tensor] = []
        seq_lens: list[int] = []

        cursor = 0
        v_cursor = 0
        a_cursor = 0

        vid_coords = get_coords((t_steps, h_steps, w_steps), (t_steps, h_steps, w_steps), device=device)
        if aud_len > 0:
            aud_ref_t = max(1, ((aud_len - 1) // 8 + 1) // pt)
            aud_coords = get_coords((aud_len, 1, 1), (aud_ref_t, 1, 1), device=device)
        else:
            aud_coords = torch.empty((0, 9), device=device, dtype=torch.float32)

        # FlowUniPCMultistepScheduler.set_timesteps integer-truncates (sigmas * 1000).to(int64)
        # before normalizing back by 1000, keeping sinusoidal_embedding_1d on exact integer positions.
        ts_raw = timestep.reshape(-1).float().to(device=device)
        if ts_raw.numel() == 1 and batch_size > 1:
            ts_raw = ts_raw.expand(batch_size)
        t_flat = torch.floor(ts_raw) / 1000.0

        for b in range(batch_size):
            item_start = cursor
            ts_b = ts_raw[b]
            sigma_b = t_flat[b]

            # 1. Generated video tokens
            video_parts.append(vid_tokens[b])
            video_idx_parts.append(torch.arange(cursor, cursor + vid_tok_per_item, device=device, dtype=torch.long))
            gen_video_slices.append((v_cursor, v_cursor + vid_tok_per_item))
            v_cursor += vid_tok_per_item
            modality_parts.append(torch.full((vid_tok_per_item,), int(Modality.VIDEO), device=device, dtype=torch.int32))
            coords_parts.append(vid_coords)
            if vid_mask_tokens is not None:
                raw_t_parts.append(torch.floor(vid_mask_tokens[b] * ts_b) / 1000.0)
            else:
                raw_t_parts.append(torch.full((vid_tok_per_item,), sigma_b.item(), device=device, dtype=torch.float32))
            cursor += vid_tok_per_item

            # 2. Generated audio tokens
            if aud_len > 0:
                audio_parts.append(aud_tokens[b])
                audio_idx_parts.append(torch.arange(cursor, cursor + aud_len, device=device, dtype=torch.long))
                gen_audio_slices.append((a_cursor, a_cursor + aud_len))
                a_cursor += aud_len
                modality_parts.append(torch.full((aud_len,), int(Modality.AUDIO), device=device, dtype=torch.int32))
                coords_parts.append(aud_coords)
                if aud_mask_tokens is not None:
                    raw_t_parts.append(torch.floor(aud_mask_tokens[b] * ts_b) / 1000.0)
                else:
                    raw_t_parts.append(torch.full((aud_len,), sigma_b.item(), device=device, dtype=torch.float32))
                cursor += aud_len
            else:
                gen_audio_slices.append((a_cursor, a_cursor))

            # 3. Text tokens
            txt_len_b = int(txt_lens[b]) if txt_lens is not None and b < len(txt_lens) else context.shape[1]
            txt_b = context[b, :txt_len_b]
            if txt_len_b > 0:
                text_parts.append(txt_b)
                text_idx_parts.append(torch.arange(cursor, cursor + txt_len_b, device=device, dtype=torch.long))
                modality_parts.append(torch.full((txt_len_b,), int(Modality.TEXT), device=device, dtype=torch.int32))
                coords_parts.append(get_coords((txt_len_b, 1, 1), (1, 1, 1), offset_thw=(-txt_len_b, 0, 0), device=device))
                raw_t_parts.append(torch.zeros((txt_len_b,), device=device, dtype=torch.float32))
                cursor += txt_len_b

            # 4. Optional reference images (<Figure k> special token + Wan2.2 latent patches)
            item_refs = ref_images[b] if isinstance(ref_images, list) and b < len(ref_images) else (ref_images or [])
            for k_idx, ref_entry in enumerate(item_refs):
                ref_lat = ref_entry["latent"].to(device=device)  # [1, 48, 1, H_r, W_r] or [48, 1, H_r, W_r]
                if ref_lat.ndim == 5:
                    ref_lat = ref_lat[0]
                h_r, w_r = ref_lat.shape[-2] // ph, ref_lat.shape[-1] // pw
                ref_tok = (
                    ref_lat.view(self.video_in_channels, 1, pt, h_r, ph, w_r, pw)
                    .permute(1, 3, 5, 2, 4, 6, 0)
                    .reshape(h_r * w_r, pt * ph * pw * self.video_in_channels)
                )
                special_tok = ref_entry.get("special_token")
                if special_tok is None:
                    special_tok = torch.zeros((1, self.text_in_channels), device=device, dtype=context.dtype)
                else:
                    special_tok = special_tok.to(device=device).reshape(1, self.text_in_channels)

                t_off = t_steps + 2 + k_idx
                # Leading <Figure k> token routes as TEXT at (t_off, -1, -1)
                text_parts.append(special_tok)
                text_idx_parts.append(torch.tensor([cursor], device=device, dtype=torch.long))
                modality_parts.append(torch.tensor([int(Modality.TEXT)], device=device, dtype=torch.int32))
                coords_parts.append(
                    torch.tensor([[t_off, -1.0, -1.0, 1.0, float(h_r), float(w_r), 1.0, float(h_r), float(w_r)]], device=device, dtype=torch.float32)
                )
                raw_t_parts.append(torch.zeros((1,), device=device, dtype=torch.float32))
                cursor += 1

                # Reference image patches route as VIDEO with diffusion time 0
                n_ref = ref_tok.shape[0]
                video_parts.append(ref_tok)
                video_idx_parts.append(torch.arange(cursor, cursor + n_ref, device=device, dtype=torch.long))
                v_cursor += n_ref
                modality_parts.append(torch.full((n_ref,), int(Modality.VIDEO), device=device, dtype=torch.int32))
                coords_parts.append(get_coords((1, h_r, w_r), (1, h_r, w_r), offset_thw=(t_off, 0, 0), device=device))
                raw_t_parts.append(torch.zeros((n_ref,), device=device, dtype=torch.float32))
                cursor += n_ref

            seq_lens.append(cursor - item_start)

        video_cat = torch.cat(video_parts, dim=0) if video_parts else video.new_empty((0, pt * ph * pw * self.video_in_channels))
        audio_cat = torch.cat(audio_parts, dim=0) if audio_parts else video.new_empty((0, self.audio_in_channels))
        text_cat = torch.cat(text_parts, dim=0) if text_parts else context.new_empty((0, self.text_in_channels))
        video_idx = torch.cat(video_idx_parts, dim=0) if video_idx_parts else torch.empty((0,), device=device, dtype=torch.long)
        audio_idx = torch.cat(audio_idx_parts, dim=0) if audio_idx_parts else torch.empty((0,), device=device, dtype=torch.long)
        text_idx = torch.cat(text_idx_parts, dim=0) if text_idx_parts else torch.empty((0,), device=device, dtype=torch.long)
        modality_mapping = torch.cat(modality_parts, dim=0)
        coords_mapping = torch.cat(coords_parts, dim=0)
        raw_t = torch.cat(raw_t_parts, dim=0)
        time_emb = sinusoidal_embedding_1d(self.time_channel_dim, raw_t) if self.time_channel_dim > 0 else None

        return {
            "video_cat": video_cat,
            "audio_cat": audio_cat,
            "text_cat": text_cat,
            "video_idx": video_idx,
            "audio_idx": audio_idx,
            "text_idx": text_idx,
            "gen_video_slices": gen_video_slices,
            "gen_audio_slices": gen_audio_slices,
            "modality_mapping": modality_mapping,
            "coords_mapping": coords_mapping,
            "time_emb": time_emb,
            "seq_lens": seq_lens,
            "total_tokens": cursor,
            "grid": (batch_size, t_steps, h_steps, w_steps, aud_len),
        }

    def _run_layers(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        seq_lens: list[int],
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
                            args["seq_lens"],
                            args["modality_dispatcher"],
                            args["transformer_options"],
                        )
                    }

                x = patches_replace[("double_block", idx)](
                    {
                        "img": x,
                        "rope": rope,
                        "seq_lens": seq_lens,
                        "modality_dispatcher": modality_dispatcher,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )["img"]
            else:
                x = layer(x, rope, seq_lens, modality_dispatcher, transformer_options)
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
        denoise_mask: Optional[torch.Tensor] = None,
        audio_denoise_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        is_av = not torch.is_tensor(x)
        video_in = x[0] if is_av else x
        audio_in = x[1] if is_av and len(x) > 1 else None

        orig_t, orig_h, orig_w = video_in.shape[2], video_in.shape[3], video_in.shape[4]
        video_padded = dinkster_inference.ldm.common_dit.pad_to_patch_size(video_in, self.patch_size)
        compute_dtype = video_padded.dtype

        packed = self._pack_batch(
            video_padded,
            audio_in,
            timestep,
            context,
            magi2_payload,
            denoise_mask=denoise_mask,
            audio_denoise_mask=audio_denoise_mask,
        )
        modality_dispatcher = ModalityDispatcher(packed["modality_mapping"], num_modalities=3)

        h, rope = self.pre_adapter(
            packed["video_cat"],
            packed["audio_cat"],
            packed["text_cat"],
            packed["video_idx"],
            packed["audio_idx"],
            packed["text_idx"],
            packed["coords_mapping"],
            packed["total_tokens"],
        )
        if packed["time_emb"] is not None:
            h[:, : self.time_channel_dim] = packed["time_emb"].to(h.dtype)

        h = modality_dispatcher.permute(h.to(compute_dtype))
        h = self._run_layers(h, rope, packed["seq_lens"], modality_dispatcher, transformer_options)
        h = modality_dispatcher.inv_permute(h)

        v_all, a_all = self.post_adapter(h, packed["video_idx"], packed["audio_idx"])

        batch_size, t_steps, h_steps, w_steps, aud_len = packed["grid"]
        pt, ph, pw = self.patch_size
        v_items = []
        for s_idx, e_idx in packed["gen_video_slices"]:
            v_b = (
                v_all[s_idx:e_idx]
                .view(t_steps, h_steps, w_steps, pt, ph, pw, self.video_in_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.video_in_channels, t_steps * pt, h_steps * ph, w_steps * pw)
            )
            v_items.append(v_b[:, :orig_t, :orig_h, :orig_w])
        video_out = torch.stack(v_items, dim=0).to(compute_dtype)

        if not is_av:
            return video_out

        if aud_len > 0:
            a_items = [a_all[s_idx:e_idx].transpose(0, 1).contiguous() for s_idx, e_idx in packed["gen_audio_slices"]]
            audio_out = torch.stack(a_items, dim=0).to(audio_in.dtype)
        else:
            audio_out = torch.zeros_like(audio_in)
        return [video_out, audio_out]
