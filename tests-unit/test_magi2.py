from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors.torch import save_file
import torch

if not torch.cuda.is_available():
    import dinkster_inference.cli_args

    dinkster_inference.cli_args.args.cpu = True
import torch.nn.functional as F
import pytest

import dinkster_inference.latent_formats
import dinkster_inference.ldm.magi2.common as magi2_common
import dinkster_inference.ldm.magi2.moe as magi2_moe
import dinkster_inference.ldm.magi2.preview as magi2_preview
import dinkster_inference.ldm.magi2.refiner as magi2_refiner
import dinkster_inference.ldm.magi2.sampling as magi2_sampling
import dinkster_inference.ldm.magi2.turbo_vae as magi2_turbo_vae
import dinkster_inference.model_detection
import dinkster_inference.ops
import dinkster_inference.samplers
import dinkster_inference.sd
import dinkster_inference.text_encoders.magi2 as magi2_te
from tools.quantize_magi2 import calibrate_fp8_input_scales, convert_and_quantize


def test_swiglu7_clamping_and_bias_asymmetric():
    # Interleaved [gate, up]: gate clamps only above +7.0 (not below -7.0), up clamps [-7.0, +7.0] and adds +1.0
    x = torch.tensor([[-10.0, -10.0, 10.0, 10.0, 2.0, 0.0]], dtype=torch.bfloat16)
    out = magi2_common.swiglu7(x, out_dtype=torch.float32)

    # Pair 0: gate = -10.0 (unclamped below!), up = -7.0 -> (-10 * sigmoid(-17.02)) * (-7 + 1)
    g0 = -10.0 * (1.0 / (1.0 + math.exp(17.02))) * (-6.0)
    # Pair 1: gate = +7.0 (clamped), up = +7.0 (clamped) -> (7 * sigmoid(11.914)) * 8
    g1 = 7.0 * (1.0 / (1.0 + math.exp(-7.0 * 1.702))) * 8.0
    # Pair 2: gate = 2.0, up = 0.0 -> (2 * sigmoid(3.404)) * 1.0
    g2 = 2.0 * (1.0 / (1.0 + math.exp(-3.404))) * 1.0
    expected = torch.tensor([[g0, g1, g2]], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_mhc_sinkhorn_doubly_stochastic_and_partial_rope():
    torch.manual_seed(7)
    logits = torch.randn(17, 4, 4, dtype=torch.float32) * 1.0
    ds = magi2_preview._sinkhorn_knopp_fp32(logits, num_iters=20, eps=1e-12)
    assert torch.allclose(ds.sum(dim=-1), torch.ones(17, 4), atol=1e-5)
    assert torch.allclose(ds.sum(dim=-2), torch.ones(17, 4), atol=1e-4)

    # Zero-SNR DDPM sigma endpoints and index 220 sanity check
    sigmas = magi2_sampling.zero_snr_ddpm_sigmas(1000)
    assert math.isclose(float(sigmas[0].item()), 0.9995749, rel_tol=1e-4)
    assert float(sigmas[-1].item()) == 0.0
    assert 0.80 < float(sigmas[220].item()) < 0.90

    # Exact ODE integration check for FlowUniPC: on constant velocity v = x_init - target_x0,
    # model(x, sigma) = x - sigma * v = target_x0 -> FlowUniPC must land on target_x0 to <1e-5!
    torch.manual_seed(31)
    x_init = torch.randn(2, 16)
    target_x0 = torch.randn(2, 16)
    f_sigmas = magi2_sampling.flow_unipc_sigmas(10, shift=7.0)
    assert math.isclose(float(f_sigmas[0].item()), 7.0 * 0.999 / (1.0 + 6.0 * 0.999), rel_tol=1e-6)
    assert float(f_sigmas[-1].item()) == 0.0
    solved = magi2_sampling.sample_flow_unipc(lambda x_cur, _s: target_x0, x_init, f_sigmas)
    assert torch.allclose(solved, target_x0, atol=1e-5)

    # Partial 96/128 split-half RoPE: trailing 32 dims must stay bit-identical
    rope_embed = magi2_common.ElementWiseFourierEmbed(128)
    rope_embed.bands.copy_(magi2_common.default_rope_bands(128))
    coords = magi2_common.get_coords((2, 3, 4), (2, 3, 4))
    rope = rope_embed(coords)  # [24, 96]

    q = torch.randn(1, 24, 6, 128, dtype=torch.float32)
    k = torch.randn(1, 24, 6, 128, dtype=torch.float32)
    q_rot, k_rot = magi2_common.apply_magi2_rope(q, k, rope)

    assert torch.equal(q_rot[..., 96:], q[..., 96:])
    assert torch.equal(k_rot[..., 96:], k[..., 96:])

    # Independent reference split-half formula on [:96]
    sin_e, cos_e = rope.tensor_split(2, dim=-1)
    cos_rep = torch.cat((cos_e, cos_e), dim=-1).unsqueeze(0).unsqueeze(2)
    sin_rep = torch.cat((sin_e, sin_e), dim=-1).unsqueeze(0).unsqueeze(2)
    q1, q2 = q[..., :48], q[..., 48:96]
    q_ref = torch.cat((torch.cat((-q2, q1), dim=-1) * sin_rep + q[..., :96] * cos_rep, q[..., 96:]), dim=-1)
    assert torch.allclose(q_rot, q_ref, atol=1e-5)


def test_moe_router_uses_bias_only_for_selection():
    # Expert 0 has tiny raw score (logit -4 -> sigmoid ~0.018) + huge bias (+10.0)
    # Expert 1 has high raw score (logit +4 -> sigmoid ~0.982) + zero bias
    # Expert 2 has medium raw score (logit 0 -> sigmoid 0.5) + zero bias
    # With top_k=2, experts {0, 1} must be selected, but normalized weights must come from UNBIASED scores!
    logits = torch.tensor([[[ -4.0, 4.0, 0.0, -5.0 ]]], dtype=torch.float32)  # [H=1, S=1, E=4]
    bias = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    probs, idx = magi2_moe._compute_topk_routing(
        logits, top_k=2, expert_bias=bias, score_func="sigmoid", route_norm=True, route_scale=4.9
    )
    chosen = set(idx.reshape(-1).tolist())
    assert chosen == {0, 1}
    s0 = 1.0 / (1.0 + math.exp(4.0))
    s1 = 1.0 / (1.0 + math.exp(-4.0))
    expected_map = {0: 4.9 * s0 / (s0 + s1), 1: 4.9 * s1 / (s0 + s1)}
    for slot in range(2):
        eid = int(idx[0, 0, slot].item())
        assert math.isclose(float(probs[0, 0, slot].item()), expected_map[eid], rel_tol=1e-5)

    if torch.cuda.is_available():
        moe_ops = dinkster_inference.ops.mixed_precision_ops({}, torch.bfloat16)
        moe = magi2_moe.CoreMultiHeadMoE(
            hidden_size=128, num_heads=2, num_experts=4, top_k=2, expert_intermediate_size=128,
            dtype=torch.bfloat16, device="cuda", moe_ops=moe_ops,
        )
        sd = {
            "gate": torch.randn(8, 64, device="cuda", dtype=torch.float32) * 0.1,
            "router.expert_bias": torch.randn(8, device="cuda", dtype=torch.float32) * 0.05,
            "W_gate.weight": torch.randn(8, 128, 64, device="cuda", dtype=torch.bfloat16) * 0.05,
            "W_up.weight": torch.randn(8, 128, 64, device="cuda", dtype=torch.bfloat16) * 0.05,
            "W_down.weight": torch.randn(8, 64, 128, device="cuda", dtype=torch.bfloat16) * 0.05,
        }
        moe.load_state_dict(sd)
        x_in = torch.randn(37, 128, device="cuda", dtype=torch.bfloat16)
        out_fused = moe(x_in)
        xh = x_in.view(37, 2, 64).contiguous()
        tp, ti = moe._route(xh)
        out_seq = moe._forward_sequential_banks_triton(xh, tp, ti).reshape(37, 128)
        with moe.W_gate.bank_resident(x_in) as wg, moe.W_up.bank_resident(x_in) as wu, moe.W_down.bank_resident(x_in) as wd:
            out_eager = moe._forward_eager(xh, tp, ti, wg, wu, wd).reshape(37, 128)
        assert torch.allclose(out_fused.float(), out_eager.float(), atol=1e-2, rtol=1e-2)
        assert torch.allclose(out_seq.float(), out_eager.float(), atol=1e-2, rtol=1e-2)


def test_refiner_window_attention_matches_explicit_3d_block_mask():
    # Non-multiple T=15 (blocks of 8 -> [8, 7]), H=24 (6 blocks of 4), W=24 (6 blocks of 4)
    # Since 6 > 5 (2*radius + 1), boundary blocks must NOT attend to opposite-corner blocks!
    torch.manual_seed(11)
    t_num, h_num, w_num = 15, 24, 24
    bs = (8, 4, 4)
    radius = (2, 2, 2)
    tail_len = 5
    vid_tokens = t_num * h_num * w_num
    total_tokens = vid_tokens + tail_len

    _, _, block_ranges, grid_shape = magi2_refiner._build_refiner_block_order(
        (t_num, h_num, w_num), bs, total_tokens, torch.device("cpu")
    )
    q = torch.randn(1, 4, total_tokens, 32, dtype=torch.float32)
    k = torch.randn(1, 2, total_tokens, 32, dtype=torch.float32)
    v = torch.randn(1, 2, total_tokens, 32, dtype=torch.float32)

    out_win = magi2_refiner._window_attention_eager(
        q, k, v, block_ranges, grid_shape, radius, vid_tokens
    )

    # Independent dense boolean attention mask over [total_tokens, total_tokens]
    nt, nh, nw = grid_shape
    allow = torch.zeros((total_tokens, total_tokens), dtype=torch.bool)
    allow[vid_tokens:, :] = True
    allow[:, vid_tokens:] = True
    ranges = block_ranges.tolist()
    for qt in range(nt):
        for qh in range(nh):
            for qw in range(nw):
                qb = qt * (nh * nw) + qh * nw + qw
                qs, qe = ranges[qb]
                for kt in range(max(0, qt - 2), min(nt, qt + 3)):
                    for kh in range(max(0, qh - 2), min(nh, qh + 3)):
                        for kw in range(max(0, qw - 2), min(nw, qw + 3)):
                            kb = kt * (nh * nw) + kh * nw + kw
                            ks, ke = ranges[kb]
                            allow[qs:qe, ks:ke] = True
    # Confirm window actually excludes distant blocks
    assert not bool(allow.all())
    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, enable_gqa=True).transpose(1, 2)
    assert torch.allclose(out_win, out_ref, atol=1e-5)


def test_magi2_tokenizer_cjk_split_json_markdown_and_figure_span():
    tok = magi2_te.Magi2Tokenizer()
    structured = {
        "global_layer": {
            "context": "studio",
            "description": "A person greets the camera in Chinese: \u4f60\u597d\u4e16\u754c",
        }
    }
    tokens = tok.tokenize_with_weights(json.dumps(structured, ensure_ascii=False), magi2_figure_ids=[1])
    assert tokens["magi2_txt_len"] > 0
    span = tokens["magi2_figure_spans"][1]
    assert len(span) >= 3  # `<Figure 1>` spans multiple BPE tokens


def _make_raw_sandai_preview_sd(hidden_size: int = 128, num_heads_q: int = 2, head_dim: int = 64) -> dict[str, torch.Tensor]:
    """Synthesize a miniature raw upstream SandAI MAGI-2 Preview checkpoint (unsplit mm + untransposed MoE)."""
    torch.manual_seed(123)
    ref = magi2_preview.Magi2PreviewModel(
        num_layers=4,
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_q,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=5120,
        intermediate_factor=4,
        mm_layers=(0, 3),
        moe_layers=(1, 2),
        num_stream=4,
        moe_num_heads=2,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=128,
        dtype=torch.bfloat16,
        device="cpu",
        operations=dinkster_inference.ops.manual_cast,
    )
    norm_sd = {}
    for k, v in ref.state_dict().items():
        if k == "pre_adapter.rope.bands":
            norm_sd[k] = magi2_common.default_rope_bands(head_dim)
        elif "norm" in k or "sinks" in k or "mhc_bias" in k:
            norm_sd[k] = torch.randn_like(v, dtype=torch.float32) * 0.05
        elif "mhc_alpha" in k:
            norm_sd[k] = torch.full_like(v, 0.01, dtype=torch.float32)
        else:
            norm_sd[k] = (torch.randn_like(v, dtype=torch.float32) * 0.02).to(v.dtype if v.is_floating_point() else torch.bfloat16)

    # Convert back to raw SandAI disk format to test full conversion + auto-normalization!
    raw_sd: dict[str, torch.Tensor] = {}
    grouped_buckets: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for k, v in norm_sd.items():
        if ".linears." in k:
            pre, rest = k.split(".linears.", 1)
            idx_str, param = rest.split(".", 1)
            grouped_buckets.setdefault(f"{pre}.{param}", []).append((int(idx_str), v))
        elif k.endswith((".moe_mlp.W_gate.weight", ".moe_mlp.W_up.weight", ".moe_mlp.W_down.weight")):
            raw_key = k[: -len(".weight")]
            raw_sd[raw_key] = v.transpose(1, 2).contiguous().to(torch.bfloat16)
        elif k.endswith(".moe_mlp.router.expert_bias"):
            # Put garbage in raw expert_bias and true bias in expert_bias_ema!
            raw_sd[k] = torch.full_like(v, -999.0)
            raw_sd[f"{k}_ema"] = v.clone()
        else:
            raw_sd[k] = v

    for gk, indexed in grouped_buckets.items():
        indexed.sort(key=lambda iv: iv[0])
        raw_sd[gk] = torch.cat([t for _, t in indexed], dim=0).contiguous()

    # MoEExperts.weight starts as None until loaded; populate raw SandAI [E, in, out] MoE banks!
    d_head = hidden_size // 2
    for m_layer in (1, 2):
        pref = f"block.layers.{m_layer}.mlp.moe_mlp"
        raw_sd[f"{pref}.W_gate"] = (torch.randn(8, d_head, 128) * 0.02).to(torch.bfloat16)
        raw_sd[f"{pref}.W_up"] = (torch.randn(8, d_head, 128) * 0.02).to(torch.bfloat16)
        raw_sd[f"{pref}.W_down"] = (torch.randn(8, 128, d_head) * 0.02).to(torch.bfloat16)
    return raw_sd


def _make_raw_sandai_refiner_sd() -> dict[str, torch.Tensor]:
    """Miniature Refiner with hidden_size=192 -> intermediate_size=504 (`504 % 16 == 8`!) to exercise NVFP4 pad_16x and INT8 gs=4."""
    torch.manual_seed(456)
    ref = magi2_refiner.Magi2RefinerModel(
        num_layers=4,
        hidden_size=192,
        head_dim=64,
        num_heads_q=3,
        num_heads_kv=1,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=5120,
        mm_layers=(0, 3),
        dtype=torch.bfloat16,
        device="cpu",
        operations=dinkster_inference.ops.manual_cast,
    )
    raw_sd: dict[str, torch.Tensor] = {}
    grouped_buckets: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for k, v in ref.state_dict().items():
        if k == "pre_adapter.rope.bands":
            t = magi2_common.default_rope_bands(64)
        elif "norm" in k:
            t = torch.randn_like(v, dtype=torch.float32) * 0.05
        else:
            t = (torch.randn_like(v, dtype=torch.float32) * 0.02).to(v.dtype)
        if ".linears." in k:
            pre, rest = k.split(".linears.", 1)
            idx_str, param = rest.split(".", 1)
            grouped_buckets.setdefault(f"{pre}.{param}", []).append((int(idx_str), t))
        else:
            raw_sd[k] = t
    for gk, indexed in grouped_buckets.items():
        indexed.sort(key=lambda iv: iv[0])
        raw_sd[gk] = torch.cat([t for _, t in indexed], dim=0).contiguous()
    return raw_sd


@pytest.mark.parametrize("kind", ["preview", "refiner"])
def test_magi2_end_to_end_loader_ksampler_and_quantization(tmp_path: Path, kind: str):
    raw_dir = tmp_path / f"raw_{kind}"
    raw_dir.mkdir()
    raw_sd = _make_raw_sandai_preview_sd() if kind == "preview" else _make_raw_sandai_refiner_sd()
    # Embed text_in_channels=64 test override only via pre_adapter.text_embedder shape? Note model_detection uses default 5120 unless detected;
    # let's make text_in_channels=5120 on tiny preview/refiner or detect text_in_channels!
    save_file(raw_sd, str(raw_dir / "model.safetensors"))

    bf16_file = tmp_path / f"{kind}_bf16.safetensors"
    convert_and_quantize(raw_dir, bf16_file, model_kind=kind, fmt="bf16", device_str="cpu")
    assert bf16_file.exists()

    # Verify streaming multi-part merge path too by forcing small flush threshold
    formats = ["bf16", "int8", "fp8"]
    if torch.cuda.is_available():
        formats.append("nvfp4")

    outputs_by_fmt: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for fmt in formats:
        out_file = bf16_file if fmt == "bf16" else (tmp_path / f"{kind}_{fmt}.safetensors")
        scales_path = None
        if fmt == "fp8" and "bf16" in outputs_by_fmt:
            patcher_cal = dinkster_inference.sd.load_diffusion_model(str(bf16_file))
            dev = patcher_cal.load_device
            dt = patcher_cal.model.get_dtype_inference()
            dinkster_inference.model_management.load_models_gpu([patcher_cal])
            v_c = torch.randn(1, 48, 2, 4, 4, device=dev, dtype=dt)
            a_c = torch.randn(1, 64, 6, device=dev, dtype=dt)
            ctx_c = torch.randn(1, 5, patcher_cal.model.diffusion_model.text_in_channels, device=dev, dtype=dt)
            scales = calibrate_fp8_input_scales(
                patcher_cal,
                [([v_c, a_c], torch.tensor([500.0], device=dev), ctx_c, {"txt_lens": [5]})],
            )
            assert len(scales) > 0
            scales_path = tmp_path / f"{kind}_fp8_scales.json"
            scales_path.write_text(json.dumps(scales))

        if fmt != "bf16":
            convert_and_quantize(
                raw_dir,
                out_file,
                model_kind=kind,
                fmt=fmt,
                device_str="cuda" if torch.cuda.is_available() else "cpu",
                input_scales_path=scales_path,
            )

        patcher = dinkster_inference.sd.load_diffusion_model(str(out_file))
        dyn_units, last_units = patcher.model.get_dynamic_vram__units()
        assert len(dyn_units) == 4 and len(last_units) == 1

        text_dim = patcher.model.diffusion_model.text_in_channels
        torch.manual_seed(99)
        pos_ctx = torch.randn(1, 7, text_dim)
        neg_ctx = torch.randn(1, 3, text_dim)  # Different length to test CONDMagi2CrossAttn batching!
        ref_lat = torch.randn(1, 48, 1, 4, 4)
        ref_tok = torch.randn(text_dim)

        positive = [[pos_ctx, {"magi2_txt_len": 7, "magi2_ref_images": [{"latent": ref_lat, "special_token": ref_tok}]}]]
        negative = [[neg_ctx, {"magi2_txt_len": 3, "magi2_ref_images": [{"latent": torch.zeros_like(ref_lat), "special_token": ref_tok}]}]]

        v_init = torch.zeros(1, 48, 2, 4, 4)
        a_init = torch.zeros(1, 64, 6)
        latent_av = dinkster_inference.nested_tensor.NestedTensor((v_init, a_init))
        torch.manual_seed(2026)
        noise_av = dinkster_inference.nested_tensor.NestedTensor(
            (torch.randn_like(v_init), torch.randn_like(a_init))
        )

        sampler = dinkster_inference.samplers.KSAMPLER(magi2_sampling.sample_flow_unipc)
        sigmas = magi2_sampling.flow_unipc_sigmas(3, shift=7.0 if kind == "preview" else 5.0)
        sampled = dinkster_inference.samplers.sample(
            patcher,
            noise_av,
            positive,
            negative,
            cfg=3.0,
            device=patcher.load_device,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent_av,
            disable_pbar=True,
            seed=42,
        )
        v_out, a_out = sampled.unbind()
        assert v_out.shape == v_init.shape
        assert a_out.shape == a_init.shape
        assert torch.isfinite(v_out).all()
        assert torch.isfinite(a_out).all()
        outputs_by_fmt[fmt] = (v_out.detach().cpu().float(), a_out.detach().cpu().float())

    v_ref, _ = outputs_by_fmt["bf16"]
    for fmt, (v_q, _) in outputs_by_fmt.items():
        if fmt == "bf16":
            continue
        cos = F.cosine_similarity(v_ref.flatten(), v_q.flatten(), dim=0).item()
        assert cos > 0.95, f"{kind} {fmt} cosine similarity too low: {cos}"


def test_magi2_turbo_vae_detection_and_sliding_decode():
    cfg = {
        "latent_channels": 48,
        "out_channels": 3,
        "decoder_block_out_channels": [32, 32, 64, 64],
        "decoder_layers_per_block": [1, 1, 1, 1, 1],
        "decoder_spatio_temporal_scaling": [True, True, True, False],
        "decoder_spatio_only": [False, True, False, False],
        "patch_size": 2,
        "first_chunk_size": 3,
        "step_size": 5,
    }
    dec = magi2_turbo_vae.Magi2TurboVAEDecoder(**cfg)
    sd = {k: torch.randn_like(v) * 0.02 for k, v in dec.state_dict().items()}
    vae = dinkster_inference.sd.VAE(sd=sd, metadata={"config": json.dumps(cfg)}, device=torch.device("cpu"), dtype=torch.float32)
    z = torch.randn(1, 48, 9, 4, 4)  # 9 frames -> exercises 2 sliding chunks + padding removal!
    pixels = vae.decode(z)
    # 9 Wan2.2 latent frames -> 9 * 4 - 3 = 33 pixel frames, 4 * 16 = 64x64 spatial
    assert pixels.shape == (1, 33, 64, 64, 3)
    assert torch.isfinite(pixels).all()


def test_magi2_letterbox_reference_image_matches_upstream_resizepad_order():
    # A 16:9 (1920x1080 -> scaled 192x108) dark reference image at 512x896 has raw_w=896, raw_h=504
    # (exact 16:9 fit -> zero white bars in resizepad), then aligns to (480, 896) without adding white side bars.
    dark_16_9 = torch.full((1, 108, 192, 3), 0.15, dtype=torch.float32)
    keep_out = magi2_sampling.letterbox_reference_image(dark_16_9, 512, 896, resize_type="keep_ratio")
    assert keep_out.shape == (1, 480, 896, 3)
    assert keep_out[:, :, 0, :].max().item() < 0.25
    assert keep_out[:, :, -1, :].max().item() < 0.25

    square_out = magi2_sampling.letterbox_reference_image(dark_16_9, 512, 896, resize_type="square")
    assert square_out.shape == (1, 320, 320, 3)
    assert torch.allclose(square_out[:, 0, :, :], torch.ones_like(square_out[:, 0, :, :]))
    assert square_out[:, 160, 160, :].max().item() < 0.25


def test_magi2_core_moe_two_slot_cast_buffer_no_overwrite():
    """Verify 3-bank CoreMultiHeadMoE never overwrites W_gate with W_down when NUM_STREAMS=2."""
    torch.manual_seed(1234)
    devices_and_dims = [(torch.device("cpu"), torch.float32, 16, 8)]
    if torch.cuda.is_available():
        devices_and_dims.append((torch.device("cuda"), torch.bfloat16, 128, 64))

    for dev, dt, hidden_size, d_exp in devices_and_dims:
        ops = dinkster_inference.ops.mixed_precision_ops({}, dt)
        moe = dinkster_inference.ldm.magi2.moe.CoreMultiHeadMoE(
            hidden_size=hidden_size,
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=d_exp,
            dtype=dt,
            device=dev,
            moe_ops=ops,
        )
        d_head = hidden_size // 2
        sd = {
            "gate": torch.randn(8, d_head, device=dev, dtype=torch.float32) * 0.2,
            "router.expert_bias": torch.zeros(8, device=dev, dtype=torch.float32),
            "W_gate.weight": torch.randn(8, d_exp, d_head, device=dev, dtype=dt) * 0.2,
            "W_up.weight": torch.randn(8, d_exp, d_head, device=dev, dtype=dt) * 0.2,
            "W_down.weight": torch.randn(8, d_head, d_exp, device=dev, dtype=dt) * 0.2,
        }
        moe.load_state_dict(sd)

        x = torch.randn(19, hidden_size, device=dev, dtype=dt)
        expected = moe(x).clone()

        max_numel = 8 * d_exp * d_head
        ring_buffers = [
            torch.empty(max_numel, device=dev, dtype=dt),
            torch.empty(max_numel, device=dev, dtype=dt),
        ]
        counter = [0]

        def make_ring_cast_bank(bank_mod):
            def _fake_cast_bank(_inp):
                slot = counter[0] % 2
                counter[0] += 1
                src = bank_mod.weight
                buf = ring_buffers[slot][: src.numel()].view_as(src)
                buf.copy_(src)

                class _RingCtx:
                    state = (buf, None, ("fake_stream", buf, None))

                    def __enter__(self):
                        return buf, None

                    def __exit__(self, *args):
                        return None

                return _RingCtx()

            return _fake_cast_bank

        moe.W_gate._cast_bank = make_ring_cast_bank(moe.W_gate)
        moe.W_up._cast_bank = make_ring_cast_bank(moe.W_up)
        moe.W_down._cast_bank = make_ring_cast_bank(moe.W_down)

        actual = moe(x)
        tol = 2e-2 if dt == torch.bfloat16 else 1e-5
        assert torch.allclose(actual, expected, atol=tol, rtol=tol)


def test_magi2_preview_packed_av_latent_roundtrip(tmp_path: Path):
    """Verify packed 3D (B, 1, V+A) latents from utils.pack_latents normalize/unnormalize video only."""
    raw_dir = tmp_path / "raw_prev"
    raw_dir.mkdir()
    save_file(_make_raw_sandai_preview_sd(), str(raw_dir / "model.safetensors"))
    bf16_file = tmp_path / "prev_bf16.safetensors"
    convert_and_quantize(raw_dir, bf16_file, model_kind="preview", fmt="bf16", device_str="cpu")
    patcher = dinkster_inference.sd.load_diffusion_model(str(bf16_file))
    model = patcher.model

    torch.manual_seed(99)
    v_raw = torch.randn(1, 48, 2, 4, 4)
    a_raw = torch.randn(1, 64, 6)
    packed, shapes = dinkster_inference.utils.pack_latents([v_raw, a_raw])
    assert packed.ndim == 3
    model.latent_shapes = shapes

    normed = model.process_latent_in(packed)
    v_norm_u, a_norm_u = dinkster_inference.utils.unpack_latents(normed, shapes)
    assert not torch.allclose(v_norm_u, v_raw)
    assert torch.allclose(v_norm_u, model.latent_format.process_video_in(v_raw), atol=1e-5)
    assert torch.equal(a_norm_u, a_raw)

    unnormed = model.process_latent_out(normed)
    v_out_u, a_out_u = dinkster_inference.utils.unpack_latents(unnormed, shapes)
    assert torch.allclose(v_out_u, v_raw, atol=1e-5)
    assert torch.equal(a_out_u, a_raw)


def test_flow_unipc_supports_nested_tensor():
    torch.manual_seed(77)
    nt_init = dinkster_inference.nested_tensor.NestedTensor(
        (torch.randn(1, 4, 2, 2, 2), torch.randn(1, 8, 6))
    )
    nt_target = dinkster_inference.nested_tensor.NestedTensor(
        (torch.randn(1, 4, 2, 2, 2), torch.randn(1, 8, 6))
    )
    f_sigmas = magi2_sampling.flow_unipc_sigmas(6, shift=7.0)
    solved = magi2_sampling.sample_flow_unipc(lambda _x, _s: nt_target, nt_init, f_sigmas)
    s_v, s_a = solved.unbind()
    t_v, t_a = nt_target.unbind()
    assert torch.allclose(s_v, t_v, atol=1e-5)
    assert torch.allclose(s_a, t_a, atol=1e-5)


def test_magi2_core_moe_int8_full_precision_mm():
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for dev in devices:
        dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
        d_head, d_exp = (64, 64) if dev.type == "cuda" else (16, 16)
        ops = dinkster_inference.ops.mixed_precision_ops({}, dt)
        moe = dinkster_inference.ldm.magi2.moe.CoreMultiHeadMoE(
            hidden_size=d_head * 2,
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=d_exp,
            dtype=dt,
            device=dev,
            moe_ops=ops,
        )
        torch.manual_seed(123)
        sd_ref = {
            "gate": torch.randn(8, d_head, device=dev, dtype=torch.float32) * 0.2,
            "router.expert_bias": torch.zeros(8, device=dev, dtype=torch.float32),
            "W_gate.weight": torch.randn(8, d_exp, d_head, device=dev, dtype=dt) * 0.2,
            "W_up.weight": torch.randn(8, d_exp, d_head, device=dev, dtype=dt) * 0.2,
            "W_down.weight": torch.randn(8, d_head, d_exp, device=dev, dtype=dt) * 0.2,
        }
        moe.load_state_dict(sd_ref)
        x = torch.randn(11, d_head * 2, device=dev, dtype=dt)
        expected = moe(x).clone()

        for name in ("W_gate", "W_up", "W_down"):
            bank = getattr(moe, name)
            w2d = sd_ref[f"{name}.weight"].float().reshape(8 * bank.out_features, bank.in_features)
            q, params = TensorWiseINT8Layout.quantize(w2d, is_weight=True, per_channel=True, convrot=False)
            bank.weight = torch.nn.Parameter(
                QuantizedTensor(q.to(dev), "TensorWiseINT8Layout", params), requires_grad=False
            )
            bank._full_precision_mm = True

        actual = moe(x)
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        cos = float(torch.nn.functional.cosine_similarity(actual.detach().float().reshape(1, -1), expected.detach().float().reshape(1, -1)).item())
        assert cos > 0.99


def test_magi2_uncond_zero_ref_image_stays_zero_normalized(tmp_path: Path):
    raw_dir = tmp_path / "raw_prev_ref"
    raw_dir.mkdir()
    save_file(_make_raw_sandai_preview_sd(), str(raw_dir / "model.safetensors"))
    bf16_file = tmp_path / "prev_ref_bf16.safetensors"
    convert_and_quantize(raw_dir, bf16_file, model_kind="preview", fmt="bf16", device_str="cpu")
    patcher = dinkster_inference.sd.load_diffusion_model(str(bf16_file))
    model = patcher.model

    pos_lat = torch.ones(1, 48, 1, 4, 4) * 1.5
    zero_lat = torch.zeros_like(pos_lat)
    tok = torch.ones(64)

    ec_pos = model.extra_conds(magi2_ref_images=[{"latent": pos_lat, "special_token": tok}])
    ec_zero = model.extra_conds(magi2_ref_images=[{"latent": zero_lat, "special_token": tok}])
    pos_out = ec_pos["magi2_payload"].cond["ref_images"][0][0]["latent"]
    zero_out = ec_zero["magi2_payload"].cond["ref_images"][0][0]["latent"]
    assert torch.allclose(pos_out, model.latent_format.process_video_in(pos_lat), atol=1e-6)
    assert float(zero_out.abs().max().item()) == 0.0


def test_magi2_multi_figure_prompt_normalization_and_packing(tmp_path: Path):
    tok = magi2_te.Magi2Tokenizer()

    # Single figure fallback keeps exact first-frame hint
    single_norm = magi2_te.normalize_magi2_prompt("A calm lake.", ensure_figure_ids=[1])
    assert "The first frame refers to <Figure 1>" in single_norm

    # Multi-figure fallback appends distinct Reference k hints for missing figures without touching present ones
    multi_partial = magi2_te.normalize_magi2_prompt(
        "Character <Figure 1> enters the room.", ensure_figure_ids=[1, 2, 3]
    )
    assert multi_partial.count("<Figure 1>") == 1
    assert "Reference 2 refers to <Figure 2>" in multi_partial
    assert "Reference 3 refers to <Figure 3>" in multi_partial

    toks = tok.tokenize_with_weights(multi_partial, magi2_figure_ids=[1, 2, 3])
    for fid in (1, 2, 3):
        assert len(toks["magi2_figure_spans"][fid]) >= 3
    assert toks["magi2_figure_spans"][1] != toks["magi2_figure_spans"][2]
    assert toks["magi2_figure_spans"][2] != toks["magi2_figure_spans"][3]

    # Verify _pack_batch places multiple reference images at t_steps + 2 + k with raw_t == 0
    raw_dir = tmp_path / "raw_multi_ref"
    raw_dir.mkdir()
    save_file(_make_raw_sandai_preview_sd(), str(raw_dir / "model.safetensors"))
    bf16_file = tmp_path / "prev_multi_ref.safetensors"
    convert_and_quantize(raw_dir, bf16_file, model_kind="preview", fmt="bf16", device_str="cpu")
    patcher = dinkster_inference.sd.load_diffusion_model(str(bf16_file))
    dm = patcher.model.diffusion_model

    video = torch.randn(1, 48, 3, 2, 2)
    audio = torch.randn(1, 64, 5)
    ctx = torch.randn(1, 4, dm.text_in_channels)
    refs = [
        {"latent": torch.randn(1, 48, 1, 2, 2), "special_token": torch.ones(dm.text_in_channels)},
        {"latent": torch.randn(1, 48, 1, 2, 4), "special_token": torch.full((dm.text_in_channels,), 2.0)},
    ]
    packed = dm._pack_batch(
        video, audio, torch.tensor([500.0]), ctx, {"txt_lens": [4], "ref_images": [refs]}
    )
    # 12 video + 5 audio + 4 text + (1 + 4) ref1 + (1 + 8) ref2 = 35 tokens
    assert packed["total_tokens"] == 35
    coords = packed["coords_mapping"]
    # Ref 0 sits at t_off = 3 + 2 + 0 = 5; Ref 1 sits at t_off = 3 + 2 + 1 = 6
    assert float(coords[21, 0].item()) == 5.0
    assert float(coords[26, 0].item()) == 6.0
    # Clean conditioning tokens (text + refs) have raw_t == 0 -> cos(0) == 1, sin(0) == 0
    half_t = dm.time_channel_dim // 2
    assert torch.allclose(packed["time_emb"][21:, :half_t], torch.ones(14, half_t))
    assert torch.allclose(packed["time_emb"][21:, half_t:], torch.zeros(14, half_t))


def test_magi2_per_token_av_denoise_mask_sampling(tmp_path: Path):
    raw_dir = tmp_path / "raw_mask_prev"
    raw_dir.mkdir()
    save_file(_make_raw_sandai_preview_sd(), str(raw_dir / "model.safetensors"))
    bf16_file = tmp_path / "prev_mask_bf16.safetensors"
    convert_and_quantize(raw_dir, bf16_file, model_kind="preview", fmt="bf16", device_str="cpu")
    patcher = dinkster_inference.sd.load_diffusion_model(str(bf16_file))
    dm = patcher.model.diffusion_model

    # 1. Verify _pack_batch per-token video & audio diffusion times
    v_in = torch.randn(1, 48, 3, 2, 2)
    a_in = torch.randn(1, 64, 6)
    v_mask = torch.ones(1, 1, 3, 2, 2)
    v_mask[:, :, 0] = 0.0  # Anchor frame 0
    a_mask = torch.ones(1, 1, 6)
    a_mask[:, :, :2] = 0.0  # Anchor first 2 audio tokens
    ctx = torch.randn(1, 3, dm.text_in_channels)
    packed = dm._pack_batch(
        v_in,
        a_in,
        torch.tensor([750.0]),
        ctx,
        {"txt_lens": [3]},
        denoise_mask=v_mask,
        audio_denoise_mask=a_mask,
    )
    half_t = dm.time_channel_dim // 2
    # Frame 0 (first 4 video tokens) must have t=0 -> cos=1, sin=0; frames 1..2 must have t=0.75
    assert torch.allclose(packed["time_emb"][:4, :half_t], torch.ones(4, half_t))
    assert torch.allclose(packed["time_emb"][:4, half_t:], torch.zeros(4, half_t))
    expected_750 = magi2_common.sinusoidal_embedding_1d(dm.time_channel_dim, torch.tensor([0.75]))[0]
    assert torch.allclose(packed["time_emb"][4:12], expected_750.expand(8, -1), atol=1e-6)
    # Audio tokens 0..1 at indices 12..14 must have t=0; 2..5 at indices 14..18 must have t=0.75
    assert torch.allclose(packed["time_emb"][12:14, half_t:], torch.zeros(2, half_t))
    assert torch.allclose(packed["time_emb"][14:18], expected_750.expand(4, -1), atol=1e-6)

    # 2. End-to-end samplers.sample: verify clean video frame 0 and clean audio prefix enter
    # forward un-noised AND remain preserved in the final output while unmasked tokens denoise.
    seen_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig_forward = dm.forward

    def spy_forward(x, timestep, context, **kwargs):
        seen_inputs.append((x[0].detach().clone(), x[1].detach().clone()))
        return orig_forward(x, timestep, context, **kwargs)

    dm.forward = spy_forward
    try:
        torch.manual_seed(314)
        v_guide = torch.randn(1, 48, 3, 2, 2) * 1.25
        a_guide = torch.randn(1, 64, 6) * 0.8
        latent_av = dinkster_inference.nested_tensor.NestedTensor((v_guide.clone(), a_guide.clone()))
        noise_av = dinkster_inference.nested_tensor.NestedTensor(
            (torch.randn_like(v_guide), torch.randn_like(a_guide))
        )
        mask_av = dinkster_inference.nested_tensor.NestedTensor((v_mask.clone(), a_mask.clone()))

        text_dim = dm.text_in_channels
        pos = [[torch.randn(1, 4, text_dim), {"magi2_txt_len": 4}]]
        neg = [[torch.randn(1, 2, text_dim), {"magi2_txt_len": 2}]]
        sampler = dinkster_inference.samplers.KSAMPLER(magi2_sampling.sample_flow_unipc)
        sigmas = magi2_sampling.flow_unipc_sigmas(3, shift=7.0)

        sampled = dinkster_inference.samplers.sample(
            patcher,
            noise_av,
            pos,
            neg,
            cfg=2.5,
            device=patcher.load_device,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent_av,
            denoise_mask=mask_av,
            disable_pbar=True,
            seed=19,
        )
        v_out, a_out = sampled.unbind()
        v_out = v_out.detach().cpu()
        a_out = a_out.detach().cpu()
        # Masked regions (mask == 0) must equal the original clean guide latents!
        assert torch.allclose(v_out[:, :, :1], v_guide[:, :, :1], atol=1e-5)
        assert torch.allclose(a_out[:, :, :2], a_guide[:, :, :2], atol=1e-5)
        # Unmasked regions (mask == 1) must have evolved away from guide latents!
        assert not torch.allclose(v_out[:, :, 1:], v_guide[:, :, 1:], atol=1e-2)
        assert not torch.allclose(a_out[:, :, 2:], a_guide[:, :, 2:], atol=1e-2)

        # Verify model forward received normalized clean v_guide[:, :, :1] and clean a_guide[:, :, :2] (NOT sigma-noised!)
        expected_v0_norm = patcher.model.latent_format.process_video_in(v_guide)[:, :, :1]
        expected_a0 = a_guide[:, :, :2]
        for v_step, a_step in seen_inputs:
            assert torch.allclose(v_step[:1, :, :1].cpu().float(), expected_v0_norm.float(), atol=2e-2)
            assert torch.allclose(a_step[:1, :, :2].cpu().float(), expected_a0.float(), atol=2e-2)
    finally:
        dm.forward = orig_forward
