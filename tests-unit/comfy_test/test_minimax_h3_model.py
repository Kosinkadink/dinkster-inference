import weakref

import pytest
import torch
from torch import nn

import dinkster_inference.ldm.minimax.vae as minimax_vae
from dinkster_inference.ldm.minimax.model import MiniMaxH3Model, PackedLayout, time_shift_sigma
from dinkster_inference.model_sampling import CONST


def make_model(video_output, audio_output):
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    nn.Module.__init__(model)
    model.sigma_shift_video = 12.0
    model.sigma_shift_audio = 3.0
    model._forward = lambda *args, **kwargs: [video_output.clone(), audio_output.clone()]
    return model


def test_forward_scales_velocity_to_mask_timestep():
    video_output = torch.full((1, 2, 1, 2, 2), 2.0)
    audio_output = torch.full((1, 2, 2, 3), 3.0)
    video_mask = torch.tensor([[[[[1.0, 0.75], [0.5, 0.25]]]]])
    audio_mask = torch.tensor([[[[1.0, 0.5, 0.25], [0.75, 0.5, 0.0]]]])
    sigma = torch.tensor([0.5])
    clean = torch.arange(video_output.numel(), dtype=torch.float32).reshape_as(video_output)
    model_input = clean + sigma.reshape(1, 1, 1, 1, 1) * video_mask * video_output
    model = make_model(video_output, audio_output)

    out = model(
        [model_input, torch.zeros_like(audio_output)],
        sigma * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 1.0},
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )

    torch.testing.assert_close(out[0], video_output * video_mask)
    torch.testing.assert_close(out[1], audio_output * audio_mask)
    denoised = CONST.calculate_denoised(None, sigma, out[0], model_input)
    torch.testing.assert_close(denoised, clean)


def test_forward_scales_audio_velocity_before_carry_conversion():
    video_output = torch.ones((1, 1, 1, 1, 1))
    audio_output = torch.full((1, 1, 2, 2), 3.0)
    audio_src = torch.full_like(audio_output, 2.0)
    audio_mask = torch.tensor([[[[0.75, 0.5], [0.25, 0.0]]]])
    model = make_model(video_output, audio_output)
    sigma_v = torch.tensor(0.5)
    sigma_a = time_shift_sigma(sigma_v, 12.0, 3.0)
    carry = sigma_a / sigma_v

    out = model(
        [torch.zeros_like(video_output), audio_src],
        sigma_v.reshape(1) * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 4.0},
        audio_denoise_mask=audio_mask,
    )

    expected = -3.0 * audio_src * carry + (1.0 + 3.0 * sigma_a) * audio_output * audio_mask
    torch.testing.assert_close(out[1], expected)


@pytest.mark.parametrize("conditioning", ["none", "refs", "keyframes"])
def test_embed_and_pack_releases_intermediates(conditioning):
    model = MiniMaxH3Model(
        hidden_size=8, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8,
        latents_dim=2, audio_latents_dim=3, text_dim=5,
        timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
        dtype=torch.float32, device="cpu", operations=nn,
    )
    video = torch.randn(1, 2, 1, 4, 6)
    audio = torch.randn(1, 3, 2, 2)
    context = torch.randn(1, 4, 8)
    payload = {}
    if conditioning != "none":
        cond_video = torch.randn_like(video)
        cond_audio = torch.randn(1, 3, 2, 1)
        payload.update(cond_video_latents=[cond_video], cond_audio_latents=[cond_audio])
        if conditioning == "refs":
            payload["refs"] = [{"kind": "image", "latent_h": 4, "latent_w": 6},
                               {"kind": "audio", "ref_audio_t": 1}]
        else:
            payload["keyframes"] = [{"resolved_frame_index": 0, "latent": cond_video,
                                      "audio_latent": cond_audio}]
    layout = PackedLayout(4, 1, 4, 6, 2, refs=payload.get("refs"), keyframes=payload.get("keyframes"))
    outputs, refs = {}, []

    def remember(name):
        def hook(module, inputs, output):
            outputs[name] = output.clone()
            refs.extend([weakref.ref(inputs[0]), weakref.ref(output)])
        return hook

    handles = [model.video_patch_proj.register_forward_hook(remember("video")),
               model.audio_patch_proj.register_forward_hook(remember("audio"))]
    with torch.no_grad():
        packed = model._embed_and_pack(video, audio, context, layout, payload, {})
    for handle in handles:
        handle.remove()

    assert all(ref() is None for ref in refs)
    assert packed.shape == (layout.seq_len, 8)
    assert packed.dtype == torch.float32
    slices = {kind: torch.cat([packed[a:b] for a, b, name in layout.segments if name in kinds])
              for kind, kinds in {"video": {"video", "ref_img", "cond"},
                                  "audio": {"audio", "ref_audio", "cond_audio"},
                                  "text": {"text"}}.items()}
    assert torch.equal(slices["video"], outputs["video"])
    assert torch.equal(slices["audio"], outputs["audio"])
    assert torch.equal(slices["text"], context[0])


class RecordingOperations:
    class Linear(nn.Linear):
        def forward(self, x):
            self.call = x
            self.output = torch.arange(1, self.out_features + 1, dtype=x.dtype, device=x.device).expand(*x.shape[:-1], -1)
            return self.output


def test_vae_feed_forward_delegates_norm_activation_and_residual():
    block = minimax_vae.FeedForward(dim=2, mult=1, operations=RecordingOperations)
    pre_norm = minimax_vae.dinkster_inference.ops.manual_cast.RMSNorm(2, eps=0.0125)
    pre_norm.weight = nn.Parameter(torch.tensor([0.75, 1.25]))
    x = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    residual = x + 3
    residual_scale = torch.tensor([0.25, 0.75])

    out = block(x, pre_norm, residual=residual, residual_scale=residual_scale)

    torch.testing.assert_close(block.w1.call, torch.nn.functional.rms_norm(x, (2,), pre_norm.weight, pre_norm.eps))
    gate, value = block.w1.output.chunk(2, dim=-1)
    torch.testing.assert_close(block.w2.call, torch.nn.functional.silu(gate) * value)
    torch.testing.assert_close(out, residual + block.w2.output * residual_scale)


def test_vae_attention_delegates_norm_and_residual(monkeypatch):
    block = minimax_vae.Attention(heads=1, dim_head=2, operations=RecordingOperations)
    pre_norm = minimax_vae.dinkster_inference.ops.manual_cast.RMSNorm(2, eps=0.0125)
    pre_norm.weight = nn.Parameter(torch.tensor([0.75, 1.25]))
    x = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    residual = x + 7
    residual_scale = torch.tensor([0.125, 0.875])

    def optimized_attention(query, key, value, heads, skip_reshape):
        return query.transpose(1, 2).reshape(1, 2, 2)

    monkeypatch.setattr(minimax_vae, "optimized_attention", optimized_attention)

    out = block(x, None, pre_norm, residual=residual, residual_scale=residual_scale)

    torch.testing.assert_close(block.to_qkv.call, torch.nn.functional.rms_norm(x, (2,), pre_norm.weight, pre_norm.eps))
    query = block.to_qkv.output[..., :2]
    torch.testing.assert_close(block.to_out.call, torch.nn.functional.rms_norm(query, (2,), block.norm_q.weight, block.norm_q.eps))
    torch.testing.assert_close(out, residual + block.to_out.output * residual_scale)


def test_input_act_weight_casts_offloaded_norm_with_its_own_epsilon(monkeypatch):
    ops = minimax_vae.dinkster_inference.ops
    pre_norm = ops.manual_cast.RMSNorm(2, eps=0.0125)
    x = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    cast_weight = object()
    context_modules = []

    class CastContext:
        def __init__(self, module, input, offloadable):
            context_modules.append((module, input, offloadable))

        def __enter__(self):
            return cast_weight, None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(ops, "CastBiasWeightContext", CastContext)

    with ops._input_act_weight(x, "rms_norm", pre_norm, 0.75) as (weight, eps):
        assert weight is cast_weight
        assert eps == pre_norm.eps

    assert context_modules == [(pre_norm, x, True)]
