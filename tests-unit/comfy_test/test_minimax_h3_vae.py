import pytest
import torch

from dinkster_inference.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import dinkster_inference.quant_ops
from dinkster_inference.ldm.minimax import vae as minimax_vae
from dinkster_inference.ldm.minimax.vae import Attention, MiniMaxH3VideoVAE


class _OffloadedScale:
    """Stands in for a buffer that comfy's dynamic-VRAM offload left on another
    device: it only tracks its device label and can be moved with .to(), like
    the real qk_norm_scale tensor the bug report is about."""

    def __init__(self, device):
        self.device = torch.device(device)

    def to(self, device):
        return _OffloadedScale(device)


class _FakeCK:
    def rms_rope_split_half(self, query, key, freqs, scale, epsilon, rot_dim):
        return self._check(query, key, scale)

    def rms_rope_split_half_(self, query, key, freqs, scale, epsilon, rot_dim):
        return self._check(query, key, scale)

    @staticmethod
    def _check(query, key, scale):
        if scale.device != query.device:
            # mirrors comfy_kitchen's eager backend, which does no device
            # alignment and refuses to run the fused kernel on mixed devices
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got weight is on "
                f"{scale.device}, different from other tensors on {query.device}"
            )
        return query, key


def test_attention_moves_offloaded_qk_norm_scale_to_input_device(monkeypatch):
    monkeypatch.setattr(dinkster_inference.quant_ops, "ck", _FakeCK(), raising=False)

    heads, dim_head = 2, 4
    attn = Attention(heads=heads, dim_head=dim_head)
    # simulate a partially-offloaded VAE: qk_norm_scale sits on a different
    # device than the tensors flowing through the fused rope kernel
    attn._buffers["qk_norm_scale"] = _OffloadedScale("meta")

    batch_size, seq_len = 1, 3
    x = torch.arange(1, 25, dtype=torch.float32).reshape(batch_size, seq_len, heads * dim_head)
    rotary_pos_emb = torch.randn(batch_size, 1, seq_len, dim_head // 2, 2, 2)

    pre_norm = dinkster_inference.ops.manual_cast.RMSNorm(heads * dim_head, elementwise_affine=False, eps=1e-5)
    qkv_outputs = []
    hook = attn.to_qkv.register_forward_hook(lambda module, inputs, output: qkv_outputs.append(output))
    with torch.no_grad():
        attn.to_qkv.weight.copy_(torch.eye(heads * dim_head).repeat(3, 1))
        attn.to_qkv.bias.zero_()
        attn.to_out.weight.copy_(torch.eye(heads * dim_head))
        attn.to_out.bias.zero_()
        out = attn.forward(
            x, rotary_pos_emb, pre_norm=pre_norm, residual=None, residual_scale=None
        )
    hook.remove()

    assert out.device == x.device
    normalized = torch.nn.functional.rms_norm(x, (heads * dim_head,), eps=pre_norm.eps)
    expected = torch.nn.functional.linear(normalized, attn.to_qkv.weight, attn.to_qkv.bias)
    unnormalized = torch.nn.functional.linear(x, attn.to_qkv.weight, attn.to_qkv.bias)
    assert len(qkv_outputs) == 1
    torch.testing.assert_close(qkv_outputs[0], expected)
    assert not torch.allclose(qkv_outputs[0], unnormalized)



def test_still_decodes_as_first_latent_of_a_full_clip():
    with torch.device("meta"):
        vae = MiniMaxH3VideoVAE()
    vae.latents_mean, vae.latents_std = torch.zeros(24), torch.ones(24)
    vae.pixel_mean, vae.pixel_std = torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)
    seen = []

    def decode_pixels(z):  # every output frame holds its own index / 100
        seen.append(z.shape[2])
        frames = torch.arange(z.shape[2] * vae.vae_ratio_t, dtype=torch.float32) / 100
        return frames.view(1, 1, -1, 1, 1).expand(z.shape[0], 3, -1, z.shape[-2] * vae.vae_ratio, z.shape[-1] * vae.vae_ratio)

    vae._decode_pixels = decode_pixels
    out = vae.decode(torch.zeros(1, 24, 1, 40, 24))  # 640x384, several tiles
    assert out.shape == (1, 3, 1, 640, 384)
    assert set(seen) == {vae.tokens_chunk_size}
    torch.testing.assert_close(out, torch.full_like(out, vae.frame_pre_padding / 100))


def test_ph3_image_encode_samples_fixed_posterior_without_changing_global_rng():
    with torch.device("meta"):
        vae = minimax_vae.PH3ImageVAE(num_layers=0)
    mean = torch.linspace(-2, 3, 24).reshape(1, 24, 1, 1, 1)
    logvar = torch.linspace(-40, 40, 24).reshape_as(mean)
    vae.latents_mean = torch.linspace(0.1, 0.9, 24)
    vae.latents_std = torch.linspace(0.5, 2, 24)
    vae.pixel_mean = torch.zeros(1, 3, 1, 1, 1)
    vae.pixel_std = torch.ones(1, 3, 1, 1, 1)
    vae._adaptive_encode = lambda x: torch.cat([mean, logvar], dim=1)
    image = torch.zeros(1, 3, 1, 16, 16)
    before = torch.random.get_rng_state()
    actual = vae.encode(image, device="cpu")
    assert torch.equal(torch.random.get_rng_state(), before)
    torch.rand(7)
    torch.testing.assert_close(vae.encode(image), actual, rtol=0, atol=0)
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(42))
    expected = (mean + logvar.clamp(-30, 20).mul(0.5).exp() * noise
                - vae.latents_mean.reshape_as(mean)) / vae.latents_std.reshape_as(mean)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, (mean - vae.latents_mean.reshape_as(mean)) / vae.latents_std.reshape_as(mean))


def test_ph3_image_decode_uses_one_frame_and_writes_output_buffer():
    with torch.device("meta"):
        vae = minimax_vae.PH3ImageVAE(num_layers=0)
    vae.tiling = False
    vae.latents_mean, vae.latents_std = torch.zeros(24), torch.ones(24)
    vae.pixel_mean, vae.pixel_std = torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)
    vae._decode_pixels = lambda z: z[:, :3].repeat_interleave(16, -2).repeat_interleave(16, -1)
    z = torch.linspace(0.2, 0.95, 24).view(1, 24, 1, 1, 1).expand(1, 24, 1, 2, 3)
    expected = z[:, :3].repeat_interleave(16, -2).repeat_interleave(16, -1)
    output = torch.empty(vae.decode_output_shape(z.shape))
    assert vae.decode(z, output_buffer=output) is output
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(vae.decode(z), expected)
    with pytest.raises(ValueError, match="single-frame"):
        vae.decode(z.expand(1, 24, 2, 2, 3))
