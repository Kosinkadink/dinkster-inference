import pytest
import torch

from dinkster_inference.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from dinkster_inference.ldm.minimax import vae as minimax_vae  # noqa: E402


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
