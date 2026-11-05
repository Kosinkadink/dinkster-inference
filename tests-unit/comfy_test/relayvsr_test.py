import pytest
import torch
import torch.nn.functional as F

from dinkster_inference.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from dinkster_inference import latent_formats, model_management  # noqa: E402
from dinkster_inference.ldm.wan.model_vsr import RelayFlash  # noqa: E402
from dinkster_inference.model_detection import model_config_from_unet  # noqa: E402
from dinkster_inference.supported_models import WAN22_RelayVSR  # noqa: E402


@pytest.mark.parametrize("factor", [2, 4])
def test_relayvsr_checkpoint_detection(factor):
    state = {
        "head.modulation": torch.empty(1, 2, 128),
        "head.head.weight": torch.empty(192, 128),
        "blocks.0.ffn.0.weight": torch.empty(256, 128),
        "patch_embedding.weight": torch.empty(128, 48, 1, 2, 2),
        "lq_proj_in.conv1.weight": torch.empty(768, 3 * (4 * factor) ** 2, 1, 1),
    }
    config = model_config_from_unet(state, "")
    assert isinstance(config, WAN22_RelayVSR)
    assert config.unet_config["upscale_factor"] == factor
    latent = torch.arange(48, dtype=torch.float32).reshape(1, 48, 1, 1, 1)
    assert isinstance(config.latent_format, latent_formats.RelayVSR)
    torch.testing.assert_close(config.latent_format.process_in(latent), latent)
    torch.testing.assert_close(config.latent_format.process_out(latent), latent)


@pytest.mark.parametrize("factor,frames,keyframes,latent_size", [(2, 1, 1, (2, 4)), (4, 17, 3, (4, 6))])
def test_flash_decode_zero_detail_preserves_all_upscaled_frames(factor, frames, keyframes, latent_size):
    model = RelayFlash(factor, dim=64, depth=1, num_heads=1, kv_groups=1, hidden_dim=96, head_dim=64)
    for parameter in model.parameters():
        parameter.data.fill_(0.01)
    images = torch.linspace(0.1, 0.9, frames * 9 * 19 * 3).reshape(frames, 9, 19, 3)
    latent = torch.zeros(1, 48, keyframes, *latent_size)
    shape = (1, 3, frames, 9 * factor, 19 * factor)
    assert model.decode_output_shape(latent.shape, lq_images=images) == shape
    output = torch.full(shape, float("nan"))
    decoded = model.decode(latent, lq_images=images, output_buffer=output, detail=0)
    stride = 16 // factor
    padded = F.pad(images.movedim(-1, 1) * 2 - 1, (0, latent_size[1] * stride - 19, 0, latent_size[0] * stride - 9), mode="replicate")
    expected = F.interpolate(padded, scale_factor=factor, mode="bicubic", align_corners=False)
    expected = expected[..., :9 * factor, :19 * factor].add(1).div(2).clamp(0, 1).movedim(0, 1).unsqueeze(0)
    assert decoded is output
    torch.testing.assert_close(decoded, expected)
    with pytest.raises(ValueError, match="doesn't fit"):
        model.decode_output_shape((1, 48, keyframes + 1, *latent_size), lq_images=images)


def test_host_writer_cpu_copies_strided_frame_destinations():
    output = torch.full((2, 3, 5, 4, 7), float("nan"))
    expected = torch.arange(output.numel(), dtype=output.dtype).reshape(output.shape)
    with model_management.HostWriter(torch.device("cpu"), output, (2, 3, 4, 7)) as writer:
        for frame in range(5):
            writer.write(output[:, :, frame], expected[:, :, frame].clone())
    torch.testing.assert_close(output, expected)


def test_host_writer_allocation_failure_leaves_no_pinned_buffers(monkeypatch):
    target = torch.empty(2, 3)
    allocate = torch.empty
    allocations = 0
    registered = set()

    def fail_second_allocation(*args, **kwargs):
        nonlocal allocations
        allocations += 1
        if allocations == 2:
            raise MemoryError("host allocation failed")
        return allocate(*args, **kwargs)

    def pin(buffer, **kwargs):
        registered.add(buffer.data_ptr())
        return True

    monkeypatch.setattr(model_management, "get_data_stream", lambda device: object())
    monkeypatch.setattr(model_management, "ensure_pin_budget", lambda *args, **kwargs: True)
    monkeypatch.setattr(model_management, "pin_memory", pin)
    monkeypatch.setattr(model_management, "unpin_memory", lambda buffer: registered.remove(buffer.data_ptr()))
    monkeypatch.setattr(torch, "empty", fail_second_allocation)
    with pytest.raises(MemoryError, match="host allocation failed"):
        model_management.HostWriter(torch.device("cpu"), target, (3,))
    assert not registered
