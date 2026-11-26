from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import dinkster_inference.model_management
import dinkster_inference.ops

ops = dinkster_inference.ops.disable_weight_init


def _turbo_unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    b, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)
    x = x.view(b, channels, patch_size, patch_size, frames, height, width)
    return x.permute(0, 1, 4, 5, 3, 6, 2).contiguous().view(
        b, channels, frames, height * patch_size, width * patch_size
    )


class _ChannelRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty((dim, 1, 1, 1)))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            w = dinkster_inference.model_management.cast_to(
                self.weight, dtype=x.dtype, device=x.device
            )
            return normed.to(x.dtype) * w
        return normed.to(x.dtype)


class _Conv2dSplitUpsampler(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        pad = kernel_size // 2
        self.conv = ops.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.conv(x), self.stride)


class _NonCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        st = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.kernel_size = ks
        self.conv = ops.Conv3d(
            in_channels,
            out_channels,
            ks,
            stride=st,
            padding=(0, ks[1] // 2, ks[2] // 2),
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tk = self.kernel_size[0]
        if tk > 1:
            pad_n = (tk - 1) // 2
            x = torch.cat(
                (x[:, :, :1].repeat(1, 1, pad_n, 1, 1), x, x[:, :, -1:].repeat(1, 1, pad_n, 1, 1)),
                dim=2,
            )
        return self.conv(x)


class _DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        ks = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = ks
        self.depthwise_conv = ops.Conv3d(
            in_channels,
            in_channels,
            ks,
            stride=1,
            padding=(0, ks[1] // 2, ks[2] // 2),
            groups=in_channels,
        )
        self.pointwise_conv = ops.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tk = self.kernel_size[0]
        if tk > 1:
            pad_n = (tk - 1) // 2
            x = torch.cat(
                (x[:, :, :1].repeat(1, 1, pad_n, 1, 1), x, x[:, :, -1:].repeat(1, 1, pad_n, 1, 1)),
                dim=2,
            )
        return self.pointwise_conv(self.depthwise_conv(x))


class _TurboResNetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        eps: float = 1e-6,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
        is_upsampler_modified: bool = False,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        out_ch = out_channels or in_channels
        self.is_upsampler_modified = is_upsampler_modified
        self.norm1 = _ChannelRMSNorm(in_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.nonlinearity = nn.SiLU()
        self.replace_nonlinearity = nn.ReLU() if is_upsampler_modified else None
        if is_dw_conv:
            self.conv1 = _DepthwiseSeparableConv3d(in_channels, out_ch, kernel_size=dw_kernel_size)
            self.conv2 = _DepthwiseSeparableConv3d(out_ch, out_ch, kernel_size=dw_kernel_size)
        else:
            self.conv1 = _NonCausalConv3d(in_channels, out_ch, kernel_size=3)
            self.conv2 = _NonCausalConv3d(out_ch, out_ch, kernel_size=3)
        self.norm2 = _ChannelRMSNorm(out_ch, eps=1e-8, elementwise_affine=elementwise_affine)
        if in_channels != out_ch:
            self.norm3 = _ChannelRMSNorm(in_channels, eps=eps, elementwise_affine=elementwise_affine)
            self.conv_shortcut = _NonCausalConv3d(in_channels, out_ch, kernel_size=1)
        else:
            self.norm3 = None
            self.conv_shortcut = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.norm1(inputs)
        h = self.replace_nonlinearity(h) if self.is_upsampler_modified else self.nonlinearity(h)
        h = self.conv1(h)
        h = self.conv2(self.nonlinearity(self.norm2(h)))
        sc = inputs
        if self.norm3 is not None:
            sc = self.norm3(sc)
        if self.conv_shortcut is not None:
            sc = self.conv_shortcut(sc)
        return h + sc


class _WanResample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        out_dim = dim // 2 if upsample_out_dim is None else upsample_out_dim
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                ops.Conv2d(dim, out_dim, 3, padding=1),
            )
            self.time_conv = _NonCausalConv3d(dim, dim * 2, (3, 1, 1)) if mode == "upsample3d" else None
        else:
            self.resample = nn.Identity()
            self.time_conv = None

    def forward(self, x: torch.Tensor, is_first_chunk: bool = True) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d" and self.time_conv is not None:
            x = self.time_conv(x)
            x = x.view(b, 2, c, t, h, w).permute(0, 2, 3, 1, 4, 5).reshape(b, c, t * 2, h, w)
            if is_first_chunk:
                x = x[:, :, 1:]
        t_cur = x.shape[2]
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t_cur, c, h, w)
        if isinstance(self.resample, nn.Sequential):
            up = self.resample[0](x_2d.float()).to(x.dtype)
            x_2d = self.resample[1](up)
        return x_2d.view(b, t_cur, -1, x_2d.shape[-2], x_2d.shape[-1]).permute(0, 2, 1, 3, 4).contiguous()


class _TurboMidBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        eps: float,
        is_dw_conv: bool,
        dw_kernel_size: int,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                _TurboResNetBlock3d(
                    in_channels,
                    in_channels,
                    eps=eps,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                    elementwise_affine=elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resnets:
            x = block(x)
        return x


class _TurboUpBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        eps: float,
        spatio_temporal_scale: bool,
        is_dw_conv: bool,
        dw_kernel_size: int,
        spatio_only: bool,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self.conv_in = (
            _TurboResNetBlock3d(
                in_channels,
                out_channels,
                eps=eps,
                is_dw_conv=is_dw_conv,
                dw_kernel_size=dw_kernel_size,
                elementwise_affine=elementwise_affine,
            )
            if in_channels != out_channels
            else None
        )
        self.upsamplers = (
            nn.ModuleList(
                [
                    _WanResample(
                        dim=out_channels,
                        mode="upsample2d" if spatio_only else "upsample3d",
                        upsample_out_dim=out_channels,
                    )
                ]
            )
            if spatio_temporal_scale
            else None
        )
        self.resnets = nn.ModuleList(
            [
                _TurboResNetBlock3d(
                    out_channels,
                    out_channels,
                    eps=eps,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                    is_upsampler_modified=spatio_temporal_scale,
                    elementwise_affine=elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, is_first_chunk: bool) -> torch.Tensor:
        if self.conv_in is not None:
            x = self.conv_in(x)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x, is_first_chunk=is_first_chunk)
        for block in self.resnets:
            x = block(x)
        return x


class _TurboDecoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 48,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 384, 384),
        spatio_temporal_scaling: tuple[bool, ...] = (True, True, True, False),
        layers_per_block: tuple[int, ...] = (2, 2, 2, 2, 2),
        patch_size: int = 2,
        resnet_norm_eps: float = 1e-6,
        decoder_is_dw_conv: tuple[bool, ...] = (False, False, False, False, False),
        decoder_dw_kernel_size: int = 3,
        spatio_only: tuple[bool, ...] = (False, True, False, False),
        use_unpatchify: bool = False,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.use_unpatchify = use_unpatchify

        rev_ch = tuple(reversed(block_out_channels))
        rev_st = tuple(reversed(spatio_temporal_scaling))
        rev_layers = tuple(reversed(layers_per_block))
        rev_dw = tuple(reversed(decoder_is_dw_conv))
        rev_sp = tuple(reversed(spatio_only))

        cur_ch = rev_ch[0]
        self.conv_in = _NonCausalConv3d(in_channels, cur_ch, kernel_size=3)
        self.mid_block = _TurboMidBlock3d(
            cur_ch, rev_layers[0], resnet_norm_eps, rev_dw[0], decoder_dw_kernel_size,
            elementwise_affine=elementwise_affine,
        )
        self.up_blocks = nn.ModuleList()
        for i, next_ch in enumerate(rev_ch):
            self.up_blocks.append(
                _TurboUpBlock3d(
                    cur_ch,
                    next_ch,
                    rev_layers[i + 1],
                    resnet_norm_eps,
                    rev_st[i],
                    rev_dw[i + 1],
                    decoder_dw_kernel_size,
                    rev_sp[i],
                    elementwise_affine=elementwise_affine,
                )
            )
            cur_ch = next_ch

        if not use_unpatchify:
            self.norm_up_1 = _ChannelRMSNorm(cur_ch, eps=1e-8, elementwise_affine=False)
            self.upsampler2d_1 = _Conv2dSplitUpsampler(cur_ch, kernel_size=3, stride=2)
            cur_ch = cur_ch // 4
        else:
            self.norm_up_1 = None
            self.upsampler2d_1 = None

        self.conv_act = nn.SiLU()
        conv_out_ch = out_channels * patch_size * patch_size if (use_unpatchify and patch_size >= 2) else out_channels
        self.conv_out = _NonCausalConv3d(cur_ch, conv_out_ch, kernel_size=3)

    def forward(self, x: torch.Tensor, is_first_chunk: bool) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x))
        for up in self.up_blocks:
            x = up(x, is_first_chunk=is_first_chunk)
        if not self.use_unpatchify and self.norm_up_1 is not None and self.upsampler2d_1 is not None:
            x = self.conv_act(self.norm_up_1(x))
            x = torch.stack([self.upsampler2d_1(x[:, :, t]) for t in range(x.shape[2])], dim=2)
        var = x.float().pow(2).mean(dim=1, keepdim=True)
        x = (x.float() * torch.rsqrt(var + 1e-8)).to(x.dtype)
        x = self.conv_out(self.conv_act(x))
        if self.use_unpatchify:
            x = _turbo_unpatchify(x, self.patch_size)
        return x


class Magi2TurboVAEDecoder(nn.Module):
    """Distilled non-causal 7-frame sliding-window Wan2.2 video VAE decoder.

    Accepts unnormalized Wan2.2 latents `z` (matching `ldm/wan/vae2_2.py:WanVAE.decode`,
    since `latent_formats.Wan22.process_out` already applied `z * std + mean`).
    """

    def __init__(
        self,
        latent_channels: int = 48,
        out_channels: int = 3,
        decoder_block_out_channels: tuple[int, ...] = (128, 256, 384, 384),
        decoder_layers_per_block: tuple[int, ...] = (2, 2, 2, 2, 2),
        decoder_spatio_temporal_scaling: tuple[bool, ...] = (True, True, True, False),
        patch_size: int = 2,
        resnet_norm_eps: float = 1e-6,
        decoder_is_dw_conv: tuple[bool, ...] = (False, False, False, False, False),
        decoder_dw_kernel_size: int = 3,
        decoder_spatio_only: tuple[bool, ...] = (False, True, False, False),
        first_chunk_size: int = 3,
        step_size: int = 5,
        temporal_compression_ratio: int = 4,
        use_unpatchify: bool = False,
        elementwise_affine: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.decoder = _TurboDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=tuple(decoder_block_out_channels),
            spatio_temporal_scaling=tuple(decoder_spatio_temporal_scaling),
            layers_per_block=tuple(decoder_layers_per_block),
            patch_size=patch_size,
            resnet_norm_eps=resnet_norm_eps,
            decoder_is_dw_conv=tuple(decoder_is_dw_conv),
            decoder_dw_kernel_size=decoder_dw_kernel_size,
            spatio_only=tuple(decoder_spatio_only),
            use_unpatchify=use_unpatchify,
            elementwise_affine=elementwise_affine,
        )
        self.first_chunk_size = first_chunk_size
        self.step_size = step_size
        self.temporal_compression_ratio = temporal_compression_ratio

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "Magi2TurboVAEDecoder is a distilled decoder-only checkpoint; use Wan2.2_VAE for encoding."
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_time = z.ndim == 4
        if squeeze_time:
            z = z.unsqueeze(2)

        first_chunk = self.first_chunk_size
        step = self.step_size
        overlap_px = self.temporal_compression_ratio
        num_frames = z.shape[2]

        pad_frames = 0
        if num_frames < first_chunk:
            pad_frames = first_chunk - num_frames
        elif (num_frames - first_chunk) % step != 0:
            pad_frames = step - (num_frames - first_chunk) % step

        if pad_frames > 0:
            z = torch.cat((z, z[:, :, -1:].repeat(1, 1, pad_frames, 1, 1)), dim=2)
            num_frames += pad_frames

        if num_frames == first_chunk:
            out = self.decoder(z, is_first_chunk=True)
        else:
            chunks = [self.decoder(z[:, :, : first_chunk + 1], is_first_chunk=True)[:, :, :-overlap_px]]
            for start in range(first_chunk, num_frames, step):
                is_last = start + step == num_frames
                right = start + step if is_last else start + step + 1
                decoded = self.decoder(z[:, :, start - 1 : right], is_first_chunk=False)
                chunks.append(
                    decoded[:, :, overlap_px:] if is_last else decoded[:, :, overlap_px:-overlap_px]
                )
            out = torch.cat(chunks, dim=2)

        if pad_frames > 0:
            out = out[:, :, : -pad_frames * self.temporal_compression_ratio]
        if squeeze_time:
            out = out.squeeze(2)
        return out
