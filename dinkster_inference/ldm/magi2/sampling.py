from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.signal import resample
import torch
import torch.nn.functional as F

import dinkster_inference.model_sampling


def zero_snr_ddpm_sigmas(
    num_timesteps: int = 1000,
    linear_start: float = 0.00085,
    linear_end: float = 0.0120,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Zero-SNR DDPM sigma table matching SandAI's `ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)`."""
    betas = torch.linspace(
        linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=torch.float64
    ) ** 2
    alphas_cumprod = np.cumprod((1.0 - betas).numpy(), axis=0)
    ac_sqrt = torch.tensor(alphas_cumprod, dtype=torch.float32, device=device).sqrt()
    ac_0 = ac_sqrt[0].clone()
    ac_t = ac_sqrt[-1].clone()
    ac_sqrt = (ac_sqrt - ac_t) * (ac_0 / (ac_0 - ac_t))
    # get_sigmas flips once, then __call__(..., flip=True) flips back -> original ascending index order!
    return ac_sqrt


class Magi2RefinerModelSampling(
    dinkster_inference.model_sampling.ModelSamplingDiscreteFlow,
    dinkster_inference.model_sampling.CONST,
):
    """Flow-UniPC sampling for MAGI-2 Refiner with spherical Zero-SNR DDPM[220] initial re-noising."""

    def __init__(self, model_config=None) -> None:
        super().__init__(model_config)
        settings = model_config.sampling_settings if model_config is not None else {}
        noise_index = int(settings.get("refiner_noise_value", 220))
        sigmas_table = zero_snr_ddpm_sigmas(1000)
        self.refiner_noise_value = noise_index
        self.register_buffer("refiner_renoise_sigma", sigmas_table[noise_index].clone())

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if self.refiner_noise_value <= 0:
            return latent_image
        s_ref = self.refiner_renoise_sigma.to(device=noise.device, dtype=noise.dtype)
        return latent_image.to(noise.dtype) * s_ref + noise * torch.sqrt(1.0 - s_ref * s_ref)

    def inverse_noise_scaling(self, sigma, latent):
        return latent


def letterbox_reference_image(
    image_bhwc: torch.Tensor,
    target_height: int,
    target_width: int,
    vae_scale_factor: int = 32,
    resize_type: str = "original",
) -> torch.Tensor:
    """Letterbox an `[1, H, W, C]` RGB image in `[0, 1]` onto a white canvas matching MAGI-2 `resizepad`.

    Returns `[1, target_h, target_w, 3]` in `[0, 1]` ready for `Wan2.2_VAE.encode` in FP32.
    """
    img = image_bhwc[:1, ..., :3].float()
    _, src_h, src_w, _ = img.shape
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid reference image dimensions: {src_h}x{src_w}")

    if resize_type == "square":
        raw_h, raw_w = 320, 320
    else:
        max_side = max(target_height, target_width)
        if src_w > src_h:
            raw_w = max_side
            raw_h = max(1, int(src_h * max_side / src_w))
        else:
            raw_h = max_side
            raw_w = max(1, int(src_w * max_side / src_h))

    scale = min(raw_w / src_w, raw_h / src_h)
    fit_h = max(1, int(round(src_h * scale)))
    fit_w = max(1, int(round(src_w * scale)))

    resized = F.interpolate(
        img.movedim(-1, 1), size=(fit_h, fit_w), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    canvas = torch.ones((1, 3, raw_h, raw_w), device=img.device, dtype=torch.float32)
    top = (raw_h - fit_h) // 2
    left = (raw_w - fit_w) // 2
    canvas[:, :, top : top + fit_h, left : left + fit_w] = resized

    aligned_h = max(vae_scale_factor, (raw_h // vae_scale_factor) * vae_scale_factor)
    aligned_w = max(vae_scale_factor, (raw_w // vae_scale_factor) * vae_scale_factor)
    if (aligned_h, aligned_w) != (raw_h, raw_w):
        canvas = F.interpolate(
            canvas, size=(aligned_h, aligned_w), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
    return canvas.movedim(1, -1).contiguous()


def flow_unipc_sigmas(
    steps: int,
    shift: float = 7.0,
    num_train_timesteps: int = 1000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Exact sigma schedule of SandAI's `FlowUniPCMultistepScheduler.set_timesteps(steps, shift=shift)`."""
    sigma_max = 1.0 - (1.0 / float(num_train_timesteps))
    raw = np.linspace(sigma_max, 0.0, int(steps) + 1, dtype=np.float64)[:-1]
    shifted = (float(shift) * raw) / (1.0 + (float(shift) - 1.0) * raw)
    with_zero = np.concatenate([shifted, np.zeros((1,), dtype=np.float64)]).astype(np.float32)
    return torch.from_numpy(with_zero).to(device=device)


def sample_flow_unipc(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[dict] = None,
    callback=None,
    disable: Optional[bool] = None,
    solver_order: int = 2,
    solver_type: str = "bh2",
    lower_order_final: bool = True,
) -> torch.Tensor:
    """Rectified-Flow UniPC multistep predictor-corrector (`alpha_t = 1 - sigma_t`, `sigma_t = sigma_t`).

    Matches `MAGI-2-preview/inference/pipeline/sampler.py:FlowUniPCMultistepScheduler` step for step,
    operating on Comfy/Dinkster `CONST` denoiser `model(x, sigma)` which returns `x0_pred = x - sigma * v`.
    """
    extra = {} if extra_args is None else extra_args
    num_steps = len(sigmas) - 1
    if num_steps <= 0:
        return x

    s_in = x.new_ones((x.shape[0],))
    model_outputs: list[Optional[torch.Tensor]] = [None] * solver_order
    sigma_history: list[Optional[torch.Tensor]] = [None] * solver_order
    lower_order_nums = 0
    last_sample: Optional[torch.Tensor] = None
    this_order = 1

    def _uni_p_update(sample: torch.Tensor, step_idx: int, order: int) -> torch.Tensor:
        m0 = model_outputs[-1]
        assert m0 is not None
        sig_s0 = sigmas[step_idx].float()
        sig_t = sigmas[step_idx + 1].float()
        if float(sig_t.item()) <= 0.0:
            # Exact terminal flow step to sigma = 0 -> x_t = x0_pred (for order 1) or multistep limit
            return m0.to(sample.dtype)

        alpha_s0, alpha_t = 1.0 - sig_s0, 1.0 - sig_t
        lambda_s0 = torch.log(alpha_s0) - torch.log(sig_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sig_t)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            mi = model_outputs[-(i + 1)]
            sig_si = sigma_history[-(i + 1)]
            assert mi is not None and sig_si is not None
            lambda_si = torch.log(1.0 - sig_si) - torch.log(sig_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi.float() - m0.float()) / rk)
        rks.append(torch.ones_like(h))
        rks_t = torch.stack(rks)

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0
        B_h = hh if solver_type == "bh1" else h_phi_1

        R_rows = []
        b_vals = []
        fact = 1.0
        for i in range(1, order + 1):
            R_rows.append(torch.pow(rks_t, i - 1))
            b_vals.append(h_phi_k * fact / B_h)
            fact *= float(i + 1)
            h_phi_k = h_phi_k / hh - 1.0 / fact

        x_f = sample.float()
        x_t_base = (sig_t / sig_s0) * x_f - alpha_t * h_phi_1 * m0.float()
        if d1s:
            d1_stack = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], device=sample.device, dtype=torch.float32)
            else:
                R_mat = torch.stack(R_rows)[:-1, :-1]
                b_vec = torch.stack(b_vals)[:-1]
                rhos_p = torch.linalg.solve(R_mat, b_vec)
            pred_res = torch.einsum("k,bk...->b...", rhos_p, d1_stack)
            x_next = x_t_base - alpha_t * B_h * pred_res
        else:
            x_next = x_t_base
        return x_next.to(sample.dtype)

    def _uni_c_update(
        this_x0: torch.Tensor, prev_sample: torch.Tensor, step_idx: int, order: int
    ) -> torch.Tensor:
        m0 = model_outputs[-1]
        assert m0 is not None
        sig_s0 = sigmas[step_idx - 1].float()
        sig_t = sigmas[step_idx].float()
        alpha_s0, alpha_t = 1.0 - sig_s0, 1.0 - sig_t
        lambda_s0 = torch.log(alpha_s0) - torch.log(sig_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sig_t)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            mi = model_outputs[-(i + 1)]
            sig_si = sigma_history[-(i + 1)]
            assert mi is not None and sig_si is not None
            lambda_si = torch.log(1.0 - sig_si) - torch.log(sig_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi.float() - m0.float()) / rk)
        rks.append(torch.ones_like(h))
        rks_t = torch.stack(rks)

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0
        B_h = hh if solver_type == "bh1" else h_phi_1

        R_rows = []
        b_vals = []
        fact = 1.0
        for i in range(1, order + 1):
            R_rows.append(torch.pow(rks_t, i - 1))
            b_vals.append(h_phi_k * fact / B_h)
            fact *= float(i + 1)
            h_phi_k = h_phi_k / hh - 1.0 / fact

        if order == 1:
            rhos_c = torch.tensor([0.5], device=prev_sample.device, dtype=torch.float32)
        else:
            rhos_c = torch.linalg.solve(torch.stack(R_rows), torch.stack(b_vals))

        x_t_base = (sig_t / sig_s0) * prev_sample.float() - alpha_t * h_phi_1 * m0.float()
        corr_res = (
            torch.einsum("k,bk...->b...", rhos_c[:-1], torch.stack(d1s, dim=1))
            if d1s
            else 0.0
        )
        d1_t = this_x0.float() - m0.float()
        x_corr = x_t_base - alpha_t * B_h * (corr_res + rhos_c[-1] * d1_t)
        return x_corr.to(prev_sample.dtype)

    sample = x
    for step_idx in range(num_steps):
        sig_cur = sigmas[step_idx]
        x0_pred = model(sample, sig_cur * s_in, **extra)

        if step_idx > 0 and last_sample is not None:
            sample = _uni_c_update(x0_pred, last_sample, step_idx, this_order)

        if callback is not None:
            callback(
                {
                    "x": sample,
                    "i": step_idx,
                    "sigma": sig_cur,
                    "sigma_hat": sig_cur,
                    "denoised": x0_pred,
                }
            )

        for i in range(solver_order - 1):
            model_outputs[i] = model_outputs[i + 1]
            sigma_history[i] = sigma_history[i + 1]
        model_outputs[-1] = x0_pred
        sigma_history[-1] = sig_cur.float()

        max_order = min(solver_order, num_steps - step_idx) if lower_order_final else solver_order
        this_order = min(max_order, lower_order_nums + 1)

        last_sample = sample
        sample = _uni_p_update(sample, step_idx, this_order)
        if lower_order_nums < solver_order:
            lower_order_nums += 1

    return sample


def resample_magi2_audio(waveform_bct: torch.Tensor, ratio: float = 441.0 / 512.0) -> torch.Tensor:
    """Sinc-resample decoded Oobleck `[B, C, samples]` audio by `441 / 512` to align 25 Hz latents to 44.1 kHz."""
    if math.isclose(ratio, 1.0):
        return waveform_bct
    src_len = waveform_bct.shape[-1]
    dst_len = max(1, int(src_len * ratio))
    arr = waveform_bct.detach().cpu().float().numpy()
    resampled = resample(arr, dst_len, axis=-1)
    return torch.from_numpy(resampled).to(device=waveform_bct.device, dtype=waveform_bct.dtype)
