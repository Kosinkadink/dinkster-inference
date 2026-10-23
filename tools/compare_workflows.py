"""Compare deterministic library workflows in isolated upstream and fork processes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SD_CASES = ("sd15", "lora", "hook_keyframes", "controlnet_range", "context_windows")
PROMPT = "A red fox beside a blue ceramic teapot, detailed photograph, soft morning light"
NEGATIVE = "blurry, distorted, low quality"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_arrays(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"Array contract differs: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite workflow output")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def run_worker(args):
    if args.side == "upstream":
        sys.path.insert(0, str(args.upstream))
        # A site-installed regular comfy alias otherwise beats upstream's namespace package.
        spec = importlib.machinery.PathFinder.find_spec("comfy", [str(args.upstream)])
        if spec is None:
            raise ValueError("Upstream checkout has no comfy package")
        sys.modules["comfy"] = importlib.util.module_from_spec(spec)
        package = "comfy"
    else:
        package = "dinkster_inference"
    cli = importlib.import_module(package + ".cli_args")
    cli.args.deterministic = True
    cli.args.disable_dynamic_vram = True
    cli.args.use_pytorch_cross_attention = True
    modules = {name: importlib.import_module(package + "." + name) for name in
               ("sd", "sample", "hooks", "controlnet", "context_windows", "utils", "model_management", "nested_tensor")}
    for module in modules.values():
        is_reference = Path(module.__file__).resolve().is_relative_to(args.upstream)
        if is_reference != (args.side == "upstream"):
            raise ValueError(f"Worker imported the wrong source: {module.__file__}")
    sd, sample, hooks = modules["sd"], modules["sample"], modules["hooks"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1064 if args.case == "h3" else 459)
    started = time.monotonic()
    if args.case == "h3":
        model = sd.load_diffusion_model(str(args.h3_model))
        clip = sd.load_clip([str(args.h3_clip)], clip_type=sd.CLIPType.MINIMAX)
        vae = sd.VAE(sd=modules["utils"].load_torch_file(str(args.h3_vae)))
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
        negative = []
        # Upstream EmptyMiniMaxH3LatentAV: 124 frames -> 37 video tokens, 207 audio tokens.
        latent = modules["nested_tensor"].NestedTensor((
            torch.zeros((1, 24, 37, 48, 84)), torch.zeros((1, 32, 2, 207))))
        seed, steps, cfg = 1064, 20, 6.0
    else:
        checkpoint = args.flux_checkpoint if args.case == "flux" else args.checkpoint
        model, clip, vae, _ = sd.load_checkpoint_guess_config(str(checkpoint))
        if args.case == "lora":
            lora = modules["utils"].load_torch_file(str(args.lora), safe_load=True)
            model, clip = sd.load_lora_for_models(model, clip, lora, 0.8, 0.8)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(NEGATIVE))
        if args.case == "hook_keyframes":
            lora = modules["utils"].load_torch_file(str(args.lora), safe_load=True)
            group = hooks.create_hook_lora(lora, strength_model=0.8, strength_clip=0.0)
            keyframes = hooks.HookKeyframeGroup()
            for index in range(5):
                keyframes.add(hooks.HookKeyframe(strength=index / 4, start_percent=index / 8,
                                                guarantee_steps=1 if index == 0 else 0))
            group.set_keyframes_on_hooks(keyframes)
            positive = hooks.set_hooks_for_conditioning(positive, group)
            negative = hooks.set_hooks_for_conditioning(negative, group)
        if args.case == "controlnet_range":
            control = modules["controlnet"].load_controlnet(str(args.controlnet))
            # An asymmetric deterministic edge map, shared by conditional/unconditional paths.
            hint = torch.zeros((1, 3, 512, 512))
            hint[:, :, 80:440, 95:100] = 1
            hint[:, :, 170:175, 95:380] = 1
            control = control.copy().set_cond_hint(hint, 1.0, (0.0, 0.6))
            for conditioning in (positive, negative):
                for entry in conditioning:
                    entry[1] = {**entry[1], "control": control, "control_apply_to_uncond": False}
        if args.case == "context_windows":
            windows = modules["context_windows"]
            model = model.clone()
            model.model_options["context_handler"] = windows.IndexListContextHandler(
                context_schedule=windows.get_matching_context_schedule("standard_static"),
                fuse_method=windows.get_matching_fuse_method("pyramid"),
                context_length=16, context_overlap=4, context_stride=1, dim=0)
            windows.create_prepare_sampling_wrapper(model)
        latent = torch.zeros((32 if args.case == "context_windows" else 1, 4, 64, 64))
        if args.case == "flux":
            latent = sample.fix_empty_latent_channels(model, latent)
            positive = [[value, {**metadata, "guidance": 3.5}] for value, metadata in positive]
        seed, steps, cfg = 459, 4 if args.case == "flux" else 20, 1.0 if args.case == "flux" else 7.0
    result = sample.sample(model, sample.prepare_noise(latent, seed), steps, cfg,
                           "euler", "normal", positive, negative, latent, seed=seed, disable_pbar=True)
    tensors = result.unbind() if result.is_nested else [result]
    arrays = {f"latent_{index}": value.detach().cpu().float().numpy() for index, value in enumerate(tensors)}
    images = vae.decode(tensors[0]).detach().cpu().numpy()
    if not np.isfinite(images).all():
        raise ValueError("Non-finite decoded images")
    images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    arrays["images"] = images
    directory = args.output_dir / args.case / args.side
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "outputs.npz", **arrays)
    image_hashes = []
    for index, image in enumerate(images):
        path = directory / f"image-{index:04d}.png"
        Image.fromarray(image).save(path)
        image_hashes.append(file_sha256(path))
    latent_hash = hashlib.sha256()
    for name, array in arrays.items():
        if name.startswith("latent_"):
            latent_hash.update(array.tobytes())
    metadata = {
        "side": args.side, "case": args.case, "module_file": modules["sd"].__file__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "seed": seed, "steps": steps, "cfg": cfg, "sampler": "euler", "scheduler": "normal",
        "prompt": PROMPT, "negative_prompt": NEGATIVE, "seconds": time.monotonic() - started,
        "latent_shapes": [list(array.shape) for name, array in arrays.items() if name.startswith("latent_")],
        "latent_sha256": latent_hash.hexdigest(), "image_sha256": image_hashes,
        "packages": sorted((item.metadata["Name"], item.version) for item in importlib.metadata.distributions()),
    }
    if args.side == "fork":
        metadata["installed_from"] = importlib.metadata.distribution("dinkster-inference").read_text("direct_url.json")
    (directory / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_comparison(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if reference != args.reference:
        raise ValueError(f"Upstream checkout is {reference}, expected {args.reference}")
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True):
        raise ValueError("Upstream checkout must be clean")
    cases = args.cases or [*SD_CASES, "h3" if args.h3_model else "flux"]
    summary = {"reference": reference, "started_at": datetime.now(timezone.utc).isoformat(), "inputs": {}, "cases": {}}
    paths = ("checkpoint", "lora", "controlnet", "h3_model", "h3_clip", "h3_vae", "flux_checkpoint")
    for name in paths:
        path = getattr(args, name)
        if path is not None and path.is_file():
            summary["inputs"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    for case in cases:
        required = ("h3_model", "h3_clip", "h3_vae") if case == "h3" else (
            ("flux_checkpoint",) if case == "flux" else ("checkpoint",))
        if case in ("lora", "hook_keyframes"):
            required += ("lora",)
        if case == "controlnet_range":
            required += ("controlnet",)
        missing = [name for name in required if name not in summary["inputs"]]
        if missing:
            summary["cases"][case] = {"status": "unavailable", "reason": "Missing model files: " + ", ".join(missing)}
            continue
        record = {}
        for side in ("upstream", "fork"):
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", "--side", side,
                       "--case", case, "--upstream", str(args.upstream), "--output-dir", str(args.output_dir)]
            for name in paths:
                path = getattr(args, name)
                if path is not None:
                    command.extend(["--" + name.replace("_", "-"), str(path)])
            log = args.output_dir / f"{case}-{side}.log"
            with log.open("w") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                         env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
            if process.returncode:
                record = {"status": "error", "side": side, "returncode": process.returncode, "log": str(log)}
                break
            record[side] = json.loads((args.output_dir / case / side / "result.json").read_text())
        else:
            with np.load(args.output_dir / case / "upstream/outputs.npz") as left, np.load(args.output_dir / case / "fork/outputs.npz") as right:
                if set(left.files) != set(right.files):
                    raise ValueError("Workflow output tensors differ")
                differences = {name: compare_arrays(left[name], right[name]) for name in left.files}
            record["latent_max_abs_difference"] = max(value for name, value in differences.items() if name.startswith("latent_"))
            record["image_max_abs_difference"] = differences["images"]
            record["status"] = "identical" if (
                not any(differences.values()) and record["upstream"]["latent_sha256"] == record["fork"]["latent_sha256"]
                and record["upstream"]["image_sha256"] == record["fork"]["image_sha256"]
                and record["upstream"]["torch"] == record["fork"]["torch"]) else "different"
        summary["cases"][case] = record
        (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(case + ": " + record["status"])  # noqa: T201
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(value["status"] == "identical" for value in summary["cases"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference", default="bc5f880dbdd43e426077f42ccec53bb577ffee9f")
    parser.add_argument("--output-dir", type=Path, required=True)
    for name in ("checkpoint", "lora", "controlnet", "h3-model", "h3-clip", "h3-vae", "flux-checkpoint"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--cases", nargs="+", choices=(*SD_CASES, "h3", "flux"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--side", choices=("upstream", "fork"), help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=(*SD_CASES, "h3", "flux"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.upstream = args.upstream.resolve()
    args.output_dir = args.output_dir.resolve()
    for name in ("checkpoint", "lora", "controlnet", "h3_model", "h3_clip", "h3_vae", "flux_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.worker:
        run_worker(args)
    else:
        raise SystemExit(run_comparison(args))


if __name__ == "__main__":
    main()
