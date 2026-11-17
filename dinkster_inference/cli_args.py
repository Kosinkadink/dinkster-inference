import argparse
import enum
import dinkster_inference.options


parser = argparse.ArgumentParser()


cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--cuda-malloc", action="store_true", help="Enable cudaMallocAsync (enabled by default for torch 2.0 and up).")


fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32 (If this makes your GPU work better please report it).")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Run the diffusion model in fp32.")
fpunet_group.add_argument("--fp64-unet", action="store_true", help="Run the diffusion model in fp64.")
fpunet_group.add_argument("--bf16-unet", action="store_true", help="Run the diffusion model in bf16.")
fpunet_group.add_argument("--fp16-unet", action="store_true", help="Run the diffusion model in fp16")
fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn.")
fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2.")
fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store unet weights in fp8_e8m0fnu.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16, might cause black images.")
fpvae_group.add_argument("--fp32-vae", action="store_true", help="Run the VAE in full precision fp32.")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true", help="Store text encoder weights in fp8 (e4m3fn variant).")
fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true", help="Store text encoder weights in fp8 (e5m2 variant).")
fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16.")
fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32.")
fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store text encoder weights in bf16.")

parser.add_argument("--fp16-intermediates", action="store_true", help="Experimental: Use fp16 for intermediate tensors between nodes instead of fp32.")

parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format when inferencing the models.")

parser.add_argument("--supports-fp8-compute", action="store_true", help="ComfyUI will act like if the device supports fp8 compute.")
parser.add_argument("--enable-triton-backend", action="store_true", help="ComfyUI will enable the use of Triton backend in comfy-kitchen. Is disabled at launch by default.")
parser.add_argument("--disable-triton-backend", action="store_true", help="Force-disable the comfy-kitchen Triton backend, overriding --enable-triton-backend.")


cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument("--high-ram", action="store_true", help="Can improve performance slightly on high RAM or on systems where pagefile use is preferred over model loading.")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--use-split-cross-attention", action="store_true", help="Use the split cross attention optimization. Ignored when xformers is used.")
attn_group.add_argument("--use-quad-cross-attention", action="store_true", help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
attn_group.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use the new pytorch 2.0 cross attention function.")
attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")
attn_group.add_argument("--use-flash-attention", action="store_true", help="Use FlashAttention.")
attn_group.add_argument("--use-ck-attention", action="store_true", help="Use Comfy Kitchen attention.")

parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true", help="Force enable attention upcasting, please report if it fixes black images.")
upcast.add_argument("--dont-upcast-attention", action="store_true", help="Disable all upcasting of attention. Should be unnecessary except for debugging.")


vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--highvram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--lowvram", action="store_true", help="Doesn't do anything if dynamic vram is enabled. If dynamic vram isn't being used this option makes the text encoders run on the CPU.")
vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")

parser.add_argument("--reserve-vram", type=float, default=None, help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS.")

parser.add_argument("--async-offload", nargs='?', const=2, type=int, default=None, metavar="NUM_STREAMS", help="Use async weight offloading. An optional argument controls the amount of offload streams. Default is 2. Enabled by default on Nvidia.")
parser.add_argument("--disable-async-offload", action="store_true", help="Disable async weight offloading.")
parser.add_argument("--fast-disk", action="store_true", help="Force disk-backed dynamic loading and offload over unpinned RAM. Can be faster for users with fast NVME disks.")
parser.add_argument("--disable-fast-disk", action="store_true", help="Disable disk-backed dynamic loading and offload over unpinned RAM. Overrides --fast-disk.")
parser.add_argument("--disable-cuda-graphs", action="store_true", help="Disable CUDA graphs.")
parser.add_argument("--disable-comfy-compiler", action="store_true", help="Disable the Comfy model compiler, including its CUDA graph subfeature.")
parser.add_argument("--assert-graph-breaks", action="store_true", help="Fail on Comfy model compiler graph breaks.")

parser.add_argument("--force-non-blocking", action="store_true", help="Force ComfyUI to use non-blocking operations for all applicable tensors. This may improve performance on some non-Nvidia systems but can cause issues with some workflows.")

parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.")

parser.add_argument("--disable-smart-memory", action="store_true", help="Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can.")
parser.add_argument("--deterministic", action="store_true", help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

class PerformanceFeature(enum.Enum):
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"
    AutoTune = "autotune"

parser.add_argument("--fast", nargs="*", type=PerformanceFeature, help="Enable some untested and potentially quality deteriorating optimizations. This is used to test new features so using it might crash your comfyui. --fast with no arguments enables everything. You can pass a list specific optimizations if you only want to enable specific ones. Current valid optimizations: {}".format(" ".join(map(lambda c: c.value, PerformanceFeature))))


parser.add_argument("--disable-pinned-memory", action="store_true", help="Disable pinned memory use.")

parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap when loading ckpt/pt files.")
parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap when loading safetensors.")


parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")


if dinkster_inference.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])


if args.disable_comfy_compiler:
    args.disable_cuda_graphs = True

if args.force_fp16:
    args.fp16_unet = True


# '--fast' is not provided, use an empty set
if args.fast is None:
    args.fast = set()
# '--fast' is provided with an empty list, enable all optimizations
elif args.fast == []:
    args.fast = set(PerformanceFeature)
# '--fast' is provided with a list of performance features, use that list
else:
    args.fast = set(args.fast)
