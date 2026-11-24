"""End-to-end KedaiEX sampling with the true BF16 KEX baseline (clip7).

clip7 = moss_image_sft_bf16 with the KEX dense tail (layers.41.proj.o,
layers.41.proj.ffn_down, final_linear_video) replaced, and per-layer q/k/v fused
into one qkv tensor. Verified bit-exact against its manifest.

Arms (same noise / cond / steps):
  bf16  all tensors from clip7, CPU (does not fit in 8 GB VRAM)
  v14   deployed MI-DIT-Prism-KedaiEX1_ki8_14 e2m5 codes, BF16 activations
  v15   Peak8 packed directly from clip7 BF16, BF16 activations
  ki6   KI6 v3 26-byte groups packed directly from clip7 BF16, BF16 activations

All quantized arms take non-quantized tensors (norms, modulation, rope, biases)
from clip7, so only the weight codec differs between them and the reference.
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_read  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["bf16", "v14", "v15", "ki6", "int8", "int8a8"], required=True)
p.add_argument("--clip7", type=Path, required=True)
p.add_argument("--deployed", type=Path)
p.add_argument("--ki8", type=Path, required=True)
p.add_argument("--ki6", type=Path)
p.add_argument("--transformer", type=Path, required=True)
p.add_argument("--cond", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--latent", type=int, default=16)
p.add_argument("--steps", type=int, default=8)
p.add_argument("--seed", type=int, default=20261123)
p.add_argument("--device", default="cuda:0")
args = p.parse_args()
sys.path.insert(0, str(args.ki8))
if args.ki6 is not None:
    sys.path.insert(0, str(args.ki6))

VAE_DIR = Path(r"C:/Users/LENOVO/.zcode/workspace/07fc1b23-2cc7-4fb1-97f8-9a81dd2b3fc2/quant_test/work/vae")

spec = importlib.util.spec_from_file_location("moss_local", args.transformer / "pipeline_moss_image.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["moss_local"] = mod
spec.loader.exec_module(mod)
cfg = json.loads((args.transformer / "config.json").read_text())
with torch.device("meta"):
    model = mod.MOSSImageTransformer(**{k: v for k, v in cfg.items() if not k.startswith("_")})
linears = [n for n, m in model.named_modules() if type(m) is torch.nn.Linear]
linear_set = set(linears)
clip_keys = set(st_read.keys(args.clip7))
print("linears", len(linears), "clip7 tensors", len(clip_keys), flush=True)

# Non-quantized tensors come from clip7 for every arm (same bytes the BF16 arm sees).
t0 = time.time()
stash = {}
for k in clip_keys:
    owner, _, leaf = k.rpartition(".")
    if args.arm != "bf16" and owner in linear_set and leaf == "weight":
        continue
    if owner.endswith(".proj.qkv") and leaf == "weight":
        continue  # fused qkv, split into q/k/v below
    stash[k] = st_read.read(args.clip7, k)
print(f"stash {len(stash)} tensors in {time.time()-t0:.1f}s", flush=True)

def bias_for(name):
    b = stash.get(name + ".bias")
    return None if b is None else b.to(torch.bfloat16)

def pack_arm(name, dense):
    bias = bias_for(name)
    if args.arm == "v15":
        from ki8_lora import formats
        e, s = formats.pack(dense, 32, "peak8")
        return dict(elements=e, scales=s, bias=bias, weight_format="peak8",
                    activation_format="bf16")
    if args.arm == "ki6":
        import ki6_v3
        return ki6_v3.encode(dense.float().numpy())
    if args.arm in ("int8", "int8a8"):
        from ki8_lora import formats
        e, s = formats.pack(dense, 32, "s8")
        return dict(elements=e, scales=s, bias=bias, weight_format="s8",
                    activation_format=("bf16" if args.arm == "int8" else "s8"))
    raise ValueError(args.arm)

if args.arm == "bf16":
    for i in range(42):
        w = st_read.read(args.clip7, f"layers.{i}.proj.qkv.weight")
        for j, part in enumerate(("q", "k", "v")):
            stash[f"layers.{i}.proj.{part}.weight"] = w[j * 3072:(j + 1) * 3072].clone()
        del w
    print(f"bf16 tensors {len(stash)} in {time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(stash, strict=False, assign=True)
else:
    packed = {}
    if args.arm == "v14":
        if args.deployed is None:
            raise SystemExit("--deployed is required for --arm v14")

        def read_dep(name):
            e = st_read.read(args.deployed, name + ".weight")
            s = st_read.read(args.deployed, name + ".weight_scale")
            return dict(elements=e, scales=s, bias=bias_for(name),
                        weight_format="e2m5", activation_format="bf16")
        for i in range(42):
            e = st_read.read(args.deployed, f"layers.{i}.proj.qkv.weight")
            s = st_read.read(args.deployed, f"layers.{i}.proj.qkv.weight_scale")
            for j, part in enumerate(("q", "k", "v")):
                packed[f"layers.{i}.proj.{part}"] = dict(
                    elements=e[j * 3072:(j + 1) * 3072].contiguous(),
                    scales=s[j * 3072:(j + 1) * 3072].contiguous(), bias=None,
                    weight_format="e2m5", activation_format="bf16")
            del e, s
            for suffix in ("o", "ffn_up_gate", "ffn_down"):
                name = f"layers.{i}.proj.{suffix}"
                packed[name] = read_dep(name)
            if i % 7 == 0:
                print("v14 layer", i, flush=True)
        for name in linears:
            if name in packed:
                continue
            packed[name] = read_dep(name)
    else:
        for i in range(42):
            w = st_read.read(args.clip7, f"layers.{i}.proj.qkv.weight")
            for j, part in enumerate(("q", "k", "v")):
                name = f"layers.{i}.proj.{part}"
                packed[name] = pack_arm(name, w[j * 3072:(j + 1) * 3072].contiguous())
            del w
            for suffix in ("o", "ffn_up_gate", "ffn_down"):
                name = f"layers.{i}.proj.{suffix}"
                dense = st_read.read(args.clip7, name + ".weight")
                packed[name] = pack_arm(name, dense)
                del dense
            print("packed layer", i, flush=True)
        for name in linears:
            if name in packed:
                continue
            dense = st_read.read(args.clip7, name + ".weight")
            packed[name] = pack_arm(name, dense)
            del dense
    print(f"{args.arm} codes {len(packed)} ready in {time.time()-t0:.1f}s", flush=True)

    if args.arm in ("v14", "v15", "int8", "int8a8"):
        from ki8_lora import install_codes
        install_codes(model, packed, dtype=torch.bfloat16, output_tile=128, token_tile=256,
                      fp32_output=("final_linear_video",))
        model.load_state_dict(stash, strict=False, assign=True)
    else:
        import torch_cpu
        for name, blocks in packed.items():
            old = model.get_submodule(name)
            layer = torch_cpu.PackedLinear(blocks, old.in_features, stash.get(name + ".bias"),
                                           row_tile=512, token_tile=512)
            layer.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16),
                                              requires_grad=False)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, layer)
        model.load_state_dict(stash, strict=False, assign=True)
    del packed

left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if t.device.type == "meta"]
print("meta left", len(left), flush=True)
if left:
    raise SystemExit("meta tensors remain: " + ", ".join(left[:8]))

dev = torch.device(args.device) if args.arm != "bf16" else torch.device("cpu")
t0 = time.time()
model = model.eval().to(dev)
print(f"moved to {dev} in {time.time()-t0:.1f}s", flush=True)

cond = torch.load(args.cond, map_location="cpu", weights_only=False)
emb = cond[0][0].to(torch.bfloat16)
mask = cond[0][1]["attention_mask"]
g = torch.Generator().manual_seed(args.seed)
lat = torch.randn(1, 128, args.latent, args.latent, generator=g, dtype=torch.float32).to(dev)
sch = mod.MOSSImageScheduler()
sch.set_timesteps(args.steps, device=dev)
times = []
with torch.inference_mode():
    for t in sch.timesteps:
        s0 = time.perf_counter()
        pred = model(lat, t, emb.to(dev), mask.to(dev)).sample.float()
        lat = sch.step(pred, t, lat, return_dict=False)[0]
        times.append(time.perf_counter() - s0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0
    lat_cpu = lat.detach().cpu()
del model
torch.cuda.empty_cache()

from diffusers import AutoencoderKLFlux2
from diffusers.image_processor import VaeImageProcessor
vdev = torch.device("cuda:0")
vae = AutoencoderKLFlux2.from_pretrained(str(VAE_DIR), torch_dtype=torch.float32,
                                         local_files_only=True, low_cpu_mem_usage=True).eval().to(vdev)
with torch.inference_mode():
    z = lat_cpu.to(vdev).to(vae.dtype)
    bn = vae.bn; eps = vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    img = VaeImageProcessor(vae_scale_factor=16).postprocess(
        vae.decode(z, return_dict=False)[0].clamp(-1, 1), output_type="np")[0]

args.out.parent.mkdir(parents=True, exist_ok=True)
np.save(str(args.out.with_suffix(".npy")), img)
np.save(str(args.out.with_name(args.out.stem + "_latent.npy")), lat_cpu.numpy())
from PIL import Image
Image.fromarray((img * 255).round().astype("uint8")).save(str(args.out.with_suffix(".png")))
args.out.with_suffix(".json").write_text(json.dumps({
    "arm": args.arm, "source": str(args.clip7), "steps": args.steps, "latent": args.latent,
    "seed": args.seed, "step_seconds": times, "total_seconds": sum(times),
    "cuda_peak_allocated_mib": peak, "device": str(dev)}, indent=1))
print("done", args.out, f"{sum(times):.2f}s peak {peak:.0f} MiB", flush=True)
