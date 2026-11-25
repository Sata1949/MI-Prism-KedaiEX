# ComfyUI 原生 CUDA 解码（Peak8 / W8A16）

ComfyUI 默认用 `comfy/ki8_codec.py` 的纯 torch 路径解 Peak8 权重。3072×3072 的一层要 184 ms，269 层每步重复一遍。

这个补丁把解码换到 `peak_optimized.ptx` 的 `ki8_decode_w32`：一个 warp 处理一个 32 权重组，整数归约找峰值和奇偶校验，整层一次内核发射。

## 文件

| 文件 | 作用 |
|---|---|
| `comfy/ki8_peak_cuda.py` | ctypes 加载 PTX、建解码 LUT、`unpack(codes, heads, 32)` |
| `comfy/peak_optimized.ptx` | 编译好的 `ki8_decode_w32` / `ki8_select` |
| `comfy/ki8_codec.py` | `_decode_peak8` 在 `group=32`、`dtype=bf16`、CUDA 时先走原生 |

`install_moss.py` 的 step 7 会把这三个文件复制进 ComfyUI；已有文件内容相同就跳过，不同则先备份。

## 行为

- 只接管 **group 32 + BF16 + CUDA** 的权重解码，也就是 W8A16 的发行路径。
- `float32`、CPU、group 128（A8 激活）继续走 torch。
- 首次加载做一次 7×32 和 7×100 的自检，和 torch 输出逐位比较；不一致就禁用原生并打 warning。
- `KI8_PEAK_CUDA=0` 可强制关闭。
- 驱动需要 SM 8.0 以上，PTX 由驱动 JIT；PTX 加载失败会回退 torch，不会中断出图。

## 实测

RTX 3070 Ti Laptop 8 GB，torch 2.9.1+cu130：

| 场景 | 纯 torch | 原生 W32 |
|---|---:|---:|
| 3072×3072 单层解码（warm） | 184 ms | 1.0 ms |
| 1024×1024 / 12 步整图 | 205.8 s | **60–77 s** |
| TE（27B，GPU 99 层） | 19.1 s | 19.1 s |
| DiT + VAE | ~186 s | **~41.5 s** |

两次输出逐像素相同：PSNR 999、max diff 0。

## 验证

```bash
python - <<'PY'
import sys, torch
sys.path.insert(0, r"D:/moss-image/ComfyUI")
from safetensors import safe_open
from comfy.ki8_codec import _decode_peak8
p = r"D:/moss-image/ComfyUI/models/diffusion_models/MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors"
with safe_open(p, framework="pt", device="cpu") as f:
    e = f.get_tensor("layers.0.proj.o.weight").cuda()
    s = f.get_tensor("layers.0.proj.o.weight_scale").cuda()
out = _decode_peak8(e, s, 32, torch.bfloat16)
print(out.shape, out.dtype, float(out.float().abs().mean()))
PY
```

输出 `torch.Size([3072, 3072]) torch.bfloat16`，耗时约 1 ms 即为原生路径。
