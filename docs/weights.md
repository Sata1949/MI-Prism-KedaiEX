# DiT 权重：SFT 1.6 与 KEX 1.5

## KI8 1.6.0-rc1 · MOSS-Image-SFT · universal-a16

[分片与校验清单](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/kex-weights-v1.6rc1) · [固定的上游 revision](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9)

1.6 只量化 **SFT 主干**的线性权重。源文件 `moss_image_sft_bf16.safetensors`（10,164,664,256 B，SHA256 `b747ce4e6406b878c6ffaa21e81bffda279e173750b1cb546a2c277282c09a09`）的 463 个张量与实际打包使用的 Diffusers transformer 三分片逐字节一致。本包不含 KEX 的 clip7 微调、VAE 或外部文本编码器；打包不需要 `text_encoder`。

`universal-a16` 在 KI8 1.6 的候选权重格式中逐层选择，激活固定为 BF16（W8A16）。这次 269 层都选中了 Peak8，不是人为指定 `peak8-a16`。下方 1.5 节的 PSNR、耗时和工作流不作为本次 SFT 的实测结果。

| `inspect` 与独立 BF16 核对 | 实测 |
|---|---|
| 清单 / 覆盖 | `1.6.0-rc1`；269/269 个线性模块，标定 0，`peak8 / bf16` 269 个 |
| 主块 | 42 × 6 = 252 个投影；其余 17 个是主干内部的线性模块，含文本条件投影 |
| 原始矩阵形状（`[out, in]`） | 各块 `q/k/v/o` 均为 `[3072, 3072]`，`ffn_up_gate` 为 `[16384, 3072]`，`ffn_down` 为 `[3072, 8192]` |
| 源身份 | 全部 269 个矩阵的形状与原始 BF16 一致，清单中 269 个 `source_weight` 指纹吻合 |
| 被编码权重 | 5,081,923,584 个；元素码 + W32 尺度 5,240,733,696 B（8.25 bit/权重） |
| 逐层文件 | 269 个 `.safetensors`，每个包含 `elements` 和 `scales` 两个 U8 张量，共 538 个；文件合计 5,240,777,416 B |
| `manifest.json` | 136,458 B；SHA256 `750e03e15480bac7fcdcd6c49ca9571e17f9aeed4d4652d9bf225a6e9a69a15b` |
| `ki8-16-weightonly/` | 270 个文件，合计 **5,240,913,874 B**（含清单，不含 tar 头） |
| `ki8-16-universal-a16.tar` | **5,241,210,880 B**；SHA256 `e05c1845234dce27387da672e2ee69bb3533422c15450074ca5d1873f33019de` |

### 下载、重组与校验

使用新的空目录，避免混入别的版本或残片。分片编号 `000–039`：前 39 片各 134,217,728 B（128 MiB），末片 6,719,488 B。私仓用有读取权限的 `gh auth login`；不要把 token 写进命令或发到下载重定向的域名。

```bash
set -euo pipefail
export LC_ALL=C
mkdir sft-ki8-16 && cd sft-ki8-16
repo=Sata1949/MI-Prism-KedaiEX
tag=kex-weights-v1.6rc1
archive=ki8-16-universal-a16.tar
gh release download "$tag" --repo "$repo" -p "$archive.part*" -p SHA256SUMS
```

仓库**公开后**，也可在上述空目录用以下循环替代 `gh release download`（Git Bash / Linux / Colab，无需 token）：

```bash
base="https://github.com/$repo/releases/download/$tag"
curl -fLsS --retry 5 -O "$base/SHA256SUMS"
for i in $(seq 0 39); do
  n=$(printf '%03d' "$i")
  curl -fLsS --retry 5 -O "$base/$archive.part$n"
done
```

两种下载方式的后续相同；须看到校验全部为 `OK` 后再解包：

```bash
cat "$archive".part[0-9][0-9][0-9] > "$archive"
sha256sum -c SHA256SUMS       # 整个 tar 和 40 个分片
tar -xf "$archive"
sha256sum -c KI8-FILES.sha256 # 清单和 269 个逐层文件
```

### 加载：KI8 目录与 ComfyUI 单文件

**KI8 1.6.0-rc1 目录接口**：对同一 revision 已在 CPU 装好的原始浮点 `MOSSImageTransformer` 调用 `ki8_lora.install(dit, packed_dir)`，其中 `packed_dir` 是解包后的 `ki8-16-weightonly/`。目录只带线性层码流；归一化、bias、RoPE 等 194 个原始张量仍取自 SFT。已用 `install_codes` 在 CPU 装入所有 269 个投影、补齐 194 个张量，并抽测 3 个 BF16 投影前向；没有把这个结果当作出图测试。

**ComfyUI / `UNETLoader`**：它选择完整模型文件，不能选择 tar、`manifest.json` 或任一逐层文件。这个包的 269 层都是现有 KI8 ComfyUI codec 能识别的 Peak8/W32。先用下面的命令把 KI8 1.6 码流和原 SFT 未量化张量组成单个文件，再从 `UNETLoader` 选择 `moss_image_sft_ki8_16_universal_a16.safetensors`，`weight_dtype` 选 `default`。无需再次量化，也不必合并 q/k/v。宿主须已装支持 KI8 Peak8/BF16 的 codec 和对应的 ComfyUI 加载补丁；原版 ComfyUI 不够。此方法只针对上述已校验的清单，不推断其他 universal 包也能用同一 codec。

<details>
<summary>生成 UNETLoader 单文件（Python 3.11+、torch、safetensors）</summary>

先完成上面的分片与逐层校验，再设置 `SFT_BF16` 为原始 SFT 单文件路径、`COMFYUI_ROOT` 为已安装 KI8 codec 的 ComfyUI 根目录。需额外约 5.24 GB 磁盘；命令不会覆盖已有文件。

```bash
python - "$SFT_BF16" ki8-16-weightonly "$COMFYUI_ROOT/models/diffusion_models/moss_image_sft_ki8_16_universal_a16.safetensors" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

base, packed, out = map(Path, sys.argv[1:])
if out.exists():
    raise SystemExit(f"输出已存在：{out}")
with base.open("rb") as src:
    if hashlib.file_digest(src, "sha256").hexdigest() != "b747ce4e6406b878c6ffaa21e81bffda279e173750b1cb546a2c277282c09a09":
        raise SystemExit("需要指定 revision 对应的 SFT BF16 单文件，不能用 clip7")
manifest = (packed / "manifest.json").read_bytes()
if hashlib.sha256(manifest).hexdigest() != "750e03e15480bac7fcdcd6c49ca9571e17f9aeed4d4652d9bf225a6e9a69a15b":
    raise SystemExit("清单不是本次实测的 1.6 universal-a16 Release")
meta = json.loads(manifest)
layers = meta["layers"]
replaced = {n + ".weight" for n in layers}
with safe_open(str(base), framework="pt", device="cpu") as f:
    # 只复制未量化的 norm、bias、RoPE 等；在读取码流前释放原 BF16 映射。
    state = {k: f.get_tensor(k).clone() for k in f.keys() if k not in replaced}
for name, spec in layers.items():
    data = load_file(str(packed / spec["file"]))
    state[name + ".weight"] = data["elements"]
    state[name + ".weight_scale"] = data["scales"]
    # UNETLoader 需要逐层的布局标签。本包的 q/k/v 原本分开，无需融合或重新量化。
    tag = {"format": "ki8", "group_size": meta["weight_group"],
           "weight_format": spec["weight_format"], "activation_format": spec["activation_format"]}
    state[name + ".comfy_quant"] = torch.tensor(list(json.dumps(tag).encode("ascii")), dtype=torch.uint8)
save_file(state, str(out), metadata={"format":"pt", "ki8_version":meta["version"], "ki8_profile":"universal-a16"})
PY
```

</details>

在指定量化环境中，转换所得单文件为 **1,001 个张量、5,241,628,690 B**。已逐字节核对 269 对元素码/尺度、269 份布局标签及 194 个原始张量。这是容器校验，不是实际运行 `UNETLoader` 出图。

权重按 Apache-2.0 分发。归档内含 `LICENSE-APACHE-2.0`、`NOTICE` 和本次量化修改说明，转发权重时一并保留。旧实验文件的本地清理记录见 [README 权重段](../README.md#权重)。

## 上一版对照：KI8 1.5.0-rc3 KEX DiT — W8A16 (Peak8 / W32)

One checkpoint, split into 40 parts of 128 MiB (last part ~6.8 MiB) because GitHub
release assets are capped at 2 GiB.

| File | Format | Bytes | SHA256 |
|---|---|---:|---|
| `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors` | KI8 1.5.0-rc3 Peak8/W32, W8A16 | 5,241,596,371 | `7064ed51…e572` |

The KEX BF16 baseline is not distributed in this release; this is the shipped
KI8 1.5.0-rc3 weight.

### Colab / Linux download

私仓请先登录 `gh`，在空目录下载与该版本匹配的 40 片：

```bash
gh release download kex-weights-v1.5rc3 --repo Sata1949/MI-Prism-KedaiEX -p 'MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors.part*' -p SHA256SUMS
```

以下是公开仓库的分片循环和两种方式通用的重组命令：

```bash
BASE=https://github.com/Sata1949/MI-Prism-KedaiEX/releases/download/kex-weights-v1.5rc3
mkdir -p /content/kex && cd /content/kex

for i in $(seq 0 39); do
  n=$(printf "%03d" "$i")
  curl -fLsS --retry 5 -O "$BASE/MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors.part$n"
done

cat MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors.part* > MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors
sha256sum MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors
# 7064ed51dc966d4c01d7c0d1c16c0e242cc01217955040c116659164c4c3e572
```

`SHA256SUMS` in this release carries every part hash.

### Loading

Put the file in `ComfyUI/models/diffusion_models/`. The KI8 1.5.0-rc3 codec
loads it as **W8A16**: Peak8 weights decoded by `ki8_decode_w32`, BF16
activations. The CUDA patch for the KI8 runtime is
`patches/ki8-1.5.0-rc3-cuda.patch` in this repository; `UNETLoader` uses the
ComfyUI codec path.

### License

Derivative of MOSS-Image SFT, Apache-2.0. Keep `NOTICE` and the modification
notice (KEX tail + Peak8 quantization) with any redistribution. See
`LICENSING-LAYERS.md`.
