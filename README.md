<div align="center">

<img src="docs/assets/kedai-logo.svg" alt="Kedai Team" width="460">

# MI-Prism-KedaiEX

KEX 5B 文生图 · 8 GB 笔记本 · KI8 1.5.0-rc3 W8A16

[Release](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/kex-weights-v1.5rc3) ·
[Weights](docs/weights.md) ·
[License](LICENSING-LAYERS.md) ·
[ComfyUI workflow](examples/moss27b_e2e24576_ki815.json)

</div>

![KEX 1.5 Peak8，1024×1024，12 步，cfg 2.5](docs/assets/kex-1024.png)

KEX（KedaiEX1）是在 [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) 上继续训练的 5B 文生图模型。这个仓库把它装进 8 GB 显存：KI8 1.5.0-rc3 用 Peak8 / W32 编码 269 个线性层，权重 8.25 bit，激活保持 BF16。256×256、8 步、同噪声下，输出对 KEX BF16 参考是 43.34 dB PSNR、LPIPS-Alex 0.00756。

---

## 目录

- [结果](#结果)
- [模型](#模型)
- [权重](#权重)
- [快速开始](#快速开始)
- [采样设置](#采样设置)
- [显存与速度](#显存与速度)
- [构图与样张](#构图与样张)
- [已知限制](#已知限制)
- [仓库结构](#仓库结构)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 结果

256×256 latent、8 步、seed 20261123、同一份 27B 条件缓存。BF16 参考为 clip7 全张量，量化臂的非量化张量也来自 clip7，差异只来自权重编解码。

| 指标 | KEX BF16（clip7） | KI8 1.5 Peak8 |
|---|---:|---:|
| RGB PSNR | 基准 | 43.34 dB |
| LPIPS-Alex ↓ | 0 | 0.00756 |
| LPIPS-Squeeze ↓ | 0 | 0.00663 |
| latent PSNR | 基准 | 21.56 dB |
| latent 相关 | 1.0 | 0.9964 |
| 最大像素差 | 0 | 18 / 255 |
| 步时 | 71.91 s（CPU） | 2.999 s |
| 8 步总时 | 575.28 s | 25.15 s |
| 峰值显存 | — | 5071 MiB |

同码率（8.25 bit/w）、同权重、同噪声，Peak8 与 INT8 的对比：

| 臂 | 激活 | 步时 | RGB PSNR | LPIPS-Alex ↓ | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| KI8 1.5 Peak8 | BF16 | 2.999 s | 43.34 dB | 0.00756 | 5071 MiB |
| INT8 | W8A16 | 8.74 s | 38.72 dB | 0.01878 | 5073 MiB |
| INT8 | W8A8 | 10.99 s | 35.51 dB | 0.01663 | 5102 MiB |

Peak8 比 INT8 W8A16 高 4.62 dB，LPIPS 低 2.5 倍，步时短 2.9 倍。INT8 走逐 output tile 反量化再做 BF16 GEMM，没有专用内核。

权重和投影的 SQNR：

| 编解码 | bits/w | 权重 SQNR | W8A16 投影输出 SQNR |
|---|---:|---:|---:|
| e2m5 | 8.25 | 44.59 dB | 47.67 dB |
| Peak8 | 8.25 | 45.23 dB | 48.34 dB |
| KI6 v3 26B 组 | 6.5 | 35.38 dB | 39.97 dB |

单投影 SQNR 不能代替出图保真度。e2m5 的投影有 47.67 dB，8 步端到端只有 34.28 dB；Peak8 端到端 43.34 dB。发行口径以上面的出图数据为准。

---

## 模型

| 字段 | 值 |
|---|---|
| 名称 | MI-Prism-KedaiEX（KedaiEX1 / KEX） |
| 开发者 | sata1921、seeds、suihan、hilo114 |
| 基座 | [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)，Apache-2.0 |
| 架构 | 42 层 DiT，5B 主干；Qwen3.8-27B headless 文本编码器（5120→4096 adapter）；FLUX.2 small VAE |
| 文本条件 | 自然语言 prompt、tag 序列、JSON layout |
| 参考精度 | BF16（clip7），尾部三张量微调，qkv 融合 |
| 发行精度 | KI8 1.5.0-rc3 Peak8 / W32，W8A16，8.25 bit/w |
| 权重文件 | `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors`，5,241,596,371 B |
| 权重许可 | Apache-2.0，MOSS-Image 衍生，见 `NOTICE` |
| 代码许可 | MIT |
| 发行 | 2026-11，KI8 1.5.0-rc3 |

KEX 相对 MOSS-Image-SFT 只改尾部：替换 `layers.41.proj.o`、`layers.41.proj.ffn_down`、`final_linear_video` 三个张量，共 34,996,224 参数，并把每层 q/k/v 融成 `qkv`。其余张量与 SFT 同源。SFT 和 clip7 的逐张量 SQNR 只有 15.65–23.08 dB，微调幅度集中在这三个输出层。

---

## 权重

| 文件 | 格式 | 大小 | SHA256 |
|---|---|---:|---|
| `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors` | KI8 1.5.0-rc3 Peak8/W32，W8A16 | 5,241,596,371 B | `7064ed51…e572` |

Release `kex-weights-v1.5rc3` 里按 128 MiB 切成 40 片，单片小于 2 GiB。仓库是私有的，下载需要 `GITHUB_TOKEN`（repo 权限）：

```bash
BASE=https://github.com/Sata1949/MI-Prism-KedaiEX/releases/download/kex-weights-v1.5rc3
for i in $(seq 0 39); do
  n=$(printf "%03d" "$i")
  curl -L --fail -H "Authorization: token $GITHUB_TOKEN" \
    -O "$BASE/MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors.part$n"
done
cat MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors.part* > MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors
sha256sum MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors
```

逐片哈希和 `gh release download` 的写法在 [docs/weights.md](docs/weights.md)。KEX BF16 基线 10.16 GB，不随 Release 分发。

---

## 快速开始

### 一键安装

发行包里的 `install_moss.py` 处理 ComfyUI 侧的安装：拷 codec、打 `ops.py` 补丁、装节点包、应用 KI8 CUDA 补丁、检查 27B 与 DiT 权重，然后启动 ComfyUI 并排队工作流。

```bash
python install_moss.py --comfy <COMFYUI_DIR> --yes
```

跑完会打印图片路径。工作流本身已经实测：RTX 3070 Ti Laptop 8 GB，1024×1024、12 步、cfg 2.5，从排队到出图 **200.8–205.8 s**（两次实测），输出 `output/moss27b/e2e24576-ki815_00001_.png`。

### ComfyUI 手动

1. 安装 KI8 1.5.0-rc3 运行时和 ComfyUI 编解码/节点。运行时补丁在 [patches/ki8-1.5.0-rc3-cuda.patch](patches/ki8-1.5.0-rc3-cuda.patch)。
2. 把 `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors` 放进 `ComfyUI/models/diffusion_models/`。
3. 导入 [examples/moss27b_e2e24576_ki815.json](examples/moss27b_e2e24576_ki815.json)，改掉本机的 27B GGUF、`KedaiEX1-Adapter` 和 `first_token.npz` 路径。
4. 节点链路：

```
MossTE27BEncodePair → CONDITIONING
UNETLoader(ki8_15peak8) → KSampler(er_sde / simple) → VAEDecode → SaveImage
```

Peak8 由 ComfyUI codec 按 W8A16 装载，权重走 `ki8_decode_w32`，激活保持 BF16。

ComfyUI 目前的 Peak8 解码是纯 torch 实现，没有 LUT 和 decode cache，约 13.4 s/步。上面那个 200.8 s 就是这么跑出来的。KI8 runtime 的原生 CUDA 路径是 2.999 s/步，两者权重与输出一致。

### 命令行校验

```bash
P="python"                     # 先激活含 torch 2.9.1+cu128 的 venv
KI8="D:/moss-image/KI8-1.5.0-rc3"

KI8_REQUIRE_A16=1 "$P" "$KI8/run.py" check --device cuda:0 --require-a16

"$P" "$KI8/run.py" bench-speed --device cuda:0 --profile peak8-a16 \
  --in-features 4096 --out-features 4096 --tokens 512 \
  --token-tile 128 --output-tile 128 --iterations 5 --out work/speed_a16.json
```

`check --require-a16` 会现场打包 `347×511`（带尾组）和 `3072×9216`（真实投影）两个 `1.5.0-rc3` checkpoint，装回去后要求 `auto` 和 `reference` 逐位相同，并检查 `a16_decode == native-w32`。

### 复现保真度

```bash
C="<PATH>/moss_image_clip7_bf16.safetensors"
OUT=work; mkdir -p $OUT

KI8_REQUIRE_A16=1 "$P" -I bench/kex_bf16_arms.py --arm v15 --clip7 "$C" --ki8 "$KI8" \
  --transformer <MOSS_TRANSFORMER_DIR> --cond <COND_CACHE.pt> \
  --out $OUT/kex_w8a16/v15 --steps 8 --device cuda:0

"$P" -I bench/kex_bf16_arms.py --arm bf16 --clip7 "$C" --ki8 "$KI8" \
  --transformer <MOSS_TRANSFORMER_DIR> --cond <COND_CACHE.pt> \
  --out $OUT/kex_bf16/bf16 --steps 8

"$P" -I bench/score_psnr_lpips.py \
  --ref $OUT/kex_bf16/bf16.npy --test $OUT/kex_w8a16/v15.npy \
  --ref-latent $OUT/kex_bf16/bf16_latent.npy --test-latent $OUT/kex_w8a16/v15_latent.npy \
  --out $OUT/kex_w8a16/score_v15_vs_bf16.json
```

BF16 臂在 CPU 上跑，10.16 GB 放不进 8 GB 显存，8 步约 575 s。

---

## 采样设置

| 场景 | 分辨率 | 步数 | CFG | Sampler | 说明 |
|---|---|---|---:|---|---|
| 保真度探针 | 256×256 | 8 | — | MOSS scheduler | 同噪声比编解码误差 |
| 快速出图 | 1024×1024 | 12 | 2.5 | er_sde / simple | 发行工作流默认 |
| 视觉对比 | 1024×1024 | 25 | 2.5 | er_sde / simple | 细节和笔触更完整 |
| 文本排版 | 1024×1024 | 50 | 3.0 | Flow-matching Euler | MOSS-Image 官方默认 |

8 步只用于量化保真度。看画面用 25 步或 50 步。

---

## 显存与速度

参考机：RTX 3070 Ti Laptop 8 GB，Ryzen 7 6800H，16 GB DDR5。

| 项目 | 数值 |
|---|---:|
| 权重常驻 | 5.24 GB |
| 8 步采样峰值 | 5071 MiB |
| 8 步总时 | 25.15 s |
| 步时中位 | 2.999 s |
| BF16 基线步时（CPU） | 71.91 s |
| Peak8 单投影 4096×4096 / 512 tokens | 21.20 ms |
| e2m5 同形状 | 88.85 ms |

27B 文本编码器约 10 GB，和 DiT 不能同时驻留 8 GB 卡。链路必须先编码 cond、释放 TE，再加载 DiT 采样。

---

## 构图与样张

KEX 的微调集中在最后合成层。看四臂同 prompt 的 1024×1024 对照，变化主要在这些地方：

- 人物重心偏移。髋部有角度，头会轻微倾斜，站姿不是直上直下。
- 光线偏暖，逆光轮廓贴着发丝和肩背，海面高光收在人物附近。
- 皮肤高光偏暖，湿发和布料的反射更厚，暗部保留细节。
- 主体压在画面偏右的黄金分割位置，海岸线让位给人物。

这些差异可以量化。1024×1024、12 步、cfg 2.5、seed 20261121：

| 对 | RGB PSNR | LPIPS-Alex |
|---|---:|---:|
| KEX 1.5 Peak8 vs KEX 1.5 s8 | 31.00 dB | 0.0375 |
| 原生 INT8 SFT vs KEX 1.5 Peak8 | 24.43 dB | 0.2379 |

KEX 1.5 的两个码流差 0.0375 LPIPS-Alex，原生 SFT 和 KEX 差 0.24。构图差异来自 KEX 权重本身。

27B 编码器影响条件强度。等计算量下，27B 跑 4 步 cfg 2.0 的笔触量是 9B 跑 8 步 cfg 1.0 的 98%（3.820 vs 3.889），前向次数相同。

---

## 已知限制

- ComfyUI 的 Peak8 还没有原生 LUT，13.4 s/步；2.999 s/步属于 KI8 runtime 的 CUDA 路径。
- PSNR 和 LPIPS 来自单种子、单图，只作相对参考。
- 8 GB 卡上 TE 和 DiT 不能共存，必须两阶段。
- BF16 基线不随 Release 分发。复现 43.34 dB 需要自备 `moss_image_clip7_bf16.safetensors`。
- 私仓 Release 需要 token 才能下载。

---

## 仓库结构

```
MI-Prism-KedaiEX/
├── README.md
├── LICENSING-LAYERS.md
├── NOTICE
├── docs/
│   ├── weights.md
│   └── assets/
│       ├── kedai-logo.svg
│       └── kex-1024.png
├── examples/
│   └── moss27b_e2e24576_ki815.json
├── patches/
│   └── ki8-1.5.0-rc3-cuda.patch
├── bench/
│   ├── kex_bf16_arms.py
│   ├── score_psnr_lpips.py
│   └── st_read.py
├── configs/
├── phase1/
├── tools/
└── tests/
```

---

## 许可证

| 层 | 许可 | 覆盖 |
|---|---|---|
| 代码 | MIT | KI8 运行时、转换器、安装器、本仓库工具 |
| 权重 | Apache-2.0 | KEX DiT 衍生权重、GSQ 文本编码器 |
| ComfyUI 补丁 | GPL-3.0 | 装入 ComfyUI 后的 `ki8_codec.py`、`ops.py`、节点包 |
| VAE | Apache-2.0 | `flux2-vae_small` |

完整义务见 [LICENSING-LAYERS.md](LICENSING-LAYERS.md) 和 [docs/10-phase1-licensing.md](docs/10-phase1-licensing.md)。

---

## 致谢

基座和架构来自 [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（OpenMOSS）。文本编码器来自 Qwen3.8-27B，VAE 来自 FLUX.2 small decoder，推理和量化工具基于 Diffusers / PyTorch。

引用 MOSS-Image 技术报告：

```bibtex
@misc{ai2026mossimagetechnicalreport,
      title={MOSS-Image Technical Report},
      author={MOSI. AI},
      year={2026},
      eprint={2610.23374},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.23374},
}
```
