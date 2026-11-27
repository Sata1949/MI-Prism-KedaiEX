<div align="center">

<img src="docs/assets/kedai-logo.svg" alt="Kedai Team" width="460">

# MI-Prism-KedaiEX

基于 MOSS-Image-SFT 的 DiT 微调权重与 KI8 W8A16 低显存推理方案

[SFT 4-Layer BF16 Release](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/sft-4layers-22-26-35-41-bf16) ·
[KI8 1.6 SFT Release](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/kex-weights-v1.6rc1) ·
[KI8 1.5 KEX Release](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/kex-weights-v1.5rc3) ·
[权重文档](docs/weights.md) ·
[CUDA 解码](docs/cuda-decode.md) ·
[许可证](LICENSING-LAYERS.md)

</div>

![KEX 1.5 Peak8 实测样张，1024×1024，12 步，cfg 2.5](docs/assets/kex-1024.png)

**MI-Prism-KedaiEX** 面向 [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)（5B、42 层 DiT）提供衍生微调权重与 8 GB 显存可用的 **KI8 W8A16** 量化推理方案。当前仓库通过 GitHub Releases 分发三组权重，并配套原生 CUDA W32 解码补丁、ComfyUI 工作流与保真度复现脚本。

---

## 目录

- [已发布权重](#已发布权重)
- [模型与训练](#模型与训练)
- [KI8 量化与实测基准](#ki8-量化与实测基准)
- [快速开始](#快速开始)
- [采样参考](#采样参考)
- [已知限制](#已知限制)
- [仓库结构](#仓库结构)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 已发布权重

| Release | 模型内容 | 精度 / 格式 | 大小与分片 | 完整文件 SHA256 |
|---|---|---|---|---|
| [`sft-4layers-22-26-35-41-bf16`](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/sft-4layers-22-26-35-41-bf16) | **SFT 4 层微调**（layers 22/26/35/41，step 2000） | BF16 单文件（`sft_dit_4layers_22_26_35_41_h200-merged.safetensors`） | 10,164,664,320 B（9.47 GiB；1 GiB × 9 + 478 MiB，共 10 片） | `8f55dcec2aed68a8d6c9122a4b2d0fd66d0831c1daad0ebcf018c90846843d9e` |
| [`kex-weights-v1.6rc1`](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/kex-weights-v1.6rc1) | **原版 MOSS-Image-SFT** 主干 | KI8 1.6.0-rc1 `universal-a16`（W8A16；`ki8-16-universal-a16.tar`） | 5,241,210,880 B（4.88 GiB；128 MiB × 39 + 6.4 MiB，共 40 片） | `e05c1845234dce27387da672e2ee69bb3533422c15450074ca5d1873f33019de` |
| [`kex-weights-v1.5rc3`](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/kex-weights-v1.5rc3) | **KEX clip7** 尾部微调对照 | KI8 1.5.0-rc3 Peak8/W32（W8A16；`MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors`） | 5,241,596,371 B（4.88 GiB；128 MiB × 39 + 6.8 MiB，共 40 片） | `7064ed51dc966d4c01d7c0d1c16c0e242cc01217955040c116659164c4c3e572` |

> 所有 Release 均仅包含 **DiT 主干权重**，不含 VAE 与外部文本编码器，并随包附带 `SHA256SUMS`、`LICENSE-APACHE-2.0` 与 `NOTICE`。

### 下载与校验示例（SFT 4-Layer BF16）

在新空目录执行（私仓需先 `gh auth login`）：

```bash
set -euo pipefail
export LC_ALL=C
mkdir sft-4layers-bf16 && cd sft-4layers-bf16
f=sft_dit_4layers_22_26_35_41_h200-merged.safetensors
gh release download sft-4layers-22-26-35-41-bf16 --repo Sata1949/MI-Prism-KedaiEX -p "$f.part*" -p SHA256SUMS
cat "$f".part[0-9][0-9][0-9] > "$f"
sha256sum -c SHA256SUMS
```

其余版本的 `gh` / `curl` 下载循环、KI8 1.6 逐层校验与 ComfyUI 单文件封装脚本，见 [docs/weights.md](docs/weights.md)。

---

## 模型与训练

### 基础架构

| 字段 | 说明 |
|---|---|
| 项目 / 团队 | MI-Prism-KedaiEX（Kedai Team：sata1921、seeds、suihan、hilo114） |
| 基座模型 | [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)（Apache-2.0） |
| DiT 主干 | 42 层 DiT，约 5B 参数（269 个线性投影模块） |
| 外部组件 | 文本编码器（MOSS-Image 原生 Qwen3.5-9B / KEX 工作流 Qwen3.8-27B headless + adapter）；FLUX.2 small VAE |
| 许可 | 权重 Apache-2.0（见 `NOTICE` 与 `LICENSING-LAYERS.md`）；代码 MIT；ComfyUI 补丁 GPL-3.0 |

### 1. SFT 4-Layer DiT（最新 BF16 权重）

基于 `MOSS-Image-SFT` 在 H200 上完成 2 层（22, 35）与 4 层（22, 26, 35, 41）消融对比后，对第 **22、26、35、41** 四层进行了 2,000 步正式训练并导出完整 BF16 单文件权重（未融合 q/k/v）：

| 项目 | 实测 / 配置 |
|---|---|
| 微调层 | `layers.22`、`layers.26`、`layers.35`、`layers.41` |
| 训练步数与超参 | 2,000 steps，`batch_size=2`，`grad_acc=2`，末期 `lr=3e-6`，目标面积 `1024×1024`（`TARGET_AREA=1048576`） |
| 训练终值（step 2000） | `loss=0.5388`，`val_loss=0.6924`（`σ=0.2/0.4/0.6/0.8` 分别为 `0.8820 / 0.7406 / 0.5985 / 0.5485`） |
| H200 吞吐与耗时 | `922.4 TFLOPS`，`10.94 samples/s`，总耗时 `806.2 s`，峰值显存分配 `22.83 GiB` |
| 导出产物 | `sft_dit_4layers_22_26_35_41_h200-merged.safetensors`（10,164,664,320 B，BF16） |

### 2. KI8 1.6.0-rc1（原版 SFT W8A16）

对固定 revision（[`f972bd4`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9)）的原版 `MOSS-Image-SFT` 主干执行 `universal-a16` 逐层格式选择，激活固定 BF16（W8A16）。269/269 个线性层均自动选中 `peak8 / bf16`，元素码 + W32 尺度共 5,240,733,696 B（8.25 bit/权重），可通过 `ki8_lora.install(dit, packed_dir)` 加载，或按 [docs/weights.md](docs/weights.md) 中的脚本与原 SFT 非量化张量合成 ComfyUI 单文件。

### 3. KEX 1.5（clip7 尾部微调对照）

仅替换 `layers.41.proj.o`、`layers.41.proj.ffn_down`、`final_linear_video` 三个尾部张量（34,996,224 参数）并融合每层 `qkv`，以 KI8 1.5.0-rc3 Peak8/W32（W8A16）打包分发。

---

## KI8 量化与实测基准

> 以下端到端保真度、速度与显存数据均来自 **KEX 1.5.0-rc3（clip7）** 实测，不作为 SFT 4-Layer BF16 或 SFT 1.6 的出图指标。

### 量化保真度（256×256 latent，8 步，seed 20261123，同噪声同条件）

参考臂为 KEX BF16（clip7，CPU 步时 71.91 s，8 步总时 575.28 s）；测试平台为 RTX 3070 Ti Laptop 8 GB：

| 方案 | 激活精度 | 步时 | 8 步总时 | RGB PSNR ↑ | LPIPS-Alex ↓ | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|
| **KI8 1.5 Peak8** | BF16（W8A16） | **2.999 s** | **25.15 s** | **43.34 dB** | **0.00756** | 5071 MiB |
| INT8（逐 tile 反量化） | BF16（W8A16） | 8.74 s | — | 38.72 dB | 0.01878 | 5073 MiB |
| INT8 | INT8（W8A8） | 10.99 s | — | 35.51 dB | 0.01663 | 5102 MiB |

KI8 1.5 Peak8 相对 KEX BF16 的 latent PSNR 为 `21.56 dB`、latent 相关 `0.9964`、LPIPS-Squeeze `0.00663`、最大像素差 `18 / 255`。

### ComfyUI 端到端速度（1024×1024，12 步，cfg 2.5，RTX 3070 Ti Laptop 8 GB）

| 解码后端 | 整图耗时 | 其中 27B TE | 其中 DiT + VAE | 与纯 torch 输出差异 |
|---|---:|---:|---:|---|
| 纯 torch Peak8 解码 | 205.8 s | 19–35 s | ~186 s | 基准 |
| **原生 CUDA W32（`ki8_decode_w32`）** | **60–77 s** | 19–35 s | **~41 s** | 逐像素一致（max diff 0） |

3072×3072 单层热解码从 184 ms 降至 1.0 ms。实现细节、回退条件与复测命令见 [docs/cuda-decode.md](docs/cuda-decode.md)。

---

## 快速开始

### 使用 SFT 4-Layer BF16 单文件

1. 按上方命令从 [`sft-4layers-22-26-35-41-bf16`](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/sft-4layers-22-26-35-41-bf16) 下载并重组 `sft_dit_4layers_22_26_35_41_h200-merged.safetensors`，确认 `sha256sum -c SHA256SUMS` 全部 `OK`。
2. 将单文件放入 `ComfyUI/models/diffusion_models/`（需 ≥ 12–16 GB 显存直接运行 BF16，或先按 KI8 工具链量化为 W8A16）。

### 使用 KEX 1.5 KI8 W8A16（8 GB 显存 ComfyUI）

1. 应用仓库中的运行时 CUDA 补丁：[patches/ki8-1.5.0-rc3-cuda.patch](patches/ki8-1.5.0-rc3-cuda.patch)，并安装支持 KI8 Peak8/BF16 的 ComfyUI 编解码与节点。
2. 将 `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors` 放入 `ComfyUI/models/diffusion_models/`。
3. 导入 [examples/moss27b_e2e24576_ki815.json](examples/moss27b_e2e24576_ki815.json)，将 27B GGUF、Adapter 与 `first_token.npz` 路径替换为本机路径后运行。

节点链路：

```text
MossTE27BEncodePair → CONDITIONING
UNETLoader(ki8_15peak8) → KSampler(er_sde / simple) → VAEDecode → SaveImage
```

### 命令行校验与保真度复现（KI8 1.5）

```bash
# 1. 验证原生 W32 解码与逐位一致性
KI8_REQUIRE_A16=1 python <KI8_DIR>/run.py check --device cuda:0 --require-a16

# 2. 复现 KEX 1.5 W8A16 vs BF16 保真度（BF16 臂在 CPU 运行，约 575 s）
python -I bench/kex_bf16_arms.py --arm bf16 --clip7 <CLIP7_BF16> --ki8 <KI8_DIR> \
  --transformer <MOSS_TRANSFORMER_DIR> --cond <COND_CACHE.pt> --out work/kex_bf16/bf16 --steps 8

KI8_REQUIRE_A16=1 python -I bench/kex_bf16_arms.py --arm v15 --clip7 <CLIP7_BF16> --ki8 <KI8_DIR> \
  --transformer <MOSS_TRANSFORMER_DIR> --cond <COND_CACHE.pt> --out work/kex_w8a16/v15 --steps 8 --device cuda:0

python -I bench/score_psnr_lpips.py \
  --ref work/kex_bf16/bf16.npy --test work/kex_w8a16/v15.npy \
  --ref-latent work/kex_bf16/bf16_latent.npy --test-latent work/kex_w8a16/v15_latent.npy \
  --out work/kex_w8a16/score_v15_vs_bf16.json
```

---

## 采样参考

| 场景 | 分辨率 | 步数 | CFG | Sampler | 说明 |
|---|---|---|---:|---|---|
| 量化保真度探针 | 256×256 | 8 | — | MOSS scheduler | 仅用于同噪声编解码误差评测 |
| 快速出图 | 1024×1024 | 12 | 2.5 | `er_sde` / `simple` | ComfyUI 示例工作流默认 |
| 视觉对比 | 1024×1024 | 25 | 2.5 | `er_sde` / `simple` | 细节与笔触更完整 |
| 文本排版 / 高质量 | 1024×1024 | 50 | 3.0 | Flow-matching Euler | MOSS-Image 官方默认 |

---

## 已知限制

- **显存两阶段**：8 GB 显卡无法同时驻留 27B 文本编码器（~10 GB）与 DiT（~5.2 GB），必须先编码条件并卸载 TE，再加载 DiT 采样。
- **原生解码范围**：`ki8_decode_w32` 仅覆盖 W8A16 的 group-32 权重；A8 / group-128 路径仍走 PyTorch 回退。
- **BF16 权重体积**：最新 `sft-4layers-22-26-35-41-bf16` 为未量化的 9.47 GiB BF16 权重，8 GB 显存需先量化后使用。
- **保真度口径**：43.34 dB / 0.00756 来自 KEX 1.5 单图单种子评测；KEX 1.5 的 clip7 BF16 参考权重未随 Release 分发。
- **私仓下载权限**：仓库为私有时需使用有权限的 `gh auth login` 下载 Release 分片；公开后可直接匿名 `curl` 下载。

---

## 仓库结构

```text
MI-Prism-KedaiEX/
├── README.md
├── LICENSING-LAYERS.md
├── LICENSE
├── LICENSE-APACHE-2.0
├── NOTICE
├── docs/
│   ├── weights.md                 # 三组 Release 下载、校验与单文件封装指南
│   ├── cuda-decode.md             # 原生 CUDA W32 解码实现与复测命令
│   ├── 13-handoff-2026-11-25.md   # 交接与实测记录
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
├── licenses/
├── phase1/
├── tools/
└── tests/
```

---

## 许可证

| 层 | 许可 | 覆盖范围 |
|---|---|---|
| 代码 | MIT | 本仓库脚本、评测工具与转换脚本（见 `LICENSE`） |
| 权重 | Apache-2.0 | MOSS-Image-SFT 衍生权重（见 `LICENSE-APACHE-2.0`、`NOTICE`） |
| ComfyUI 补丁 | GPL-3.0 | 集成至 ComfyUI 后的 `ki8_codec.py`、`ops.py` 与节点包 |
| VAE | Apache-2.0 | `flux2-vae_small` |

完整合规义务与第三方声明见 [LICENSING-LAYERS.md](LICENSING-LAYERS.md) 与 [docs/10-phase1-licensing.md](docs/10-phase1-licensing.md)。

---

## 致谢

基座模型与架构来自 [OpenMOSS / MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)；文本编码器基于 Qwen3 系列，VAE 基于 FLUX.2 small decoder，推理与量化基于 PyTorch / Diffusers。

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
