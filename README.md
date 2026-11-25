<div align="center">

<img src="docs/assets/logo-kedai-team.png" alt="Kedai Team" width="520">

# MI-Prism-KedaiEX

**KEX 5B 文生图 · 8 GB 笔记本 · KI8 1.5.0-rc3 W8A16**

[Release](https://github.com/Sata1949/MI-Prism-KedaiEX/releases/tag/kex-weights-v1.5rc3) ·
[Weights](docs/weights.md) ·
[License](LICENSING-LAYERS.md) ·
[ComfyUI workflow](examples/moss27b_e2e24576_ki815.json)

</div>

![KEX 1.5 Peak8，1024×1024，12 步，cfg 2.5](docs/assets/kex-1024.png)

KEX（KedaiEX1）是在 [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) 上继续训练的 5B 文生图模型。MI-Prism-KedaiEX 把它压进 8 GB 显存：KI8 1.5.0-rc3 用 **Peak8 / W32** 把 269 个线性层编到 **8.25 bit/权重**，激活保持 BF16（W8A16）。对 KEX BF16 参考，256×256 / 8 步的端到端结果是 **43.34 dB PSNR / LPIPS-Alex 0.00756**，8 步采样峰值 **5071 MiB**。

---

## 目录

- [亮点](#亮点)
- [模型](#模型)
- [权重](#权重)
- [保真度](#保真度)
- [快速开始](#快速开始)
  - [ComfyUI](#comfyui)
  - [命令行校验与测速](#命令行校验与测速)
  - [保真度复现](#保真度复现)
- [采样设置](#采样设置)
- [显存与速度](#显存与速度)
- [构图与审美](#构图与审美)
- [局限](#局限)
- [仓库结构](#仓库结构)
- [许可证](#许可证)
- [致谢与引用](#致谢与引用)

---

## 亮点

- **KEX 的画质被保住了。** 对 KEX BF16（clip7）参考，KI8 1.5.0-rc3 W8A16 在 256×256 / 8 步 / 同噪声同条件下是 **43.34 dB PSNR、LPIPS-Alex 0.00756、latent 相关 0.9964**，最大像素差 18/255。构图、肤色、逆光轮廓和末层质感没有在量化里丢掉。
- **同码率压过 INT8。** 8.25 bit/w 下，Peak8 比 INT8 W8A16 **高 4.62 dB**、LPIPS-Alex 低 **2.5 倍**、步时快 **2.9 倍**（2.999 s vs 8.74 s）。
- **8 GB 卡能跑。** 42 层 DiT、269 个线性层重编码后权重 5.24 GB，8 步采样峰值 **5071 MiB**；27B 文本编码器单独进程加载，编码完即释放。
- **W8A16 是硬口径。** `peak8-a16` 检查会现场打包 `1.5.0-rc3` checkpoint 并回环，`auto` 与 `reference` 逐位相同；失败不会静默回退 PyTorch。
- **构图是 KEX 的资产。** 1024×1024 四臂同 prompt 对照里，三个 KEX 臂彼此 LPIPS-Alex 只有 0.027–0.041，和原生 SFT 差 0.23。量化必须保住的是这个签名。

---

## 模型

| 字段 | 值 |
|---|---|
| 名称 | MI-Prism-KedaiEX（KedaiEX1 / KEX） |
| 开发者 | sata1921、seeds、suihan、hilo114 |
| 基座 | [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)（Apache-2.0） |
| 架构 | 42 层 DiT，5B 主干；Qwen3.8-27B headless 文本编码器（5120→4096 adapter）；FLUX.2 small VAE |
| 文本条件 | 自然语言 prompt、tag 序列、JSON layout（MOSS-Image 原生格式） |
| 参考精度 | BF16（clip7，尾部三张量微调，qkv 融合） |
| 发行精度 | **KI8 1.5.0-rc3 Peak8 / W32，W8A16，8.25 bit/w** |
| 权重文件 | `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors`，5,241,596,371 B |
| 权重许可 | Apache-2.0（MOSS-Image 衍生，见 `NOTICE`） |
| 代码许可 | MIT（KI8 运行时、转换器、安装器、本仓库工具） |
| 发行 | 2026-11，KI8 1.5.0-rc3 |

KEX 相对 MOSS-Image-SFT 的改动只在尾部：替换 `layers.41.proj.o`、`layers.41.proj.ffn_down`、`final_linear_video` 三个张量（共 34,996,224 参数），并把每层 q/k/v 融成 `qkv`。其余张量与 SFT 同源；SFT 与 clip7 的逐张量 SQNR 只有 15.65–23.08 dB，说明微调幅度全部集中在这三个输出层。

---

## 权重

| 文件 | 格式 | 大小 | SHA256 |
|---|---|---:|---|
| `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors` | KI8 1.5.0-rc3 Peak8/W32，W8A16 | 5,241,596,371 B | `7064ed51…e572` |

权重按 128 MiB 分片放在 GitHub Release `kex-weights-v1.5rc3`（40 片，单片 < 2 GiB）。仓库是私有的，下载需要 `GITHUB_TOKEN`（repo 权限）：

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

逐片哈希、加载说明和 `gh release download` 写法见 [docs/weights.md](docs/weights.md)。KEX BF16 基线（10.16 GB）不随本 Release 分发。

---

## 保真度

**端到端：256×256 latent、8 步、seed 20261123、同噪声、同 27B 条件缓存。** BF16 参考是 clip7 全张量；量化臂的非量化张量也全部取自 clip7，所以差异只来自权重编解码。

| 臂 | 精度 | 步时 | RGB PSNR | LPIPS-Alex ↓ | LPIPS-Squeeze ↓ | latent 相关 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **KI8 1.5 Peak8** | **W8A16** | **2.999 s** | **43.34 dB** | **0.00756** | **0.00663** | **0.9964** | **5071 MiB** |
| INT8 | W8A16 | 8.74 s | 38.72 dB | 0.01878 | 0.01484 | 0.9898 | 5073 MiB |
| INT8 | W8A8 | 10.99 s | 35.51 dB | 0.01663 | 0.01448 | 0.9941 | 5102 MiB |
| KEX BF16（clip7） | BF16 | 71.91 s | 基准 | — | — | — | —（CPU） |

**权重 / 投影 SQNR（clip7 权重 + KEX 真实激活）：**

| 编解码 | bits/w | 权重 SQNR | W8A16 投影输出 SQNR |
|---|---:|---:|---:|
| e2m5 | 8.25 | 44.59 dB | 47.67 dB |
| **Peak8** | 8.25 | **45.23 dB** | **48.34 dB** |
| KI6 v3 26B 组 | 6.5 | 35.38 dB | 39.97 dB |

单投影 SQNR 不能替代出图保真度：e2m5 的投影有 47.67 dB，端到端只有 34.28 dB（8 步累积）；Peak8 端到端 43.34 dB。上面的出图数据才是发行口径。

---

## 快速开始

### ComfyUI

1. 安装发行包里的 KI8 1.5.0-rc3 运行时与 ComfyUI 编解码/节点；KI8 运行时补丁见 [patches/ki8-1.5.0-rc3-cuda.patch](patches/ki8-1.5.0-rc3-cuda.patch)。
2. 把 `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors` 放进 `ComfyUI/models/diffusion_models/`。
3. 导入 [examples/moss27b_e2e24576_ki815.json](examples/moss27b_e2e24576_ki815.json)，按本机路径修改 27B GGUF、`KedaiEX1-Adapter` 和 `first_token.npz`。
4. 节点链路：

```
MossTE27BEncodePair → CONDITIONING
UNETLoader(ki8_15peak8) → KSampler(er_sde / simple) → VAEDecode → SaveImage
```

Peak8 由 KI8 codec 按 W8A16 装载：权重走 `ki8_decode_w32`，激活保持 BF16，`quantize_input=False`。

> ComfyUI 里的 Peak8 目前是纯 torch 模拟解码（无 LUT、无 decode cache），约 13.4 s/步；原生 CUDA 路径的 2.999 s/步见下面的命令行与显存表。要上线到原生速度，需要给 ComfyUI 接 Peak8 LUT 或打开 decode cache。

### 命令行校验与测速

```bash
P="python"                     # 先激活含 torch 2.9.1+cu128 的 venv
KI8="D:/moss-image/KI8-1.5.0-rc3"

# W8A16 硬验证：真实 1.5.0-rc3 checkpoint 回环 + 原生 W32
KI8_REQUIRE_A16=1 "$P" "$KI8/run.py" check --device cuda:0 --require-a16

# 单投影 W8A16 速度
"$P" "$KI8/run.py" bench-speed --device cuda:0 --profile peak8-a16 \
  --in-features 4096 --out-features 4096 --tokens 512 \
  --token-tile 128 --output-tile 128 --iterations 5 --out work/speed_a16.json
```

`check --require-a16` 会打包 `347×511`（带尾组）和 `3072×9216`（真实投影）两个 `1.5.0-rc3` checkpoint，装回去后要求 `auto` 与 `reference` **逐位相同**，并检查 `a16_decode == native-w32`。

### 保真度复现

```bash
C="<PATH>/moss_image_clip7_bf16.safetensors"     # KEX BF16 基线，需自备
OUT=work; mkdir -p $OUT

# 候选：KI8 1.5 W8A16
KI8_REQUIRE_A16=1 "$P" -I bench/kex_bf16_arms.py --arm v15 --clip7 "$C" --ki8 "$KI8" \
  --transformer <MOSS_TRANSFORMER_DIR> --cond <COND_CACHE.pt> \
  --out $OUT/kex_w8a16/v15 --steps 8 --device cuda:0

# 基线：BF16（CPU，约 575 s / 8 步）
"$P" -I bench/kex_bf16_arms.py --arm bf16 --clip7 "$C" --ki8 "$KI8" \
  --transformer <MOSS_TRANSFORMER_DIR> --cond <COND_CACHE.pt> \
  --out $OUT/kex_bf16/bf16 --steps 8

# PSNR / LPIPS
"$P" -I bench/score_psnr_lpips.py \
  --ref $OUT/kex_bf16/bf16.npy --test $OUT/kex_w8a16/v15.npy \
  --ref-latent $OUT/kex_bf16/bf16_latent.npy --test-latent $OUT/kex_w8a16/v15_latent.npy \
  --out $OUT/kex_w8a16/score_v15_vs_bf16.json
```

---

## 采样设置

| 场景 | 分辨率 | 步数 | CFG | Sampler | 说明 |
|---|---|---|---:|---|---|
| 保真度探针 | 256×256 | 8 | — | MOSS scheduler | 同噪声比编解码误差 |
| 快速出图 | 1024×1024 | 12 | 2.5 | er_sde / simple | 发行工作流默认 |
| 视觉对比 | 1024×1024 | 25 | 2.5 | er_sde / simple | 细节与笔触更完整 |
| 文本/排版 | 1024×1024 | 50 | 3.0 | Flow-matching Euler | MOSS-Image 官方默认 |

> 8 步是量化保真度探针，不是最终视觉质量；看画面用 25 步或 50 步。

---

## 显存与速度

参考机：RTX 3070 Ti Laptop 8 GB / Ryzen 7 6800H / 16 GB DDR5，`cuda:0`。

| 项目 | 数值 |
|---|---:|
| 权重常驻 | 5.24 GB |
| 8 步采样峰值 | 5071 MiB |
| 8 步总时 | 25.15 s |
| 步时中位 | 2.999 s |
| BF16 基线步时（CPU） | 71.91 s |
| Peak8 单投影（4096×4096，512 tokens） | 21.20 ms |
| e2m5 同形状 | 88.85 ms |

8 GB 卡的硬约束：27B 文本编码器（约 10 GB）和 DiT 不能同时驻留。先编码 cond，释放 TE，再加载 DiT 采样；这是两阶段链路，不是可选优化。

---

## 构图与审美

KEX 的微调集中在**最后合成层**，改变的是：

- **人物重心**：髋部偏移、头部倾斜、肩线角度，站姿不是直上直下；
- **光线取向**：暖调逆光、轮廓光贴着发丝和肩背走，海面高光更集中；
- **材质与肤色**：皮肤高光偏暖，布料和湿发的反射更厚，暗部不死黑；
- **画面重心**：主体压在黄金分割附近，太阳/海岸线让位给人物，而不是平均铺开。

这个取向是可测的。1024×1024、12 步、cfg 2.5、seed 20261121 的四臂同 prompt 对照里：

| 对 | RGB PSNR | LPIPS-Alex |
|---|---:|---:|
| KEX 1.5 Peak8 vs KEX 1.5 s8 | 31.00 dB | 0.0375 |
| 原生 INT8 SFT vs KEX 1.5 Peak8 | 24.43 dB | 0.2379 |

KEX 1.5 的两个码流（Peak8 / s8）彼此只差 0.0375 LPIPS-Alex；原生 SFT 和 KEX 差 0.24。**构图签名来自 KEX 权重，不是量化噪声。** 27B 编码器再把条件约束抬一档：等计算量下，27B 跑 4 步 cfg 2.0 的笔触量是 9B 跑 8 步 cfg 1.0 的 98%（3.820 vs 3.889），前向次数相同。

---

## 局限

- **ComfyUI 的 Peak8 还没有原生 LUT。** 当前是纯 torch 模拟解码，约 13.4 s/步；2.999 s/步是 KI8 runtime 原生 CUDA 路径。两者权重与输出一致，速度不同。
- **单种子、单图指标。** PSNR / LPIPS 只作相对参考；质量结论以多臂一致性和逐像素差值为准。
- **8 GB 卡不能同时放 TE 和 DiT。** 必须两阶段；16 GB 内存也会限制同时驻留的权重数量。
- **BF16 基线不随 Release 分发。** 复现 43.34 dB 需要自备 `moss_image_clip7_bf16.safetensors`（10.16 GB）；仓库只发行 KI8 1.5.0-rc3 量化权重。
- **私仓下载需要 token。** Release 资产在私有仓库下不公开；Colab 里需要 `GITHUB_TOKEN` 或 `gh auth login`。

---

## 仓库结构

```
MI-Prism-KedaiEX/
├── README.md
├── LICENSING-LAYERS.md        四层许可边界
├── NOTICE                     上游署名与修改声明
├── docs/
│   ├── weights.md             分片下载、哈希、加载
│   └── assets/kex-1024.png    1024² 样图
├── examples/
│   └── moss27b_e2e24576_ki815.json   ComfyUI 工作流
├── patches/
│   └── ki8-1.5.0-rc3-cuda.patch      W32 解码 + W8A16 校验
├── bench/
│   ├── kex_bf16_arms.py       clip7 基线 / v15 W8A16 采样
│   ├── score_psnr_lpips.py    PSNR / LPIPS-Alex / LPIPS-Squeeze
│   └── st_read.py             原样 safetensors 读取器
├── configs/                   27B 预算模型配置
├── phase1/                    组件清单与哈希
├── tools/                     预算模型、硬件探测、文档审计、许可审计
└── tests/                     预算模型 / 文档 / 许可测试
```

---

## 许可证

| 层 | 许可 | 覆盖 |
|---|---|---|
| 代码 | MIT | KI8 运行时、转换器、安装器、本仓库工具 |
| 权重 | Apache-2.0 | KEX DiT 衍生权重、GSQ 文本编码器（见 `NOTICE`） |
| ComfyUI 补丁 | GPL-3.0 | 装入 ComfyUI 后的 `ki8_codec.py` / `ops.py` / 节点包 |
| VAE | Apache-2.0 | `flux2-vae_small`（与 FLUX.2-small-decoder 字节一致） |

完整义务与文件清单见 [LICENSING-LAYERS.md](LICENSING-LAYERS.md)、[docs/10-phase1-licensing.md](docs/10-phase1-licensing.md)。

---

## 致谢与引用

基座与架构来自 [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（OpenMOSS）。文本编码器来自 Qwen3.8-27B；VAE 来自 FLUX.2 small decoder；推理与量化工具基于 Diffusers / PyTorch。

如果这个项目对你有帮助，请优先引用 MOSS-Image 技术报告：

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

**KEX 负责好看，KI8 1.5.0-rc3 负责在 8 GB 卡上把这份好看原样跑出来。**
