# MI-Prism-KedaiEX

**8 GB 笔记本上的 KEX 出图栈：KI8 1.5.0-rc3，W8A16，8.25 bit/权重。对 KEX BF16 参考 43.34 dB PSNR / LPIPS-Alex 0.00756。**

- 发行版：**KI8 1.5.0-rc3**（Peak8 / W32，W8A16）
- 基线：**KEX BF16（clip7）**
- 参考机：RTX 3070 Ti Laptop 8 GB / Ryzen 7 6800H / 16 GB DDR5
- 许可：MIT / Apache-2.0 / GPL-3.0 分层，见 [LICENSING-LAYERS.md](LICENSING-LAYERS.md)

---

## 结论

同一张 256×256、同噪声、同条件、8 步、seed 20261123：

| 指标 | KEX BF16（clip7） | KI8 1.5.0-rc3 W8A16 |
|---|---:|---:|
| RGB PSNR | 基准 | **43.34 dB** |
| LPIPS-Alex ↓ | 0 | **0.00756** |
| LPIPS-Squeeze ↓ | 0 | **0.00663** |
| latent PSNR | 基准 | 21.56 dB |
| latent 相关 | 1.0 | **0.9964** |
| 最大像素差 | 0 | 18 / 255 |
| 显存峰值 | —（CPU 采样） | **5071 MiB** |
| 步时 | 71.91 s（CPU） | **2.999 s** |
| 8 步总时 | 575.28 s | **25.15 s** |

**43.34 dB / LPIPS 0.00756 是逐像素基本分不出来的水平。** KEX 的构图、肤色、逆光轮廓和末层质感，在 8.25 bit/权重下保住了。

同码率（8 位值 + 2 位/权重的 E5F3 尺度 = 8.25 bit/w）、同权重、同噪声，KI8 1.5 的 Peak8 对 INT8：

| 臂 | 激活 | 步时 | RGB PSNR | LPIPS-Alex ↓ | 显存 |
|---|---:|---:|---:|---:|---:|
| **KI8 1.5 Peak8（W8A16）** | BF16 | **2.999 s** | **43.34 dB** | **0.00756** | 5071 MiB |
| INT8 W8A16 | BF16 | 8.74 s | 38.72 dB | 0.01878 | 5073 MiB |
| INT8 W8A8 | INT8 | 10.99 s | 35.51 dB | 0.01663 | 5102 MiB |

同码率下 Peak8 比 INT8 W8A16 **高 4.62 dB、LPIPS 低 2.5 倍、快 2.9 倍**。INT8 没有专用内核，走逐 output tile 反量化再 BF16 GEMM；Peak8 走专用 PTX 内核。

---

## 画质：KEX 的构图与审美

KEX 是在 MOSS-Image SFT 上继续训练出来的成像模型：只替换尾部三个张量（`layers.41.proj.o`、`layers.41.proj.ffn_down`、`final_linear_video`，共 34,996,224 参数），并把每层 q/k/v 融成 `qkv`。改动集中在**最后合成层**，改变的是：

- **人物重心**：髋部偏移、头部倾斜、肩线角度，站姿不是直上直下；
- **光线取向**：暖调逆光、轮廓光贴着发丝和肩背走，海面高光更集中；
- **材质与肤色**：皮肤高光偏暖，布料和湿发的反射更厚，暗部不死黑；
- **画面重心**：主体通常压在黄金分割附近，太阳/海岸线让位给人物，而不是平均铺开。

![KEX 1.5 Peak8，1024×1024，12 步，cfg 2.5](docs/assets/kex-1024.png)

*KEX 1.5 Peak8，1024×1024，12 步，cfg 2.5，seed 20261121。*

1024×1024、12 步、cfg 2.5、seed 20261121 的四臂同 prompt 对照里，这个取向是可测的：三个 KEX 臂彼此 LPIPS-Alex 只有 **0.027–0.041**，而同 prompt 的原生 INT8 SFT 与 KEX 臂差 **0.23**、PSNR 只有约 24.5 dB。**构图签名来自 KEX 权重，不是量化噪声。**

27B 文本编码器把条件约束再抬一档。等计算量对照（同样前向次数）：

| 配置 | 笔触量 `stroke_band` |
|---|---:|
| 9B，cfg 1.0，8 步 | 3.889 |
| 27B，cfg 2.0，4 步 | 3.820 |
| 9B，cfg 1.0，4 步 | 2.766 |

27B 跑 4 步、cfg 2.0，笔触量是 9B 跑 8 步、cfg 1.0 的 **98%**，前向次数相同。条件更强，构图更听话，细节不必靠加步数硬堆。

**KEX 的构图和审美是这条线的上限；KI8 1.5 的工作是别把它抹掉。** 43.34 dB / LPIPS 0.00756 就是这句话的证据。

---

## KI8 1.5.0-rc3

### Peak8：整组编码，8.25 bit/w

普通 int8 按固定间隔切格子，少数大值把量程撑开后，大量小值挤进少数格子。Peak8 以 32 个权重为一组：

- 组内先存一个 5+3 bit 组头和一个 112..127 的峰值标记；
- 其余幅值 0..111，用三 bit syndrome 选半径，最多改一个相邻码；
- 解码端从组头和 syndrome 还原峰值与半径，不需要额外位置表。

结果还是 **8 bit 值 + 0.25 bit/权重的尺度 = 8.25 bit/w**，但每组都能跟住自己的峰值。clip7 上 14 个张量的权重 SQNR：

| 编解码 | bits/w | 权重 SQNR |
|---|---:|---:|
| e2m5 | 8.25 | 44.59 dB |
| **Peak8** | 8.25 | **45.23 dB** |
| KI6 v3 26B 组 | 6.5 | 35.38 dB |

真实激活、真实投影输出（clip7 权重 + KEX 前向激活）：

| 臂 | 输出 SQNR |
|---|---:|
| e2m5 权重 + BF16 激活 | 47.67 dB |
| **Peak8 权重 + BF16 激活（W8A16）** | **48.34 dB** |
| e2m5 权重 + e2m5 激活（W8A8） | 43.18 dB |
| Peak8 权重 + Peak8 激活（W8A8） | 44.70 dB |

### W8A16：激活不量化

发行口径是 **Peak8 权重 + BF16 激活**。激活不量化，少一层 A8 舍入漂移；权重走 W32 整组解码，配专用 CUDA 内核。

- `peak8-a16` 是 rc3 的正式 profile，checkpoint 的 `manifest.version` 是 `1.5.0-rc3`；
- 装载时验证每层真的是 `("peak8", "bf16")`，不是“名字叫 A16、实际走 A8”；
- 端到端 8 步累计误差 43.34 dB / LPIPS 0.00756，说明 W8A16 路径没有把权重优势吃掉。

### CUDA 原生 W32 解码

`peak_optimized.ptx` 里的 `ki8_decode_w32` 用 warp 整数指令找峰值、算奇偶校验，替代两段 shuffle 归约；列宽是 32 的倍数时每 lane 正好一个值。`ki8_select` 把候选舍入和半径选择压进一次内核发射。

本机实测（4096×4096，512 tokens，rank-16 LoRA，5 次中位数）：

| 路径 | 中位 |
|---|---:|
| e2m5 `fast` | 88.85 ms |
| Peak8 `reference`（PyTorch） | 191.84 ms |
| **Peak8 `auto`（原生 W32）** | **21.20 ms** |
| Peak8 `release`（对照） | 20.63 ms |

相对 e2m5 是 **4.19×**；原生 W32 与 release 在该形状下持平（21.20 vs 20.63 ms，差值在噪声内），但 `auto` 路径的输出与 reference **逐位相同**，不是近似。

### 结构与显存

- q/k/v 融合成 `qkv`，42 层每层少两次矩阵乘调用，权重读取从三次连续读变成一次；
- 全模型 269 个线性层重编码，非量化张量逐位复制；8 步采样峰值 **5071 MiB**；
- 27B 编码器由 llama.cpp 单独进程加载，编码完即关，不和 DiT 同时占 8 GB 卡。

---

## 安装

```bash
P="python"                     # 先激活含 torch 2.9.1+cu128 的 venv
KI8="D:/moss-image/KI8-1.5.0-rc3"
REPO="$(pwd)"                  # 本仓库根目录
INSTALLER="<发行包>/install_moss.py"

# 1. 应用 CUDA 内核
git -C "$KI8" apply "$REPO/patches/ki8-1.5.0-rc3-cuda.patch"

# 2. 安装并校验 W8A16
"$P" "$INSTALLER" --comfy <COMFYUI> --yes --no-launch --ki8-root "$KI8"
```

安装器会定位 `KI8-1.5.0-rc3` 树、应用补丁、备份被改文件，然后跑一次真实 `1.5.0-rc3` checkpoint 回环：

- `347×511`（带尾组，token_tile 31 / output_tile 43）
- `3072×9216`（真实投影形状）

两个用例都要求 `auto` 与 `reference` **逐位相同**，并检查 `a16_decode == native-w32`；失败会还原所有改动，包括新增文件。

本机校验输出：

```json
"wire_check": {
  "backend": "peak_cuda.ptx",
  "w32_decode": "native-w32",
  "radius_selection": "fused",
  "wire_elements": 216224,
  "identical": true
},
"a16_checkpoint": {
  "version": "1.5.0-rc3",
  "cases": [
    {"shape": [347, 511],   "weight_format": "peak8", "activation_format": "bf16", "bit_identical": true},
    {"shape": [3072, 9216], "weight_format": "peak8", "activation_format": "bf16", "bit_identical": true}
  ],
  "bit_identical": true,
  "backend": {"a16_decode": "native-w32"}
}
```

---

## 使用

```bash
P="python"                     # 先激活含 torch 2.9.1+cu128 的 venv
KI8="D:/moss-image/KI8-1.5.0-rc3"

# W8A16 硬验证
KI8_REQUIRE_A16=1 "$P" "$KI8/run.py" check --device cuda:0 --require-a16

# 单投影速度
"$P" "$KI8/run.py" bench-speed --device cuda:0 --profile peak8-a16 \
  --in-features 4096 --out-features 4096 --tokens 512 \
  --token-tile 128 --output-tile 128 --iterations 5 --out work/speed_a16.json

# 端到端：先跑 KEX BF16 基线（CPU，约 575 s），再跑 W8A16
C="D:/moss-image/clip7/moss_image_clip7_bf16.safetensors"
OUT=work; mkdir -p $OUT

"$P" -I bench/kex_bf16_arms.py --arm bf16 --clip7 "$C" --ki8 "$KI8" \
  --transformer D:/moss-image/moss-bf16/transformer \
  --cond D:/moss-image/ComfyUI/models/cond_cache/kedaiEX1-pos.pt \
  --out $OUT/kex_bf16/bf16 --steps 8

KI8_REQUIRE_A16=1 "$P" -I bench/kex_bf16_arms.py --arm v15 --clip7 "$C" --ki8 "$KI8" \
  --transformer D:/moss-image/moss-bf16/transformer \
  --cond D:/moss-image/ComfyUI/models/cond_cache/kedaiEX1-pos.pt \
  --out $OUT/kex_w8a16/v15 --steps 8 --device cuda:0

# PSNR / LPIPS
"$P" -I bench/score_psnr_lpips.py \
  --ref $OUT/kex_bf16/bf16.npy --test $OUT/kex_w8a16/v15.npy \
  --ref-latent $OUT/kex_bf16/bf16_latent.npy --test-latent $OUT/kex_w8a16/v15_latent.npy \
  --out $OUT/kex_w8a16/score_v15_vs_bf16.json
```

BF16 基线在 CPU 上跑（10.16 GB 放不进 8 GB 显存），8 步约 575 s；量化臂在 `cuda:0` 上跑。

---

## 数据口径

- 端到端：256×256 latent、8 步、seed 20261123、同噪声、同 27B 条件缓存；BF16 为 clip7 全张量，量化臂的非量化张量也全部取自 clip7。
- LPIPS 0.1.4，Alex / Squeeze 两个 backbone；PSNR 按 8-bit RGB 计算，最大差按 0..255 计。
- 速度：同机、同一脚本、`cuda:0`；8 步取每步耗时中位数。跨机器不直接比。
- 1024×1024 四臂：ComfyUI，12 步，cfg 2.5，er_sde，seed 20261121，同 prompt、同噪声。
- 单种子、单图指标只作相对参考；质量结论以多臂一致性和逐像素差值为准。

---

## 仓库结构

```
MI-Prism-KedaiEX/
├── README.md
├── LICENSING-LAYERS.md        四层许可边界
├── NOTICE                     上游署名与修改声明
├── patches/
│   └── ki8-1.5.0-rc3-cuda.patch   W32 解码 + W8A16 校验
├── bench/
│   ├── kex_bf16_arms.py       clip7 基线 / v15 W8A16 端到端采样
│   ├── score_psnr_lpips.py    PSNR / LPIPS-Alex / LPIPS-Squeeze
│   └── st_read.py             原样 safetensors 读取器
├── configs/                   27B 预算模型配置
├── docs/                      预算模型、硬件档位、评测协议、许可说明
├── phase1/                    组件清单与哈希
├── tools/                     预算模型、硬件探测、文档审计、许可审计
└── tests/                     预算模型 / 文档 / 许可测试
```

---

## 许可

| 层 | 许可 | 覆盖 |
|---|---|---|
| 代码 | MIT | KI8 运行时、转换器、安装器、本仓库工具 |
| 权重 | Apache-2.0 | KEX DiT 衍生权重、GSQ 文本编码器（见 `NOTICE`） |
| ComfyUI 补丁 | GPL-3.0 | 装入 ComfyUI 后的 `ki8_codec.py` / `ops.py` / 节点包 |
| VAE | Apache-2.0 | `flux2-vae_small`（与 FLUX.2-small-decoder 字节一致） |

完整义务与文件清单见 [LICENSING-LAYERS.md](LICENSING-LAYERS.md)、[docs/10-phase1-licensing.md](docs/10-phase1-licensing.md)。

---

**KEX 负责好看，KI8 1.5.0-rc3 负责在 8 GB 卡上把这份好看原样跑出来。**
