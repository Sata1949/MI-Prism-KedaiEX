# MI-Prism-KedaiEX（第一阶段存档）

> 当前发行版是 KI8 1.5.0-rc3（W8A16），主文档见 [README.md](README.md)。本文件保留为 2026-11-22 第一阶段记录，数字不再更新。

通过激进的模型改动，让原本跑不动的模型能在 8 GB 显存 + 16 GB 内存的笔记本上本地出图，
并且尽量不损失画质。

- 状态：第一阶段完成（2026-11-22）
- 参考机：RTX 3070 Ti Laptop 8 GB / Ryzen 7 6800H / 16 GB DDR5
- 显存下限 6 GB。4 GB 不可行。
- 许可：MIT / Apache-2.0 / GPL-3.0，见 [LICENSING-LAYERS.md](LICENSING-LAYERS.md)

---

## 项目在做什么

主流开源模型越做越大，8 GB 卡能跑的东西和旗舰模型之间的差距肉眼可见。
这个项目不接受"小机器就该用小模型"，做法是把大模型拆开改，压到能跑，再尽量把能力留住。

三个阶段：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 一 | Dense 改进 + 27B 转换 | 完成 |
| 二 | Dense 大改 + 原生 27B 支持 | 未开始 |
| 三 | Dense 转 MoE 化 + 原生 27B 支持 | 探索中 |

第二、三阶段要写自定义 kernel，本机缺编译器，尚未开始。
下面的内容全部是第一阶段，已经在 8 GB 卡上跑通。

---

## 8 GB 卡上的问题

第一阶段的目标模型是 [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)，一个 5B 参数的文生图模型。
官方给两种权重：BF16（9.47 GiB）和 int8 量化版（5.04 GiB）。
文本编码器是 Qwen3.5-9B，BF16 要 17.98 GiB，量化版要 6.84 GiB。

把这几个数加起来，加上 VAE 和运行时的开销：

| 配置 | 权重总量 |
|---|---:|
| 官方 BF16 全套 | 27.76 GiB |
| 官方 int8 全套 | 12.11 GiB |
| 参考机实测可用显存 | 7.00 GiB |

两套都装不下。硬装的结果是权重在内存和显存之间反复换页，出图时间从秒级掉到分钟级。

---

## 第一阶段做了什么

两处改动。

第一处针对 DiT。换成自研的 KI8 1.4 量化格式，文件从 9.47 GiB 降到 4.88 GiB，
量化后的画面与 BF16 参考相差 55.56 dB PSNR。55 dB 这个量级，两张图放在一起逐像素基本分不出来。
同时还把注意力里的 q/k/v 三个投影合并成一个，减少每层的运算次数。

第二处针对文本编码器。把 Qwen3.8-27B 这个语言模型"去头"，
只保留到最后一层归一化之前的隐状态，中间接一层自训的投影器，
把 27B 的 5120 维输出接到 MOSS 需要的 4096 维条件空间。这一步叫 27B 转换。

两处合起来，8 GB 卡上常驻 5.11 GiB，画质接近 BF16，条件约束比原来的 9B 强。

---

## DiT

### 量化格式：E2M5

常规 int8 量化的做法是按固定间隔把权重切成格子。问题在于权重分布很不均匀：
少数大值把量程撑开之后，大量小值只能挤进少数几个格子，误差很大。

KI8 1.4 换了个思路，用"块浮点"。每 32 个权重共用一个小尺码（E5F3 编码，
值域是 `2^(code/8 − 16)`，也就是 1/8 倍频程一档），这个尺码能跟住每个块的峰值。
32 个权重内部再用 E2M5 表示：1 位符号、2 位指数、5 位尾数，最大表示 7.875。

一句话概括：用 8 bit 存数、另加 0.25 bit 存尺码，合 8.25 bit/权重。

支持的四种元素格式：

| 格式 | 结构 | 最大绝对值 | 用途 |
|---|---|---:|---|
| E4M3FN | 1 符号 4 指数 3 尾数 | 448 | 1.0 到 1.3 的旧格式 |
| E3M4 | 1 符号 3 指数 4 尾数 | 31 | 1.4 |
| E2M5 | 1 符号 2 指数 5 尾数 | 7.875 | 1.4 默认 |
| S8 | 对称整数 | 127 | 1.4 |

### 结构：q/k/v 合并

官方权重里，每层的注意力有三个独立投影 `q`、`k`、`v`，各是 3072×3072。
KedaiEX1 把这三个合并成一个 `qkv`，形状 9216×3072。

运行时先做一次矩阵乘，再把结果按最后一维切成三段当作 q、k、v。
42 层算下来，每层少两次矩阵乘调用，权重读取也从三次连续读变成一次。

这是纯粹的运行时收益，不改变数学结果。

### 逐投影选格式

不是整个模型统一用 E2M5。每个投影单独在四种格式里挑，权重和激活各挑一次，
最多 16 种组合，用两批独立的输入分别做拟合和验证。

实测下来 MOSS 的三个真实投影都选了 E2M5/E2M5，Qwen 那边有一层选了 S8 权重。
格式是跑出来的，没有预设。

### 量化后还剩多少

用 BF16 权重加 FP32 运算当参考，同一噪声、同一调度完整采样：

| 项目 | 结果 |
|---|---|
| 成图 PSNR（对 BF16 参考，160 张） | 55.56 dB |
| 相对上一版 KI8 1.3 | +8.13 dB，160 张全胜 |
| 独立训练副本复验 | +7.39 dB，160 张全胜 |

真实投影上的对照，用 MOSS 和 Qwen-Image-2.1 的实际权重：

| 保留集 | 上一版 | 纯 int8 | KI8 1.4 | 业界常见的 MXFP8 |
|---|---:|---:|---:|---:|
| MOSS 三个真实投影 | 29.87 dB | 37.03 dB | 39.86 dB | 28.46 dB |
| Qwen 两个真实投影 | 30.01 dB | 37.27 dB | 40.02 dB | 28.51 dB |

对 MXFP8 高 11.4 到 11.5 dB，对纯 int8 高 2.8 dB。

### 弱 LoRA 不再被吃掉

原版在最后一层输出时会做一次 BF16 舍入。LoRA 强度很低的时候（0.05），
增量比舍入误差还小，等于白加。

KI8 1.4 加了"锚定出口"：先把基础输出舍入成 BF16，再提升到 FP32 加增量。
强度为 0 时和原版输出逐元素完全相等，所以基础画面不变，弱 LoRA 的细节留下来了。

| 出口 | 强度 0.05 的增量信噪比 |
|---|---:|
| 原版 BF16 出口 | −18.68 dB |
| KI8 1.4 锚定出口 | 64.91 dB |

---

## 文本编码器

MOSS 原本用 9B 的语言模型当编码器。这里换成 27B，权重从 6.84 GiB 涨到 10.30 GiB。

27B 由 llama.cpp 单独起进程加载，编码完就关掉，不和 DiT 同时占显存。
8 GB 卡上这两者装不下同一个时刻，编码和采样必须分两阶段跑。

### 笔触与细节

厚涂题材，1024×1024，同提示词同 seed。`stroke_band` 越高，笔触越强：

| 配置 | stroke_band |
|---|---:|
| 9B，cfg 1.0，4 步 | 2.766 |
| 27B，cfg 1.0，4 步 | 2.595 |
| 9B，cfg 1.0，8 步 | 3.889 |
| 27B，cfg 2.0，4 步 | 3.820 |

27B 跑 4 步、cfg 2.0，笔触量是 9B 跑 8 步、cfg 1.0 的 98%。
两者前向次数相同（cfg 2.0 每步走两次），所以这组对比是等计算量的。

### 人物一致性

27B 在 cfg 1.0 下跑到 12 步会"换人"，面部色调距离 0.177，肉眼能看出来。
把 cfg 提到 2.0 之后降到 0.021。

原因是条件更强之后，抬 cfg 放大的是有效信号，不是噪声。

---

## 出图参数

| 场景 | 尺寸 | 步数 | CFG | 采样器 |
|---|---|---:|---:|---|
| 标准出图 | 1024×1024 | 25 | 2.5 | euler / simple |
| 高分辨率 | 1920×1088 | 25 | 2.5 | er_sde / simple |
| 2K | 2560×1440 | 25 | 2.5 | er_sde / simple |
| 试构图 | 1024×1024 | 4 | 1.0 | euler / simple |

4 步的蒸馏 LoRA 只覆盖约 4096 个 latent token，正好是 1024×1024 对应的量。
分辨率一上去，画面就会发软、皮肤出塑料感，所以定稿用 25 步。

CFG 按题材调：厚涂和油画用 2.0，笔触量能涨 47% 到 53%；二次元 1.6 到 1.8；
写实摄影 1.5 到 1.8；快速草稿用 1.0，但那时负向词不生效。

负向词要跟题材对齐。厚涂写 `smooth, plastic, airbrushed, digital painting, 3d render`，
不要写 `painterly / brushstroke / oil painting`，那是反向操作。

---

## 显存与速度

| | DiT | 文本编码器 | VAE | 合计 |
|---|---:|---:|---:|---:|
| 官方 BF16 | 9.47 GiB | 17.98 GiB | 0.31 GiB | 27.76 GiB |
| 官方 int8 | 5.04 GiB | 6.84 GiB | 0.23 GiB | 12.11 GiB |
| 本项目 | 4.88 GiB | 10.30 GiB | 0.23 GiB | 15.41 GiB |

参考机可用 7.00 GiB。采样阶段 DiT 加 VAE 峰值 6.19 GiB，可以常驻。
编码器不常驻，它单独起进程，用完就关。

27B 编码器比 9B 大 3.5 GiB。冷启动要把 11 GB 权重读一遍，本机顺序读速度 1.83 GB/s，
光读盘就 6 秒，加上上传和编码，实测编码阶段 16.5 到 18.9 秒。9B 是 5.43 秒。
批量出图时能摊掉这部分开销，一次启动编 8 条合 3.56 秒一条。

1024×1024、25 步的实测：复用条件 78.46 秒，新提示词含编码 97.4 秒，首图端到端 107.64 秒。

---

## 交付件

| # | 组件 | 文件 | 体积 | 许可 |
|---|---|---|---:|---|
| 1 | KI8 1.4 量化核心 | `KI8-1.4.0-rc1/KI8` | 2.51 MiB | MIT |
| 2 | KedaiEX1 DiT | `MI-DIT-Prism-KedaiEX1_ki8_14.safetensors` | 4.88 GiB | Apache-2.0 |
| 3 | 转换器 | `KedaiEX1-Adapter.safetensors` | 168 MiB | MIT |
| 4 | 去头文本编码器 | `enc27b_gsq_rco_iq3s_headless.gguf` | 10.30 GiB | Apache-2.0 |
| 5 | 整合安装器 | `MOSS-Image_KI8_安装包` | 456 KiB | MIT |

另外需要 `flux2-vae_small.safetensors`（238 MiB，Apache-2.0）、
`llama-server.exe` 与配套 dll（约 1.4 GiB，MIT）、ComfyUI 0.44.0（GPL-3.0，自行安装）。

校验用：

```bash
python -I tools/license_audit.py
```

---

## 安装

```bash
# 1. 放模型。安装器按文件名搜索，位置不限
#    MI-DIT-Prism-KedaiEX1_ki8_14.safetensors  → models/diffusion_models/
#    flux2-vae_small.safetensors               → models/vae/
#    enc27b_gsq_rco_iq3s_headless.gguf
#    KedaiEX1-Adapter.safetensors
#    first_token.npz
#    llama-server.exe + ggml-cuda.dll

# 2. 安装
python install_moss.py --comfy D:/ComfyUI

# 无人值守
python install_moss.py --comfy D:/ComfyUI --yes --wait 300

# 3. 卸载或回滚
python uninstall_moss.py --comfy D:/ComfyUI
python install_moss.py --comfy D:/ComfyUI --rollback
```

安装器分八步，每步有独立退出码。第 6 步会检查六个模型文件，缺一个就停下，
不会装出一半再让你在出图时才看到报错。

启动参数：

```bash
run_comfy_kex_fast.cmd     # --use-sage-attention --reserve-vram 0.35 --cache-lru 8
```

---

## 限制

- 只在 Windows 上测过（Python 3.11 / 3.12，torch 2.9.1+cu130 与 2.11+cu128）。
- 跨 ComfyUI 重启出图不复现。同 seed、同参数，重启前后 PSNR 只有 35.48 dB。
  比画质要在同一个进程里连着出图。
- 27B 编码器冷启动有物理下限，见上面速度一节。
- 内存低于 12 GiB 时安装器直接拒绝。
- 1080P 和 2K 的 25 步耗时按 12 步线性推算，未实测。

---

## 致谢

- [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)：5B 文生图模型与推理代码（Apache-2.0）。
  模型架构、聊天模板与 sink 约定都来自这里。
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)：文本编码器基座（Apache-2.0）。
- [ISTA DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)：GSQ-RCO 混合精度量化（Apache-2.0）。
- [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder)：FLUX.2 small decoder VAE（Apache-2.0）。
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)：执行宿主（GPL-3.0）。
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：GGUF 推理（MIT）。
- [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)：量化验证用的第二个真实模型，研究用途，不随本项目分发。

Qwen、MOSS、FLUX 是各自所有者的商标。本项目与它们没有隶属或背书关系。

---

## 许可

| 用途 | 允许 | 条件 |
|---|---|---|
| 商用出图 | 是 | 保留 MIT 版权行、Apache NOTICE、GPL 层开源 |
| 闭源分发 KI8 / 转换器 / 安装器 | 是 | MIT，保留版权行 |
| 再分发 DiT / 编码器 / VAE | 是 | Apache-2.0 第 4b 条标注修改、第 4c 条保留 NOTICE |
| 分发节点包 | 是 | GPL-3.0，提供源码 |
| 分发 Qwen-Image-2.1 或其 LoRA | 否 | 非商用，且本项目不发布 |
