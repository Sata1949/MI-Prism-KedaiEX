# 分许可对照表

**原创内容 MIT（四人团队署名），衍生权重 Apache-2.0，单独节点 GPL-3.0。Qwen-Image-2.1 不在发布范围内。**

按你拿到的是哪一层查表，不要假设整包一个许可。

机器可校验版本：`phase1/components.json`（`schema: ...@3`）。跑 `python -I tools/license_audit.py` 验证。

---

## 署名

```
Copyright (c) 2026 sata1921, seeds, suihan, hilo114
```

四人团队是本项目全部原创内容的 MIT 版权持有人。sata1921 为项目负责人。

---

## 第 1 层 · MIT — 原创内容

**版权人：sata1921, seeds, suihan, hilo114。** 可商用、可闭源、可再许可。保留版权行即可。

| 交付物 | 路径 | 说明 |
|---|---|---|
| KI8 1.4 核心 | `D:\moss-image\KI8-1.4.0-rc1\KI8` | 量化编解码器、LoRA 运行时、标定与格式选择（75 文件 / 2,635,539 B） |
| 转换器 adapter + 脚本 | `D:\moss-image\moss_converter_24576_bundle_20261117`（**`refs/` 除外**） | 14 张量 / 23,081,985 可训练参数 / step 900，全部从随机初始化训练 |
| 整合安装器 | `D:\moss-image\kedaiEX1_20261122\pkg\MOSS-Image_KI8_安装包` | install / uninstall / 分支测试（32 文件 / 466,271 B） |
| 本仓库工具与文档 | `tools/`、`docs/`、`configs/`、`notes/` | 预算模型、审计器、架构配置 |

---

## 第 2 层 · Apache-2.0 — 衍生权重与解码器

可以商用，但必须做 §4b（标注修改）+ §4c（保留 NOTICE）。

### 2a · 修改后的 MOSS-Image SFT DiT（KedaiEX1）

| 项 | 值 |
|---|---|
| 文件 | `MI-DIT-Prism-KedaiEX1_ki8_14.safetensors` |
| 体积 / sha256 | 5,241,594,706 B / `3dc0b5f82a85c91e…` |
| 张量 | 750，KI8 1.4 e2m5 |
| 上游 | `OpenMOSS-Team/MOSS-Image-SFT` @ `f972bd4b4b9e` |
| 修改 | KI8 1.4 权重量化（W32 组 / e2m5 元素网格 / E5F3 块尺度） |

衍生作品，**整份文件**按 Apache-2.0，包括团队自己贡献的量化部分。不能因为量化代码是 MIT 就声称权重也是 MIT。

### 2b · GSQ 文本编码器

| 项 | 值 |
|---|---|
| 文件 | `enc27b_gsq_rco_iq3s_headless.gguf` |
| 体积 / sha256 | 11,056,384,896 B / `0758183b9990a9d4…` |
| 上游 | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` ← `Qwen/Qwen3.8-27B` |
| 修改 | tap 移到 `last_rmsnorm`（hidden 5120）；去 LM head |

许可写在文件自己的 GGUF KV 表里，是全部条目中证据最硬的一类：

```
general.license          = apache-2.0
general.quantized_by     = ISTA DASLab
general.base_model.0.url = https://huggingface.co/Qwen/Qwen3.8-27B
```

### 2c · VAE — `flux2-vae_small.safetensors`（**2026-11-22 更正**）

| 项 | 值 |
|---|---|
| 文件 | `ComfyUI/models/vae/flux2-vae_small.safetensors` |
| 体积 / sha256 | 249,521,340 B / `d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b` |
| 上游 | `black-forest-labs/FLUX.2-small-decoder` → `diffusion_pytorch_model.safetensors` |
| 许可 | **Apache-2.0** |

**逐字节相同**：HF tree API 报 `size = 249,521,340`、`lfs.oid = d8d52ba0…013b`，与本地 sha256 完全一致。

上游 README 两处明说：

> 5. Available under the Apache 2.0 license.
>
> # License
> This model is licensed under the https://www.apache.org/licenses/LICENSE-2.0.

架构也对得上：本地 `moss_vae_config.json` 是 `"_class_name": "AutoencoderKLFlux2"`，正是上游 README 里 `AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-small-decoder")` 的那个类。

**我上一轮错在哪**：只看了 `Comfy-Org/Flux-2-Dev` 的**仓库级** `flux-1-dev-non-commercial-license` 标签。那个标签管的是该 repack 里的 DiT 和**全尺寸**解码器，不管这个文件 —— BFL 把同一份字节单独以 Apache-2.0 发布在 `FLUX.2-small-decoder`。**文件级溯源压过 repack 者的仓库级标签。**

### 2d · `refs/` 单列

转换器 bundle 标 MIT，但 `refs/` **不是原创**，MIT 不覆盖：

| 文件 | 字节 | 来源 |
|---|---:|---|
| `qwen38_tokenizer.json` | 12,809,320 | `Qwen/Qwen3.8-27B` |
| `moss_tokenizer.json` | 19,989,325 | `OpenMOSS-Team/MOSS-Image-Base` |
| `moss_tokenizer_config.json` | 15 | 同上 |
| `moss_te_config.json` | 2,813 | 同上 |
| `moss_transformer_config.json` | 240 | 同上 |
| `moss_vae_config.json` | 838 | 同上 |
| `moss_model_index.json` | 412 | 同上 |

7 文件 / 32,802,963 B / tree sha256 `edaf2c59292647af…` / **Apache-2.0，逐字节拷贝，零修改**。

---

## 第 3 层 · GPL-3.0 — 单独节点

**只在"作为独立 ComfyUI 节点包分发"这个形态下适用。** 7 个包 / 27 个节点：

| 包 | 节点数 |
|---|---:|
| `ComfyUI-KI8` | 3 |
| `ComfyUI-KI8-Triton` | 2 |
| `ComfyUI-MOSS-Cond` | 3 |
| `ComfyUI-MOSS-MoE` | 1 |
| `ComfyUI-MOSS-MoE16` | 1 |
| `ComfyUI-MOSS-NoLoRA` | 13 |
| `ComfyUI-MOSS-TE27B` | 4 |

它们 `import comfy.*` / `comfy_api.latest`，不能脱离宿主运行，所以这些包按 GPL-3.0 分发。

**落地清单**：每个包 `__init__.py` 顶部加 `# SPDX-License-Identifier: GPL-3.0-or-later`，随包附 `licenses/LICENSE-GPL-3.0.txt` 全文。

装入后同样变成 GPL 衍生的两个文件（不属于"单独节点"，也不在 MIT 授权内）：`comfy/ops.py`（打补丁）、`comfy/ki8_codec.py`（放进 `comfy/` 包内）。

---

## 不发布 · 仅本地

这三项**不进仓库**，因此不属于任何一层的授权范围：

| 项 | 为什么不发 |
|---|---|
| `Qwen-Image-2.1` | Qwen Research License，§2a 仅非商用 |
| Qwen-Image-2.1 的 LoRA | 继承基座非商用；其中两个文件上游不可考 |
| `moss_image_sft_4_steps_lora_v2`（Kijai 派生） | `Kijai/MOSS-image_experimental` 无任何许可声明，默认全权保留 |

KI8 README 里引用 Qwen-Image-2.1 的基准数字保留 —— 那是研究用途测量，不是分发。

---

## 隔离 · 全尺寸 VAE（不发布）

`flux2-vae.safetensors`（336,213,556 B / `d64f3a68…`）已移到：

```
ComfyUI/models/vae/_noncommercial_do_not_ship/
```

移出 `models/vae/` 是为了让 ComfyUI **扫不到**它 —— 文件在模型目录里就还能被加载。

**为什么非商用**：全尺寸解码器没有 Apache-2.0 发布。查过 `FLUX.2-klein-4B` / `FLUX.2-klein-9B`，都只发 DiT 单文件，不含独立 VAE。所以 `FLUX.2-small-decoder` 是 BFL 唯一以 Apache-2.0 发布的分量解码器。

---

## 一张图看完

```
┌─ MIT            Copyright (c) 2026 sata1921, seeds, suihan, hilo114 ─┐
│  KI8 核心 · 转换器 adapter/脚本 · 安装器源码 · 本仓库工具              │  商用 ✅ 闭源 ✅
└──────────────────────────────────────────────────────────────────────┘
┌─ Apache-2.0 ─────────────────────────────────────────────────────────┐
│  KedaiEX1 DiT · GSQ 编码器 · flux2-vae_small（FLUX.2 small decoder）   │  商用 ✅ 需署名+标注修改
│  refs/ 7 文件（单列，勿并进 MIT）                                      │
└──────────────────────────────────────────────────────────────────────┘
┌─ GPL-3.0 ────────────────────────────────────────────────────────────┐
│  7 个单独节点包 / 27 节点                                             │  商用 ✅ 须开源同许可
└──────────────────────────────────────────────────────────────────────┘
┌─ 不发布 ─────────────────────────────────────────────────────────────┐
│  Qwen-Image-2.1 · 其 LoRA · Kijai 派生 LoRA                           │  —  仅本地
└──────────────────────────────────────────────────────────────────────┘
┌─ 隔离 ───────────────────────────────────────────────────────────────┐
│  全尺寸 flux2-vae（非商用，移到 _noncommercial_do_not_ship/）          │  不参与出图
└──────────────────────────────────────────────────────────────────────┘
```

**商用结论变了**：上一版说"整条链路不能商用"是**错的**，错在把 VAE 判成非商用。现在三层（MIT / Apache-2.0 / GPL-3.0）都允许商用，各自的义务是署名、标注修改、开源同许可。装配完整出图链路可以商用。
