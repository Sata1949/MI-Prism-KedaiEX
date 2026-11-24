# 第一阶段许可与来源核查 · 2026-11-22

> 核查对象：KI8 1.4.0-rc1、KedaiEX1 DiT 权重、5120→4096 转换器、去头 GSQ 文本编码器、整合安装器。
> 方法：本地制品头部解析（GGUF KV / safetensors `__metadata__`）+ 逐文件 sha256 + 上游 HF/GitHub API 回查。
> 原则：**结论标注来源；查不到的写"未确认"，不推定。**

机器可校验版本：`phase1/components.json`（登记表）+ `tools/license_audit.py`（审计器）。本文是它的散文版，两者不一致时以审计器为准。

---

## 1. 结论速查

| # | 组件 | 版本 | 体积 | 许可 | 商用 | 可再分发 | 依据 |
|---|---|---|---|---:|---|---|---|
| 1 | KI8 核心（编解码器 + LoRA 运行时） | 1.4.0-rc1 | 2.51 MiB | **MIT** | ✅ | ✅ | 本项目原创 |
| 2 | KedaiEX1 DiT 权重 | ki8_14 | 4.88 GiB | **Apache-2.0** | ✅ | ✅ | 派生自 MOSS-Image-SFT |
| 3 | 转换器（KedaiEX1-Adapter） | e2e_24576 / step 900 | 168 MiB | **MIT** | ✅ | ✅ | 原创训练参数 |
| 3a | ↳ 转换器 `refs/`（tokenizer 等） | — | 31.3 MiB | **Apache-2.0** | ✅ | ✅ | 上游原样拷贝 |
| 4 | 去头 GSQ 文本编码器 | GSQ-RCO IQ3_S | 10.30 GiB | **Apache-2.0** | ✅ | ✅ | GGUF 头部实测 |
| 5 | 整合安装器 | 2026-11-22 | 455 KiB | **MIT** | ✅ | ✅ | 本项目原创 |
| — | ComfyUI 0.44.0（宿主，不随包发） | 0.44.0 | — | **GPL-3.0** | ✅ | ✅（传染） | GitHub API |
| — | `flux2-vae_small`（管线 VAE） | — | 238 MiB | **Apache-2.0** | ✅ | ✅ | 与 BFL `FLUX.2-small-decoder` 逐字节相同 |

**一句话**：五个组件都是 MIT / Apache-2.0，**整条出图链路可以商用**。唯一需要分层处理的是 ComfyUI 节点（GPL-3.0）—— 它要求同许可开源，不禁止商用。详见 §4。

**2026-11-22 更正（重要）**：本文先前判定 VAE 为非商用，**这是错的**。逐文件溯源后确认：

- 管线用的 `flux2-vae_small.safetensors`（249,521,340 B，sha256 `d8d52ba0…013b`）**与 `black-forest-labs/FLUX.2-small-decoder/diffusion_pytorch_model.safetensors` 逐字节相同**，该仓库 `cardData.license = apache-2.0`，README 明写 "Available under the Apache 2.0 license" 与 "This model is licensed under the https://www.apache.org/licenses/LICENSE-2.0"。
- 全尺寸 `flux2-vae.safetensors` 才是非商用，且**没有 Apache-2.0 发布**（`FLUX.2-klein-4B/9B` 只发 DiT 单文件，无独立 VAE）。已移到 `models/vae/_noncommercial_do_not_ship/`，不参与出图。

错因：只看 `Comfy-Org/Flux-2-Dev` 的**仓库级**非商用标签，没查 BFL 对同一份字节的独立 Apache-2.0 发布。**文件级溯源压过 repack 者的仓库级标签。**
MIT 版权署名已改为四人团队 `sata1921, seeds, suihan, hilo114`。
**商用结论随之翻转**：装配完整出图链路**可以商用**。

---

## 2. 五个组件逐个核实

### 2.1 KI8 1.4.0-rc1 — MIT

75 个文件 / 2,635,539 字节 / tree sha256 `beba38db…`。

纯原创：`ki8_lora/` 七个模块（codec / adapters / calibration / checkpoint / formats / runtime / tuning）、`tools/` 十三个脚本、`tests/` 两套测试、`run.py` 与 `ki8.bat`。没有从 ComfyUI 或任何 GPL 项目复制的代码路径。

**缺口**：发布包目录 `KI8-1.4.0-rc1/KI8/` 内**没有 LICENSE 文件**。只有仓库根的 MIT 覆盖它。修复：在打包步骤里把 `LICENSE` 复制进 `KI8/`。

### 2.2 KedaiEX1 DiT 权重 — Apache-2.0

5,241,594,706 字节 / sha256 `3dc0b5f8…` / 750 个张量。

文件头 `__metadata__`：

```json
{"encoder": "comfy.ki8_codec", "version": "1.4.0-rc1", "format": "ki8", "element": "e2m5"}
```

**没有 license 字段。** 这是 ComfyUI 生态 safetensors 的常态，不能拿本地文件判断许可。许可来自上游：

```bash
$ curl -sL "https://hf-mirror.com/api/models/OpenMOSS-Team/MOSS-Image-SFT" | \
    python -c "import json,sys;d=json.load(sys.stdin);print(d['cardData']['license'])"
apache-2.0
```

Apache-2.0 **允许**量化与再分发，附带 §4b（标注修改）+ §4c（保留 NOTICE）两项义务。发布时必须在模型卡写明：KI8 1.4 量化、W32 分组、e2m5 元素网格、E5F3 块尺度。

### 2.3 转换器 — MIT（参数）+ Apache-2.0（`refs/`）

37 个文件 / 662,600,769 字节 / tree sha256 `9df6e7e8…`。

拆成两块看，这是本组件最容易被搞错的地方：

| 部分 | 内容 | 许可 | 理由 |
|---|---|---|---|
| 训练产物 | `adapter_e2e_24576.safetensors`（176,305,196 B，sha256 `d9530032…`）<br>`train_state.safetensors`、`report.json`、`train_meta.json` | **MIT** | 14 个张量、23,081,985 可训练参数，全部为本项目从随机初始化训练产生。**不含** Qwen 或 MOSS 的任何权重复制，只以小模型特征为监督信号。训练数据的来源不免 A 及 B。 |
| 脚本 | `scripts/*.py`（5 个）、`src/moss27b/*.py`（7 个） | **MIT** | 原创 |
| `refs/` | `qwen38_tokenizer.json`(12.8 MB)、`moss_tokenizer.json`(20.0 MB)、5 个 config JSON | **Apache-2.0** | **上游原样拷贝**，逐字节复制。不适用本项目的 MIT 授权。 |

`refs/` 那 31.3 MiB 是最容易出错的部分：整个 bundle 标 MIT，里面却装着别人的文件。已单列为 `converter-refs` 组件（tree sha256 `edaf2c59…`，7 个文件）。

### 2.4 去头 GSQ 文本编码器 — Apache-2.0（文件内实证）

11,056,384,896 字节 / sha256 `0758183b…`。

**这是全部五项里证据最硬的一个**，因为许可写在文件自己的字节里：

```
$ python tools/gguf_peek.py enc27b_gsq_rco_iq3s_headless.gguf --license
general.license                    = apache-2.0
general.name                       = Qwen3.8-27B GSQ-RCO
general.quantized_by               = ISTA DASLab
general.base_model.0.name          = Qwen3.8 27B
general.base_model.0.organization  = Qwen
general.base_model.0.repo_url      = https://huggingface.co/Qwen/Qwen3.8-27B
```

GGUF v3 的 KV 表未加密，读它不需要加载 10.3 GiB 权重（毫秒级）。这条链是：Qwen3.8-27B（Apache-2.0）→ ISTA DASLab GSQ-RCO 量化（Apache-2.0，HF API `cardData.license = apache-2.0` 已复核）→ 本项目去头重导出。

**`NOTICE` 现有内容有误**：只提 Unsloth，没提 ISTA DASLab。本机同时存在 `Qwen3.8-27B-UD-IQ3_S.gguf`（Unsloth）和 `Qwen3.8-27B-GSQ-RCO-IQ3_S`（ISTA）两条线，两者都必须署名。已修，见 §6。

### 2.5 整合安装器 — MIT

32 个文件 / 465,901 字节 / tree sha256 `3c2d26fb…`。

`install_moss.py` / `uninstall_moss.py` / `test_branches.py` / `MANIFEST.json` 全部原创。MIT 可以标。

**但它的产物不是 MIT**：安装器会往目标 ComfyUI 树里写 `comfy/ki8_codec.py`、改 `comfy/ops.py`、铺 7 个 custom_nodes 包。见 §4。

---

## 3. 许可文本齐备情况

| 许可 | 文件 | 字符数 | 状态 |
|---|---|---:|---|
| MIT | `LICENSE` | 1,086 | ✅ 仓库根 |
| Apache-2.0 | `LICENSE-APACHE-2.0` | 11,358 | ✅ 仓库根 |
| GPL-3.0 | `licenses/LICENSE-GPL-3.0.txt` | 35,149 | ✅ 本次新增 |
| Qwen Research License | `licenses/LICENSE-QWEN-RESEARCH.txt` | 7,821 | ✅ 本次新增 |
| FLUX.1-dev Non-Commercial | `licenses/LICENSE-FLUX-NONCOMMERCIAL.md` | 18,510 | ✅ 本次新增 |

三份新文本取自权威源：GPL-3.0 复制自本机 ComfyUI 自带的 `LICENSE`（与 gnu.org 标准文本一致）；Qwen 与 FLUX 从 HF 镜像原始 blob 拉取。

> HF 直连被 DNS 拒绝（`curl: (28) Connection timed out`），全部查询走 `hf-mirror.com`。镜像返回同一 API 的 JSON，`license` 字段来自源站。正式商用前建议在能访问源站的环境复核一次。

---

## 4. 冲突一：ComfyUI 是 GPL-3.0，安装器在造衍生作品

### 4.1 事实

```bash
$ curl -sL https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/LICENSE | head -3
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007
$ curl -s https://api.github.com/repos/comfyanonymous/ComfyUI | \
    python -c "import json,sys;print(json.load(sys.stdin)['license']['spdx_id'])"
GPL-3.0
```

本机 `D:\moss-image\ComfyUI\LICENSE` 是同一份 GPL-3.0 全文，`comfyui_version.py` 写 `__version__ = "0.44.0"`。

### 4.2 三条 GPL 接触面

| 接触面 | 具体动作 | GPL 后果 |
|---|---|---|
| `comfy/ki8_codec.py` | 安装器把文件**放进 ComfyUI 自己的 `comfy/` 包内**，该文件第 434 行 `from comfy.ops import _eager_input_act, _linear_residual`，第 45 行 `from comfy_kitchen.tensor import ...` | 与 GPL 代码同进程、同包、双向调用 → 整体视为 GPL-3.0 组合作品 |
| `comfy/ops.py` 补丁 | 在 GPL 文件里插入两段带标记的代码（`LOAD_BEGIN`/`SAVE_BEGIN`，共 4 个标记常量） | 直接修改 GPL 源文件 → 修改后的 `ops.py` **本身**就是 GPL-3.0 衍生作品 |
| 7 个 custom_nodes | 全部 `import comfy.*` 或 `from comfy_api.latest import ...`（`invert_nodes.py` 4 处、`ref_nodes.py` 3 处、`vision27b_nodes.py` 3 处、`textmask_nodes.py` 2 处、`unified_nodes.py`/`cond_nodes.py` 各 1 处） | 典型的 GPL 宿主插件，只在 ComfyUI 内可运行 |

### 4.3 这意味着什么

- **安装器源码**可以继续 MIT。它本身不包含 GPL 代码。
- **安装后的 ComfyUI 树**是 GPL-3.0 组合作品。对**外分发**这棵树，必须整棵按 GPL-3.0 提供源码。
- **`custom_nodes` 包**单独发布也是 GPL-3.0（import 宿主内部模块）。ComfyUI 生态这么做的项目很多，原因就是这条。
- 现在的 README 写"许可：代码 MIT"没有区分层次。这句话对 KI8、转换器、安装器成立，对节点包和 `ki8_codec.py` 的**分发形态**不成立。

### 4.4 处理方案（推荐 A）

| 方案 | 做法 | 代价 |
|---|---|---|
| **A（推荐）** | 三层分许可：**MIT** = KI8 核心 + 转换器 + 安装器源码 + DiT 权重（Apache-2.0）；**GPL-3.0** = `comfy/ki8_codec.py` 落地形态 + 7 个节点包；**不随包分发** = ComfyUI 本体（用户自装） | 需要在 README、每个节点包头部、安装器 step-1 banner 三处标明 |
| B | 整包改 GPL-3.0 | 最简单，但把 MIT 的 KI8 核心也拖进 copyleft，损失不必要的自由度 |
| C | 安装器改为运行时从网络拉取 GPL payload | 增加部署复杂度，且离线场景不可用；不解决"改 ops.py"这一点 |

**方案 A 的落地清单**：
1. `phase1/components.json` 已把 `comfyui-ki8-codec` 和 `comfyui-nodes` 单列为 `license: GPL-3.0` / `license_status: combined-work`。
2. 每个 custom_nodes 的 `__init__.py` 顶部加 `# SPDX-License-Identifier: GPL-3.0-or-later`。
3. 安装器 step-1 打印一行：`本包装入 ComfyUI 后，受管文件按 GPL-3.0 分发；安装器与 KI8 核心自身仍为 MIT`。
4. README 的 "许可" 一行改成三层表述。

---

## 5. VAE 溯源：一次判错与更正（2026-11-22）

> **结论：VAE 是 Apache-2.0。本节原来的"VAE 非商用，整条链路不能商用"是错的，已作废。**
> 保留过程记录，因为这个错误很容易再犯一次。

### 5.1 判错的原因

先前只查了 `Comfy-Org/Flux-2-Dev` 的**仓库级**标签：

```bash
$ curl -s "https://huggingface.co/api/models/Comfy-Org/Flux-2-Dev" | ...
  license: other | license_name: flux-1-dev-non-commercial-license
```

于是把整仓的标签套到了 VAE 文件头上。**这一步是错的**：仓库级标签管的是该 repack 里的 DiT 和全尺寸解码器，不必然覆盖每一个文件。

### 5.2 正确做法：按 sha256 反查文件级出处

管线用的文件是 `flux2-vae_small.safetensors`：

```
本地           249,521,340 B   d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b
```

拿这个 sha256 去反查，命中的是：

```
black-forest-labs/FLUX.2-small-decoder
  diffusion_pytorch_model.safetensors   249,521,340 B   d8d52ba0…013b   ← 逐字节相同
  cardData.license                     = apache-2.0
  tags                                 = license:apache-2.0
```

README 两处明写：

> 5. Available under the Apache 2.0 license.
>
> # License
> This model is licensed under the https://www.apache.org/licenses/LICENSE-2.0.

架构也闭合：本地 `refs/moss_vae_config.json` 是

```json
{"_class_name": "AutoencoderKLFlux2", "latent_channels": 32, ...}
```

正是上游 README 里 `AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-small-decoder")` 的那个类。

### 5.3 全尺寸解码器确实非商用，但没有 Apache 替代

`flux2-vae.safetensors`（336,213,556 B / `d64f3a68…`）只有两个官方出处，都是非商用：`black-forest-labs/FLUX.2-dev` 与 `Comfy-Org/Flux-2-Dev`。

查过是否存在 Apache-2.0 的全尺寸版本 —— `FLUX.2-klein-4B` / `FLUX.2-klein-9B` 都只发 DiT 单文件（7.75 GB / 18.16 GB），**不含独立 VAE**。所以 `FLUX.2-small-decoder` 是 BFL 唯一以 Apache-2.0 发布的分量解码器。

处理：已移到 `ComfyUI/models/vae/_noncommercial_do_not_ship/`，移出模型目录是为了让 ComfyUI 扫不到它。

### 5.4 影响

| 项 | 更正前 | 更正后 |
|---|---|---|
| 管线 VAE 许可 | FLUX 非商用 | **Apache-2.0** |
| 整链商用 | ❌ 受阻 | **✅ 允许** |
| 全尺寸 VAE | 默认，被使用 | 隔离，不参与出图 |
| 剩余义务 | — | §4b 标注修改、§4c 保留 NOTICE、GPL 层开源同许可 |

### 5.5 教训

**文件级溯源压过 repack 者的仓库级标签。** 判断单个权重文件的许可，必须拿它自己的 sha256 去反查发布源，不能继承它所在仓库的 tag —— 一个 repack 仓库里可能同时装着不同许可的文件。

同理，`Qwen-Image-2.1` 是 Qwen Research License（§2a 仅非商用）。该项**不进仓库**，只留在开发机；KI8 README 引用它的基准数字属研究用途测量。

---

## 6. NOTICE 修正

现有 `NOTICE` §2 只写了 Unsloth。本机实际使用的 27B 文本编码器来自 **ISTA DASLab** 的 GSQ-RCO 量化，两条线都要署名。已改为：

```
2. Quantization
--------------------------------------------------------------------------------
Qwen3.8-27B-GSQ-RCO-GGUF (IQ3_S, mixed-precision GSQ)
  Quantized by: ISTA DASLab (Institute of Science and Technology Austria)
  License: Apache License 2.0 (inherited; quantization is a derivative work)
  Source:  https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
  Used as: the 27B text encoder, re-exported headless (tap at last_rmsnorm)

  Modifications by MI-Prism-KedaiEX:
    - hidden-state tap moved to last_rmsnorm
    - LM head removed (headless export)

Qwen3.8-27B-GGUF (UD-IQ3_S, UD-IQ4_XS)
  Quantized by: Unsloth
  License: Apache License 2.0 (inherited; quantization is a derivative work)
  Source:  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
```

---

## 7. 待办（按阻塞程度排序）

| # | 项 | 阻塞什么 | 怎么修 |
|---|---|---|---|
| 1 | VAE 非商用 | 商用发布 | §5 方案 A 或 B |
| 2 | 节点包 GPL-3.0 标注 | 对外分发 | §4.4 四条 |
| 3 | `KI8-1.4.0-rc1/KI8/` 缺 LICENSE | 独立发布 KI8 | 打包时复制 |
| 4 | `converter-refs` 未单列进发布说明 | Apache §4a/§4c | 发布页写明这 7 个文件是上游拷贝 |
| 5 | Kijai LoRA 无许可 | 分发 `moss_image_sft_4_steps_lora_v2` | 找作者或换 akhaliq Apache-2.0 源 |
| 6 | HF 直连不可用 | 许可复核的最终确认 | 换网络环境复核一次 |
| 7 | Qwen-Image-2.1 基准的措辞 | 商业宣传合规 | README 加非商用范围注 |

`tools/license_audit.py` 已把这 7 项里可机检的部分（1/2/3/4 的登记一致性）纳入 20 项检查。剩下的是需要人决策的商业判断。

---

## 8. 复现

```bash
cd D:/MI-Prism-KedaiEX

# 全量审计：路径 / 体积 / sha256 / GGUF 头部 / 许可文本篇幅
python -I tools/license_audit.py

# 只查结构，不算哈希（秒级）
python -I tools/license_audit.py --skip-hashes

# 出机器可读报告
python -I tools/license_audit.py --json phase1/license_audit.json

# 单独读 GGUF 许可（不加载权重，毫秒级）
python -I tools/gguf_peek.py "D:/moss-image/moss27b/models/enc27b_gsq_rco_iq3s_headless.gguf" --license

# 上游许可回查
curl -s "https://hf-mirror.com/api/models/OpenMOSS-Team/MOSS-Image-SFT" | \
  python -c "import json,sys;print(json.load(sys.stdin)['cardData']['license'])"
curl -s "https://api.github.com/repos/comfyanonymous/ComfyUI" | \
  python -c "import json,sys;print(json.load(sys.stdin)['license']['spdx_id'])"
```
