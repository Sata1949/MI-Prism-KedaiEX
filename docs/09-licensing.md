# 许可证清单

> 核查日期：2026-11-17
> 方法：本地文件头部解析 + HuggingFace API（经 hf-mirror 镜像，见 §5）。
> 原则：每个结论都标来源。查不到的写"未确认"，不猜。

> **⚠ 本文结论已被 2026-11-22 的第一阶段核查修正。**
> 下面 §1 速查表说"整条链都是 Apache-2.0 或 MIT，没有非商用限制" —— **这句不成立**。
> 2026-11-22 逐个回查上游后发现两处：
> - `flux2-vae_small`（安装器硬依赖）是 **FLUX 非商用许可**
> - `Qwen-Image-2.1` 是 **Qwen Research License（仅非商用）**
>
> 当前有效的结论见 **[docs/10-phase1-licensing.md](10-phase1-licensing.md)** 与
> `LICENSING-LAYERS.md`。本文保留作 2026-11-17 的历史记录；§2–§6 里关于 MIT、
> Apache-2.0、GGUF 头部取证的内容仍然有效。

---

## 1. 结论速查（2026-11-17 版，部分已被修正，见上方警示）

| 对象 | 许可证 | 商用 | 改再分发 | 来源 |
|---|---|---|---|---|
| **MI-Prism-KedaiEX 本项目** | **MIT** | ✅ | ✅ | 本仓库 `LICENSE` |
| Qwen3.8-27B（基座） | **Apache-2.0** | ✅ | ✅ | GGUF 头部 `general.license` |
| Qwen3.8-27B GGUF 量化（Unsloth） | 继承 Apache-2.0 | ✅ | ✅ | GGUF 头部 + 量化不改变许可 |
| Qwen3.5-9B（MOSS-Image 文本编码器） | **Apache-2.0** | ✅ | ✅ | HF API `license:apache-2.0` |
| MOSS-Image-Base / SFT（出图主模型） | **Apache-2.0** | ✅ | ✅ | HF API `cardData.license` |
| MOSS-Image ComfyUI 构建（Comfy-Org） | **Apache-2.0** | ✅ | ✅ | HF API，`base_model` 指向 OpenMOSS-Team |

**一句话：整条链都是 Apache-2.0 或 MIT，没有 copyleft，没有非商用限制。**

---

## 2. 本项目：MIT

`LICENSE` 是标准 MIT 全文（21 行，sha256 前缀 `55cf3f92`），逐条核对：

| 条款 | 状态 |
|---|---|
| 标题 `MIT License` | ✅ |
| 版权行 `Copyright (c) 2026 MI-Prism-KedaiEX contributors` | ✅ |
| 授权（use/copy/modify/merge/publish/distribute/sublicense/sell） | ✅ |
| "without restriction" | ✅ |
| 保留声明要求 | ✅ |
| AS IS 无担保 | ✅ |
| 免责（含 merchantability / fitness / noninfringement） | ✅ |

**注意一个笔误的历史**：第一版校验脚本因为「条款内含换行」把免责段误判为缺失。已改为先规范化空白再匹配。这不是许可缺陷，是检查方法的缺陷。

### 2.1 权重与代码的许可分离

README 写"许可：MIT（权重另行发布，不随仓库分发）"。这句话需要更精确，因为：

- **代码/文档** = MIT（本仓库）
- **衍生权重** = 受基座许可约束，即 **Apache-2.0**，不是 MIT

Apache-2.0 比 MIT 多了两项义务，做发布时必须遵守：

1. **保留 NOTICE 与版权声明**（Apache-2.0 §4c）
2. **标注修改**（§4b）—— 剪枝、量化、词表裁剪都属于"修改"，必须在发布页写明

建议发布衍生权重时附带：

```
本权重派生自 Qwen/Qwen3.8-27B (Apache-2.0)。
修改：层剪枝 65→26、FFN 17408→14336、词表 248320→131072、4.0bpw 量化、QAD 蒸馏。
原始许可全文见 https://www.apache.org/licenses/LICENSE-2.0
```

---

## 3. 基座模型：Apache-2.0（硬证据）

### 3.1 GGUF 头部直接读取

```bash
$ python tools/gguf_peek.py "D:/moss-image/moss27b/models/ms_dl/Qwen3.8-27B-UD-IQ3_S.gguf" --license

file        : .../Qwen3.8-27B-UD-IQ3_S.gguf  (11.21 GiB)
  general.license                    = apache-2.0
  general.name                       = Qwen3.8-27B
  general.quantized_by               = Unsloth
  general.base_model.0.name          = Qwen3.8 27B
  general.base_model.0.organization  = Qwen
  general.base_model.0.repo_url      = https://huggingface.co/Qwen/Qwen3.8-27B
```

这是**文件里写的字节**，不是我的推断。GGUF v3 的 KV 表未加密，读取不需要加载 11 GiB 权重。

### 3.2 下载来源

`models/ud/dl.sh` 里的实际下载地址：

```bash
BASE="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"
```

### 3.3 量化不改变许可

Unsloth 做的是格式转换 + 量化，属于 Apache-2.0 §4 所说的"derivative work"。Apache-2.0 **允许**这样做，但要求：

- 保留原版权与许可声明
- 标注修改（"quantized to IQ3_S"）

GGUF 头部已经把这些字段写进去了（`general.quantized_by = Unsloth`），符合要求。

---

## 4. MOSS-Image 栈：Apache-2.0

本地有三个相关文件，逐个查：

| 本地文件 | 用途 | 上游 | 许可证 |
|---|---|---|---|
| `ComfyUI/models/diffusion_models/moss_image_base_int8_convrot.safetensors` | 出图主模型 | `Comfy-Org/MOSS-Image` | Apache-2.0 |
| `ComfyUI/models/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors` | 文本编码器（9B 教师） | `Qwen/Qwen3.5-9B` | Apache-2.0 |
| `moss27b/models/ms_dl/Qwen3.8-27B-UD-IQ3_S.gguf` | 27B 学生 | `unsloth/Qwen3.8-27B-GGUF` | Apache-2.0 |

### 4.1 本地 safetensors 没有许可元数据

```python
# moss_image_base_int8_convrot.safetensors
# __metadata__ 为空；757 个张量，无 license 字段
```

这是 ComfyUI 生态的常态：单文件 safetensors 通常只留 `format` 字段。**所以不能靠本地文件判断许可**，必须回查上游仓库。

### 4.2 上游 API 查询结果

```bash
$ curl -sL "https://hf-mirror.com/api/models/OpenMOSS-Team/MOSS-Image-Base"
  license tags: ['license:apache-2.0']
  cardData.license: apache-2.0
  gated: False | private: False
```

```bash
$ curl -sL "https://hf-mirror.com/api/models/OpenMOSS-Team/MOSS-Image-SFT"
  license tags: ['license:apache-2.0']
  cardData.license: apache-2.0
  gated: False
```

```bash
$ curl -sL "https://hf-mirror.com/api/models/Comfy-Org/MOSS-Image"
  license tags: ['base_model:OpenMOSS-Team/MOSS-Image-Base', 'license:apache-2.0']
  cardData.license: apache-2.0
  siblings: [... 'diffusion_models/moss_image_base_int8_convrot.safetensors' ...]
```

注意第三条的 `siblings` 里**正好有本地那个文件名**，确认本地文件来自 Comfy-Org 的官方构建，而不是第三方重打包。

### 4.3 9B 文本编码器

```bash
$ curl -sL "https://hf-mirror.com/api/models/Qwen/Qwen3.5-9B"
  license tags: ['license:apache-2.0']
```

---

## 5. 网络访问记录（可复现）

本机无法直连 HuggingFace：

```bash
$ curl -sS --max-time 15 https://huggingface.co
curl: (28) Connection timed out
$ nslookup huggingface.co
*** UnKnown can't find huggingface.co: Query refused
```

改用镜像：

```bash
$ curl -sS --max-time 8 -o /dev/null -w "%{http_code}" https://hf-mirror.com
200
```

**所有 HF API 查询都走 `hf-mirror.com`。** 镜像返回的是同一 API 的 JSON，`license` 字段来自源站，可信度足够做法律判断的初步依据；正式商用前建议再直连核对一次（或在能访问源站的环境下复核）。

---

## 6. Apache-2.0 的实际义务（发布前自查清单）

Apache-2.0 宽松，但不是"随便用"。发布本项目衍生权重前逐条确认：

| # | 义务 | 条款 | 本项目怎么做 |
|---|---|---|---|
| 1 | 附许可全文 | §4a | `LICENSE-APACHE-2.0` 随权重发布 |
| 2 | 保留版权/NOTICE | §4c | 保留 Qwen + Unsloth 声明 |
| 3 | 标注修改 | §4b | 在模型卡写明剪枝/量化/蒸馏动作 |
| 4 | 专利授权与终止 | §3 | 无需动作（不会主动起诉即可） |
| 5 | 不使用商标 | §6 | 命名不得暗示 Qwen 官方背书 |

**第 5 条对本项目有直接影响**：项目名叫"MI-Prism-KedaiEX"，与 Qwen 无关，没问题；但如果宣传里写"Qwen 官方优化版"就违规了。

---

## 7. 需要补的

| 项 | 状态 | 怎么补 |
|---|---|---|
| `LICENSE-APACHE-2.0` 全文 | 未加入仓库 | 从 apache.org 取标准文本放入仓库根 |
| 衍生权重的 NOTICE 文件 | 未写 | 发布时随权重附 |
| 直连 HF 复核 | 未做 | 换能访问的网络环境跑一次 |
| `moss_image_sft_int8_convrot` 的许可 | 按 SFT 仓库推定 Apache-2.0 | 已由 `MOSS-Image-SFT` 的 cardData 确认 |
| 第三方 LoRA 的许可 | **未查** | 本机 `models/loras/` 若有第三方 LoRA，需逐个查 |

最后一条是真缺口：本机装了若干 LoRA（含社区 NSFW 微调），这些**不一定是 Apache-2.0**，混进发布物会污染许可。发布前必须逐个核对。

---

## 8. 复现命令

```bash
# 本项目 LICENSE
python -I - <<'EOF'
import pathlib, re
t = re.sub(r"\s+", " ", pathlib.Path("LICENSE").read_text(encoding="utf-8"))
for k, s in [("title","MIT License"),
             ("grant","Permission is hereby granted, free of charge"),
             ("as-is","THE SOFTWARE IS PROVIDED \"AS IS\""),
             ("liability","IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE")]:
    print(("OK  " if s in t else "MISS"), k)
EOF

# GGUF 许可（不加载权重）
python tools/gguf_peek.py "D:/moss-image/moss27b/models/ms_dl/Qwen3.8-27B-UD-IQ3_S.gguf" --license

# 上游仓库许可
curl -sL "https://hf-mirror.com/api/models/OpenMOSS-Team/MOSS-Image-Base" | \
  python -c "import json,sys;d=json.load(sys.stdin);print(d['cardData'].get('license'))"
```
