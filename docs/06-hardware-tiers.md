# 硬件档次覆盖

> 项目定位是"中低端机"，但"中低端"本身是个模糊词。本文把它变成一张表：
> 每档显存的机器上，四套配置各自跑多少 tok/s，能不能过 30 tok/s 门槛。

---

## 1. 为什么需要这张表

参考机是 RTX 3070 Ti Laptop（448 GB/s 峰值带宽），这在 8 GB 卡里属于**偏高端**。用它验证通过的配置，搬到真正的"中低端机"上可能直接跌破门槛：

```
P1（10.15B / 4.0bpw）在参考机：51.1 tok/s  ✅
P1 在 RTX 3050 Laptop（224 GB/s 峰值）：25.4 tok/s  ❌ 低于 30 门槛
```

**这不是小数点问题，是可用性问题。** 所以必须为低档位单独准备一套配置。

---

## 2. 解码速度的物理公式

单 token 的解码时间由带宽决定（详见 `docs/02-budget-frontier.md` §1）：

```
tok/s = 1 / (每 token 读取字节 / 显存带宽)
```

每 token 读取字节由模型结构决定，与 GPU 无关。所以**换显卡只是换带宽，速度线性缩放**。这让我们可以用一台机器上的实测推定全系列：

```
tok/s(GPU_B) = tok/s(GPU_A) × 带宽_B / 带宽_A
```

这个线性关系在 batch=1、带宽受限时成立；batch>1 或算力受限时不成立（见 §5）。

---

## 3. GPU 档位与带宽预算

带宽预算取峰值的 62%（理由见 `docs/BENCHMARKS.md`：GEMV 链达不到峰值）。

| GPU | 显存 | 位宽 / 速率 | 峰值 GB/s | 预算 GB/s（62%） |
|---|---:|---|---:|---:|
| RTX 4090 Laptop | 16 GB | 256-bit GDDR6 @18 | 576 | 357 |
| RTX 4080 Laptop | 12 GB | 192-bit GDDR6 @18 | 432 | 268 |
| RTX 3070 Ti / 3070 Laptop | 8 GB | 256-bit GDDR6 @14 | 448 | 278 |
| RTX 3060 Laptop | 6 GB | 192-bit GDDR6 @14 | 336 | 208 |
| RTX 4060 Laptop | 8 GB | 128-bit GDDR6 @17 | 272 | 169 |
| RTX 3050 Laptop | 8 GB / 4 GB | 128-bit GDDR6 @14 | 224 | 139 |
| GTX 1650 Laptop | 4 GB | 128-bit GDDR5 @8 | 192 | 119 |

**注意 RTX 4060 Laptop 的带宽比 3070 Ti Laptop 低 39%**（128-bit vs 256-bit）。这是 40 系的真实情况：架构更新但位宽砍半。所以"新一代卡更快"在带宽受限的场景下不成立。

---

## 4. 四套配置 × 八个档位

`tok/s` 全部由 `--set bw_vram_gbps=<预算>` 计算得出，取整到一位小数。

| GPU | 预算 | P1 (10.15B) | **P1-low (8.43B)** | P2 (9.26B) | P3 (28.4B MoE) |
|---|---:|---:|---:|---:|---:|
| RTX 4090 Laptop 16G | 357 | 65.2 | **86.3** | 46.8 | 9.6 |
| RTX 4080 Laptop 12G | 268 | 48.9 | **64.7** | 38.5 | 9.3 |
| RTX 3070 Ti Laptop 8G | 278 | 50.7 | **67.1** | 39.5 | 9.4 |
| RTX 3060 Laptop 6G | 208 | 38.0 | **50.4** | 32.1 | 9.1 |
| RTX 4060 Laptop 8G | 169 | 30.8 | **40.8** | 27.2 | 8.9 |
| RTX 3050 Laptop 8G | 139 | **25.4 ✗** | **33.6** | 23.3 ✗ | 8.7 |
| RTX 3050 Laptop 4G | 139 | 显存不够 | 显存不够 | 显存不够 | 显存不够 |
| GTX 1650 Laptop 4G | 119 | 显存不够 | 显存不够 | 显存不够 | 显存不够 |

（粗体 = 该档位的最佳选择；✗ = 低于 30 tok/s 门槛）

### 读出三条结论

**1. P1-low 是唯一覆盖全部 ≥6 GB 档位的配置。**
在 RTX 3050 Laptop 8G（本表最低的 8 GB 卡）上跑 33.6 tok/s，刚过门槛。

**2. P1 需要 RTX 4060 Laptop 及以上。**
4060 上是 30.8 tok/s，勉强过线；3050 上是 25.4，不够。所以 `configs/p1_target.json` 的适用条件是"8 GB 且带宽预算 ≥ 169 GB/s"。

**3. P3 的速度在所有档位都被显存带宽限制在 8.4-9.6 tok/s。**
因为它抄近路：显存只放 4.82 GiB 主体，专家流式读 —— 瓶颈在内存带宽不在显存带宽。**换更快的 GPU 对 P3 几乎没有帮助**（357 GB/s 时仍是 9.6）。这条见 `notes/decisions/0004`。

---

## 5. 4 GB 显存的硬墙

4 GB 卡可用显存只有约 3.2 GiB。在这个包线内，最大能放下的 Dense 配置是：

| 配置 | 参数 | 显存 | 缺口 |
|---|---:|---:|---:|
| 20L / ffn 14336 / vocab 64k @3.5bpw | 6.30B | 4.04 | −0.84 |
| 22L / ffn 12288 / vocab 64k @3.5bpw | 6.17B | 4.05 | −0.85 |
| 24L / ffn 14336 / vocab 64k @3.5bpw | 7.49B | 4.65 | −1.45 |

**全部超包线。** 最接近的 6.30B 方案仍差 0.84 GiB，需要溢出到 PCIe — 而 4 GB 卡上的溢出会把速度从估算的 30+ 拖到个位数（PCIe 只 11 GB/s，是显存带宽的 1/13）。

所以项目的明确边界是：

> **最低支持 6 GB 显存的机器（RTX 3060 Laptop 档）。4 GB 卡不在支持范围内。**

4 GB 卡要跑这个模型，需要另立方案（不共享第一阶段的剪枝路径），目前不做。

---

## 6. 门槛的定义与例外

`decode_tok_s_floor = 30` 是写死在 `configs/*.json` 的 `acceptance` 段里的。它的依据是"能用来写代码"的交互体验阈值，不是绝对的性能标准。

**如果你的用法不同，门槛应该不同：**

| 用法 | 合理门槛 | 说明 |
|---|---:|---|
| 交互式写代码 / 对话 | 30 tok/s | 默认值 |
| 批量离线生成 | 10 tok/s | 可以等 |
| 长文摘要（输入远大于输出） | prefill 主导，见 `docs/04-eval-protocol.md` §4.1 | decode 门槛不适用 |

改门槛只需改配置文件的 `decode_tok_s_floor`，门禁会按新值判定：

```bash
# 举例：把 P1 的门槛降到 20，看它在 3050 上能否通过
python tools/eval_gate.py configs/p1_target.json --set bw_vram_gbps=139
```

---

## 7. 复现

```bash
# 单档位
python tools/budget_model.py --arch p1l --set bw_vram_gbps=139

# 生成上表的任一行
python tools/budget_model.py --arch p1  --set bw_vram_gbps=169
python tools/budget_model.py --arch p1l --set bw_vram_gbps=169
python tools/budget_model.py --arch p2  --set bw_vram_gbps=169
python tools/budget_model.py --arch p3  --set bw_vram_gbps=169

# 4 GB 包线检查
python tools/budget_model.py --arch p1l --set vram_gib=3.2

# 门禁在低档位上的判定（这才是交付时的实际检查）
python tools/eval_gate.py configs/p1_low.json --set bw_vram_gbps=139
```

上表全部 24 个 tok/s 数字由 `tools/audit_docs.py` 的 GPU tier 段逐项复核（`TIER_BW_FACTOR = 0.62`）。

---

## 8. 已知局限

**这张表全是模型预测，没有一档是实测。** 原因：

- 只有一台参考机（3070 Ti Laptop），无法实测其他档位
- 62% 的带宽效率是经验值，不是实测（`docs/05-risks.md` R4）

所以表里的绝对数字可能整体偏高或偏低 20-30%，但**相对关系（档位排序、配置排序）是可靠的**，因为所有数字乘的是同一个系数。

补齐方式：在至少一台低端机（3050 档）上跑一次真实 decode，把实测值写进 `docs/BENCHMARKS.md`，并反推真实的效率系数。
