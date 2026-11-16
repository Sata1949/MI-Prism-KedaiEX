# ADR 0004 · P1 与 P3 的优化方向相反

- 状态：已接受
- 日期：2026-11-17
- 相关：`docs/02-budget-frontier.md` §5、`docs/03-roadmap.md` 阶段三

## 背景

在做带宽敏感性分析时发现，两个阶段的瓶颈位置完全不同：

| 配置 | `bw_vram_gbps` 280→400 | `bw_host_gbps` 17→68 |
|---|---|---|
| P1（10.15B Dense，全常驻） | 51.1 → **73.0**（+43%） | 51.1 → 51.1（0%） |
| P3（28.4B MoE，专家在内存） | 9.4 → **9.6**（+2%） | 9.4 → **30.0**（+219%） |

## 决策

**P1 的工程投入全部放在 GPU 侧；P3 的工程投入全部放在 CPU 侧与预取。两者不可复用。**

## 依据

1. **P1 是完全常驻的**。所有权重和 KV 都在显存里，所以每 token 的耗时 = 5.48 GiB / VRAM 带宽。提升 VRAM 带宽直接提速，提升内存带宽毫无用处。P1 对 `bw_host_gbps` 的敏感度是**精确的 0**（模型里 host 项为 0，因为 `arch.experts == 0`）。

2. **P3 的显存只占 4.82 GiB，且读取量只有 2.48 GB/token**。它的 97.7 ms/token 几乎全部花在从内存读专家（1.66 GB）。提升 VRAM 带宽只影响那 8.9 ms 的部分，所以 280 → 400 只带来 0.2 tok/s。

3. **这个差异会决定工程优先级**。如果误以为 P3 也需要 GPU 优化，会把时间浪费在 kernel 调优上，而真正的瓶颈（CPU 反量化 0.8 GB/s、缺少预取重叠）不受影响。

## 具体后果

| 阶段 | 该做什么 | 不该做什么 |
|---|---|---|
| 阶段一（P1） | 高效的 4-bit GEMV kernel、CUDA graph、KV 布局优化 | 优化 CPU 侧、写预取逻辑 |
| 阶段三（P3） | C/AVX2 反量化 kernel、双缓冲预取、路由预测 | 优化 GPU kernel（收益 < 2%） |

## 已知边界

这个结论在两种情况下会失效：

1. **P3 改小**：如果专家库缩小到能放进显存（比如 6 个专家而不是 12 个），P3 会向 P1 靠拢，GPU 侧变得重要。
2. **P1 改大**：如果 P1 溢出（任何 `spill > 0` 的配置），PCIe 会立刻主导（见 27B：95% 时间在 PCIe），此时 CPU 侧与 PCIe 优化变得关键。

所以这条 ADR 的准确表述是：**"在各自当前的配置下"方向相反**，不是"永远相反"。

## 复现

```bash
python tools/budget_model.py --arch p1 --set bw_vram_gbps=400
python tools/budget_model.py --arch p1 --set bw_host_gbps=68
python tools/budget_model.py --arch p3 --set bw_vram_gbps=400
python tools/budget_model.py --arch p3 --set bw_host_gbps=68
```
