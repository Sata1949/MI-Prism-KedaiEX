# 硬件实测记录

> 本文只记录**实测**数字。没有实测的写 `未测`。
> 生成命令：`python tools/probe_hardware.py`（含 `--json`）。
> 每次换机器、换驱动、换内存条都要重跑并追加一节。

---

## 2026-11-16 · 参考机（Windows 11 / RTX 3070 Ti Laptop）

### 环境

| 项 | 值 |
|---|---|
| OS | Windows 11 (10.0.26200), AMD64 |
| Python | 3.13.17 |
| numpy | 2.5.4（torch / transformers / llama_cpp 均**未安装**） |
| 驱动 | 617.14 |

### GPU

| 项 | 值 | 来源 |
|---|---|---|
| 型号 | NVIDIA GeForce RTX 3070 Ti Laptop GPU | `nvidia-smi` |
| 显存总量 | 8192 MiB | `nvidia-smi` |
| 显存空闲 | 7178-7204 MiB（多次采样） | `nvidia-smi` |
| Compute Capability | 8.6（Ampere） | `nvidia-smi --query-gpu=compute_cap` |
| PCIe 代数（max） | Gen 4 | `nvidia-smi -q` |
| PCIe 链路宽度（max） | x16 | `nvidia-smi -q` |
| PCIe 链路宽度（当前） | **x8** | `nvidia-smi -q` |
| PCIe 代数（空闲时） | Gen 1（省电降速） | `nvidia-smi -q` |

**为什么记 PCIe 当前 x8：** 这不是省电降速（代数会降，宽度不会）。x8 是笔记本主板的物理走线。预算模型按 Gen4 x8 = 15.75 GB/s 理论、11 GB/s 有效计算。

### CPU

| 项 | 值 |
|---|---|
| 型号 | AMD Ryzen 7 6800H with Radeon Graphics |
| 核心 / 线程 | 8 / 16 |
| 架构 | Zen 3+（支持 AVX2，**不支持 AVX-512**） |

### 内存

| 项 | 值 | 来源 |
|---|---|---|
| 物理总量 | 17179869184 B = 16.00 GiB | `Win32_ComputerSystem.TotalPhysicalMemory` |
| 可见总量 | 15557 MiB = **15.19 GiB** | `Win32_OperatingSystem.TotalVisibleMemorySize` |
| 空闲（测试时） | 8290 MiB | 同上 |
| 条数 | **1 条** | `Win32_PhysicalMemory` 只有一行 |
| 标称频率 | 5600 MT/s | `Win32_PhysicalMemory.Speed` |
| 实际频率 | **4800 MT/s** | `Win32_PhysicalMemory.ConfiguredClockSpeed` |
| 厂商 | Micron Technology | 同上 |

**两个关键点：**
1. 16 GB 标称只对应 15.19 GiB 可见 —— 差额被核显（Radeon 680M）保留。
2. 单条意味着单通道。DDR5-4800 单通道理论 38.4 GB/s，实测 19.3 = 50% 效率。

### 内存带宽（numpy，含 0.7 s 预热）

```
mem copy : 22.1 GB/s
mem read : 19.3 GB/s  (by threads: {1: 6.0, 4: 13.3, 8: 19.0, 16: 19.3})
```

多次重复（`--quick` 模式，三次连续运行）：

| 次数 | copy GB/s | read GB/s（8T） | GEMV GFLOPS |
|---|---:|---:|---:|
| 1 | 23.8 | 19.2 | 10.90 |
| 2 | 23.7 | 18.8 | 10.58 |
| 3 | 25.4 | 20.5 | 10.96 |
| **区间** | **23.7-25.4** | **18.8-20.5** | **10.58-10.96** |

全量模式（`--quick` 关）单次：copy 22.1 / read 19.3（16T）/ GEMV 10.46。

**稳定区间是 18.8-20.5 GB/s 读、10.5-11.0 GFLOPS GEMV。** 任何落在这个区间外的数字都应重测。

**线程扩展性（最重要的工程结论）**：

| 线程数 | 读带宽 | 相对单线程 |
|---:|---:|---:|
| 1 | 6.0 GB/s | 1.0x |
| 4 | 13.3 GB/s | 2.2x |
| 8 | 19.0 GB/s | 3.2x |
| 16 | 19.3 GB/s | 3.2x（饱和） |

**8 线程就够了，16 线程无收益。** 这直接决定 P3 的反量化 kernel 应该开 8 线程。

### GEMV（numpy fp32，含 0.7 s 预热）

```
gemv : 10.46 GFLOPS fp32 4096x4096
```

多次重复：10.46 / 10.58 / 10.83 / 10.90 / 10.96 / 11.19 GFLOPS。取 **10.46-11.19** 作为区间。

**这个数字的用途是判死 CPU 计算，不是验证 GPU 带宽**：

```
4096×4096 GEMV 计算受限时间 = 2×4096×4096 / 10.46e9 = 3.21 ms
同矩阵带宽受限时间 @280GB/s  = 64 MiB / 280e9      = 0.24 ms
比例 13.4x → CPU 侧是算力瓶颈，不是带宽瓶颈
```

所以 CPU 在本项目里的唯一合法角色是**反量化与数据搬运**，不是矩阵乘。任何"CPU 也参与计算"的设计都需要先解决这 13 倍的算力差距。

### 量化路径吞吐（Python/NumPy）

```
int8 -> fp32 反量化式逐块转换：0.6 / 0.7 / 0.8 GB/s
```

**比内存带宽（19.3）低 24-32 倍。** 这是 Python 逐元素循环的开销，不是内存问题。结论：P3 的反量化必须用 C/C++ 扩展，且必须 8 线程。

### 磁盘

| 项 | 值 |
|---|---|
| C 盘总量 | 199 GiB |
| 已用 | 184 GiB |
| 可用 | **15.5 GiB** |

**这是阶段一的第一个阻断项**（见 `docs/05-risks.md` R1）。27B 在 4.5 bpw 下 14.39 GiB，加中间产物完全放不下。

### 工具链可用性

| 工具 | 状态 |
|---|---|
| git | 2.55.0.windows.3 ✅ |
| python | 3.13.17 ✅ |
| numpy | 2.5.4 ✅ |
| safetensors | 0.8.0 ✅ |
| torch | ❌ 未安装 |
| transformers | ❌ 未安装 |
| llama_cpp | ❌ 未安装 |
| cupy | ❌ 未安装 |
| nvcc (CUDA) | ❌ 不存在 |
| gcc / clang / cl | ❌ 全部不存在 |

**阶段二与阶段三的 3.4/3.5 全部阻塞在此**（`docs/05-risks.md` R2）。

---

## 探针方法的已知局限

### 1. 预热是必需的

同一次会话中，`--quick` 第一次跑出：

```
mem copy : 11.2 GB/s
mem read : 7.6 GB/s  (by threads: {1: 3.7, 8: 7.6})
gemv     : 3.81 GFLOPS
```

三次重跑稳定在 11.1-11.3 / 8.0 / 4.16-4.25，**比预热后低 2-3 倍**。原因是笔记本 CPU 在第一次突发负载时还在 base clock，没有升到 boost。

修复方式（已实现）：计时前空转 0.7 s（`_warm()`）。修复后 18.8-20.5 GB/s / 10.5-11.0 GFLOPS。

**任何不预热的基准都不可信。** 这也是为什么探针代码里保留了这段注释和实现。

### 2. `--quick` 只改轮次，不改缓冲区

早期版本让 `--quick` 同时缩小缓冲区（128 MiB vs 512 MiB），导致数字系统性偏低（缓冲区太小，无法触发完整的内存并行度）。现在 `--quick` 只减少重复次数与线程扫描范围，缓冲区固定 256 MiB。

### 3. numpy 的 GEMV 不是最优实现

10.46 GFLOPS 是在 numpy 的默认 BLAS 上测的。换成 OpenBLAS/MKL 的优化实现可能高 2-3 倍。**但这个数字的用途只是"判死 CPU 计算"，不需要更精确。** 真要优化 CPU 计算，应该直接上量化 kernel 而不是 fp32 GEMV。

### 4. 没有 GPU 侧测量

当前没有 CUDA/python-torch，所以**GPU 的实际带宽与算力完全没有测**。预算模型里 `BW_VRAM_GBPS = 280` 是"448 理论峰值的 62%"这个经验值，**不是本机实测**。

**这是一个明确的待办：装好 torch 后必须补测**
- `torch.cuda` 的显存拷贝带宽（`cudaMemcpy` D2D）
- 一个 4 bit 量化的 GEMV kernel 实测吞吐
- 真实 decode 速度（对比预算模型的 51.1 tok/s）

补齐后 `docs/05-risks.md` 的 R4（速度预测偏差）才能关闭。

---

## 附录：完整探针输出

```
$ python tools/probe_hardware.py
Moss KedaiEX hardware probe
  host     : Windows 11 (AMD64)
  gpu      : NVIDIA GeForce RTX 3070 Ti Laptop GPU  8192 MiB (free 7178 MiB)  cc 8.6  driver 617.14
  pcie     : Gen1 x8 current, Gen4 x16 max  [idle link-down; budget Gen4]
  cpu      : AMD Ryzen 7 6800H with Radeon Graphics 8c/16t
  ram      : 15.19 GiB total
  mem copy : 22.1 GB/s
  mem read : 19.3 GB/s  (by threads: {1: 6.0, 4: 13.3, 8: 19.0, 16: 19.3})
  gemv     : 10.46 GFLOPS fp32 4096x4096
  disk     : 15.5 GiB free of 199.0 GiB
```

JSON 结构（`--json`）：

```json
{
  "timestamp": "...",
  "platform": {"system": "Windows", "release": "11", "machine": "AMD64", "python": "3.13.17"},
  "gpu": {"available": true, "name": "...", "vram_total_mib": 8192, "pcie_gen_max": "4",
          "pcie_width_max": "16", "pcie_width_current": "8", "compute_capability": "8.6"},
  "ram_cpu": {"available": true, "ram_total_gib": 15.19, "CPU": "...", "CORES": "8", "THREADS": "16"},
  "bandwidth": {"available": true, "copy_gbps": 22.1, "stream_read_gbps": 19.3,
                "stream_read_by_threads": {"1": 6.0, "4": 13.3, "8": 19.0, "16": 19.3}},
  "gemv": {"available": true, "shape": "4096x4096", "gflops": 10.46, "dtype": "float32"},
  "disk": {"available": true, "total_gib": 199.0, "free_gib": 15.5}
}
```
