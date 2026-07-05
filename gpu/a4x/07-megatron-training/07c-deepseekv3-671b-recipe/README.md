# DeepSeek V3 — GB200 NVL72 128 GPU HybridEP 训练复现指南

> forrest 集群 30+ 组实验, 从 300 优化到 **975 TFLOPs/GPU (v1)** / **985 TFLOPs/GPU (v2)**，达 NVIDIA 256 GPU 参考 1,106 TFLOPs 的 **89%**。
>
> 来源: [奚老师完整报告](https://doc.maxwell-x.dev/dsv3-hybridep-128g-optimization?t=9IBu8bMJhPhILN-CztzcKs&theme=gcloud) (2026-07-05)

## 核心优化路径

```
alltoall dispatcher (300) → HybridEP (+58%, 474) → CUDA graph partial capture (+96%, 928) → mxfp8 + fp32 optimizer (+105%, 975)
```

**关键限制**: 全量 61 层模型 CUDA graph 会 OOM (184 GB HBM 不够)，必须缩到 32 层 (~221B)。

## 1. 最佳配置速查

### v1 — 975 TFLOPs/GPU (MCore 0.16, TE 2.9)

| 参数 | 值 |
|---|---|
| 模型 | DSv3 缩减 32 层 ~221B, H=7168, 256 experts top-8, MLA |
| 层频率 | `([0]*3+[1]*29)` (3 dense + 29 MoE) |
| 并行 | **PP=2 EP=64** TP=1, seq=8192, MBS=1, GBS=2048 |
| FP8 | **mxfp8** e4m3 + **fp8-param-gather** + reuse-grad-buf-for-mxfp8-param-ag |
| CUDA Graph | `--cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess` |
| HybridEP | hybridep-num-sms=32, RANKS_PER_DOMAIN=64, USE_MNNVL=1 |
| Optimizer | fp32 main-grads + fp32 main-params, bf16 exp-avg + bf16 exp-avg-sq |
| Recompute | selective: moe_act, mlp |
| NCCL | **NVLS=0** GRAPH_REGISTER=0 MNNVL=0 |

### v2 — 985 TFLOPs/GPU (MCore 0.17, TE 2.15)

与 v1 仅 3 处差异:
- `--cuda-graph-modules attn` (v1 用 `--cuda-graph-scope attn moe_router moe_preprocess`)
- `--cross-entropy-fusion-impl native` (v1 用 `te`)
- `--moe-router-padding-for-quantization` (v1 无此参数)

### 3 个致命参数

| 参数 | 必须值 | 错误值后果 |
|---|---|---|
| `--cuda-graph-impl transformer_engine` | 必须显式设 | 漏掉 → impl=none, graph 静默禁用, 只有 879 不是 975 |
| `NCCL_GRAPH_REGISTER` | 0 | 1 → AssertionError crash (与 expandable_segments 冲突) |
| `NCCL_NVLS_ENABLE` | 0 | 1 → iter 20-40 后性能渐降 30-50% (时间相关退化 bug) |

## 2. 硬件与软件

| 项目 | 规格 |
|---|---|
| GPU | 128× GB200 (184 GB HBM3e), 32 nodes × 4 GPU |
| NVLink | 2 cliques × 16 nodes, NVL72 域内全互联 |
| RDMA | 4× MRDMA 400Gb/s per node (RoCE), GIB v1.1.2 |
| IMEX | Host nvidia-imex daemon (非 ComputeDomain), per-clique 独立 |

|  | v1 | v2 |
|---|---|---|
| Base | pytorch:25.09-py3 (CUDA 13.0) | pytorch:26.05-py3 (CUDA 13.2) |
| MCore | 0.16.0 (effebd81) | 0.17 dev (bfa3326) |
| TE | 2.9 (custom 7dd3914) | 2.15 (NGC 内置) |
| DeepEP | hybrid-ep 3f601f7 | 同 |

## 3. 全部实验汇总

### 3.1 DSv3 61L PP=4 EP=32 (v1)

| # | 配置变化 | TFLOPs | 备注 |
|---|---|---|---|
| 1 | baseline (alltoall, BF16) | 300 | |
| 1b | + mxfp8 | 298 | mxfp8 对 BF16 baseline 无提升 |
| 2 | HybridEP + blockwise + Ring | **432** | +44% |
| 3 | 同上, algo=auto | 474 (±30) | +58%, 波动大 |
| 4 | + NVLS=1 + GRAPH_REGISTER=1 | 488 (peak 518) | **61L 最高** |
| — | 61L + CUDA graph | OOM (SIGKILL) | 184 GB 不够 |

### 3.2 DSv3 32L PP=2 EP=64 (v1)

| # | 配置变化 | TFLOPs | 备注 |
|---|---|---|---|
| — | baseline (无 graph) | 783 | |
| 7 | + CUDA graph (attn+router+preprocess) | **928** | +18.5% |
| 8 | + mxfp8 + fp32 optimizer + fp8-param-gather | **970** | +24% |
| — | **复现 run** | **975** | **v1 最终** |
| 9 | PP=4 EP=32 (vs PP=2 EP=64) | 955 | PP=2 更优 |
| 22 | seq=4096 MBS=2 | 905 | seq 影响大 |
| 19 | + numactl 绑 NUMA 0 | 950 | -3%, Grace NUMA 延迟低不需要 |
| 18 | recompute mlp only (去 moe_act) | 772 | -20%, 内存压力 |

### 3.3 DSv3 32L PP=2 EP=64 (v2)

| # | 配置变化 | TFLOPs | 备注 |
|---|---|---|---|
| 10 | baseline 无 graph | **935** | v2 baseline 已很高 |
| 11 | graph attn+router+preprocess | 365 (-62%) | **v2 MoE graph 退化!** |
| 12 | **graph attn only** | **985** (+5.3%) | **v2 最终** |
| 13 | + NCCL_MIN_CTAS=32 | 916 (-7%) | CTA 挤占 SM |
| 14 | graph attn+router | 369 (-61%) | router 也不行 |
| 15 | seq=4096 | 710 (-28%) | 计算密度不够 |
| 16-17 | seq=12288 + offload | OOM | |

### 3.4 NVLS 退化排查 (7 组对照实验)

| # | 变量 | TFLOPs | 退化? |
|---|---|---|---|
| 27 | ComputeDomain + NVLS=1 | 427→hang | ✅ iter 44 |
| 28 | host IMEX + NVLS=1 | 460→260 | ✅ iter 33 |
| 30 | **NVLS=0 (baseline)** | **474 稳态** | **❌ 零退化** |
| 31 | NVLS=1 + 禁 GC | 526→368 | ✅ iter 19 |
| 32 | NVLS=1 + GRAPH_REG=0 | 441→333 | ✅ iter 27 |
| 33 | NVLS=1 + GPU 监控 | 446→344 | ✅ iter 41 |

**结论**: 退化在 NVLS transport 内部，排除了 multicast slot、GC、内存泄漏、thermal throttling。必须 NVLS=0。

### 3.5 Qwen3 235B 参考 (奚老师在同硬件测)

| hidden_size | seq | TFLOPs |
|---|---|---|
| 4096 | 4096 | 219 |
| 4096 | 8192 | 325 |
| **7168 (DSv3)** | **8192** | **985** |

> 计算密度差 3x (H=7168 vs 4096)，是 235B TFLOP/s 低的根因。

## 4. 关键发现

### 4.1 MoE CUDA Graph v1 vs v2 差异

**v1 (MCore 0.16)**: graph capture 含 MoE dispatch 全链路，零额外 sync → **975 TFLOPs**

**v2 (MCore 0.17)**: 引入 `MoECudaGraphPartialCaptureSignal` 截断 graph，MoE 回退到 graph 外，每层 3+ 次 graph boundary sync → **365 TFLOPs (-62%)**。v2 只能 graph attn → **985 TFLOPs**

### 4.2 PP=2 EP=64 优于 PP=4 EP=32

32L PP=2: 每 GPU 4 experts, EP 通信量小 → **975**
32L PP=4: 每 GPU 8 experts, 通信量翻倍 → **955 (-2%)**

### 4.3 ComputeDomain vs Host IMEX

CD 稳态低 5% 且 iter 44 hang。Host IMEX 更稳定。CD→host IMEX 切换需 kubelet restart 清理 IMEX session 残留。

### 4.4 Silicon Binning

cuBLAS BF16 跨节点 spread 8.8%。gb200-05 全 4 GPU 慢 4.7-7.2%。128 GPU 实验自然平均，影响 <1%。单节点 A/B 对比必须 pin 同一节点。

## 5. 通用 NCCL 配置

```bash
export NCCL_NET=gIB NCCL_MNNVL_ENABLE=0 NCCL_CUMEM_ENABLE=1
export NCCL_IB_GID_INDEX=3 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=52 NCCL_IB_FIFO_TC=84 NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_PXN_C2C=1
export NCCL_NVLS_ENABLE=0          # 必须禁用 (时间退化 bug)
export NCCL_GRAPH_REGISTER=0       # 必须 =0 (与 CUDA graph + expandable_segments 冲突)
export NCCL_SET_STACK_SIZE=1

# HybridEP (独立于 NCCL)
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64
export USE_MNNVL=1
export NVLINK_DOMAIN_SIZE=72
```

## 6. 已排除优化方向

| 方向 | 结果 | 原因 |
|---|---|---|
| NCCL_MIN_CTAS=32 | -7% | CTA 占 SM 过多 |
| bindpcie / numactl | -3% | Grace CPU NUMA 延迟极低 |
| seq > 8192 + offload | OOM | offload 释放不够 |
| recompute mlp only | -20% | 内存压力拖慢 allocator |
| optimizer-cuda-graph | CRASH | grad_norm 在 graph capture 中非法 |
| VPP + CUDA graphs | OOM | graph pool + VPP activation 超 184 GB |
| PP=8 EP=16 | -19% | EP=16 每 GPU 16 experts 通信量翻倍 |
| CUDA graph local impl | CRASH | HybridEP tensor view assert |
| NVLS=1 | 退化 | 时间相关退化 bug |
| GRAPH_REGISTER=1 | CRASH | 与 expandable_segments assert |

## 7. 对 Qwen3 235B 的启示

基于奚老师报告,以下优化可应用于 Qwen3 235B (NeMo recipe):

| 优化 | 奚老师结果 | 我们的状态 | 适用性 |
|---|---|---|---|
| MNNVL=0 + USE_MNNVL=1 | 标准配置 | ✅ 已用, 685 | 已验证 |
| NVLS=0 | 必须 | ✅ 已用 | 已验证 |
| GRAPH_REGISTER=0 | 必须 | ✅ 已用 | 已验证 |
| fp8-param-gather | 970→975 | ❌ 未测 | NeMo recipe 可能已包含 |
| fp32 main-grads + main-params | 标准配置 | ❌ 未测 | NeMo 默认 bf16 |
| PP=2 EP=64 (vs PP=2 EP=32) | 975 vs 955 | 我们 EP=32 (64 GPU) | 需 64 GPU 才能 EP=64 |
| 32L 缩减模型 | 解锁 CUDA graph | N/A | NeMo recipe 跑完整模型 |

> **计算密度差异**: DSv3 H=7168 vs Qwen3 235B H=4096,计算密度差 3x。这是 Qwen3 235B 无法达到 900+ TFLOPs 的根本原因,与优化无关。
