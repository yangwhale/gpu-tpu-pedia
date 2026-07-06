# DeepSeek V3 — GB200 NVL72 128 GPU HybridEP 训练复现指南

> forrest 集群 40+ 组实验, 从 300 优化到 **981 TFLOPs/GPU (v3.1, peak 993)**，达 NVIDIA 256 GPU 参考 1,106 TFLOPs 的 **89%**。
>
> 来源: [奚老师完整报告](https://doc.maxwell-x.dev/dsv3-hybridep-128g-optimization?t=9IBu8bMJhPhILN-CztzcKs&theme=gcloud) (持续更新)

## 版本演进

v1 (MCore 0.16, 975) → v2 (MCore 0.17 dev, 985, 需 runtime patch) → v2.1 (patch baked, 956) → **v3.1 (MCore 0.18.0, 981, 推荐)**

## 核心优化路径

```
alltoall dispatcher (300) → HybridEP (+58%, 474) → CUDA graph partial capture (+96%, 928)
→ mxfp8 + fp32 optimizer (+105%, 975) → MCore 0.18.0 + graph attn (+109%, 981)
```

**关键限制**: 全量 61 层模型 CUDA graph 会 OOM (184 GB HBM 不够)，必须缩到 32 层 (~221B)。

## 1. 最佳配置速查

### v3.1 — 981 TFLOPs/GPU (推荐, MCore 0.18.0)

| 参数 | 值 |
|---|---|
| 模型 | DSv3 缩减 32L ~221B, H=7168, 256 experts top-8, MLA |
| 并行 | **PP=2 EP=64** TP=1, seq=8192, MBS=1, GBS=2048 |
| FP8 | mxfp8 e4m3 + **fp8-param-gather** + reuse-grad-buf |
| CUDA Graph | `--cuda-graph-impl transformer_engine --cuda-graph-modules attn` |
| HybridEP | hybridep-num-sms=32, RANKS_PER_DOMAIN=64, USE_MNNVL=1 |
| Optimizer | fp32 main-grads + fp32 main-params, bf16 exp-avg/sq |
| Recompute | selective: moe_act, mlp |
| NCCL | **NVLS=0** GRAPH_REGISTER=0 MNNVL=0 |
| Patch | nvidia-resiliency-ext 0.6.0 + fused_a2a.py non_blocking 删除 |

### v1 — 975 TFLOPs/GPU (MCore 0.16)

与 v3.1 差异: `--cuda-graph-scope attn moe_router moe_preprocess` (含 full MoE graph capture)，无需 patch。

### v2 — 985 TFLOPs/GPU (MCore 0.17 dev, 需 runtime patch)

与 v3.1 差异: `--cross-entropy-fusion-impl native`, `--moe-router-padding-for-quantization`。v2 image 启动即 crash 需 sed patch。

### v2.1 — 956 TFLOPs/GPU (patches baked)

同 v2 但 patches baked in Dockerfile，pin MCore 到 bfa3326。

### 4 个致命参数

| 参数 | 必须值 | 错误值后果 |
|---|---|---|
| `--cuda-graph-impl transformer_engine` | 必须显式设 | 漏掉 → graph 静默禁用 (v3.1: 981→836) |
| `NCCL_GRAPH_REGISTER` | 0 | 1 → AssertionError crash |
| `NCCL_NVLS_ENABLE` | 0 | 1 → iter 20-40 后性能渐降 30-50% |
| `--sequence-parallel` | 保留 | 关掉 → -17 TFLOPs (与 NVIDIA ref 建议相反) |

### MCore 版本选择 (关键)

| MCore | graph 参数 | HybridEP 状态 |
|---|---|---|
| 0.16 (v1) | `--cuda-graph-scope` | ✅ full graph 975 |
| 0.17.0/0.17.1 | `--cuda-graph-scope` | ❌ **hang** (HybridEP 集成不完整) |
| **0.18.0 (v3.1)** | `--cuda-graph-modules` | ✅ attn-only 981 |
| dev bfa3326 (v2/v2.1) | `--cuda-graph-modules` | ✅ attn-only 956-985 |

**v0.17.0/v0.17.1 无论 graph/无 graph 都 hang**，必须跳过。

### v2.1/v3.1 需要的 2 个 patch

1. `nvidia-resiliency-ext` 0.6.0 from GitHub (NGC 26.05 只有 0.5.0, MCore 要求 ≥0.6.0)
2. `fused_a2a.py` 删除 `non_blocking=non_blocking,` (MCore 加了此参数, DeepEP hybrid-ep 不支持)

## 2. 全部实验汇总

### 2.1 DSv3 61L PP=4 EP=32 (v1)

| # | 配置变化 | TFLOPs | 备注 |
|---|---|---|---|
| 1 | baseline (alltoall, BF16) | 300 | |
| 2 | HybridEP + blockwise + Ring | **432** | +44% |
| 3 | algo=auto | 474 (±30) | +58%, 波动大 |
| 4 | + NVLS=1 + GRAPH_REG=1 | 488 (peak 518) | **61L 最高** |
| — | 61L + CUDA graph | OOM | 184 GB 不够 |

### 2.2 DSv3 32L PP=2 EP=64 (v1)

| # | 配置变化 | TFLOPs | 备注 |
|---|---|---|---|
| — | baseline (无 graph) | 783 | |
| 7 | + CUDA graph (attn+router+preprocess) | **928** | +18.5% |
| 8 | + mxfp8 + fp32 optimizer + fp8-param-gather | **970** | +24% |
| — | **复现 run** | **975** | **v1 最终** |
| 9 | PP=4 EP=32 | 955 | PP=2 更优 |

### 2.3 DSv3 32L PP=2 EP=64 (v2/v2.1)

| # | 配置变化 | TFLOPs | 备注 |
|---|---|---|---|
| 10 | baseline 无 graph (v2) | **935** | |
| 11 | graph attn+router+preprocess (v2) | 365 (-62%) | **v2 MoE graph 退化!** |
| 12 | **graph attn only (v2)** | **985** (+5.3%) | **v2 最终** |
| — | v2.1 patches baked 复现 | **956** | |

### 2.4 DSv3 32L PP=2 EP=64 (v3.1, MCore 0.18.0)

| # | 配置变化 | TFLOPs | 备注 |
|---|---|---|---|
| E1 | baseline 无 graph | 836 | |
| **E2** | **graph attn** | **981** (peak 993) | **v3.1 推荐** |
| E4 | 去掉 sequence-parallel | 964 (-17) | SP 在 TP=1 仍有益 |
| E5 | recompute mla_up_proj | 977 (±1.5, 更稳定) | |

### 2.5 Phase 2 优化尝试 (v1, 07-06)

| # | 配置变化 | TFLOPs | 备注 |
|---|---|---|---|
| 34 | hybridep-num-sms=16 | 850 (-13%) | EP 带宽不足 |
| 35 | hybridep-num-sms=24 | 896 (-8%) | |
| 37 | activation offload | 终止 | 15min 无 iter1 |
| 40 | optimizer CPU offload (去 fp8-param-gather) | 774 (-21%) | 去 fp8-param-gather 致命 |
| 41/42 | delayed FP8 | CRASH | TE 2.9/2.15 各种不兼容 |
| 43 | `--cuda-graph-modules attn moe` | CRASH | assert: 只支持 drop-padding |

### 2.6 NVLS 退化排查 (7 组对照)

| 变量 | TFLOPs | 退化? |
|---|---|---|
| NVLS=0 (baseline) | 474 稳态 | ❌ 零退化 |
| NVLS=1 (各种组合, 6 组) | 427-526 → 260-368 | ✅ iter 19-44 |

退化在 NVLS transport 内部，排除了 slot 耗竭、GC、内存泄漏、thermal throttling。

## 3. 关键发现

### MoE CUDA Graph 深度分析

**v1 全 graph 的原因**: MCore 0.16 (PR #1917) 无 `MoECudaGraphPartialCaptureSignal`，TE `make_graphed_callables()` 把整层包成一个 graph，scope 只声明"包含什么"不创建截断边界。HybridEP dispatch 是 GPU-native 设计: 固定 launch config (32 SM), kernel 从 GPU memory 读 routing tensor, NVLink buffer 预分配固定大小, 无 CPU-GPU sync。

**v2+ 截断的原因**: PR #4292 引入 `MoECudaGraphPartialCaptureSignal` 用 exception 截断 graph capture。原因是通用安全——不是所有 dispatcher 都有固定 kernel config，dropless MoE 理论上 routing 可导致 buffer 溢出。

**v1 full graph 的正确性风险**: benchmark (mock data) 完全安全。真实训练中极端 routing 偏斜理论上可能导致 NVLink buffer 溢出 (silent corruption 或 SIGABRT)。DSv3 256 expert top-8 统计波动小，实际风险极低。

### sequence-parallel 在 TP=1 时仍有益

v3.1 E4 实验: 关掉 sequence-parallel 降了 17 TFLOPs (964 vs 981)。与 NVIDIA reference 建议相反。原因待查，可能跟 distributed optimizer 的通信模式有关。

### 计算密度决定 TFLOPs

| 模型 | hidden_size | seq | TFLOPs |
|---|---|---|---|
| Qwen3-235B | 4096 | 4096 | 219 |
| Qwen3-235B | 4096 | 8192 | 325 |
| **DSv3-32L** | **7168** | **8192** | **981** |

H=7168 vs H=4096 计算密度差 3x。

## 4. 已排除方向 (完整)

| 方向 | 结果 | 原因 |
|---|---|---|
| NCCL_MIN_CTAS=32 | -7% | CTA 占 SM |
| numactl | -3% | Grace NUMA 延迟低 |
| seq > 8192 + offload | OOM | |
| recompute mlp only | -20% | 内存压力 |
| optimizer-cuda-graph | CRASH | grad_norm 非法 |
| VPP + CUDA graphs | OOM | |
| PP=8 EP=16 | -19% | 通信翻倍 |
| 关 sequence-parallel | -17 TFLOPs | TP=1 仍有益 |
| vboost | N/A | GB200 不支持 |
| MCore 0.17.0/0.17.1 | hang | HybridEP 集成不完整 |
| delayed FP8 | CRASH | 各版本不兼容 |
| hybridep sms=16/24 | -13%/-8% | sms=32 最优 |
| activation offload | 终止 | 延迟叠加 |
| optimizer CPU offload | -21% | 去 fp8-param-gather 致命 |
| `attn moe` full graph (v3.1) | CRASH | 只支持 drop-padding |

## 5. 对 Qwen3 235B 的启示

| 优化 | DSv3 结果 | Qwen3 235B 状态 |
|---|---|---|
| MNNVL=0 + USE_MNNVL=1 | 标准配置 | ✅ 685, 跨域验证 |
| NVLS=0 | 必须 | ✅ |
| GRAPH_REGISTER=0 | 必须 | ✅ |
| fp8-param-gather | 928→975 | ✅ NeMo recipe 自动开 |
| sequence-parallel | 关了 -17 | ✅ NeMo recipe 默认开 |
| PP=2 EP=64 vs PP=2 EP=32 | 975 vs 955 | 我们 64 GPU 最多 EP=32 |
| 32L 缩减模型 | 解锁 CUDA graph | N/A, NeMo 跑完整模型 |

> **计算密度差异**: DSv3 H=7168 vs Qwen3 235B H=4096, 计算密度差 3x。685 是 H=4096 下的天花板。
