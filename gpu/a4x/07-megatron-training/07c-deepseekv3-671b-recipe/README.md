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

## 6. 为什么 Megatron Bridge 能达到 1106 而 raw Megatron-LM 只能 981

NVIDIA 256 GPU 参考值 1106 TFLOPs 使用 **full_iteration CUDA Graph + PP=8 + VPP=3 + HybridEP + dropless MoE**。奚老师用 raw Megatron-LM 在 128 GPU 上最高 981（32L 缩减版）。差距 11% 不是调参问题，是技术栈差异。

### Megatron Bridge 的 4 项专有技术

根据 Megatron-Core MoE 论文 [[1]](#ref1)：

**1. Sync-Free Device-Initiated Kernels**

Dropless MoE 每次 routing 产生动态 token count，传统做法需要 GPU→CPU 拷贝 count 再由 CPU 决定 kernel launch config，这个 device-to-host sync 让 CUDA Graph 无法 capture。Bridge 重写了 Grouped GEMM 和 HybridEP dispatch 为 device-initiated——kernel 自己从 GPU memory 读 shape 信息决定怎么跑，无需 CPU 参与。整个 MoE 层零 CPU-GPU 同步，可被 full_iteration graph 完整 capture。

**2. ECHO (Expert Cloning for Higher Occupancy)**

Full graph 需要按 worst-case token count 预分配 buffer。ECHO 动态复制热门 expert 到空闲 GPU，减少 token 分配不均衡，让 worst-case 预分配接近 average 实际使用量，省内存。

**3. Paged Stashing**

在 CUDA Graph 内部做细粒度内存管理。预分配 buffer 中没用到的部分被回收给其他操作复用，相当于 graph 内的动态内存池。

**4. Flexible PP Layout**

支持不均匀 pipeline stage 切分。61 层 PP=8 VP=3 不需整除，Bridge 按如 8+8+8+7+8+8+7+7 分配。raw Megatron-LM 要求整除。

### PP + CUDA Graph 的内存机制

有 PP 时每个 microbatch 必须独立 graph（否则 forward 覆盖 backward 的保存上下文）。总 graph 数 = L × M × 2（层数 × microbatch 数 × forward/backward）。Bridge 用 buffer reuse 按 PP 执行顺序回收已完成 microbatch 的 buffer 给下一个 microbatch 复用，控制内存峰值。

### 981 vs 1106 的差距归因

| 技术 | Bridge (1106) | raw Megatron-LM (981) | 影响 |
|------|-------------|---------------------|------|
| CUDA Graph | full_iteration (整个 step) | TE scoped (只 attn) | 主要差距 |
| VPP | VP=3 减少 bubble | 不可用（层数不整除 + OOM） | 次要 |
| CuTeDSL | 融合 MoE grouped MLP | 环境变量设了但可能未完整调用 | 待确认 |
| overlap-moe + delay-wgrad | 开（依赖 VPP） | 不可用（无 VPP） | 次要 |
| Sync-free kernels | 有 | 无（MoECudaGraphPartialCaptureSignal 截断） | 核心技术差距 |
| ECHO + Paged Stashing | 有 | 无 | 内存保障 |

> ~~结论：128 GPU 上达到 1106 是不可能的。~~ **已推翻**：通过 NeMo Bridge（run_script.py -cv v2），64 GPU 即可达到 1124 TFLOPs，超过 NVIDIA 256 GPU 参考值 1106。详见下方实测验证。
>
> 981 是 **raw Megatron-LM (pretrain_gpt.py)** 的极限。切换到 **NeMo Bridge (run_script.py)** 可突破。

## 7. 实测验证：NeMo Bridge V2 recipe 在 64 GPU 上跑到 1124 TFLOPs

### 背景

基于 Section 6 的分析，Bridge 的 sync-free kernel、paged stash 等优化理论上可以解锁 full_iteration graph。通过 dump NeMo recipe 配置发现，**V1 和 V2 config variant 是两套完全不同的技术栈**：

| 配置项 | V1 (`-cv v1`) | V2 (`-cv v2`) |
|---|---|---|
| cuda_graph_impl | transformer_engine (TE scoped) | **full_iteration** |
| moe_paged_stash | False | **True** |
| moe_expert_rank_capacity_factor | — | **1.5** |
| cuda_graph_modules | full (被 TE scoped 限制) | **full** |

V1 是保守配置（TE scoped graph、无 paged stash），V2 启用了 Bridge 的全部优化。

### 实测结果

在 baker 集群 pool-7 + pool-2 跨 2 个 NVL72 域，64 GPU (8+8 节点)：

```bash
run_script.py -m qwen -mr qwen3_235b_a22b --task pretrain \
  -g gb200 -c fp8_mx -ng 64 --data mock --max_steps 20 \
  -cv v2 \
  --pipeline_model_parallel_size 2 \
  --expert_model_parallel_size 32 \
  --global_batch_size 512 --micro_batch_size 1
```

| Recipe | Graph 模式 | Paged Stash | TFLOPs | Step Time | 提升 |
|---|---|---|---|---|---|
| V1 (`-cv v1`) | TE scoped (attn only) | 关 | 685 | 7.1s | baseline |
| **V2 (`-cv v2`)** | **full_iteration** | **开** | **1124** | **4.31s** | **+64%** |

稳态 **1117-1125 TFLOPs/GPU**，峰值 **1125.7**。20 步全跑完正常退出。**超过 NVIDIA 256 GPU 参考值 1106**。

### 为什么 V2 能在 64 GPU PP=2 + HybridEP 上跑 full_iteration graph

奚老师在 raw Megatron-LM (`pretrain_gpt.py`) 上测试 full_iteration + HybridEP + PP>1 = crash。这个结论**只对 raw Megatron-LM 成立**。

NeMo Bridge (`run_script.py`) 有 3 项 raw Megatron-LM 没有的技术：

1. **Sync-Free Device-Initiated Kernels**：Grouped GEMM 和 HybridEP dispatch 从 GPU memory 自主读 shape，零 CPU-GPU 同步 → 整个 MoE 层可被 graph capture
2. **Paged Stashing** (`moe_paged_stash=True`)：graph 内动态回收未使用 buffer → full_iteration graph 不 OOM
3. **Expert Rank Capacity Factor** (`=1.5`)：按 1.5 倍 worst-case 预分配 per-expert buffer，配合 paged stash 回收

### 1124 的提升来源

**full_iteration CUDA Graph (+40-50%，最大贡献)**：V1 的 TE scoped graph 只 capture attention，MoE 层的上千个 kernel 每次从 host 逐个 launch。full_iteration 把整个 training step 录成一张 graph，所有 launch overhead 归零。Step time 7.1s → 4.31s。

**Paged Stashing (使能 full graph 的前提)**：没有 paged stash，94 层 × 128 expert × 1.5 倍余量的 buffer 超过 184 GB HBM → OOM。Paged stash 在 graph 执行中动态回收未使用空间。

### 版本要求

| 组件 | 最低版本 | 说明 |
|---|---|---|
| NeMo 容器 | **nemo:26.06** | 必须用 NeMo 容器（含 Bridge） |
| Megatron Core | **0.18.0+** | 0.18.0 首次支持 `--cuda-graph-modules`；0.17.x hang |
| 入口脚本 | **`run_script.py`** | 不能用 `pretrain_gpt.py`（无 Bridge 优化） |
| Config variant | **`-cv v2`** | V1 不启用 full graph / paged stash |

**入口脚本决定技术栈**：同一个 NeMo 容器，`run_script.py` 走 Bridge 有完整优化，`pretrain_gpt.py` 走 raw Megatron-LM 无 Bridge 优化。选错入口差 64%。

### 核心教训

1. **Recipe config variant 是技术栈选择**，不只是并行度配置。V1→V2 切换了 sync-free kernel + paged stash + full_iteration graph 的完整组合
2. **"full_iteration + HybridEP + PP>1 不兼容"只对 raw Megatron-LM 成立**。NeMo Bridge 的 sync-free kernel 解决了这个限制
3. **dump config 是必要的诊断步骤**。不 dump 就不知道 V1 和 V2 底层差了什么
4. 奚老师的 981 是 raw Megatron-LM 的极限，不是硬件极限。换 NeMo Bridge 可突破

### 未来方向

1. **MCore 开源 sync-free kernels**: 论文 Section 4.3.7 的技术如果合入开源版，raw Megatron-LM 也能开 full MoE graph
2. **NCCL 修复 NVLS 退化**: 解锁 NVLink SHARP 硬件加速，预计 +3-5%
3. **256 GPU 测试**: PP=8 + VPP=3 + full graph，理论上 > 1124（更多 GPU 减少跨域通信比例）
4. **通知奚老师切 NeMo Bridge**: 他的 128 GPU forrest 集群用 `run_script.py -cv v2` 预计从 981 涨到 1100+

## 参考文献

<a id="ref1">[1]</a> *Scalable Training of Mixture-of-Experts Models with Megatron Core*, arXiv:2603.07685v2, NVIDIA, 2026. [[arxiv]](https://arxiv.org/abs/2603.07685)
