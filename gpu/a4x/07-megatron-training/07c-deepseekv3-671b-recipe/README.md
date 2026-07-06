# DeepSeek V3 — GB200 NVL72 128 GPU HybridEP 训练复现指南

> **突破性进展**: NeMo Bridge `run_script.py -cv v2` 在 64 GPU 上跑到 **1124 TFLOPs**（DSv3 16L: 1114），超过 NVIDIA 256 GPU 参考值 1106。raw Megatron-LM 最高 981。
>
> 40+ 组实验, 300 → 981 (raw Megatron-LM) → **1124** (NeMo Bridge)。
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
| H=4096 模型 | 4096 | 4096 | 219 |
| H=4096 模型 | 4096 | 8192 | 325 |
| **DSv3-32L** | **7168** | **8192** | **981** |

计算密度由 hidden_size 决定，H=7168 vs H=4096 差约 3x。

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

## 5. 突破性进展：NeMo Bridge 解锁 full_iteration graph（981 → 1124）

### 5.1 发现过程

奚老师在 raw Megatron-LM (`pretrain_gpt.py`) 上做了 40+ 组实验，从 300 优化到 981。我们基于他的报告分析 981 vs NVIDIA 参考值 1106 的差距原因，发现差距来自技术栈而非调参。通过阅读 Megatron-Core MoE 论文 [[1]](#ref1)，识别出 Megatron Bridge 的 4 项专有技术，然后 dump NeMo recipe 配置验证这些技术是否可用，最终发现 **`-cv v1` 和 `-cv v2` 是两套完全不同的技术栈**。切换到 `-cv v2` 后，64 GPU 即跑出 1124 TFLOPs。

### 5.2 raw Megatron-LM 被限制在 981 的根因

raw Megatron-LM 没有 Bridge 的 3 项关键技术：

**1. Sync-Free Device-Initiated Kernels（无同步设备端自主 kernel）**

Dropless MoE 每次 routing 产生动态 token count。传统做法：GPU 算完 routing → GPU→CPU 拷贝 per-expert token count → CPU 决定 Grouped GEMM 的 launch config（grid size、tile size）→ 这个 device-to-host 同步**阻断 CUDA Graph capture**。

Bridge 的解法：重写 Grouped GEMM 和 HybridEP dispatch 为 device-initiated——kernel 自己从 GPU memory 读 shape 信息决定怎么跑，无需 CPU 参与。整个 MoE 层**零 CPU-GPU 同步**，可被 full_iteration graph 完整 capture。

raw Megatron-LM 没有这些重写的 kernel → MCore 0.17+ 引入 `MoECudaGraphPartialCaptureSignal` 主动截断 graph capture 来保证安全 → 只能 graph attn → 981。

**2. Paged Stashing（分页暂存）**

Full_iteration graph 需要按 worst-case 为每个 expert 预分配固定 buffer。模型层数 × 256 expert × 1.5 倍余量的 buffer 总量超过 184 GB HBM → OOM。

Bridge 的 Paged Stashing 在 CUDA Graph 执行过程中**动态回收未使用的 buffer 空间**给其他操作复用，相当于静态 graph 里的动态内存池。让内存峰值可控。

raw Megatron-LM 没有 → full_iteration graph 分配内存时 OOM。

**3. Flexible PP Layout（灵活流水线切分）**

raw Megatron-LM 要求 `num_layers % (PP × VP) == 0`。DSv3 61 层 PP=8 VP=3 不整除。Bridge 支持不均匀 stage 切分（如 8+8+8+7+8+8+7+7）。

另外论文还描述了 **ECHO**（动态复制热门 expert 到空闲 GPU 减少 load imbalance）作为内存优化的补充。

### 5.3 关键发现：V1 vs V2 recipe 配置 dump 对比

通过在容器内 dump NeMo recipe 实际配置，发现 V1 和 V2 是两套技术栈：

| 配置项 | V1 (`-cv v1`) | V2 (`-cv v2`) | 影响 |
|---|---|---|---|
| cuda_graph_impl | transformer_engine (TE scoped) | **full_iteration** | **核心差异：决定 graph 覆盖范围** |
| moe_paged_stash | False | **True** | **使能 full graph 的内存前提** |
| moe_expert_rank_capacity_factor | — | **1.5** | worst-case buffer 预分配倍率 |
| moe_paged_stash_buffer_size_factor_cuda | 1.1 | **1.2** | graph 内 buffer 回收比例 |
| cuda_graph_modules | full (被 TE impl 限制) | **full** (full_iteration 下完整生效) | |
| moe_pad_experts_for_cuda_graph_inference | False | — | |

> V1 是保守配置（TE scoped graph 只 capture attn，无 paged stash）。V2 启用了 Bridge 的**全部优化**。差距不是调参，是技术栈切换。

dump config 的方法：
```python
from configs.deepseek.deepseek_llm_pretrain import deepseek_v3_pretrain_config_gb200
cfg = deepseek_v3_pretrain_config_gb200(precision="fp8_mx", config_variant="v2")
m = cfg.model
print(m.cuda_graph_impl)          # 必须是 full_iteration
print(m.moe_paged_stash)          # 必须是 True
```

### 5.4 提升原理分解

**full_iteration CUDA Graph（+40-50%，最大贡献）**

TE scoped graph 只 capture attention 模块。MoE 层的 router + preprocess + dispatch + expert compute + combine 上千个 kernel 每次从 host 逐个 launch。full_iteration 把整个 training step（forward + backward + optimizer update）录成一张 graph，CPU 只发一条 replay 命令，所有 MoE kernel launch overhead **归零**。

PP + CUDA Graph 的内存机制：有 PP 时每个 microbatch 必须独立 graph（否则 forward 覆盖 backward 的保存上下文）。总 graph 数 = L × M × 2（层数 × microbatch 数 × forward/backward）。Bridge 用 buffer reuse 按 PP 执行顺序回收已完成 microbatch 的 buffer 给下一个 microbatch 复用。

**Paged Stashing（使能 full graph 的内存前提）**

没有 paged stash，full graph 需按 worst-case 为 94 层 × 128 expert × 1.5 倍余量预分配 buffer，超过 184 GB HBM → OOM。Paged stash 在 graph 执行中动态回收未使用空间给其他操作复用。

### 5.5 版本和参数要求

| 组件 | 最低版本 | 说明 |
|---|---|---|
| NeMo 容器 | **nemo:26.06** | 必须用 NeMo 容器（内含 Megatron Bridge） |
| Megatron Core | **0.18.0+** | 首次支持 `--cuda-graph-modules`。0.17.x 跟 HybridEP 死锁必须跳过 |
| 入口脚本 | **`run_script.py`** | `pretrain_gpt.py` 走 raw Megatron-LM，无 Bridge 优化 |
| Config variant | **`-cv v2`** | V1 不启用 full graph / paged stash |

**入口脚本决定技术栈**：同一个 NeMo 26.06 容器，`run_script.py` 走 Bridge 有完整优化，`pretrain_gpt.py` 走 raw Megatron-LM 无 Bridge 优化。选错入口差 **64%**。

### 5.6 为什么之前不能开 full graph，现在又能了

| 阶段 | 认知 | 事实 |
|---|---|---|
| 之前 | full_iteration + HybridEP + PP>1 = 不兼容 | **只对 raw Megatron-LM 成立**（无 sync-free kernel） |
| 之前 | V1 和 V2 只是并行度不同 | **V2 切换了整个 CUDA Graph 技术栈** |
| 之前 | 685/981 是硬件极限 | **是 config 选择的极限，不是硬件极限** |
| 现在 | 用 `run_script.py -cv v2` | Bridge 的 sync-free kernel + paged stash 解决了所有限制 |

### 5.7 实测验证

#### 测试 1: MoE 模型 V2 recipe（1124 TFLOPs）

在 GKE 集群跨 2 个 NVL72 域，64 GPU (8+8 节点)：

| Recipe | Graph 模式 | Paged Stash | TFLOPs | Step Time | 提升 |
|---|---|---|---|---|---|
| V1 (`-cv v1`) | TE scoped (attn only) | 关 | ~同 raw Megatron-LM | ~7s | baseline |
| **V2 (`-cv v2`)** | **full_iteration** | **开** | **1124** | **4.31s** | **+64%** |

稳态 **1117-1125 TFLOPs/GPU**，峰值 **1125.7**。20 步全跑完正常退出。**超过 NVIDIA 256 GPU 参考值 1106**。

#### 测试 2: DSv3 16L（1114 TFLOPs，5 轮调试）

用 NeMo Bridge `run_script.py -m deepseek -mr deepseek_v3` 跑 DSv3 16 层缩减版。

**DSv3 recipe 的特殊性**：V1 和 V2 配置完全一致（都是 full_iteration + paged stash），NVIDIA 一步到位给了最强配置。但 recipe hardcoded 了 61 层的 PP layout 和 VPP=4，改层数需要同步改三个参数。

**调试过程（5 轮踩坑）**：

| 轮次 | 错误 | 根因 | 教训 |
|---|---|---|---|
| v1 | `VPP=4 assert` | recipe 默认 VPP=4，PP=2+16L 检测出 VPP=8 不匹配 | VPP 必须匹配层数和 PP |
| v2 | `61L layout assert` | `--num_layers 16` 改了层数但 PP layout 还是 hardcoded 的 61 层 | layout 也要覆盖 |
| v3 | `VPP=4 assert` | 手动设了 layout 但忘了同步覆盖 VPP | layout 和 VPP 必须一起改 |
| v4 | `MTP assert` | layout `Etttttttt\|ttttttttL` 缺 `m`（MTP 层） | DSv3 有 Multi-Token Prediction |
| **v5** | **成功** | PP=2 VPP=2 layout=`Etttt\|tttt\|tttt\|ttttmL` | **三件套同改** |

**DSv3 PP layout 格式**：`E`=embedding, `t`=transformer, `m`=MTP, `L`=loss, `|`=virtual stage 边界。改层数必须三件套同改：`--num_layers` + `-vp` + `--pipeline_model_parallel_layout`。

**Layout 计算**：16 层 PP=2 VPP=2 → 4 个 virtual stage × 4 layers = 16 decoder + 1 MTP + embedding + loss。Layout: `Etttt|tttt|tttt|ttttmL`（16/2/2=4 整除 ✓）。

**最终命令**：
```bash
run_script.py -m deepseek -mr deepseek_v3 --task pretrain \
  -g gb200 -c fp8_mx -ng 64 --data mock --max_steps 20 \
  --num_layers 16 \
  --pipeline_model_parallel_size 2 \
  --expert_model_parallel_size 32 \
  --global_batch_size 512 --micro_batch_size 1 \
  -vp 2 \
  --pipeline_model_parallel_layout "Etttt|tttt|tttt|ttttmL"
```

**结果**：

| 模型 | 层数 | PP | VPP | EP | H | Graph | Paged Stash | TFLOPs | Step Time |
|---|---|---|---|---|---|---|---|---|---|
| **DSv3-16L** | **16** | **2** | **2** | **32** | **7168** | **full_iteration** | **True** | **1114** | **2.35s** |

稳态 1110-1120，峰值 1120.1。每 5 步有一次 ~713 的抖动（VPP virtual stage 切换通信 spike）。20 步全跑完正常退出。

### 5.8 全局对比

| 入口 | 模型 | 层数 | GPU | Graph | Paged Stash | TFLOPs |
|---|---|---|---|---|---|---|
| pretrain_gpt.py | DSv3-32L | 32 | 128 | TE scoped (v3.1) | 无 | 981 |
| pretrain_gpt.py | DSv3-32L | 32 | 128 | TE scoped + MoE (v1) | 无 | 975 |
| **run_script.py** | **MoE 94L** | **94** | **64** | **full_iteration** | **True** | **1124** |
| **run_script.py** | **DSv3-16L** | **16** | **64** | **full_iteration** | **True** | **1114** |
| NVIDIA ref | DSv3-61L | 61 | 256 | full_iteration | True | 1106 |

> NeMo Bridge 的 full_iteration graph + paged stash 在 MoE 和 DSv3 上都验证生效。64 GPU 超过了 NVIDIA 256 GPU 参考值和 128 GPU raw Megatron-LM 结果。

### 5.9 核心教训

1. **Recipe config variant 是技术栈选择**，不只是并行度配置。`-cv v1` → `-cv v2` 切换了 sync-free kernel + paged stash + full_iteration graph 的完整组合
2. **"full_iteration + HybridEP + PP>1 不兼容"只对 raw Megatron-LM 成立**。NeMo Bridge 的 sync-free kernel 解决了 CPU-GPU 同步问题
3. **dump config 是必要的诊断步骤**。不 dump 就不知道 V1 和 V2 底层差了什么
4. **DSv3 改层数必须三件套同改**：`--num_layers` + `-vp` + `--pipeline_model_parallel_layout`（含 MTP 层）
5. **入口脚本决定一切**：`run_script.py` 和 `pretrain_gpt.py` 是同一个容器里两条完全不同的技术路径

### 5.10 未来方向

1. **MCore 开源 sync-free kernels**：论文 Section 4.3.7 的技术如果合入开源版，`pretrain_gpt.py` 也能开 full MoE graph
2. **NCCL 修复 NVLS 退化**：解锁 NVLink SHARP 硬件加速，预计 +3-5%
3. **256 GPU 测试**：PP=8 + VPP=3 + full graph，理论上 > 1124
4. **DSv3 全量 61 层 64 GPU**：如果 paged stash 省下足够 HBM，可能不需要缩层
5. **通知奚老师切换到 NeMo Bridge**：`run_script.py -cv v2` 预计从 981 涨到 1100+

## 参考文献

<a id="ref1">[1]</a> *Scalable Training of Mixture-of-Experts Models with Megatron Core*, arXiv:2603.07685v2, NVIDIA, 2026. [[arxiv]](https://arxiv.org/abs/2603.07685)
