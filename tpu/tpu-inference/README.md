# 大模型推理 on TPU v7x — 快速落地指南

[English](./README.en.md) | **中文**

> **定位声明**
>
> 本仓库是一份 **POC（概念验证）快速落地指南**，目标是让用户以最短路径在 TPU v7x 上跑通大模型推理。
>
> - **可执行优先**：每个模型都提供从零到推理成功的完整步骤
> - **非性能评测**：文档中出现的性能数据仅为功能验证时的副产品，**不代表优化后的生产性能**，不应作为评估指标
> - **非生产就绪**：未做生产级调优（KV cache 策略、batch 调度、PD 分离等），仅保证功能可用
>
> 如需性能对比或生产部署方案，请联系 TPU 推理团队。

## 模型总览

| 模型 | 参数量 | 架构 | 量化精度 | TPU 拓扑 | Cold Start | 文档 |
|------|--------|------|----------|----------|------------|------|
| DeepSeek R1 | 671B | MoE 256E top-8, MLA | FP4 MoE + FP8 Attn | v7x-8 | ~4-6 min | [详情](./DeepSeek-R1-671B-FP4/) |
| DeepSeek V3.2 | 671B | MoE 256E top-8, MLA | FP4 MoE + FP8 Attn | v7x-8 | ~4-6 min | [详情](./DeepSeek-V3.2-671B-FP4/) |
| GLM-5.1 | 754B | MoE 180E top-8 | FP4 MoE + FP8 Attn | v7x-8 | ~3-4 min | [详情](./GLM-5.1-754B-FP4/) |
| Kimi K2.6 | 1T / 32B active | MoE, native INT4 | INT4 | v7x-16 | ~6 min | [详情](./Kimi-K2.6-1T-A32B-INT4/) |
| Qwen3.5 | 397B / 17B active | Hybrid GDN+Attn, 512E | FP8 | v7x-8 | ~7 min | [详情](./Qwen3.5-397B-A17B-FP8/) |
| Qwen3-Coder | 480B / 35B active | MoE, FP8 native | FP8 | v7x-8 | ~7 min | [详情](./Qwen3-Coder-480B/) |

**硬件基线**：TPU v7x-8 = 4 chips / 8 devices / 768 GB HBM / ~944 GB 主机内存。

## 验证状态

| 模型 | 推理验证 | 质量评测 | 已知限制 |
|------|----------|----------|----------|
| DeepSeek R1 | ✅ 通过 | GSM8K 94.92% | — |
| DeepSeek V3.2 | ✅ 通过 | Smoke test | 需热补丁注册 V32 架构 |
| GLM-5.1 | ✅ 通过 | GSM8K 89.46% | 首次 JIT 编译 ~13 min |
| Kimi K2.6 | ✅ 通过 | Smoke test | 全量 61 层需 v7x-16；v7x-8 仅跑 40 层 |
| Qwen3.5 | ✅ 通过 | GSM8K 93.93% | Chat 路径不稳定，仅 completion 模式可靠 |
| Qwen3-Coder | ✅ 通过 | Smoke test | — |

## 模型架构与特性矩阵

> ✅ 模型支持且已实现　🔇 模型支持但已绕过/未启用　— 模型不含此特性　⏳ 待验证

| 模型 | Attention | MoE | Layers | Hidden | Pos Enc | MTP | DSA | Vision | Hybrid KV |
|------|-----------|-----|--------|--------|---------|:---:|:---:|:------:|:---------:|
| DeepSeek R1 | MLA | 256E top-8 | 61 (3D+58M) | 7168 | RoPE+YaRN | 🔇 | — | — | — |
| DeepSeek V3.2 | MLA | 256E top-8 | 61 (3D+58M) | 7168 | RoPE+YaRN | 🔇 | 🔇 | — | — |
| GLM-5.1 | MLA | 256E top-8 | 78+1 (3D+75M+MTP) | 6144 | RoPE (θ=1M) | 🔇 | 🔇 | — | — |
| Kimi K2.6 | MLA | 384E+1S top-8 | 61 (1D+60M) | 7168 | RoPE+YaRN | — | — | 🔇 | — |
| Qwen3.5 | GQA (32Q/2KV) | 512E+1S top-10 | 60 (45 GDN+15 GQA) | 4096 | YaRN+mrope | — | — | 🔇 | ✅ |
| Qwen3-Coder | GQA (40Q/8KV) | 128E top-8 | 94 | 5120 | RoPE | — | — | — | — |
| MiMo-V2-Flash | MHA | Dense | ⏳ | ⏳ | RoPE | ⏳ | — | — | — |

> **层数缩写**：D = Dense 层, M = MoE 层, MTP = Multi-Token Prediction 层, GDN = Gated Delta Network 层（线性注意力）, S = Shared Expert

### 已绕过特性说明

| 特性 | 影响模型 | 绕过方式 | 潜在影响 |
|------|----------|----------|----------|
| **MTP** | R1, V3.2, GLM-5.1 | TPU 推理未启用 MTP 头；GLM 第 78 层显式跳过 | 未利用 30-50% decode 吞吐提升 |
| **DSA** | V3.2, GLM-5.1 | V3.2: indexer 权重跳过 (`skip_substrs=['indexer']`)；GLM: SparseAttnIndexer 为 GPU 专用路径，TPU 不生效 | 长文本注意力稀疏优化未启用 |
| **Vision** | K2.6, Qwen3.5 | `--limit-mm-per-prompt='{"image":0,"video":0}'` 禁用 | K2.6 的 MoonViT 400M、Qwen3.5 的多模态均未使用 |

## 部署能力

| 模型 | 单机推理 | PD 分离 | 多机推理 | 备注 |
|------|:--------:|:-------:|:--------:|------|
| DeepSeek R1 | ✅ | ⏳ | ⏳ | v7x-8, EP=8 |
| DeepSeek V3.2 | ✅ | ⏳ | ⏳ | v7x-8, EP=8 |
| GLM-5.1 | ✅ | ✅ | ⏳ | v7x-8, EP=8；PD 分离需 vLLM v1 scheduler（V0 DPScheduler 已废弃） |
| Kimi K2.6 | ❌ | ⏳ | ✅ | v7x-8 全量 61 层 OOM（权重+KV cache 超 HBM）；完整推理仅 v7x-16 |
| Qwen3.5 | ✅ | ✅ | ✅ | 三种模式均已验证 |
| Qwen3-Coder | ✅ | ✅ | ✅ | 多机 TP=16 吞吐下降 15-63%，不推荐 |

> ✅ 已验证可用　⚠️ 不稳定/有已知问题　⏳ 待验证　❌ 不可用

## 性能概览（POC 参考数据）

> ⚠️ **以下数据为功能验证时的副产品，不代表优化后的生产性能。** 所有测试均使用 `--enforce-eager`（未启用 XLA 编译优化），未做 batch 调度策略调优。

### 1K Input / 1K Output

| 模型 | TTFT (c=1) | TPOT (c=1) | 单用户 tok/s | 峰值吞吐 | @ 并发 |
|------|-----------|-----------|-------------|---------|--------|
| DeepSeek R1 | 480 ms | 26 ms | 37.6 | 7,309 tok/s | c=2048 |
| DeepSeek V3.2 | ~480 ms ¹ | ~26 ms ¹ | ~37.6 ¹ | ~7,309 ¹ | c=2048 |
| GLM-5.1 | 534 ms | 35 ms | 28.4 | 6,504 tok/s | c=1024 |
| Kimi K2.6 ² | 1,142 ms | 49 ms | 20.0 | 592 tok/s | c=32 |
| Qwen3.5 | — | ~20 ms | 49.6 | 2,097 tok/s | c=128 |
| Qwen3-Coder | 95 ms | 20.6 ms | 48.0 | 1,478 tok/s | c=64 |

¹ V3.2 与 R1 架构相同，引用 R1 实测数据，V3.2 独立压测尚未进行
² Kimi K2.6 数据来自 v7x-16（全量 61 层）；v7x-8 仅能跑 40 层

### 8K 场景

| 模型 | 8K In / 1K Out (tok/s) | 1K In / 8K Out (tok/s) | 测试并发 |
|------|:----------------------:|:----------------------:|:--------:|
| DeepSeek R1 | — | — | 未测 |
| DeepSeek V3.2 | — | — | 未测 |
| GLM-5.1 | — | — | 未测 |
| Kimi K2.6 | — | 581 ³ | c=32 |
| Qwen3.5 | 850 | 1,702 | c=64 |
| Qwen3-Coder | 943 | 1,623 | c=64 |

³ Kimi K2.6 实测为 1K In / 7K Out

## 现状总结

所有 6 个模型已在 TPU v7x 上完成**推理功能验证**，质量评测达到预期水平（已测模型 GSM8K 89-95%）。当前性能处于**"功能可用，但未经优化"**阶段：

- **单用户延迟**（TPOT 20-50 ms）可满足交互式对话场景
- **系统吞吐**有初步数据，但所有测试均在 `enforce_eager` 模式下运行，未启用 XLA 编译图优化
- **长文本**（8K+）仅 Qwen 系列完成测试，其余模型待补充
- **PD 分离 / 多机部署**仅 Qwen3.5 和 Qwen3-Coder 完整验证，Kimi K2.6 多机可用，其余模型暂无

如需生产级性能数据或调优方案，请联系 TPU 推理团队。

## 快速开始

```
获取权重          准备 Cache（FP4 模型需要）       启动 vLLM
   │                      │                         │
   ▼                      ▼                         ▼
HuggingFace 下载   gen_fp4_cache_cpu_parallel.py   vllm serve \
  或 GCS 拷贝       + extract_non_moe_weights.py     --tensor-parallel-size 8 \
                   拷贝到 /dev/shm                    --quantization fp8 ...
```

**三种量化路径**：

| 路径 | 适用模型 | Cache 生成 | /dev/shm 需求 |
|------|----------|-----------|---------------|
| **FP4 MoE** | R1, V3.2, GLM-5.1 | 需要（CPU 并行脚本，~28 min） | ~610-735 GB |
| **INT4 MoE** | Kimi K2.6 | 需要（模型自带转换） | ~532 GB |
| **FP8 Native** | Qwen3.5, Qwen3-Coder | **不需要**（直读权重） | 推荐预加载（见下文） |

### 启动加速: 权重预加载到 /dev/shm

FP8 Native 模型（Qwen3.5, Qwen3-Coder）从网络存储（Lustre/GCS）直接启动时，vLLM 需要串行加载 50-94 个 safetensors shards。当主机可用 RAM 不足以预取整个 checkpoint 时（例如 /dev/shm 被 FP4 cache 占用），**每 shard 加载耗时可达 90 秒，总启动时间超过 15 分钟**。

**最佳实践：先拷贝权重到 /dev/shm，再指向本地路径启动。**

```bash
# 方式 1: 从 Lustre 拷贝（推荐，网内速度快）
cp -r /lustre/models/Qwen3.5-397B-A17B-FP8 /dev/shm/qwen35-weights/

# 方式 2: 从 GCS 直接拷贝（gcloud 4.5 GB/s）
gcloud storage cp -r gs://<BUCKET>/models/qwen3.5-397b-a17b-fp8/weights/ /dev/shm/qwen35-weights/

# 启动 vLLM 指向 /dev/shm
vllm serve /dev/shm/qwen35-weights/ \
  --tensor-parallel-size 8 --enable-expert-parallel ...
```

| 模型 | 权重大小 | 推荐策略 | Cold Start |
|------|---------|---------|------------|
| Qwen3.5 | 379 GB | /dev/shm 预加载 (拷贝 ~4.5 min) | ~4 min (vs 15+ min 直读) |
| Qwen3-Coder | 450 GB | Lustre 直读 (空 /dev/shm) | ~10 min |

> ⚠️ **Qwen3-Coder 不适合 /dev/shm 预加载**：450 GB 权重占 /dev/shm 57%，加上 vLLM 加载时的 MoE requantization 临时内存，总量超过 920 GiB 容器内存限制，会触发 OOM Killed。正确做法是保持 /dev/shm 空（800 GB 可用），从 Lustre 直读——此时 OS 有足够 RAM 做 auto-prefetch，每 shard ~5-9 秒（vs RAM 不足时 90 秒）。
>
> ⚠️ **一次只测一个模型**。FP4 模型（R1/V3.2/GLM）的 cache (~610-735 GB) 也在 /dev/shm，两者不能同时放。

## 环境变量速查

### FP4 MoE 模型（DeepSeek R1 / V3.2 / GLM-5.1）

```bash
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm
```

### INT4 MoE 模型（Kimi K2.6）

```bash
export K26_USE_V16=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm/k26_cache_v2
```

### FP8 Native 模型（Qwen3.5 / Qwen3-Coder）

```bash
export MODEL_IMPL_TYPE=vllm
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0
```

## 模型存储（GCS）

权重和预计算 Cache 统一存放在 GCS 对象存储桶中，按以下结构组织：

```
gs://<YOUR_BUCKET>/models/
├── deepseek-r1-671b/
│   ├── weights/                          # 163 safetensors shards (~642 GB)
│   └── cache/fp4/ep8_tp1_.../            # 58 层 FP4 npy_v1 + non_moe_weights.safetensors
├── deepseek-v3.2-671b/
│   ├── weights/                          # 163 safetensors shards (~643 GB)
│   └── cache/fp4/ep8_tp1_.../            # 58 层 FP4 npy_v1 + non_moe_weights.safetensors
├── glm-5.1-754b/
│   ├── weights/                          # 143 safetensors shards (~705 GB)
│   └── cache/fp4/ep8_tp1_.../            # 76 层 FP4 npy_v1 + non_moe_weights.safetensors
├── kimi-k2.6/
│   ├── weights/                          # 64 safetensors shards (~555 GB)
│   └── cache/fp4/                        # 60 层 INT4 cache (~532 GB)
├── qwen3-coder-480b-fp8/
│   └── weights/                          # 49 safetensors shards (~449 GB)
├── qwen3.5-397b-a17b-fp8/
│   └── weights/                          # 94 safetensors shards (~378 GB)
└── MiMo-V2-Flash/
    └── weights/                          # 145 safetensors shards (~292 GB)
```

> FP8 native 模型（Qwen 系列）无需 cache 目录，直接从权重推理。

## 工具脚本

| 脚本 | 功能 | 依赖 | 适用模型 |
|------|------|------|----------|
| `gen_fp4_cache_cpu_parallel.py` | 从 safetensors 生成 FP4 MoE cache | 纯 CPU (numpy) | R1, V3.2, GLM-5.1 |
| `extract_non_moe_weights.py` | 提取非 MoE 权重为单文件 | 纯 CPU (torch) | R1, V3.2, GLM-5.1 |
| `validate_weights.py` | 验证权重完整性 | torch | GLM-5.1 |

这些脚本位于各模型子目录中，纯 CPU 运行，不需要 TPU/GPU。

## 基础设施

TPU VM 创建和存储配置参考 [TPU-VM 指南](./TPU-VM/)，包括：

- Hyperdisk ML 数据盘创建与挂载
- TPU v7x-8 VM 实例创建
- fio 磁盘性能基准测试
- GCS 存储桶配置

## 目录结构

```
tpu-inference/
├── README.md                          # 本文（总纲）
├── README.en.md                       # English version
├── DeepSeek-R1-671B-FP4/              # DeepSeek R1 推理指南
├── DeepSeek-V3.2-671B-FP4/            # DeepSeek V3.2 推理指南
├── GLM-5.1-754B-FP4/                  # GLM-5.1 推理指南
├── Kimi-K2.6-1T-A32B-INT4/            # Kimi K2.6 推理指南
├── Qwen3.5-397B-A17B-FP8/             # Qwen3.5 推理指南
├── Qwen3-Coder-480B/                  # Qwen3-Coder 推理指南
└── TPU-VM/                            # TPU VM 基础设施指南
```

## 已发现的 Upstream 兼容性问题（2026-05-01 验证）

> 以下问题在 upstream merge（`507cfa16..0b9f5583`，63 commits）后的 smoke test 中发现并修复/规避。影响的是 **JAX flax_nnx 推理路径**（R1, V3.2, GLM-5.1, K2.6）；PyTorch+TorchAX 路径（Qwen3.5, Qwen3-Coder）仅受问题 3 影响。

### 问题 1: USE_MOE_EP_KERNEL 语义变化 → FUSED_MOE 报错

- **现象**: `NotImplementedError: Unsupported moe backend: MoEBackend.FUSED_MOE`
- **原因**: upstream 将 `USE_MOE_EP_KERNEL=1` + `use_ep=True` 映射到新的 `FUSED_MOE` 后端，但 JAX 路径不支持该后端
- **影响**: 所有 FP4 MoE 模型（R1, V3.2, GLM-5.1）及 Kimi K2.6
- **修复**: **不设 `USE_MOE_EP_KERNEL=1`**。`use_ep=True` 时自动 fallback 到 `GMM_EP`（正确路径）
- **代码位置**: `tpu_inference/layers/jax/moe/utils.py:select_moe_backend()`
- **详细踩坑**: 见 [R1 README 踩坑 #23](./DeepSeek-R1-671B-FP4/)

### 问题 2: additional-config 不完整 → GMM_TP 而非 GMM_EP

- **现象**: 推理输出乱码/精度崩溃（MoE 走了 Tensor Parallel 而非 Expert Parallel）
- **原因**: `--additional-config` 中 `sharding_strategy` 缺少 `expert_parallelism:8` 和 `tensor_parallelism:1`，导致 `use_ep=False`
- **影响**: 所有 MLA + EP 模型（R1, V3.2, GLM-5.1, K2.6）
- **修复**: `additional-config` 必须同时包含三个参数：
  ```json
  {
    "enable_dp_attention": true,
    "sharding_strategy": {
      "expert_parallelism": 8,
      "tensor_parallelism": 1
    }
  }
  ```
- **详细踩坑**: 见 [R1 README 踩坑 #24](./DeepSeek-R1-671B-FP4/)

### 问题 3: dp_scheduler.py hash_block_size 不兼容

- **现象**: `TypeError: Scheduler.__init__() got unexpected keyword argument 'hash_block_size'`
- **原因**: PR [#2412](https://github.com/vllm-project/tpu-inference/pull/2412) 在 `DPScheduler` 中透传了 `hash_block_size`，但基础 vLLM `Scheduler` 不接受此参数
- **影响**: 所有使用 `DPScheduler` 的场景（**全部 6 个模型**）
- **修复**: `inspect.signature` 动态检查参数是否存在再传递
- **Patch 位置**: `tpu_inference/core/sched/dp_scheduler.py`（已提交到 fork）
- **详细踩坑**: 见 [R1 README 踩坑 #25](./DeepSeek-R1-671B-FP4/)

### 影响矩阵

| 问题 | R1 | V3.2 | GLM-5.1 | K2.6 | Qwen3.5 | Qwen3-Coder |
|------|:--:|:----:|:-------:|:----:|:-------:|:-----------:|
| USE_MOE_EP_KERNEL | ✅ | ✅ | ✅ | ✅ | — | — |
| additional-config | ✅ | ✅ | ✅ | ✅ | — | — |
| hash_block_size | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> ✅ = 受影响　— = 不受影响（走 PyTorch+TorchAX 路径）

---

## 最近上游更新与待验证项

> **同步时间**：2026-05-01 | **上游 commit 范围**：`507cfa16..0b9f5583`（63 commits）
>
> 以下为 [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) 上游最近合入的重要更新，已同步到 fork `chrisya/main` 分支，尚未在各模型上逐一验证。

### 1. KV Cache Offload to Host Memory

| PR | 内容 | 状态 |
|----|------|:----:|
| [#2390](https://github.com/vllm-project/tpu-inference/pull/2390) | KV cache 从 HBM 卸载到 host DRAM，含异步传输 + staging buffer | ⏳ |
| [#2454](https://github.com/vllm-project/tpu-inference/pull/2454) | KV offloading 性能测试修复 | ⏳ |

**潜在收益**：
- **Kimi K2.6**：目前 v7x-8 单机 OOM（权重 84.58 GB/chip + KV cache 超 HBM），offload 后可能解除单机限制，部署门槛从 v7x-16 降至 v7x-8
- **DeepSeek R1/V3.2**：高并发（c=2048）时 KV cache 是瓶颈，offload 可支持更大并发
- **GLM-5.1**：每 device 仅剩 36 GB 给 KV cache，offload 扩大容量
- **Qwen3.5**：长文本（262K）场景受益

相关环境变量：`TPU_OFFLOAD_SKIP_JAX_PRECOMPILE`、`TPU_OFFLOAD_DECODE_SAVE`、`TPU_OFFLOAD_NUM_CPU_CHUNKS`、`TPU_OFFLOAD_NUM_STAGING_BLOCKS`

### 2. Qwen3.5 GDN 修复（4 个 PR）

| PR | 内容 | 影响 | 状态 |
|----|------|------|:----:|
| [#2408](https://github.com/vllm-project/tpu-inference/pull/2408) | GDN l2norm + sigmoid 升 fp32 对齐 GPU FLA 精度 | CoT 推理准确率修复（GPQA-Diamond 验证） | ⏳ |
| [#2431](https://github.com/vllm-project/tpu-inference/pull/2431) | fused GDN kernel 正确处理 has_initial_state | chunked-prefill 续传正确性 | ⏳ |
| [#2416](https://github.com/vllm-project/tpu-inference/pull/2416) | Compact mamba KV cache — GDN 层只分配 active 请求数 slot | 45 层 GDN 的 HBM 占用大幅减少 | ⏳ |
| [#2469](https://github.com/vllm-project/tpu-inference/pull/2469) | 清理释放的 slot id 为 null | 避免 GDN 状态残留污染输出 | ⏳ |

**验证建议**：重新跑 GSM8K eval 对比精度（之前 93.93%），同时测 compact KV 后高并发下的 HBM 占用。

### 3. MLA / DeepSeek 修复

| PR | 内容 | 受益模型 | 状态 |
|----|------|---------|:----:|
| [#2462](https://github.com/vllm-project/tpu-inference/pull/2462) | 修复 upstream break for mla_attention | R1, V3.2, GLM-5.1, K2.6 | ⏳ |
| [#2407](https://github.com/vllm-project/tpu-inference/pull/2407) | 重新启用 AG-FP8（All-Gather FP8），之前临时禁用 | R1, V3.2 | ⏳ |
| [#2343](https://github.com/vllm-project/tpu-inference/pull/2343) | MoE 返回 routed expert IDs | 所有 MoE 模型 | ⏳ |
| [#2412](https://github.com/vllm-project/tpu-inference/pull/2412) | 为 DeepSeek V4 预埋 input_ids + hash_block_size | 前瞻性 | — |

**注意**：#2462 是兼容性修复，不同步可能导致 MLA 模型在新版本上无法运行。

### 4. Multi-host / PD 分离改进

| PR | 内容 | 受益模型 | 状态 |
|----|------|---------|:----:|
| [#2414](https://github.com/vllm-project/tpu-inference/pull/2414) | 修复多机 device_put 用法错误 | K2.6 多机, 所有多机部署 | ⏳ |
| [#2435](https://github.com/vllm-project/tpu-inference/pull/2435) | PD 启动交错，缓解 host memory 压力 | GLM PD, Qwen3.5 PD, Qwen3-Coder PD | ⏳ |
| [#2392](https://github.com/vllm-project/tpu-inference/pull/2392) | attn_dp_expert 扩展模拟 attn_dp | R1, V3.2 | ⏳ |

### 5. 量化 + MoE 基础设施

| PR | 内容 | 受益模型 | 状态 |
|----|------|---------|:----:|
| [#2236](https://github.com/vllm-project/tpu-inference/pull/2236) | W4A8 FP8 linear layers（compressed tensors） | 潜在新量化路径 | ⏳ |
| [#2270](https://github.com/vllm-project/tpu-inference/pull/2270) | Jax native UnquantizedFusedMoE sharding 修复 | 所有 Jax native MoE 模型 | ⏳ |
| [#2398](https://github.com/vllm-project/tpu-inference/pull/2398) | DCP sharding axis + KV cache 支持 | 基础设施 | ⏳ |

### 6. 稳定性 + 可维护性

| PR | 内容 | 状态 |
|----|------|:----:|
| [#2399](https://github.com/vllm-project/tpu-inference/pull/2399) | deepcopy model_config 防止 config mutation | ⏳ |
| [#2441](https://github.com/vllm-project/tpu-inference/pull/2441) | Pin transformers==5.5.3 | ✅ 已合入 |
| [#2417](https://github.com/vllm-project/tpu-inference/pull/2417) | MLA KV cache text_config 适配多模态模型 | ✅ 已合入（替代我们的 K2.6 fallback） |
| [#2418](https://github.com/vllm-project/tpu-inference/pull/2418) | Kimi K2.6 加入 nightly CI | ✅ 已合入 |
| [#2396](https://github.com/vllm-project/tpu-inference/pull/2396) | Qwen3.5 jittable vision tower | ⏳ |
| [#2346](https://github.com/vllm-project/tpu-inference/pull/2346) | Kernel tuning 基础设施开源 | ⏳ |

### 模型影响矩阵

| 模型 | KV Offload | GDN 修复 | MLA Fix | AG-FP8 | Multi-host Fix | PD 优化 | Compact KV |
|------|:----------:|:--------:|:-------:|:------:|:--------------:|:-------:|:----------:|
| DeepSeek R1 | 中 | — | **高** | **高** | — | — | — |
| DeepSeek V3.2 | 中 | — | **高** | **高** | — | — | — |
| GLM-5.1 | 中 | — | **高** | — | — | 中 | — |
| Kimi K2.6 | **高** | — | **高** | — | **高** | — | — |
| Qwen3.5 | 低 | **高** | — | — | — | 中 | **高** |
| Qwen3-Coder | 低 | — | — | — | — | 中 | — |

> **高** = 直接影响功能正确性或解除硬件限制　**中** = 性能/容量提升　**低** = 边际改善　**—** = 不适用

### 验证优先级建议

1. **P0（必须验证）**：#2462 MLA fix — 所有 MLA 模型的兼容性修复，不验证后续可能跑不起来
2. **P0（高价值）**：#2390 KV Offload + K2.6 单机 — 如果可行则 K2.6 部署门槛降一半
3. **P1（质量提升）**：#2408 + #2431 Qwen3.5 GDN 精度 — 重新跑 GSM8K 对比
4. **P1（性能提升）**：#2407 AG-FP8 恢复 + #2416 Compact KV — R1/V3.2 吞吐和 Qwen3.5 HBM 效率
5. **P2（PD 改善）**：#2435 PD 启动优化 + #2414 多机 fix — GLM/Qwen PD 和 K2.6 多机
