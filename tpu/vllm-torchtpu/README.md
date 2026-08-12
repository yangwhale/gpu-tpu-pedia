# 大模型推理 on TPU —— vLLM TorchTPU 路线

**中文**

> ### 🚀 只想跑起来？→ [**QUICKSTART.md**](./QUICKSTART.md)
>
> 35 分钟从零到第一个 benchmark 数字，所有已知坑已预先填平，照抄即可，不需要排查任何问题。
> 本文件讲的是**架构与全景**；踩坑过程与证据在
> [RUNLOG](./Qwen3.5-397B-A17B-FP8/RUNLOG-20260812.md)。

> **定位声明**
>
> 本目录记录 **`vllm-torchtpu`**（vLLM 的 TPU platform plugin，PyTorch 路线）上的模型验证。
> 它与同级的 [`tpu-inference`](../tpu-inference/) 目录是**两条不同的技术路线**，不是替代关系：
>
> | | `tpu-inference` | `vllm-torchtpu`（本目录） |
> |---|---|---|
> | 前端 | JAX 原生 | PyTorch（vLLM 原样，模型定义不改） |
> | Kernel | Pallas | Pallas（同一套技术） |
> | 运行时 | JAX / PJRT | `torch_tpu` → PJRT |
> | 适合谁 | 已在 JAX 生态里的团队 | **想让现有 PyTorch/vLLM 栈零改动上 TPU 的团队** |
>
> 两条路线的 **kernel 层其实是同一套 Pallas 技术**，差别在前端语言和运行时桥接。所以 kernel 的算法工作可迁移，接入代码不可迁移。

---

## 这条路线是怎么搭起来的

理解下面这张分层图，后面每个模型文档里的参数和踩坑才有归属：

```
┌──────────────────────────────────────────────────────────────┐
│ ① vLLM 上游   engine / scheduler / 模型定义 / CustomOp 基类     │
│    纯 PyTorch，设备无关。TPU 相关代码一行都不在这里。            │
└──────────────────────────────────────────────────────────────┘
                   ↓  entry_points: vllm.platform_plugins
┌──────────────────────────────────────────────────────────────┐
│ ② vllm-torchtpu   vLLM 的 TPU platform plugin                 │
│    platforms/  TpuPlatform · 拓扑映射 · token padding · patch  │
│    layers/     forward_oot 覆盖 · custom_ops 桥接              │
│    kernels/    全部用 JAX + Pallas 写                          │
│    distributed/ TP · DP · EP · PCP · KV transfer               │
└──────────────────────────────────────────────────────────────┘
       ↓ 路径 A：普通 torch 算子      ↓ 路径 B：Pallas kernel
┌────────────────────────────────────────┐
│ ③ torch_tpu   PyTorch 的 TPU 设备后端    │
│    ops/  逐个 ATen 算子实现（C++ 为主）   │
│    custom_kernels.cc                    │
│      把 Pallas 的 MLIR 模块注入 HLO 图   │
└────────────────────────────────────────┘
                   ↓  两条路径在这里合流成同一张图
┌──────────────────────────────────────────────────────────────┐
│ ④ StableHLO / MLIR → XLA    SPMD 分区（Shardy）· fusion · 布局  │
└──────────────────────────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────────────────┐
│ ⑤ PJRT 运行时   →   ⑥ TPU 硬件（v5e / v6e / v7x）              │
└──────────────────────────────────────────────────────────────┘
```

三个容易误判的点，先说在前面：

1. **TorchTPU 不是 torch_xla 的延续。** `vllm-torchtpu` 全仓库 `import torch_xla` 出现 **0 次**，`import jax` 出现 **98 次**，75 个文件用 Pallas。它是另起炉灶的一套后端。
2. **TorchTPU 作用在 ATen 算子层。** 左边接 PyTorch ATen 约定与 `torch.compile`，右边接 StableHLO/MLIR → XLA → PJRT。它不做调度、不做 KV cache 管理、不做模型定义。
3. **torch 与 JAX 的公约数是 MLIR，不是什么 Python 桥。** Pallas kernel 被降级成 MLIR 模块后由 `custom_kernels.cc` **注入进 torch 的 HLO 图**一起编译，能参与 XLA 的布局选择和 SPMD 分区，两侧之间不需要跨运行时搬数据。

---

## 模型验证状态

> ✅ 本目录已复现并记录　📋 上游有 CI baseline、本目录待复现　🧩 零件齐备但无端到端证据　— 未见支持

| 模型 | 本目录状态 | 上游 perf baseline | 上游 eval baseline | 文档 |
|---|---|---|---|---|
| **Qwen3.5-397B-A17B-FP8** | 📋 待复现 | 3 份（tp1dp8 / tp2dp4 / tp8dp1）× 6 负载点 | MMLU · MMLU-Pro · HumanEval+ · MBPP+ | [详情](./Qwen3.5-397B-A17B-FP8/) |
| Qwen3.5-35B-A3B-FP8 | 📋 待复现 | 3 份（dp4tp2 / dp8 / tp4） | MMLU · MMLU-Pro | 待写 |
| Qwen3-Coder-480B-A35B | 📋 待复现 | 3 份（FP8 ×2 / NVFP4 ×1） | MMLU · MMLU-Pro · HumanEval+ · MBPP+ | 待写 |
| Qwen3-Coder-30B-A3B | 📋 待复现 | 2 份（BF16 / FP8） | MMLU · MMLU-Pro | 待写 |
| Qwen3-30B-A3B-NVFP4 | 📋 待复现 | 1 份 | MMLU · MMLU-Pro | 待写 |
| Gemma-4-26B-FP8 | 📋 待复现 | 1 份（text only） | MMLU · MMLU-Pro | 待写 |
| DeepSeek-V2 | 📋 待复现 | 1 份 tp8-ep | MMLU · MMLU-Pro | 待写 |
| DeepSeek V3.2 / V4 | 🧩 零件级 | **无** | **无** | 待写 |
| Kimi K3 | 🧩 零件级 | **无** | **无** | 待写 |
| GLM 5.2 | 🧩 在途 | **无** | **无** | — |
| Qwen3-VL | 🧩 部分 | **无** | **无** | — |

**读法**：📋 表示上游 CI 里有可信的回归基线，说明这条路被跑通过、且在持续被守护；本目录的工作是独立复现并把过程写成可执行步骤。🧩 表示 kernel 和单元测试齐备，但没有任何一次完整前向被记录过——这两者的差距比看上去大。

---

## 架构与特性覆盖

> ✅ 有专用 Pallas kernel　🔁 走通用 torch 算子路径　— 不适用　❌ 无实现

| 模型 | Attention | MoE | 量化 | 特殊组件 | 缺口 |
|---|---|---|---|---|---|
| Qwen3.5 | GQA + **GDN**（线性注意力）✅ | 512E + 1S ✅ SparseCore 加速 | FP8 ✅ | Hybrid KV ✅ | — |
| Qwen3-Coder | GQA ✅ | 128E ✅ | FP8 ✅ / NVFP4 ✅ | — | — |
| DeepSeek-V2 | MLA ✅ | MoE ✅ | KV FP8 ✅ | — | — |
| DeepSeek V3.2/V4 | MLA ✅ · SWA ✅ · **DSA indexer** ✅ | hash 路由 + sqrt-softplus ✅ | FP8 ✅ / MXFP4 ✅ | — | **mHC ❌** |
| Kimi K3 | **KDA**（Kimi Delta Attention）✅ | ✅ | — | 短卷积 ✅ | 端到端未验证 |
| Gemma-4 | ✅ | — | FP8 ✅ | 多模态未覆盖 | — |

### Kernel 家底（`src/vllm_torchtpu/kernels/`，行数）

| 目录 | 行数 | 服务对象 |
|---|---|---|
| `ragged_paged_attention` | 10,744 | 通用 paged attention（v2 / v3 / hd64 特化） |
| `gdn` | 9,028 | Gated DeltaNet 线性注意力（Qwen3.5 系） |
| `experimental` | 7,621 | 实验中 |
| `mla` | 6,767 | MLA 通用（DeepSeek / GLM） |
| `sparse_core` | 4,734 | SparseCore ragged gather/scatter/reduce（MoE 加速） |
| `deepseek_v4` | 3,904 | V4 专用：MLA / MLA+SWA / DSA topk / compressor |
| `kimi_k3` | 2,039 | KDA：chunk / ragged / decode 三形态 |
| `megablox` | 1,917 | MoE grouped matmul |
| `quantized_matmul` | 1,682 | 量化 GEMM |
| `fused_moe` | 1,563 | MoE 融合 |
| `collectives` | 843 | 通信 |
| `flash_attention` | 771 | — |
| `causal_conv1d` | 631 | 短卷积（Mamba / KDA 系） |

### 一个必须知道的成本比例

`ragged_paged_attention/v3/` 里，`kernel.py` 是 **1,926 行**，同目录的 `tuned_block_sizes.py` 是 **4,147 行**（v2 另有 1,493 行）。那张表是五层嵌套：

```
device_name → page_size → q_{dtype}_kv_{dtype}
            → q_head-N_kv_head-M_head-D → max_model_len
            ⇒ (num_kv_pages_per_block, num_queries_per_block)
```

覆盖 TPU v5 / v6e / v7 三代。**写 kernel 只是一半工作，另一半是按「芯片代 × page size × 精度组合 × head 配置 × 上下文长度」穷举调 block size**，任何一维变了都要重调。评估「支持一个新模型/新架构组件要多久」时，这个比例应该直接算进排期。

---

## 通用约束（所有模型适用）

### 静态形状税

TPU 要求静态形状，token 数必须补齐到预设桶，否则每个新形状触发一次重编译。`vllm-torchtpu` 用 `_get_token_paddings` / `_get_exponential_token_paddings` 做指数分桶。**首次运行的 cold start 里有相当一部分是在编译各个桶。**

### torch 算子在 TPU 上的绕行清单

`platforms/tpu_platform.py` 的 `_apply_model_specific_patches` 挂着一串 monkey patch，等于一份「哪些 torch 算子在 vLLM 的用法下不能直接用」的清单：

```
cumsum · tensor_cumsum · repeat_interleave
masked_scatter / masked_scatter_
get_rope_index · vision_attn_forward
```

有意思的是 `torch_tpu/ops/` 下正好有 `cumsum` 和 `masked_scatter` 的实现目录——说明底层实现了，但在 vLLM 的具体调用形态下仍需绕行。遇到这几个算子相关的报错，先看这里。

### benchmark 必须 temperature=0

`vllm bench serve` 默认走服务端采样（这些模型上是 temp 0.7），在 decode-heavy 的负载上**明显更慢**。仓库里所有 config 都显式设 `BENCHMARK_TEMPERATURE=0`。拿别处的数字来比之前，先确认对方也是贪心解码。

### `RANDOM_RANGE_RATIO` 的语义陷阱

golden 客户端（`benchmark_serving.py`）里 `0.8` 表示采样长度落在 `[0.8*len, len]`；而 `vllm bench serve` 原生的 `0.8` 表示 `[0.2*len, 1.8*len]`——**会溢出 `max-model-len`**。配置里用 `RANGE_RATIO_STYLE=min` 让 runner 做这层翻译。自己写 benchmark 命令时这是最容易踩的一脚。

### TTFT 在高并发 cell 上是被有意丢弃的

baseline JSON 里高并发的 cell **没有 `median_ttft_ms` 字段**，这不是记录失败。`check_regression.py` 的注释解释得很清楚：并发饱和时 median TTFT 反映的是排队位置的算术，不是延迟信号，所以 `--gate-ttft-max-concurrency` 会把它从 baseline 里 pop 掉让门禁报 SKIP。

> ⚠️ 用脚本读这些 JSON 时**不要用 `dict.get("median_ttft_ms", 0)`**——字段缺失会被默认值伪装成「实测 0 ms」。要判断的是字段在不在，不是值等不等于 0。

---

## 目录约定

每个模型一个目录，命名 `<模型>-<规模>-<精度>`，内含：

```
<Model>-<Size>-<Precision>/
├── README.md              # 中文主文档（可执行 runbook）
├── README.en.md           # 英文版（中文定稿后再翻译）
├── manifests/             # K8s YAML（如涉及 GKE 部署）
└── scripts/               # 补丁与工具脚本
```

写作约定沿用 `tpu-inference` 目录的风格：

- **实测数据在前，介绍在后**——读者是来复现的，不是来了解模型的
- **每条命令后跟「期望输出」注释**，让每一步可自验
- **踩坑就地标注**，不集中到文末
- **patch 精确到 PR 号 / commit hash**
- **明确区分「实测」与「引用上游」**——本目录尚未自测的数字一律标注来源
