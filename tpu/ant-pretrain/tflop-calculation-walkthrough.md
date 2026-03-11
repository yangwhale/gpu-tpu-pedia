# MaxText TFLOP/s 计算代码详解

> ALModel vs DeepSeek V3 全路径拆解，含 Bug 分析

## 目录

1. [概述：TFLOP/s 是怎么得到的](#1-概述)
2. [入口：metric_logger.py](#2-入口)
3. [主函数：calculate_tflops_training_per_device()](#3-主函数)
4. [Step 1：FFN FLOP 计算](#4-ffn-flop-计算)
5. [Step 2：Attention FLOP 计算](#5-attention-flop-计算)
6. [Step 3：Embedding FLOP 计算](#6-embedding-flop-计算)
7. [Step 4：层聚合 — 关键差异点](#7-层聚合)
8. [ALModel vs DeepSeek V3 完整数值对比](#8-数值对比)
9. [Bug 分析：ALModel TFLOP 虚高 9.5 倍](#9-bug-分析)
10. [附录：完整代码索引](#10-附录)

---

## 1. 概述

MaxText 的 TFLOP/s 是一个**公式估算值**，不是 profiler 实测值。计算流程：

```
模型初始化时:  config → calculate_tflops_training_per_device() → total_tflops (常量)
每个训练 step: TFLOP/s/device = total_tflops / step_time_seconds
```

total_tflops 包含三部分：

| 组件 | 说明 |
|------|------|
| learnable_weight_tflops | 所有可学习权重的矩阵乘法（FFN + QKV + Projection + Embedding） |
| attention_tflops | QK^T 和 Attention·V 的计算（受 causal mask 影响除以 2） |
| reference_model_tflops | 仅 DPO 训练时存在（额外一次 forward pass） |

**3x 乘子**：forward + backward 的 FLOPs 约为 forward 的 3 倍（forward=1x, backward=2x），因此 `learnable_weight_tflops` 最后乘 3。

---

## 2. 入口：metric_logger.py

### 2.1 初始化时计算 total_tflops（计算一次，全局复用）

```python
# metric_logger.py:269-272
def write_setup_info_to_tensorboard(self, params):
    num_model_parameters = max_utils.calculate_num_params_from_pytree(params)
    self.metadata[MetadataKey.PER_DEVICE_TFLOPS], _, _ = \
        maxtext_utils.calculate_tflops_training_per_device(self.config)
    self.metadata[MetadataKey.PER_DEVICE_TOKENS] = \
        maxtext_utils.calculate_tokens_training_per_device(self.config)
```

### 2.2 每步计算 TFLOP/s

```python
# metric_logger.py:305-312
def record_train_metrics(self, metrics, step, step_time):
    metrics["scalar"].update({"perf/step_time_seconds": step_time})
    if step >= self.config.rampup_end_step:
        metrics["scalar"].update({
            "perf/per_device_tflops": self.metadata[MetadataKey.PER_DEVICE_TFLOPS]
        })
        metrics["scalar"].update({
            "perf/per_device_tflops_per_sec":
                self.metadata[MetadataKey.PER_DEVICE_TFLOPS] / step_time
        })
```

### 2.3 日志输出

```python
# metric_logger.py:163
f"TFLOP/s/device: {scalars['perf/per_device_tflops_per_sec']:.3f}"
```

> **关键点**：`total_tflops` 在初始化时算一次，之后每步只做一次除法。如果公式有 bug，所有 step 的 TFLOP/s 都会系统性地偏高或偏低。

---

## 3. 主函数：calculate_tflops_training_per_device()

**文件**: `maxtext_utils.py:526-641`

整体结构如下：

```python
def calculate_tflops_training_per_device(config, log=True):
    # ① FFN flops
    if config.num_experts > 1:
        if config.decoder_block in (DEEPSEEK, LLAMA4, LING2, AL_MODEL):
            total_ffn_flops = calculate_routed_and_shared_ffn_tflops_per_device(config)
        else:
            # 通用 MoE 路径
            total_ffn_flops = gate_flops + ffn_flops * num_experts_per_tok
    else:
        total_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim)

    # ② Attention flops
    if config.attention_type == "mla":
        qkv_flops, noncausal_attention_flops, projection_flops = calculate_mla_tflops_per_device(config)
    else:
        # 标准 MHA 路径
        qkv_flops = ...
        noncausal_attention_flops = ...
        projection_flops = ...

    causal_attention_flops = noncausal_attention_flops / 2  # causal mask

    # ③ Embedding flops
    embedding_flops = 2 * B * S * E * V

    # ④ 按 decoder_block 类型聚合（关键差异点！）
    if decoder_block == DEEPSEEK / LING2:
        learnable_weight_tflops = (total_ffn_flops + (qkv + proj) * L + emb) * 3 / 1e12
        attention_tflops = causal_attn * L * 3 / 1e12
    else:  # 通用路径
        learnable_weight_tflops = ((total_ffn + qkv + proj) * L + emb) * 3 / 1e12
        attention_tflops = causal_attn * L * 3 / 1e12

    # ⑤ gradient_accumulation_steps 乘子
    learnable_weight_tflops *= config.gradient_accumulation_steps
    attention_tflops *= config.gradient_accumulation_steps

    total_tflops = learnable_weight_tflops + attention_tflops
    return total_tflops, learnable_weight_tflops, attention_tflops
```

---

## 4. FFN FLOP 计算

### 4.1 基础 FFN matmul（单层，单 expert）

```python
# maxtext_utils.py:348-360
def calculate_ffn_mamtul_tflops_per_device(config, mlp_dim):
    """单层 FFN 的 matmul FLOPs（不含 3x 乘子）"""
    # Gate/Up projection: hidden → mlp_dim (silu + linear = 2 个并行投影)
    ffn1_flops = 2 * B * S * mlp_dim * E * len(config.mlp_activations)
    # Down projection: mlp_dim → hidden
    ffn2_flops = 2 * B * S * mlp_dim * E
    return ffn1_flops + ffn2_flops
```

> **为什么乘 2**：每个矩阵乘法 `[M×K] · [K×N]` 的 FLOPs = `2 * M * K * N`（乘法 + 加法各一次）。

> **`mlp_activations`**：SwiGLU 风格使用 `["silu", "linear"]`，即有 2 个并行的 gate/up 投影，所以 ffn1 乘以 `len(mlp_activations)=2`。

单层 FFN 公式展开（假设 `mlp_activations=["silu","linear"]`）：

```
ffn_per_layer = 2 * B * S * E * mlp_dim * 2  +  2 * B * S * mlp_dim * E
              = 2 * B * S * E * mlp_dim * 3
              = 6 * B * S * E * mlp_dim
```

### 4.2 MoE 模型的 FFN（DeepSeek/ALModel 路径）

```python
# maxtext_utils.py:363-373
def calculate_routed_and_shared_ffn_tflops_per_device(config):
    """DeepSeek 风格 MoE FFN FLOPs（已乘以层数！）"""
    gate_flops = 2 * B * S * E * num_experts  # Router gate

    num_dense_layers, num_moe_layers = get_dense_moe_layers(config)

    # Dense 层: 用 mlp_dim (大), 乘以 dense 层数
    dense_ffn_flops = calculate_ffn_mamtul_tflops_per_device(config, config.mlp_dim) \
                      * num_dense_layers

    # Shared experts: 用 moe_mlp_dim (小), 乘以 shared_experts 数
    shared_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) \
                           * config.shared_experts

    # Routed experts: 用 moe_mlp_dim (小), 乘以 top-k (每 token 激活的 expert 数)
    routed_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) \
                           * config.num_experts_per_tok

    # MoE 层总 FFN = (gate + shared + routed) * MoE 层数
    moe_ffn_flops = (gate_flops + shared_experts_flops + routed_experts_flops) * num_moe_layers

    # 最终 = dense层FFN + MoE层FFN
    total_ffn_flops = dense_ffn_flops + moe_ffn_flops
    return total_ffn_flops  # ← 注意：已包含层数！
```

### 4.3 Dense/MoE 层数拆分

```python
# maxtext_utils.py:376-388
def get_dense_moe_layers(config):
    if config.decoder_block in (DEEPSEEK, LING2, AL_MODEL):
        num_dense_layers = config.first_num_dense_layers
        num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
        return num_dense_layers, num_moe_layers
    elif config.decoder_block == LLAMA4:
        num_moe_layers = config.num_decoder_layers // config.interleave_moe_layer_step
        num_dense_layers = config.num_decoder_layers - num_moe_layers
    return num_dense_layers, num_moe_layers
```

### 4.4 两个模型的 FFN 参数对比

| 参数 | ALModel | DeepSeek V3 |
|------|---------|-------------|
| `emb_dim` (E) | 2048 | 7168 |
| `mlp_dim` (dense FFN) | 5120 | 18432 |
| `moe_mlp_dim` (MoE FFN) | 512 | 2048 |
| `num_experts` | 256 | 256 |
| `num_experts_per_tok` (top-k) | 8 | 8 |
| `shared_experts` | 1 | 1 |
| `first_num_dense_layers` | 1 | 3 |
| `num_decoder_layers` (L) | 20 | 61 |
| Dense 层数 | 1 | 3 |
| MoE 层数 | 19 | 58 |

---

## 5. Attention FLOP 计算

ALModel 和 DeepSeek V3 都使用 **MLA (Multi-Head Latent Attention)**，走同一条代码路径。

### 5.1 MLA Attention（per-layer，不含 3x 乘子）

```python
# maxtext_utils.py:316-345
def calculate_mla_tflops_per_device(config):
    batch_len = B * S  # per_device_batch_size * max_target_length
    qk_head_dim_sum = qk_nope_head_dim + qk_rope_head_dim  # 128 + 64 = 192

    # ── Query projection ──
    if config.q_lora_rank == 0:
        # 无 LoRA: 直接投影
        q_flops = 2 * batch_len * E * num_q_heads * qk_head_dim_sum
    else:
        # 有 LoRA: down(E→lora_rank) + up(lora_rank→heads*dim)
        q_flops = 2 * batch_len * (
            E * q_lora_rank +
            q_lora_rank * num_q_heads * qk_head_dim_sum
        )

    # ── KV projection (always LoRA) ──
    # down: E → (kv_lora_rank + qk_rope_head_dim)
    # up: kv_lora_rank → num_q_heads * (qk_nope_head_dim + v_head_dim)
    kv_flops = 2 * batch_len * (
        E * (kv_lora_rank + qk_rope_head_dim) +
        kv_lora_rank * num_q_heads * (qk_nope_head_dim + v_head_dim)
    )

    qkv_flops = q_flops + kv_flops

    # ── Attention score + value ──
    # Q·K^T + Attention·V，包含 nope 和 rope 两部分
    attention_flops = 2 * batch_len * S * num_q_heads * (qk_head_dim_sum + v_head_dim)

    # ── Output projection ──
    projection_flops = 2 * batch_len * E * num_q_heads * v_head_dim

    return qkv_flops, attention_flops, projection_flops
```

> **MLA 的核心思想**：用低秩投影 (LoRA) 压缩 KV cache。不像标准 MHA 直接投影到 `head_dim * num_heads`，MLA 先投到低维的 `kv_lora_rank`，再展开到多头。

### 5.2 标准 MHA Attention（对比参考）

非 MLA 模型走这条路径：

```python
# maxtext_utils.py:544-563
# ── QKV projection ──
qkv_flops = 2 * B * S * E * (num_q_heads + 2 * num_kv_heads) * head_dim

# ── Attention score (Q·K^T) + Attention·V ──
noncausal_attention_flops = 4 * B * S² * num_q_heads * head_dim

# ── Output projection ──
projection_flops = 2 * B * S * E * num_q_heads * head_dim
```

### 5.3 Causal Mask 折半

```python
# maxtext_utils.py:565-569
# 因为 causal mask，attention 矩阵只有下三角有效
# 实际计算量约为 full attention 的一半
causal_attention_flops = noncausal_attention_flops / 2
```

### 5.4 两个模型的 Attention 参数对比

| 参数 | ALModel | DeepSeek V3 |
|------|---------|-------------|
| `attention_type` | mla | mla |
| `num_query_heads` | 16 | 128 |
| `num_kv_heads` | 16 | 128 |
| `head_dim` | 128 | - |
| `q_lora_rank` | 256 | 1536 |
| `kv_lora_rank` | 512 | 512 |
| `qk_nope_head_dim` | 128 | 128 |
| `qk_rope_head_dim` | 64 | 64 |
| `v_head_dim` | 128 | 128 |
| `qk_head_dim_sum` | 192 | 192 |

---

## 6. Embedding FLOP 计算

```python
# maxtext_utils.py:572
embedding_flops = 2 * B * S * E * V
```

这是 embedding lookup 的等效矩阵乘法 FLOPs（包括输入 embedding 和输出 logits 投影）。

| 参数 | ALModel | DeepSeek V3 |
|------|---------|-------------|
| `vocab_size` (V) | 157184 | 129280 |
| `emb_dim` (E) | 2048 | 7168 |

---

## 7. 层聚合 — 关键差异点（Bug 所在）

这是整个计算最容易出错的地方。不同 `decoder_block` 类型有不同的聚合逻辑。

### 7.1 DeepSeek/LING2 路径（正确路径）

```python
# maxtext_utils.py:594-598
elif config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LING2):
    learnable_weight_tflops = (
        total_ffn_flops +                              # ← 已含层数！
        (qkv_flops + projection_flops) * L +           # ← 乘以层数
        embedding_flops                                 # ← 只有1层
    ) * 3 / 1e12
    attention_tflops = causal_attention_flops * L * 3 / 1e12
```

**关键**：`total_ffn_flops` 来自 `calculate_routed_and_shared_ffn_tflops_per_device()`，这个函数**内部已经乘过层数了**（dense_ffn * dense层数 + moe_ffn * moe层数）。所以这里不需要再乘 `L`。

而 `qkv_flops` 和 `projection_flops` 是**单层**的值（来自 `calculate_mla_tflops_per_device()`），所以需要乘以 `L`。

### 7.2 通用 else 路径（不适用于 MoE 模型）

```python
# maxtext_utils.py:599-604
else:
    learnable_weight_tflops = (
        (total_ffn_flops + qkv_flops + projection_flops) * L +  # ← 全部乘 L
        embedding_flops
    ) * 3 / 1e12
    attention_tflops = causal_attention_flops * L * 3 / 1e12
```

**区别**：这里 `total_ffn_flops` 也被乘了 `L`。对于标准 dense 模型（所有层相同），这是正确的。但对于 MoE 模型，`calculate_routed_and_shared_ffn_tflops_per_device()` 已经内部处理了层数，再乘 `L` 就**重复计算**了。

### 7.3 公式对比

用符号表示：
- `F_ffn` = `total_ffn_flops`（已含层数，来自 `calculate_routed_and_shared_ffn_tflops_per_device`）
- `F_qkv` = `qkv_flops`（单层）
- `F_proj` = `projection_flops`（单层）
- `F_attn` = `causal_attention_flops`（单层）
- `F_emb` = `embedding_flops`
- `L` = `num_decoder_layers`

| 路径 | learnable_weight_tflops | 正确性 |
|------|------------------------|--------|
| **DeepSeek** | `(F_ffn + (F_qkv + F_proj) × L + F_emb) × 3 / 1e12` | ✅ 正确 |
| **else** | `((F_ffn + F_qkv + F_proj) × L + F_emb) × 3 / 1e12` | ❌ 对 MoE 模型 F_ffn 被多乘了 L |

---

## 8. ALModel vs DeepSeek V3 完整数值对比

以 ALModel benchmark 配置为例（`per_device_batch_size=12, max_target_length=4096, gradient_accumulation_steps=2`）：

### 8.1 ALModel 数值代入

```
B=12, S=4096, E=2048, V=157184, L=20
num_q_heads=16, q_lora_rank=256, kv_lora_rank=512
qk_nope=128, qk_rope=64, v_head_dim=128
mlp_dim=5120, moe_mlp_dim=512
num_experts=256, top_k=8, shared=1, dense_layers=1, moe_layers=19
mlp_activations=2 (silu+linear)
```

**FFN 计算**：

```
单层 dense FFN (mlp_dim=5120):
  ffn1 = 2 × 12 × 4096 × 5120 × 2048 × 2 = 2,061,584,302,080
  ffn2 = 2 × 12 × 4096 × 5120 × 2048     = 1,030,792,151,040
  per_layer_dense = 3,092,376,453,120

单层 MoE FFN (moe_mlp_dim=512):
  ffn1 = 2 × 12 × 4096 × 512 × 2048 × 2 = 206,158,430,208
  ffn2 = 2 × 12 × 4096 × 512 × 2048     = 103,079,215,104
  per_layer_moe = 309,237,645,312

Gate:
  gate = 2 × 12 × 4096 × 2048 × 256 = 51,539,607,552

Dense FFN total = per_layer_dense × 1 = 3,092,376,453,120
Shared experts = per_layer_moe × 1 = 309,237,645,312
Routed experts = per_layer_moe × 8 = 2,473,901,162,496
MoE per layer = gate + shared + routed = 2,834,678,415,360
MoE total = MoE_per_layer × 19 = 53,858,889,891,840

total_ffn_flops = 3,092,376,453,120 + 53,858,889,891,840 = 56,951,266,344,960
```

**Attention 计算 (MLA)**：

```
qk_head_dim_sum = 128 + 64 = 192
batch_len = 12 × 4096 = 49,152

Q projection (有 LoRA, q_lora_rank=256):
  q_flops = 2 × 49152 × (2048 × 256 + 256 × 16 × 192)
          = 2 × 49152 × (524288 + 786432)
          = 2 × 49152 × 1310720 = 128,849,018,880

KV projection:
  kv_flops = 2 × 49152 × (2048 × (512 + 64) + 512 × 16 × (128 + 128))
           = 2 × 49152 × (2048 × 576 + 512 × 16 × 256)
           = 2 × 49152 × (1179648 + 2097152)
           = 2 × 49152 × 3276800 = 322,122,547,200

qkv_flops = 128,849,018,880 + 322,122,547,200 = 450,971,566,080 (per layer)

Attention score + value:
  attention_flops = 2 × 49152 × 4096 × 16 × (192 + 128)
                  = 2 × 49152 × 4096 × 16 × 320 = 2,061,584,302,080
  causal = attention_flops / 2 = 1,030,792,151,040 (per layer)

Output projection:
  proj_flops = 2 × 49152 × 2048 × 16 × 128 = 412,316,860,416 (per layer)
```

**Embedding**:

```
embedding_flops = 2 × 12 × 4096 × 2048 × 157184 = 31,636,488,929,280
```

**层聚合（正确路径）**：

```
learnable_weight = (total_ffn + (qkv + proj) × L + emb) × 3 / 1e12
= (56,951,266,344,960 + (450,971,566,080 + 412,316,860,416) × 20 + 31,636,488,929,280) × 3 / 1e12
= (56,951,266,344,960 + 863,288,426,496 × 20 + 31,636,488,929,280) × 3 / 1e12
= (56,951,266,344,960 + 17,265,768,529,920 + 31,636,488,929,280) × 3 / 1e12
= 105,853,523,804,160 × 3 / 1e12
= 317.561 TFLOP

attention = causal × L × 3 / 1e12
= 1,030,792,151,040 × 20 × 3 / 1e12
= 61.848 TFLOP

subtotal = 317.561 + 61.848 = 379.408 TFLOP

× gradient_accumulation_steps (2):
total_tflops = 379.408 × 2 = 758.817 TFLOP
```

> 实际 benchmark 结果：step_time ≈ 9.87s → TFLOP/s ≈ 758.8 / 9.87 ≈ **76.9 TFLOP/s/device**

### 8.2 DeepSeek V3 参数规模参考

DeepSeek V3 的计算逻辑完全相同，只是数值更大：

```
B, S 取决于训练配置
E=7168, V=129280, L=61
num_q_heads=128, q_lora_rank=1536, kv_lora_rank=512
mlp_dim=18432, moe_mlp_dim=2048
dense_layers=3, moe_layers=58
```

由于 DeepSeek V3 是 671B 参数的大模型，每步 TFLOP 数值会大得多，但计算公式路径一致。

---

## 9. Bug 分析：共发现 4 个问题

### Bug 1：ALModel 层聚合路径错误（TFLOP 虚高 9.5 倍）

**Bug 位置**

```python
# maxtext_utils.py:594 (修复前)
elif config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LING2):
    # ↑ 缺少 DecoderBlockType.AL_MODEL ！
```

ALModel 走了 `else` 分支（line 599-604），`total_ffn_flops`（≈57T FLOPs，已含层数）被再乘 `L=20`。

**数值影响**

```
错误值（else 路径）:
= ((56.95T + 0.45T + 0.41T) × 20 + 31.64T) × 3 / 1e12 = 3563.5 TFLOP

正确值（DeepSeek 路径）:
= (56.95T + (0.45T + 0.41T) × 20 + 31.64T) × 3 / 1e12 = 317.6 TFLOP

倍数差异 ≈ 11.2x (learnable_weight 部分)，总 TFLOP 比约 9.5x
```

**修复**: 将 `AL_MODEL` 加入 DeepSeek/LING2 分支。

### Bug 2：Shared Expert 中间维度错误（TFLOP 低估 ~14%）

**Bug 位置**

```python
# maxtext_utils.py:369 (修复前)
shared_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, config.moe_mlp_dim) * config.shared_experts
#                                                                      ^^^^^^^^^^^^^^^ 应该用 moe_shared_expert_intermediate_size
```

**根因**：ALModel 配置中 shared expert 的维度与 routed expert 不同：
- `base_moe_mlp_dim: 512` — routed experts 的 FFN 中间维度
- `moe_shared_expert_intermediate_size: 2048` — shared expert 的 FFN 中间维度

但 TFLOP 公式对两者都用了 `moe_mlp_dim=512`。实际模型代码 (`moe.py:2071`) 使用的是正确的维度：

```python
self.shared_experts = linears.MlpBlock(
    intermediate_dim=self.config.shared_experts * self.config.moe_shared_expert_intermediate_size,  # 1 × 2048
)
```

**数值影响**

```
每 MoE 层差额 = 6 × B × S × E × (2048 - 512) = 0.928T FLOPs
× 19 MoE layers = 17.63T FLOPs (forward)
× 3 (fwd+bwd) × GDA(2) = +105.8 TFLOP
```

**修复**：

```python
# maxtext_utils.py:370-374 (修复后)
shared_expert_mlp_dim = (
    config.moe_shared_expert_intermediate_size
    if getattr(config, 'moe_shared_expert_intermediate_size', 0) > 0
    else config.moe_mlp_dim
)
shared_experts_flops = calculate_ffn_mamtul_tflops_per_device(config, shared_expert_mlp_dim) * config.shared_experts
```

### Bug 3：MTP (Multi-Token Prediction) FLOPs 完全缺失（TFLOP 低估 ~30%）

**根因**：ALModel 配置了 `mtp_num_layers: 1`，但 `calculate_tflops_training_per_device()` 中没有任何 MTP 相关代码。

每个 MTP 层包含 3 个计算密集组件：

```
┌──────────────────────────────────────────────┐
│ MTP Layer                                    │
│                                              │
│  1. Projection: [2E, E] matmul  = 0.825T     │
│  2. Transformer Layer:                       │
│     - MoE FFN (gate+shared+routed) = 3.763T  │
│     - Attention (MLA/GLA)          ≈ 2.1T    │
│  3. Output Head: [E, V] matmul   = 31.636T   │ ← 最大！
│                                              │
│  Forward Total ≈ 38.3T FLOPs                 │
└──────────────────────────────────────────────┘
```

> Output head 占大头因为 vocab_size=157184 巨大。MTP 层需要计算完整的 logits 分布用于下一个 token 的损失。

**数值影响**

```
MTP forward = 38.3T FLOPs
× 3 (fwd+bwd) × GDA(2) = +229.9 TFLOP
```

**修复**：新增 `calculate_mtp_tflops_per_device()` 函数并集成到主函数。

### Bug 4：Hybrid Attention 未建模（已知局限）

ALModel 使用 `layer_group_size: 5` 的混合注意力架构：

```python
# ALModel/al_model.py:47-54
def is_mla(num_layers, layer_group_size, layer_index):
    is_last_of_group = (layer_index + 1) % layer_group_size == 0
    is_last_of_model = (layer_index == num_layers - 1)
    return is_last_of_group or is_last_of_model
```

20 层中：**4 层 MLA**（full attention, S² 复杂度）+ **16 层 GLA**（linear attention）。

当前公式把所有 20 层都当 MLA 算。实际：
- GLA 投影 FLOPs 更高（多了 gate projection: 2×B×S×E×H×d）
- GLA attention FLOPs 远低（线性复杂度，无 S² 项）
- 净效果：约 +18.5 TFLOP（GLA 投影增量 > attention 减少量）

> 此项未在当前修复中实现，需要更深入的按层类型分别计算。标记为 TODO。

### 9.5 修复后完整数值对比（TPU v7 实测验证 ✅）

**TPU v7 (Ironwood) peak**: 2307 TFLOP/s BF16 per chip, **1153.5 TFLOP/s per device** (TensorCore)

**实测配置**: batch=12, GDA=2, FSDP=64, 30 steps on 32-chip TPU v7 (2x4x4)

| 指标 | Bug 1 原始 | Bug 1 修复 | 全部修复 (1-3) | TPU 实测验证 ✅ |
|------|:----------:|:--------:|:----------------:|:----------:|
| total_tflops | ~7,200 | ~759 | ~1,113 | **1,093.39** |
| step_time | ~9.87s | ~9.87s | ~9.87s | **9.875s** |
| TFLOP/s/device | ~731 | ~76.9 | ~112.8 | **110.7** |
| Tokens/s/device | ~9,960 | ~9,960 | ~9,960 | **9,955** |
| MFU (vs 1153.5) | 63.4% ❌虚高 | 6.7% ❌虚低 | ~9.8% | **9.6%** ✅ |

> **step_time 不变**（~9.87s）——step_time 是实测值，不受公式影响。
> 变的是 `total_tflops`（公式计算值）和由此派生的 `TFLOP/s/device` 和 `MFU`。

**公式结构验证**: 88.12% learnable weight + 11.88% attention flops。

#### TFLOP 组成分解（batch=12, GDA=2）

```
MoE Routed Experts (top-8)   282 T  (25.8%)  ████████████
MTP (all)                    229 T  (20.9%)  ██████████
Embedding                    190 T  (17.4%)  ████████
MoE Shared Expert            141 T  (12.9%)  ██████
Attention Core               124 T  (11.3%)  █████
MLA QKV                       54 T  ( 4.9%)  ██
MLA Out Proj                  49 T  ( 4.5%)  ██
Dense FFN                     19 T  ( 1.7%)
MoE Gate                       6 T  ( 0.5%)
─────────────────────────────────────────────
TOTAL                       1093 T
```

Active params ≈ 1.85B (from FLOPs/2/tokens)，forward FLOPs/token ≈ 3.71 GFLOP。

#### MFU 9.6% 的原因分析

对于小 MoE 模型在大规模设备上训练，~10% MFU 是合理的：
- **模型太小**：1.85B active params，每层参数量不足以饱和 TPU
- **FSDP=64**：模型被切分到 64 个设备，通信开销占比大
- **MoE all-to-all**：256 experts 的 dispatch/combine 通信开销
- **参考**：增大 batch/GDA 或模型规模可以提高 MFU

> 独立验证脚本：`scripts/standalone_tflop_calc.py`（无 MaxText 依赖，纯 Python 公式复现）
> TPU 实测脚本：`bench-tflop-v3` JobSet (2026-03-06)

---

## 10. 附录：完整代码索引

| 函数 | 文件位置 | 说明 |
|------|---------|------|
| `calculate_tflops_training_per_device()` | `maxtext_utils.py:595-710` | 主入口，汇总所有 FLOP |
| `calculate_mla_tflops_per_device()` | `maxtext_utils.py:316-345` | MLA 注意力 FLOP（per-layer） |
| `calculate_ffn_mamtul_tflops_per_device()` | `maxtext_utils.py:348-360` | 单层 FFN matmul FLOP |
| `calculate_routed_and_shared_ffn_tflops_per_device()` | `maxtext_utils.py:363-379` | MoE FFN FLOP（含层数，已修复 shared expert dim） |
| `calculate_mtp_tflops_per_device()` | `maxtext_utils.py:397-457` | **新增** MTP FLOP 计算 |
| `get_dense_moe_layers()` | `maxtext_utils.py:382-394` | Dense/MoE 层数拆分 |
| `write_setup_info_to_tensorboard()` | `metric_logger.py:269-277` | 初始化时计算 total_tflops |
| `record_train_metrics()` | `metric_logger.py:305-317` | 每步计算 TFLOP/s |
| `DecoderBlockType` | `common_types.py:79-99` | 模型类型枚举 |

### 模型配置文件

| 模型 | 配置路径 |
|------|---------|
| ALModel | `MaxText/configs/models/al_model.yml` |
| DeepSeek V3 671B | `MaxText/configs/models/deepseek3-671b.yml` |
| DeepSeek V2 236B | `MaxText/configs/models/deepseek2-236b.yml` |

---

*文档版本: 2026-03-06 v2 | 新增 Bug 2-4 分析，TPU v7 peak specs | 基于 MaxText commit at ant-pretrain repo*
