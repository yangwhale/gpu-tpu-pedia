# DeepSeek V3.2 (671B MoE) — TPU v7 Training Guide

基于 MaxText 框架在 TPU v7 (Ironwood) 上训练 DeepSeek V3.2 671B MoE 模型的完整指南。

包含官方步骤 + 实测验证 + 踩坑记录，确保首次操作可复现。

## 目录

- [概述](#概述)
- [硬件需求](#硬件需求)
- [Step 1: 环境准备](#step-1-环境准备)
- [Step 2: Checkpoint 转换](#step-2-checkpoint-转换)
- [Step 3: Pre-training / Fine-tuning](#step-3-pre-training--fine-tuning)
- [Step 4: V3.2 Sparse Attention 两阶段训练](#step-4-v32-sparse-attention-两阶段训练)
- [Step 5: Decoding 验证](#step-5-decoding-验证)
- [Step 6: 正确性验证](#step-6-正确性验证)
- [MoE 策略说明](#moe-策略说明)
- [实测结果（TPU v7 4x4x4）](#实测结果tpu-v7-4x4x4)
- [踩坑记录](#踩坑记录)
- [参考链接](#参考链接)

## 概述

DeepSeek V3.2 是 DeepSeek 开源 MoE 模型家族的最新成员，引入了 **DeepSeek Sparse Attention (DSA)**，在保持模型性能的同时降低长上下文场景的计算复杂度。

**关键特性：**

| 特性 | 说明 |
|------|------|
| MLA (Multi-Head Latent Attention) | 低秩压缩 KV，减少 KV cache 内存 |
| MoE (Mixture of Experts) | 256 routed experts, 每 token 激活 top-8 |
| MTP (Multi-Token Prediction) | 投机解码，提升生成效率 |
| DSA (DeepSeek Sparse Attention) | Lightning Indexer 选择关键 token，减少注意力计算量 |
| FP8 混合精度 | 原始权重为 FP8，训练需转为 BF16 |

**MaxText 支持的 DeepSeek 模型：** V2-Lite (16B), V3 (671B), R1 (671B), V3.1 (671B), V3.2 (671B)。

> **官方参考文档：** [MaxText Run_DeepSeek.md](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/deepseek/Run_DeepSeek.md)

## 硬件需求

### 内存估算 (BF16 训练)

| 组件 | 计算 | 大小 |
|------|------|------|
| 模型参数 (BF16) | 671B × 2 bytes | ~1.34 TB |
| Adam 优化器 (FP32) | 671B × 2 × 4 bytes | ~5.36 TB |
| 梯度 (BF16) | 671B × 2 bytes | ~1.34 TB |
| 激活值 (估算) | 取决于 batch size / seq len | ~2-4 TB |
| **合计** | | **~10-12 TB** |

### TPU 配置对比

| | v5p-256 (官方示例) | v7 4x4x4 (实测) |
|--|------|------|
| 芯片数 | 128 | 64 |
| HBM/chip | 95 GB (HBM2e) | 192 GB (HBM3e) |
| 总 HBM | 12.16 TB | 12.29 TB |
| BF16 TFLOPS/chip | ~459 | ~2,306 |
| 总算力 | ~58.8 PFLOPS | ~147.2 PFLOPS |
| Devices | 128 (1 dev/chip) | 128 (2 dev/chip) |
| `ici_fsdp_parallelism` | 128 | 128 |

> **注意：** TPU v7 每 chip 有 2 个 TensorCore (device)，所以 64 chips = 128 devices，与 v5p-256 的 device 数量相同。

## Step 1: 环境准备

### GKE 集群

```bash
# 集群需要以下组件：
# - TPU node pool (v7x, topology 4x4x4)
# - JobSet controller
# - Kueue (可选，用于队列管理)
```

### Docker 镜像

使用包含 MaxText 和 DeepSeek 支持的镜像，例如：
```
us-docker.pkg.dev/cloud-tpu-multipod-dev/maxtext/maxtext-tpu-parambole-deeseek-custom:latest
```

> **⚠️ 踩坑 #1：** 确认镜像中的 LIBTPU 版本。旧版 LIBTPU（如 cl/831091709, Nov 2025）编译 671B MoE sparse matmul 极慢（6+ 小时未完成）。

## Step 2: Checkpoint 转换

DeepSeek V3.2 的 HuggingFace 权重为 FP8 格式，训练需要转换为 BF16 的 MaxText Orbax 格式。

### 2.1 下载模型权重

```bash
huggingface-cli download deepseek-ai/DeepSeek-V3.2 --local-dir <local_fp8_path>
```

> 模型约 700 GB (FP8)，下载时间取决于网络带宽。

### 2.2 FP8 → BF16 反量化

```bash
python3 -m maxtext.checkpoint_conversion.standalone_scripts.deepseek_fp8_to_bf16 \
    --input-fp8-hf-path=<local_fp8_path> \
    --output-bf16-hf-path=<local_bf16_path>
```

也可以用 DeepSeek 官方脚本 [fp8_cast_bf16.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) 在 GPU 上转换。

> 输出约 1.3 TB (BF16)。需要足够的磁盘空间。

### 2.3 转换为 MaxText Orbax 格式

**训练用（scanned 格式）：**
```bash
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=deepseek3.2-671b \
    scan_layers=true \
    attention=dot_product \
    base_output_directory=$BASE_OUTPUT_PATH \
    hf_access_token=$HF_TOKEN \
    hardware=cpu \
    skip_jax_distributed_system=True \
    --hf_model_path=$DEQUANTIZED_LOCAL_WEIGHTS \
    --eager_load_method=safetensors \
    --save_dtype=bfloat16
```

**解码用（unscanned 格式）：**
```bash
# 同上，但设 scan_layers=false
```

> **⚠️ 踩坑 #2：** 转换需要大量 RAM（模型 1.3 TB BF16），推荐在高内存机器上执行（如 B200 VM，3.8 TiB RAM）。

### 2.4 上传到 GCS

```bash
# 转换后上传到 GCS，供 TPU 训练作业读取
gsutil -m cp -r <orbax_ckpt_path> gs://<bucket>/deepseek-v3.2/maxtext-ckpt/
```

## Step 3: Pre-training / Fine-tuning

### Pre-training（从头训练）

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=deepseek_pre_training \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    model_name=deepseek3-671b \
    ici_fsdp_parallelism=128 \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    dataset_type=synthetic
```

### Fine-tuning（从 checkpoint 继续训练）

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=deepseek_fine_tuning \
    dataset_path=${DATASET_PATH} \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    per_device_batch_size=1 \
    model_name=deepseek3-671b \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    enable_checkpointing=true \
    ici_expert_parallelism=128 \
    ici_fsdp_parallelism=1
```

### Fine-tuning with MTP

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=deepseek_mtp_finetuning \
    dataset_path=${DATASET_PATH} \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    per_device_batch_size=1 \
    model_name=deepseek3-671b \
    steps=10000 \
    max_target_length=2048 \
    ici_fsdp_parallelism=128 \
    attention=flash \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    mtp_num_layers=1 \
    mtp_loss_scaling_factor=0.1
```

### Supervised Fine-tuning (SFT)

仅支持 HuggingFace conversational datasets：

```bash
python3 -m maxtext.trainers.post_train.sft.train_sft_deprecated src/maxtext/configs/post_train/sft.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    run_name=deepseek_sft \
    per_device_batch_size=1 \
    model_name=deepseek3-671b \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    enable_checkpointing=true \
    ici_expert_parallelism=128 \
    ici_fsdp_parallelism=1 \
    dataset_type=hf
```

## Step 4: V3.2 Sparse Attention 两阶段训练

DeepSeek V3.2 引入 DSA (DeepSeek Sparse Attention)，通过 **Lightning Indexer** 选择 top-k token 参与注意力计算。训练采用两阶段策略：

| | Dense Warmup (Stage 1) | Sparse Training (Stage 2) |
|--|------|------|
| 目的 | 让 indexer 学会选择关键 token | 正式训练，利用 sparse attention |
| 模型权重 | **冻结**，仅 indexer 可训练 | **全部可训练** |
| `indexer_sparse_training` | `False` | `True` |
| `trainable_parameters_mask` | `['.*indexer.*']` | 不设置 |

### Stage 1: Dense Warmup

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=indexer_dense_warmup \
    model_name=deepseek3.2-671b \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3.2 \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    async_checkpointing=false \
    ici_fsdp_parallelism=128 \
    steps=5 \
    max_target_length=4096 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=True \
    sparse_matmul=True \
    dataset_type=synthetic \
    indexer_loss_scaling_factor=0.01 \
    indexer_sparse_training=False \
    trainable_parameters_mask=['.*indexer.*']
```

### Stage 2: Sparse Training

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=indexer_sparse_training \
    model_name=deepseek3.2-671b \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3.2 \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    async_checkpointing=false \
    ici_fsdp_parallelism=128 \
    steps=5 \
    max_target_length=4096 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=True \
    sparse_matmul=True \
    dataset_type=synthetic \
    indexer_loss_scaling_factor=0.01 \
    indexer_sparse_training=True
```

> **⚠️ 踩坑 #3：** `attention=flash` + `use_tokamax_splash` 组合在 TPU v7 上 XLA 编译 70+ 分钟未完成。当前推荐使用 `attention=dot_product` 作为替代（编译快但性能略差）。详见[踩坑记录](#踩坑记录)。

## Step 5: Decoding 验证

```bash
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=decode \
    model_name=deepseek3-671b \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    hf_access_token=${HF_TOKEN} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    scan_layers=False \
    enable_checkpointing=true \
    async_checkpointing=false \
    per_device_batch_size=1 \
    max_prefill_predict_length=100 \
    max_target_length=1024 \
    attention=dot_product \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    ici_tensor_parallelism=128 \
    ici_fsdp_parallelism=1 \
    prompt="An attention function can be described as mapping a query and a set of key-value pairs to an output"
```

## Step 6: 正确性验证

### Logit 对比

```bash
# 1. 生成 HuggingFace golden logits
python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
    --model-id=deepseek-ai/DeepSeek-V2-Lite \
    --output-path=golden_DeepSeek-V2-Lite.jsonl \
    --prompts='I love to;Today is a;What is the'

# 2. 对比 MaxText vs HuggingFace logits
python3 -m tests.utils.forward_pass_logit_checker \
    src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=forward_pass_test \
    model_name=deepseek2-16b \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V2-Lite \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    scan_layers=false \
    per_device_batch_size=1 \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    sparse_matmul=False \
    dtype=float32 \
    activations_in_float32=true \
    matmul_precision=high \
    --max_kl_div=2e-4 \
    --golden_logits_path=golden_DeepSeek-V2-Lite.jsonl
```

### MMLU Benchmark

参考 [MaxText API Server Benchmark README](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/api_server/README.md)。

## MoE 策略说明

MaxText 支持多种 MoE routing 策略：

| 策略 | 参数 | 说明 |
|------|------|------|
| MegaBlocks Dropless | `sparse_matmul=True megablox=True` | 推荐，省内存 |
| JAX ragged_dot Dropless | `sparse_matmul=True megablox=False` | JAX 原生实现 |
| Dense Matmul Dropless | `sparse_matmul=False capacity_factor=-1` | 通用密集矩阵 |
| Dropping | `sparse_matmul=False capacity_factor=1.0~1.25` | 丢弃超出容量的 token |

> **⚠️ 踩坑 #4：** Dense MoE (`sparse_matmul=False megablox=False`) 会导致 OOM（111 GB > 94.75 GB/device），因为所有 256 experts 的权重都复制到每个 device。**必须使用 sparse matmul 或 EP 策略。**

## 实测结果（TPU v7 4x4x4）

### 测试环境

| 项目 | 配置 |
|------|------|
| GKE 集群 | `chrisya-v7x-v134` (us-central1) |
| TPU | v7x, 4x4x4 (64 chips = 128 devices), Spot |
| HBM 总量 | 64 × 192 GB = 12.3 TB |
| Docker 镜像 | `maxtext-tpu-parambole-deeseek-custom:latest` |
| JAX | 0.8.3 |
| 并行策略 | `ici_fsdp_parallelism=128` (纯 FSDP) |
| MoE | MegaBlocks dropless (`megablox=True sparse_matmul=True`) |

### 验证结果

| 指标 | 结果 |
|------|------|
| 模型参数 | 671.878B (验证通过) |
| HBM 使用 | 29.35 GB / 94.75 GB per device (**31%**) |
| Steps 完成 | 5/5 |
| MoE routing | 正常 |
| Pathways | **不需要**，标准 JobSet + 多控制器 JAX 即可 |

### 性能数据（synthetic data, dot_product attention）

| Step | 时间 (s) | TFLOP/s/device | Tokens/s/device | 说明 |
|------|----------|----------------|-----------------|------|
| 0 | 1.708 | 609 | 2,399 | 含 XLA 编译 |
| 1 | 0.311 | 3,344 | 13,163 | 稳态参考值 |

> **注意：** 以上为 from-scratch random init + synthetic data 的初步验证数据。真实性能需加载 checkpoint + 真实数据后测量。

### 已验证的参数组合

```
model_name=deepseek3.2-671b
attention=dot_product            # flash 在 v7x 上有编译问题
megablox=True sparse_matmul=True # 省内存，推荐
ici_fsdp_parallelism=128         # 纯 FSDP，128 devices
scan_layers=true
dtype=bfloat16 weight_dtype=bfloat16
per_device_batch_size=1
dataset_type=synthetic
```

### 不可行的组合（已排除）

| 组合 | 失败原因 |
|------|----------|
| `attention=flash` (任何 matmul 模式) | v7x LIBTPU GMM BoundsCheck crash 或 XLA 编译 crash |
| `megablox=False sparse_matmul=False` | Dense MoE OOM: 111 GB > 94.75 GB/device |
| `ici_fsdp_parallelism=128` + dense MoE | OOM: 所有 256 experts 复制到每个 device |

## 踩坑记录

### #1: LIBTPU 版本导致编译超时

**现象：** `sparse_matmul=True` 的 HLO 图在旧版 LIBTPU (cl/831091709, Nov 2025) 上编译 6+ 小时未完成。

**原因：** 671B MoE 的 sparse matmul HLO 计算图极其复杂（2.8 MB before_optimizations），旧版 XLA 优化器处理不过来。

**解决：** 等待更新的 Docker 镜像 / LIBTPU 版本，或在 Step 0 完成后预期 Step 1 编译需要较长时间。

### #2: DSA indexer 需要 use_tokamax_splash

**现象：** `NotImplementedError: Sparse indexer is only supported dot_product attention or flash attention with tokamax splash.`

**原因：** DSA indexer 要求 `attention=dot_product` 或 `attention=flash` + `use_tokamax_splash=True`，官方文档未明确说明。

**解决：** 添加 `use_tokamax_splash=True`，或改用 `attention=dot_product`。

### #3: Flash + tokamax_splash 编译 70+ 分钟

**现象：** Step 0 完成 (25.3s)，但 Step 1 卡在 XLA 编译不出来（等了 70+ 分钟）。

**原因：** tokamax splash + MegaBlocks + DSA indexer + trainable_parameters_mask 的组合导致 XLA 计算图极度复杂。

**解决：** 暂用 `attention=dot_product`。需向 MaxText team 确认 tokamax splash 在 v7 上的编译优化计划。

### #4: Dense MoE OOM

**现象：** `megablox=False sparse_matmul=False` 时 HBM 使用 111 GB > 94.75 GB/device，OOM。

**原因：** Dense 模式下所有 256 expert 的权重都会复制到每个 device。

**解决：** 必须使用 `sparse_matmul=True`（MegaBlocks 或 ragged_dot）或 `ici_expert_parallelism` 策略。

### #5: GCS 写权限不足

**现象：** `403 Forbidden: Provided scope(s) are not authorized`，无法写 GCS 输出目录。

**原因：** TPU node pool 的 oauth scope 只有 `devstorage.read_only`。

**解决：** 重建 node pool 时添加 `--scopes=cloud-platform`，或使用 Workload Identity。临时绕过：输出到本地 `/tmp/` 目录。

### #6: xpk device type 命名

**现象：** xpk 创建作业时找不到 TPU。

**原因：** TPU v7x 的 device type 是 `tpu7x-64`，不是 `v7-64` 或 `tpu7x-4x4x4`。

**解决：** 使用正确的 device type 名称，或直接用 kubectl + JobSet YAML 提交。

### #7: xpk ECP Proxy 拦截

**现象：** gLinux 上 xpk 提交作业失败，Docker image 验证被拦截。

**原因：** gLinux 的 ECP (Enterprise Certificate Proxy) 拦截 Docker registry 请求。

**解决：** 直接用 `kubectl apply -f jobset.yaml` 提交，绕过 xpk。

### #8: Orbax checkpoint 加载报 incomplete

**现象：** 加载转换好的 Orbax checkpoint 时报 "incomplete checkpoint" 错误。

**原因：** 可能是 Orbax 版本不匹配。

**状态：** 待验证，可能需要重新转换。

## GCS 数据路径

| 数据 | 路径 | 大小 |
|------|------|------|
| HF BF16 权重 | `gs://chrisya-v7x-us-central1/deepseek-v3.2/hf-bf16/` | 1.25 TiB |
| Orbax checkpoint | `gs://chrisya-v7x-us-central1/deepseek-v3.2/maxtext-ckpt/` | 665 GiB |

## 当前状态

**暂停中** — XLA 编译瓶颈待解决。

**Next Steps：**
1. 确认是否有更新的 Docker 镜像 / LIBTPU（需要 v7x sparse matmul 编译优化）
2. 验证 Orbax checkpoint 加载（可能需重新转换）
3. 向 MaxText team 确认 `use_tokamax_splash` 在 v7 上的编译优化计划
4. 加载 checkpoint + 真实数据，测量稳态 TFLOP/s 和 MFU
5. 完整两阶段训练（Dense Warmup → Sparse Training）

## 参考链接

| 链接 | 说明 |
|------|------|
| [MaxText Run_DeepSeek.md](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/deepseek/Run_DeepSeek.md) | MaxText 官方 DeepSeek 训练文档 |
| [CC Pages 验证报告](https://cc.higcp.com/pages/deepseek-v3.2-tpu-v7-complete-20260416.html) | 2026-04-15~16 TPU v7 实测全记录 |
| [DeepSeek V3.2 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | 模型权重下载 |
| [DSA 论文](https://arxiv.org/pdf/2512.02556) | DeepSeek Sparse Attention 原始论文 |
| [MaxText GitHub](https://github.com/AI-Hypercomputer/maxtext) | MaxText 训练框架 |
