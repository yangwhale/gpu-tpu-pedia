# ALModel 8B Training on TPU v7 (Ironwood)

> ant-pretrain 项目（基于 MaxText fork）在 TPU v7 上训练 ALModel 8B 的完整记录。
> 包含从环境搭建到实际训练的所有步骤、踩坑记录和性能数据。

## 目录

- [模型架构](#模型架构)
- [硬件环境](#硬件环境)
- [训练步骤](#训练步骤)
- [训练结果](#训练结果)
- [性能对比](#性能对比)
- [踩坑记录](#踩坑记录)
- [参考文件](#参考文件)

---

## 模型架构

**ALModel 8B** — 混合注意力 + MoE 架构

| 参数 | 值 |
|------|-----|
| 架构 | MLA/KDA Hybrid Attention + MoE |
| 总参数量 | ~8B |
| Hidden dim | 2560 |
| Layers | 28 |
| Query heads | 20 |
| KV heads | 20 (MLA) |
| Head dim | 128 |
| Vocab size | 163,840 |
| Experts | 64 (top-6 routing) |
| Shared experts | 1 |
| Expert FFN dim | 1280 |
| Dense FFN dim | 6656 |
| 注意力 | 每 4 层一组，最后一层 MLA（full attention），其余 KDA（linear attention） |
| RoPE | YaRN, theta=50000, factor=32, max_pos=131072 |

### MLA (Multi-Head Latent Attention) 参数
- `q_lora_rank`: 512
- `kv_lora_rank`: 256
- `qk_nope_head_dim`: 128
- `qk_rope_head_dim`: 64
- `v_head_dim`: 128

### MoE 路由
- Score function: sigmoid
- Routed bias: enabled (update rate 0.001)
- Scaling factor: 2.827
- 第 1 层 dense，第 2-28 层 MoE

---

## 硬件环境

### GKE 集群
- **集群**: `chrisya-v7x-training` (us-central1)
- **项目**: `cloud-tpu-multipod-dev`
- **容器镜像**: `gcr.io/cloud-tpu-multipod-dev/chrisya-maxtext-runner`

### TPU 配置

| 配置 | 小规模测试 (2x2x1) | 中规模测试 (2x4x4) |
|------|---------------------|---------------------|
| Chips | 4 | 32 |
| Devices | 8 (2 TCs/chip) | 64 (2 TCs/chip) |
| Hosts | 1 | 8 |
| 物理拓扑 | 2x2x1 | 2x4x4 |
| 物理 mesh | [2, 2, 1, 2] | [2, 4, 4, 2] |
| Machine type | tpu7x-ultranet-4t | tpu7x-ultranet-4t |
| HBM/device | 94.75 GB | 94.75 GB |

### GCS 路径
- 代码: `gs://chrisya-v7x-us-central1/ant-pretrain/`
- 训练输出: `gs://chrisya-v7x-us-central1/almodel-training-output/`
- 测试数据集: `gs://chrisya-v7x-us-central1/ant-pretrain/test-dataset.jsonl` (500 行)

---

## 训练步骤

### 前置条件
1. GKE 集群已创建（`xpk cluster create`）
2. TPU v7 预留可用
3. ant-pretrain 代码和 tokenizer 已上传到 GCS
4. 容器镜像已构建（基于 MaxText 官方镜像 + 自定义 layer）

### Step 1: 配置 Kueue ResourceFlavor

```bash
# xpk cluster adapt 会创建 Kueue 的 ResourceFlavor 和 ClusterQueue
# 注意: tpu7x-N 中 N 是 device 数（= 2 × chip 数）
xpk cluster adapt \
  --cluster=chrisya-v7x-training \
  --project=cloud-tpu-multipod-dev \
  --zone=us-central1-ai1a \
  --tpu-type=tpu7x-64    # 32 chips = 64 devices
```

### Step 2: 创建 Workload Policy (Placement Policy)

TPU v7 multi-host 用 workload policy（命名仍叫 placement-policy）：

```bash
gcloud alpha compute workload-policies create tpu7x-64-2x4x4-placement-policy \
  --project=cloud-tpu-multipod-dev \
  --region=us-central1 \
  --type=HIGH_THROUGHPUT \
  --acceleratorTopology=2x4x4 \
  --acceleratorType=TPU_V7
```

> **Gang size** = num_hosts = chips / 4 = 32 / 4 = 8

### Step 3: 创建 Node Pool

```bash
gcloud beta container node-pools create np-tpu7x-64 \
  --cluster=chrisya-v7x-training \
  --project=cloud-tpu-multipod-dev \
  --zone=us-central1 \
  --node-locations=us-central1-ai1a \
  --machine-type=tpu7x-ultranet-4t \
  --num-nodes=8 \
  --placement-policy=tpu7x-64-2x4x4-placement-policy \
  --reservation-affinity=specific \
  --reservation=cloudtpu-20260211010000-500993041 \
  --scopes=storage-full,gke-default \
  --enable-gvnic
```

### Step 4: 准备训练脚本

训练脚本见 [reference-code/train-almodel-64dev.sh](reference-code/train-almodel-64dev.sh)

关键参数：
```bash
# 并行策略 (16 × 4 × 1 = 64 devices)
ici_fsdp_parallelism=16
ici_tensor_parallelism=4
ici_context_parallelism=1

# 训练参数（受 HBM 限制缩小）
per_device_batch_size=1
max_target_length=4096

# Optimizer
opt_type=muon
learning_rate=5e-4
adam_weight_decay=0.1
muon_weight_decay=0.1

# 重计算（MoE 必须开）
remat_policy=full
```

### Step 5: 上传脚本到 GCS

```bash
gsutil cp train-almodel.sh gs://chrisya-v7x-us-central1/ant-pretrain/train-almodel.sh
```

### Step 6: 提交训练 Workload

```bash
xpk workload create \
  --cluster=chrisya-v7x-training \
  --project=cloud-tpu-multipod-dev \
  --zone=us-central1 \
  --workload=almodel-64dev-test \
  --device-type=tpu7x-64 \
  --num-slices=1 \
  --docker-image=gcr.io/cloud-tpu-multipod-dev/chrisya-maxtext-runner \
  --command="gsutil cp gs://chrisya-v7x-us-central1/ant-pretrain/train-almodel.sh /tmp/train-almodel.sh && bash /tmp/train-almodel.sh"
```

### Step 7: 监控训练

```bash
# 查看 Pod 状态
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=almodel-64dev-test

# 查看日志（多 host 容器名是 jax-tpu-1, jax-tpu-2，不是 jax-tpu）
kubectl logs <pod-name> -c jax-tpu-1 -f
```

### Step 8: 清理

```bash
# 删 workload
xpk workload delete --workload=almodel-64dev-test \
  --cluster=chrisya-v7x-training \
  --project=cloud-tpu-multipod-dev --zone=us-central1

# 删 node pool
gcloud container node-pools delete np-tpu7x-64 \
  --cluster=chrisya-v7x-training \
  --project=cloud-tpu-multipod-dev --zone=us-central1
```

---

## 训练结果

### 测试 1: 2x2x1 (4 chips, 8 devices, 单 host)

| 参数 | 值 |
|------|-----|
| ICI | fsdp=2, tensor=2, context=1 |
| Batch | per_device=1, seq=2048 |
| HBM 使用 | 52.73G / 94.75G (55.7%) |

**训练日志**:
```
Step 0: loss=12.586, 485s (编译)
Step 1: loss=12.584, 355s
Step 2: loss=12.530, 0.05s (async pipeline)
Step 3: loss=11.992, 220s (稳定)
...
Step 9: loss=0.078, ~220s
```

| 指标 | 值 |
|------|-----|
| 稳定 step time | ~220s |
| TFLOP/s/device | 0.53 |
| 完成步数 | 10/10 |
| EXIT_CODE | 0 |

### 测试 2: 2x4x4 (32 chips, 64 devices, 8 hosts)

| 参数 | 值 |
|------|-----|
| ICI | fsdp=16, tensor=4, context=1 |
| Batch | per_device=1, seq=4096 |
| HBM 使用 | 63.9G / 94.75G (67.4%) |

**训练日志**:
```
Step 0: loss=12.153, 485s (编译)
Step 1: loss=12.154, 67.3s (warmup)
Step 2: loss=8.509,  0.076s (async pipeline)
Step 3: loss=5.272,  7.954s (稳定)
Step 4: loss=2.599,  7.450s
Step 5: loss=0.695,  7.462s
Step 6: loss=0.213,  17.534s (数据耗尽)
```

| 指标 | 值 |
|------|-----|
| 稳定 step time | ~7.5s |
| TFLOP/s/device | 32.0 |
| Tokens/s/device | 549 |
| 完成步数 | 7/10 (数据耗尽) |
| EXIT_CODE | 1 (StopIteration) |

> 注: EXIT_CODE=1 是因为 500 行测试数据不够 10 步训练（64 devices × batch=1 × seq=4096 = 每步 262,144 tokens），不是真正的错误。

---

## 性能对比

### TPU v7 小规模 vs 大规模

| 指标 | 2x2x1 (8 dev) | 2x4x4 (64 dev) | 提升倍数 |
|------|----------------|-----------------|----------|
| Devices | 8 | 64 | 8× |
| Seq length | 2048 | 4096 | 2× |
| Step time | 220s | 7.5s | 29× |
| TFLOP/s/dev | 0.53 | 32.0 | 60× |
| Tokens/s/dev | — | 549 | — |
| HBM 使用率 | 55.7% | 67.4% | — |

**分析**: 提升远超线性（8× devices → 29× step time, 60× TFLOP/s），原因：
1. 更大 seq_len (4096 vs 2048) → 更高计算密度
2. 更好的 FSDP 并行效率（16-way vs 2-way）
3. 单 host 上计算 bound 太重，多 host 更好地利用了 ICI 带宽

### vs 原始 v5e-256 基准 (参考)

来自 ant-pretrain 工作完成总结：

| 指标 | TPU v7 (64 dev) | v5e-256 DeepSeek 8B | v5e-256 Kimi 8B |
|------|-----------------|---------------------|-----------------|
| Devices | 64 | 256 | 256 |
| Tokens/s/dev | 549 | ~45,000 | ~40,000 |
| MFU | — | ~55% | ~52% |
| HBM/dev | 63.9G | ~16G | ~18G |

> **注意**: 直接对比不公平，原因：
> 1. 模型不同：ALModel (64 experts, top-6) vs DeepSeek/Kimi (32 experts, top-4)，ALModel 的 MoE 更大
> 2. 训练规模不同：batch=1/seq=4096 (受 HBM 限制) vs batch=4/seq=8192 (原始设计)
> 3. 硬件代数不同：v7 是更新的芯片，HBM 更大但架构不同
> 4. 数据量：我们只跑了 7 步 smoke test，稳态性能可能不同
> 5. TPU v7 的 549 tokens/s/dev 是在极小 batch 下的数字，增大 batch 后应该会大幅提升

---

## 踩坑记录

### 1. xpk device type 命名混乱 (TPU v7)

**问题**: `xpk --tpu-type=tpu7x-N` 中 **N 是 device 数**（不是 chip 数），但容易跟 chip 数混淆。

**表现**:
- 32 chips → 应该用 `tpu7x-64`（因为 64 devices = 32 chips × 2 TCs/chip）
- 最初用 `tpu7x-32`，结果创建了 16-chip 的 topology `2x2x4`

**教训**: TPU v7 的 device = 2 × chip。xpk 的 type 参数是 device 数。

### 2. 物理 mesh → 逻辑 mesh 映射失败

**问题**: `ici_fsdp=8, tensor=4, context=2` 无法映射到物理 mesh `[2, 4, 4, 2]`。

**错误信息**:
```
NotImplementedError: Failed to find assignment for logical_axis_index 5 of size 2
with remaining assignable mesh [0, 4, 4, 0]
```

**原因**: 逻辑 axis 的每个值必须是物理 mesh 某些 axis 的子集乘积。`context=2` 在其他 axis 占用合适的维度后，找不到匹配。

**解决**: 改为 `ici_fsdp=16, tensor=4, context=1`：
- fsdp=16 = 4×4 ✓
- tensor=4 = 2×2 ✓
- context=1 = trivial ✓

**教训**: TPU v7 上避免 `context_parallelism=2`，除非确认物理 mesh 能映射。

### 3. MoE 模型 OOM — HBM 限制严格

**问题**: ALModel 的 64 experts 极其吃内存，即使开了 `remat_policy=full` 也不够。

**迭代过程**:

| 尝试 | batch | seq | HBM 需求 | 结果 |
|------|-------|-----|----------|------|
| 1 | 4 | 8192 | 214.45 GB | OOM (2.4× 超限) |
| 2 | 1 | 8192 | 112.90 GB | OOM (1.27× 超限) |
| 3 | 1 | 4096 | 63.9 GB | 成功 (67% 利用率) |

**原因**: XLA rematerialization 无法大幅降低 MoE 的峰值内存 — 64 个 expert 的参数 + 路由 activation 占主导。

**教训**: MoE 模型 HBM 规划必须保守，64 experts 在 94.75G HBM 上只能跑 batch=1/seq=4096。要恢复原始 batch=4/seq=8192 需要 expert parallelism 或更大的 HBM。

### 4. Placement Policy 是 Workload Policy

**问题**: TPU v7 multi-host 实际用 workload policy (type: HIGH_THROUGHPUT)，但命名约定仍叫 `xxx-placement-policy`。

**关键参数**:
- `tpu7x-64-2x4x4-placement-policy`: 64 devices, topology 2x4x4, gang size=8
- `tpu7x-32-2x2x4-placement-policy`: 32 devices, topology 2x2x4, gang size=4

**Gang size** = num_hosts = chips / (chips_per_host=4)

### 5. Multi-host GKE 容器命名

**问题**: 单 host 时容器名 `jax-tpu`，多 host 时变成 `jax-tpu-1`, `jax-tpu-2`（每 host 2 个容器）。

**影响**: `kubectl logs` 需要 `-c jax-tpu-1` 指定容器名。

### 6. optax 兼容性

**问题**: 容器的 optax 版本不支持 `consistent_rms` 参数，但 ant-pretrain 的 `optimizers.py` 硬编码传了。

**解决**: 用 `inspect.signature(muon).parameters` 检测是否支持，条件传递：
```python
if 'consistent_rms' in inspect.signature(muon).parameters:
    opts['consistent_rms'] = consistent_rms_value
```

### 7. Muon Optimizer 参数名

**问题**: 传 `weight_decay` 会报 `not in valid fields`。

**解决**: ALModel + Muon 需要两个独立参数：
- `adam_weight_decay=0.1`（Adam 部分）
- `muon_weight_decay=0.1`（Muon 部分）

### 8. 数据量不足导致 StopIteration

**问题**: 500 行 JSONL 测试数据，64 devices × batch=1 × seq=4096 = 每步 262,144 tokens，不够 10 步。

**表现**: EXIT_CODE=1 (StopIteration)，但训练实际完成了 7 步。

**教训**: 做 smoke test 时要估算数据需求。公式：
```
min_lines ≈ steps × devices × batch × seq / avg_tokens_per_line
```

---

## 参考文件

### 本目录

| 文件 | 说明 |
|------|------|
| `README.md` | 本文档 |
| `reference-code/train-almodel-64dev.sh` | 64-device 最终训练脚本 |
| `reference-code/train-almodel-8dev.sh` | 8-device 小规模测试脚本（供对比） |
| `reference-code/al_model.yml` | ALModel 模型配置 |
| `reference-code/pretrain_al_model.sh` | 原始训练脚本（ant-pretrain repo） |

### ant-pretrain Repo 参考

| 文件 | 说明 |
|------|------|
| `src/MaxText/configs/models/al_model.yml` | 模型配置（完整 YAML） |
| `src/MaxText/configs/base.yml` | MaxText 基础配置 |
| `ALModel/al_model.py` | 模型实现代码 |
| `scripts/pretrain_al_model.sh` | 原始训练脚本 |
| `工作完成总结.md` | DeepSeek/Kimi 8B 性能基准 |
| `QUICKSTART_8B.md` | 快速开始指南 |
| `DEEPSEEK_KIMI_8B_PRETRAIN.md` | 详细预训练文档 |

### GCS 路径

```
gs://chrisya-v7x-us-central1/
├── ant-pretrain/                    # 完整代码
│   ├── train-almodel.sh             # 训练脚本
│   ├── test-dataset.jsonl           # 500行测试数据
│   └── src/MaxText/optimizers.py    # patched optax 兼容
└── almodel-training-output/         # 训练 checkpoint 和日志
```

---

*文档创建: 2026-02-22*
*硬件: TPU v7 (Ironwood), GKE on cloud-tpu-multipod-dev*
*框架: ant-pretrain (MaxText fork)*
